#!/usr/bin/env python3
"""One HTTP port per vLLM DP rank in front of a single-port DP vLLM server, for llm-d's data-parallel mode.

llm-d (EPP + routing sidecar) treats every (pod, port) of an InferencePool as an endpoint and expects one model-server
port per DP rank. vLLM with --data-parallel-size N and the internal load balancer has one API port and pins a request
to a rank by the `X-data-parallel-rank` header. This shim listens on BASE + i for i in 0..N-1 and:
  - forwards every request to UPSTREAM with `X-data-parallel-rank: i` (bodies streamed back, SSE included);
  - answers GET /metrics with UPSTREAM's metrics restricted to rank i (series with engine="i", plus series without an
    engine label), so the EPP sees each rank's own queue / KV usage.

--reuse-prompt-ids (on P): the toy proxy's PROXY_REUSE_PROMPT_IDS. For a prefill request from the llm-d sidecar
(kv_transfer_params.do_remote_decode), ask P for the prompt token ids (`return_token_ids`) and copy them into the
response's kv_transfer_params["prompt_token_ids"]. The sidecar (v1.0.0, nixlv2) hands P's kv_transfer_params to D
as-is, so D reuses the ids instead of templating and tokenizing the whole conversation again on the TTFT path.

--decode-router (on D): the toy proxy's PROXY_DECODE_ROUTING, ported. EPP can only add a prefix score and a load score;
the toy rule is a threshold ("the rank holding the longest prefix, unless it runs more than SLACK streams above the
idlest"), which keeps both the D prefix hit (76%) and the lockstep balance (busiest/idlest 13.0/9.7 at AgentX c512).
With it the shim picks the D rank itself for every completion request and overrides the port's rank; every D request
passes through this one process, so its in-flight counts are exact. Same block hashing as the toy proxy
(_PrefixRouter._text, 2048-char blake2b chain), same LRU size per rank (PROXY_DECODE_RANK_TOKENS 8874043), slack 2.

Packed ids: P's ids cross the Go sidecar, which runs encoding/json over every response and request it relays; a JSON
array of ~10^5 integers is its slowest input (the llm-d path's TTFT gap to the toy proxy grew with request size: +0.04 s
at p25, +0.19 s at p90). The P shim therefore hands them over as kv_transfer_params["prompt_token_ids_b64"] (int32
little-endian, base64), an opaque string to the sidecar, and the D shim turns it back into "prompt_token_ids".

Connection handling (the first version returned a handful of aiohttp 500s per AgentX c512 hour): vLLM closes idle
keep-alive connections after 5 s, so upstream connections idle for 3 s are dropped and a request that loses its
connection before any response byte is retried once; the listener keeps idle client connections for 120 s, longer
than the Go sidecar's 90 s idle timeout (the RST-on-reuse case the llm-d P/D guide warns about).

  python3 dp_rank_shim.py --upstream http://127.0.0.1:8100 --base-port 8110 --ranks 8 [--reuse-prompt-ids]
"""
import argparse
import asyncio
import base64
import collections
import hashlib
import re
import sys
import traceback

import aiohttp
import numpy as np
from aiohttp import web

try:
    import orjson
    loads, dumps = orjson.loads, orjson.dumps
except ImportError:  # pragma: no cover
    import json
    loads, dumps = json.loads, lambda o: json.dumps(o).encode()

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length", "content-encoding"}
ENGINE = re.compile(r'engine="(\d+)"')
RETRYABLE = (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, aiohttp.ClientConnectorError)
STATS = {"requests": 0, "retries": 0, "errors": 0, "ids_reused": 0, "d_routed": 0, "d_moved": 0}


class DecodeRouter:
    """toy_proxy_server._DecodeRouter (PROXY_DECODE_ROUTING=1), same hashing / capacity / slack."""
    BLOCK_CHARS = 2048
    CHARS_PER_TOKEN = 3.5

    def __init__(self, dp: int, rank_tokens: int = 8874043, slack: int = 2):
        self.dp, self.slack = dp, slack
        self.capacity = int(rank_tokens * self.CHARS_PER_TOKEN / self.BLOCK_CHARS)
        self.active = [0] * dp
        self.caches = [collections.OrderedDict() for _ in range(dp)]

    @staticmethod
    def text(req: dict):
        messages = req.get("messages")
        if isinstance(messages, list) and messages:
            parts = []
            if req.get("tools"):
                parts.append(dumps(req["tools"]).decode())
            for m in messages:
                if not isinstance(m, dict):
                    continue
                c = m.get("content", "")
                parts.append(f"\x00{m.get('role')}\x00")
                parts.append(c if isinstance(c, str) else dumps(c).decode())
                if m.get("tool_calls"):
                    parts.append(dumps(m["tool_calls"]).decode())
            return "".join(parts)
        prompt = req.get("prompt")
        return prompt if isinstance(prompt, str) else None

    def hashes(self, req: dict):
        t = self.text(req)
        if t is None:
            return None
        raw = t.encode("utf-8", "surrogatepass")
        step, prev, out = self.BLOCK_CHARS, b"", []
        for i in range(0, len(raw) - step + 1, step):
            prev = hashlib.blake2b(prev + raw[i:i + step], digest_size=8).digest()
            out.append(prev)
        return out

    def assign(self, hashes) -> int:
        matches = []
        for r in range(self.dp):
            cache, n = self.caches[r], 0
            for h in hashes:
                if h not in cache:
                    break
                n += 1
            matches.append(n)
        least = min(range(self.dp), key=lambda r: (self.active[r], -matches[r]))
        home = max(range(self.dp), key=lambda r: (matches[r], -self.active[r]))
        if matches[home] == 0:
            best = least
        elif self.active[home] - self.active[least] > self.slack:
            best = least
            STATS["d_moved"] += 1
        else:
            best = home
        cache = self.caches[best]
        for h in hashes:
            cache[h] = None
            cache.move_to_end(h)
        while len(cache) > self.capacity:
            cache.popitem(last=False)
        self.active[best] += 1
        STATS["d_routed"] += 1
        return best

    def release(self, rank: int) -> None:
        self.active[rank] = max(0, self.active[rank] - 1)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def make_app(upstream: str, rank: int, holder: dict, reuse_ids: bool, drouter) -> web.Application:
    async def metrics(request: web.Request) -> web.Response:
        async with holder["s"].get(upstream + "/metrics") as r:
            text = await r.text()
        keep = [l for l in text.splitlines() if (m := ENGINE.search(l)) is None or int(m.group(1)) == rank]
        return web.Response(text="\n".join(keep) + "\n", content_type="text/plain")

    async def proxy(request: web.Request) -> web.StreamResponse:
        STATS["requests"] += 1
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        headers["X-data-parallel-rank"] = str(rank)
        body = await request.read()
        d_rank = None
        if request.method == "POST" and body and request.path.endswith("completions") and b"prompt_token_ids_b64" in body:
            try:  # D side: unpack the ids the P shim packed (see module docstring)
                req = loads(body)
                kv = req.get("kv_transfer_params")
                if isinstance(kv, dict) and "prompt_token_ids_b64" in kv:
                    kv["prompt_token_ids"] = np.frombuffer(base64.b64decode(kv.pop("prompt_token_ids_b64")), dtype="<i4")
                    body = orjson.dumps(req, option=orjson.OPT_SERIALIZE_NUMPY)
                    STATS["ids_unpacked"] = STATS.get("ids_unpacked", 0) + 1
            except Exception as e:
                log(f"rank {rank}: unpack failed {type(e).__name__}: {e}")
        if drouter is not None and request.method == "POST" and body and request.path.endswith("completions"):
            try:
                hs = drouter.hashes(loads(body))
                if hs is not None:
                    d_rank = drouter.assign(hs)
                    headers["X-data-parallel-rank"] = str(d_rank)
            except Exception:
                d_rank = None
        try:
            return await forward(request, headers, body)
        finally:
            if d_rank is not None:
                drouter.release(d_rank)

    async def forward(request, headers, body):
        rewrite = False
        if reuse_ids and request.method == "POST" and body and request.path.endswith("completions"):
            try:
                req = loads(body)
                kv = req.get("kv_transfer_params")
                if isinstance(kv, dict) and kv.get("do_remote_decode") and not req.get("stream"):
                    req["return_token_ids"] = True
                    body, rewrite = dumps(req), True
            except Exception:  # not JSON: forward untouched
                pass
        for attempt in (0, 1):
            try:
                async with holder["s"].request(request.method, upstream + request.rel_url.path_qs,
                                               headers=headers, data=body or None) as r:
                    if rewrite and r.status == 200:
                        data = await r.read()
                        try:
                            resp = loads(data)
                            kv, ids = resp.get("kv_transfer_params"), resp.get("prompt_token_ids")
                            if isinstance(kv, dict) and ids:
                                kv["prompt_token_ids_b64"] = base64.b64encode(np.asarray(ids, dtype="<i4").tobytes()).decode()
                                # The sidecar forwards only kv_transfer_params; dropping the top-level copy halves the
                                # response it has to parse (~150k ids at AgentX lengths).
                                del resp["prompt_token_ids"]
                                STATS["ids_reused"] += 1
                            data = dumps(resp)
                        except Exception:
                            pass
                        out = web.Response(body=data, status=r.status, content_type="application/json")
                        return out
                    out = web.StreamResponse(status=r.status, reason=r.reason)
                    for k, v in r.headers.items():
                        if k.lower() not in HOP:
                            out.headers[k] = v
                    await out.prepare(request)
                    async for chunk in r.content.iter_any():
                        await out.write(chunk)
                    await out.write_eof()
                    return out
            except RETRYABLE as e:
                # Nothing has been sent to the client yet (the response object is created inside the block only
                # after the upstream answered), so one retry on a fresh connection is safe.
                if attempt == 0:
                    STATS["retries"] += 1
                    log(f"rank {rank}: retry after {type(e).__name__}: {e}")
                    continue
                STATS["errors"] += 1
                log(f"rank {rank}: upstream error {type(e).__name__}: {e}")
                return web.json_response({"error": f"dp_rank_shim upstream: {type(e).__name__}: {e}"}, status=502)
            except Exception as e:
                STATS["errors"] += 1
                log(f"rank {rank}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                raise

    app = web.Application(client_max_size=1 << 30)   # AgentX bodies carry up to ~10^6-token prompts
    app.router.add_get("/metrics", metrics)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


async def report() -> None:
    while True:
        await asyncio.sleep(300)
        log(f"dp_rank_shim stats {STATS}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--base-port", type=int, required=True)
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--reuse-prompt-ids", action="store_true")
    ap.add_argument("--decode-router", action="store_true", help="pick the D rank like the toy proxy (D side)")
    ap.add_argument("--decode-slack", type=int, default=2)
    ap.add_argument("--decode-rank-tokens", type=int, default=8874043)
    a = ap.parse_args()
    # One router for all ports: it sees every D request, so its in-flight counts are exact.
    drouter = DecodeRouter(a.ranks, a.decode_rank_tokens, a.decode_slack) if a.decode_router else None
    holder = {"s": aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
        connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=3))}
    for i in range(a.ranks):
        runner = web.AppRunner(make_app(a.upstream.rstrip("/"), i, holder, a.reuse_prompt_ids, drouter),
                               access_log=None, keepalive_timeout=120)
        await runner.setup()
        await web.TCPSite(runner, a.host, a.base_port + i).start()
    print(f"dp_rank_shim: ports {a.base_port}-{a.base_port + a.ranks - 1} -> {a.upstream} "
          f"(X-data-parallel-rank{', reuse prompt ids' if a.reuse_prompt_ids else ''}"
          f"{f', toy decode routing slack {a.decode_slack}' if drouter else ''})", flush=True)
    asyncio.create_task(report())
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
