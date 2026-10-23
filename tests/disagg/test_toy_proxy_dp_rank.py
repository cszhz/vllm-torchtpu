# SPDX-License-Identifier: Apache-2.0
"""PROXY_DECODE_ROUTING=dp_affinity: prefix-affinity DP-rank routing with a KV-aware guard."""

import importlib.util
from pathlib import Path

import pytest


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2] / "examples" / "disagg" / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server_dp", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_m = _load_toy_proxy_server()
_DPRankRouter = _m._DPRankRouter

CHUNK = 64


def _router(**kw) -> "_DPRankRouter":
    args = dict(n_hosts=1, ranks=4, chunk=CHUNK, min_frac=0.5, kv_slack_frac=0.10,
                count_slack=1.0, fallback_count_slack=1, host_slack=8)
    args.update(kw)
    return _DPRankRouter(**args)


def _conv(tag: str, n_chunks: int) -> str:
    # n_chunks full chunks of conversation-specific text
    return "".join(f"{tag}-{i:04d}".ljust(CHUNK, ".") for i in range(n_chunks))


def _route(r, text, tokens, rid, host=0):
    hs = r.hashes(text)
    return r.choose_rank(host, hs, tokens, rid)


def _fill_equal(r, n_per_rank=2, tokens=1000):
    """Puts n_per_rank unrelated requests of equal size on every rank."""
    k = 0
    for _ in range(n_per_rank):
        for _ in range(r.ranks):
            rank, why = _route(r, _conv(f"filler{k}", 8), tokens, f"f{k}")
            k += 1
    assert r.count[0] == [n_per_rank] * r.ranks


def test_next_turn_sticks_to_the_rank_of_the_previous_turn() -> None:
    r = _router()
    _fill_equal(r)
    t1 = _conv("A", 8)
    rank1, why1 = _route(r, t1, 1000, "a1")
    assert why1 == "cold"
    r.done("a1")
    # the next turn extends the previous prompt
    rank2, why2 = _route(r, t1 + _conv("A-turn2", 4), 1000, "a2")
    assert (rank2, why2) == (rank1, "sticky")


def test_cold_request_goes_to_the_least_resident_kv_rank() -> None:
    r = _router()
    for rank_tokens, rid in ((5000, "x0"), (1000, "x1"), (3000, "x2"), (4000, "x3")):
        _route(r, _conv(rid, 8), rank_tokens, rid)
    # every rank holds one request; rank of x1 holds the fewest tokens
    assert sorted(map(sum, [r.count[0]])) == [4]
    want = r.tokens[0].index(min(r.tokens[0]))
    rank, why = _route(r, _conv("cold", 8), 100, "c")
    assert (rank, why) == (want, "cold")


def test_kv_guard_moves_the_turn_off_a_kv_heavy_rank() -> None:
    r = _router(kv_slack_frac=0.10)
    _fill_equal(r, tokens=1000)
    t1 = _conv("B", 8)
    rank1, _ = _route(r, t1, 1000, "b1")
    r.done("b1")
    # a long unrelated context lands on rank1 -> rank1 far above the per-rank mean
    r.tokens[0][rank1] += 50_000
    rank2, why2 = _route(r, t1 + _conv("B2", 2), 1000, "b2")
    assert why2 == "guard_kv"
    assert rank2 != rank1


def test_kv_guard_allows_a_rank_within_the_slack() -> None:
    r = _router(kv_slack_frac=0.10)
    _fill_equal(r, tokens=1000)
    t1 = _conv("B", 8)
    rank1, _ = _route(r, t1, 1000, "b1")
    r.done("b1")
    # every rank holds 2000 tokens; +200 on rank1 -> 2200 vs 1.1 x mean 2255: inside the slack
    assert r.tokens[0] == [2000.0] * 4
    r.tokens[0][rank1] += 200
    rank2, why2 = _route(r, t1 + _conv("B2", 2), 1000, "b2")
    assert (rank2, why2) == (rank1, "sticky")


def test_count_guard_moves_the_turn_off_a_crowded_rank() -> None:
    r = _router(count_slack=1.0, kv_slack_frac=1e9)
    t1 = _conv("C", 8)
    rank1, _ = _route(r, t1, 1000, "c1")
    r.done("c1")
    # three small extra requests pinned to rank1 (count 3 vs mean 0.75 + 1)
    for i in range(3):
        r.count[0][rank1] += 1
    rank2, why2 = _route(r, t1 + _conv("C2", 2), 1000, "c2")
    assert why2 == "guard_count"
    assert rank2 != rank1


def test_fallback_respects_the_count_slack() -> None:
    r = _router(fallback_count_slack=1)
    # rank 0: 1 request with few tokens is fine; rank 3: no tokens but 3 requests -> excluded
    r.count[0] = [1, 2, 2, 3]
    r.tokens[0] = [9000.0, 8000.0, 7000.0, 0.0]
    rank, why = _route(r, _conv("D", 8), 100, "d")
    # candidates: count <= min(1)+1 = 2 -> ranks 0,1,2; least tokens among them is rank 2
    assert (rank, why) == (2, "cold")


def test_done_releases_count_and_tokens_and_is_idempotent() -> None:
    r = _router()
    rank, _ = _route(r, _conv("E", 8), 1234, "e")
    assert r.count[0][rank] == 1 and r.tokens[0][rank] == 1234
    r.done("e")
    assert r.count[0][rank] == 0 and r.tokens[0][rank] == 0
    assert r.host_unfinished == [0]
    r.done("e")
    assert r.count[0][rank] == 0 and r.host_unfinished == [0]


def test_short_shared_prefix_is_not_an_owner() -> None:
    r = _router(min_frac=0.5)
    shared = _conv("SYS", 2)
    rank1, _ = _route(r, shared + _conv("F", 2), 1000, "f1")
    # a new conversation sharing only 2 of 10 chunks: below min_frac -> cold
    rank2, why2 = _route(r, shared + _conv("G", 8), 1000, "g1")
    assert why2 == "cold"


def test_host_affinity_with_slack_and_overload_fallback() -> None:
    r = _router(n_hosts=3, host_slack=2)
    t1 = _conv("H", 8)
    h = r.choose_host(r.hashes(t1), "h1")
    assert h == 0  # all empty: lowest index
    rank1, _ = r.choose_rank(h, r.hashes(t1), 1000, "h1")
    r.done("h1")
    # owner host 0 busier than the others but within mean + slack (2 <= 0.67 + 2) -> host 0
    r.host_unfinished = [2, 0, 0]
    assert r.choose_host(r.hashes(t1 + _conv("H2", 2)), "h2") == 0
    r.done("h2")
    # overload host 0 beyond mean + slack -> least loaded host
    r.host_unfinished = [9, 1, 0]
    assert r.choose_host(r.hashes(t1 + _conv("H3", 2)), "h3") == 2


def test_turn_on_another_host_is_not_sticky() -> None:
    r = _router(n_hosts=2)
    t1 = _conv("I", 8)
    r.choose_host(r.hashes(t1), "i1")
    r.choose_rank(0, r.hashes(t1), 1000, "i1")
    r.done("i1")
    rank, why = r.choose_rank(1, r.hashes(t1 + _conv("I2", 2)), 1000, "i2")
    assert why == "other_host"


def test_owner_map_is_bounded() -> None:
    r = _router(max_owners=10)
    _route(r, _conv("J", 8), 1, "j1")
    _route(r, _conv("K", 8), 1, "k1")
    assert len(r.owners) == 10


def test_choose_rank_twice_raises() -> None:
    r = _router()
    _route(r, _conv("L", 8), 1, "l1")
    with pytest.raises(ValueError):
        _route(r, _conv("L", 8), 1, "l1")


def test_decode_headers_rank_only_on_decode_and_client_header_wins() -> None:
    fwd = {}
    assert _m._decode_headers_with_rank(fwd, 3) == {"X-data-parallel-rank": "3"}
    assert fwd == {}  # the prefill hop's dict is not mutated
    assert _m._decode_headers_with_rank({"X-data-parallel-rank": "1"}, 3) == {
        "X-data-parallel-rank": "1"
    }
    assert _m._decode_headers_with_rank({}, None) == {}


def test_gate_is_off_by_default_and_on_only_for_dp_affinity(monkeypatch) -> None:
    monkeypatch.delenv("PROXY_DECODE_ROUTING", raising=False)
    assert _m._make_dp_router(1) is None
    for v in ("round_robin", "llmd", "prefix_hash"):
        monkeypatch.setenv("PROXY_DECODE_ROUTING", v)
        assert _m._make_dp_router(1) is None
    monkeypatch.setenv("PROXY_DECODE_ROUTING", "dp_affinity")
    monkeypatch.setenv("PROXY_DP_RANKS", "4")
    r = _m._make_dp_router(3)
    assert isinstance(r, _DPRankRouter) and r.ranks == 4 and r.n_hosts == 3
    assert r.kv_slack_frac == pytest.approx(0.10) and r.count_slack == 1.0
    assert r.fallback_count_slack == 1


# ---------------------------------------------------------------- request flow, with fake backends
def _flow(monkeypatch, routing: str | None, turns: list[str]):
    """Runs chat requests through the real proxy app with httpx MockTransport backends.
    Returns (prefill request headers, decode request headers, app.state)."""
    import argparse

    import httpx
    from fastapi.testclient import TestClient

    if routing is None:
        monkeypatch.delenv("PROXY_DECODE_ROUTING", raising=False)
    else:
        monkeypatch.setenv("PROXY_DECODE_ROUTING", routing)
    monkeypatch.setenv("PROXY_LATENCY_LOG_INTERVAL", "0")
    monkeypatch.setenv("PROXY_DP_PREFIX_CHUNK", str(CHUNK))
    _m.global_args = argparse.Namespace(
        prefiller_instances=[("p0", 8000)], decoder_instances=[("d0", 7200)]
    )
    seen_p, seen_d = [], []

    def prefill(req: httpx.Request) -> httpx.Response:
        seen_p.append(dict(req.headers))
        return httpx.Response(200, json={"kv_transfer_params": {"uuid": len(seen_p), "num_tokens": 99},
                                         "usage": {"prompt_tokens": 100}})

    def decode(req: httpx.Request) -> httpx.Response:
        seen_d.append(dict(req.headers))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    with TestClient(_m.app) as client:
        st = _m.app.state
        st.prefill_clients[0]["client"] = httpx.AsyncClient(
            transport=httpx.MockTransport(prefill), base_url="http://p0:8000")
        st.decode_clients[0]["client"] = httpx.AsyncClient(
            transport=httpx.MockTransport(decode), base_url="http://d0:7200")
        for text in turns:
            r = client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 4})
            assert r.status_code == 200, r.text
        return seen_p, seen_d, st


def test_flow_dp_affinity_pins_decode_rank_and_never_prefill(monkeypatch) -> None:
    t1 = _conv("FLOW", 40)
    seen_p, seen_d, st = _flow(monkeypatch, "dp_affinity", [t1, t1 + _conv("FLOW2", 4)])
    ranks = [h.get("x-data-parallel-rank") for h in seen_d]
    assert all(r is not None and 0 <= int(r) < 4 for r in ranks)
    assert ranks[0] == ranks[1]  # second turn sticks
    assert all("x-data-parallel-rank" not in h for h in seen_p)
    s = st.dp_router.stats()
    assert s["routed"] == 2 and s["sticky"] == 1
    assert s["host_unfinished"] == [0] and s["count"] == [[0, 0, 0, 0]]


def test_flow_default_sends_no_rank_header(monkeypatch) -> None:
    seen_p, seen_d, st = _flow(monkeypatch, None, [_conv("FLOW", 40)])
    assert st.dp_router is None
    assert all("x-data-parallel-rank" not in h for h in seen_p + seen_d)
