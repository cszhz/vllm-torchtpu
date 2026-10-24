# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import importlib.util
import json
from pathlib import Path


def _load_toy_proxy_server():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "disagg"
        / "toy_proxy_server.py"
    )
    spec = importlib.util.spec_from_file_location("toy_proxy_server", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_toy_proxy_server = _load_toy_proxy_server()
_AsciiSafeStreamEncoder = _toy_proxy_server._AsciiSafeStreamEncoder
_replace_prompt_with_rendered_token_ids = (
    _toy_proxy_server._replace_prompt_with_rendered_token_ids
)
_RemoteDraftAdmissionBarrier = _toy_proxy_server._RemoteDraftAdmissionBarrier
_completion_text = _toy_proxy_server._completion_text
_record_remote_draft_verification = _toy_proxy_server._record_remote_draft_verification
send_remote_dspark_request_to_prefill = (
    _toy_proxy_server.send_remote_dspark_request_to_prefill
)


def test_remote_draft_admission_barrier_releases_complete_waves() -> None:
    async def run() -> None:
        barrier = _RemoteDraftAdmissionBarrier(3)
        released: list[int] = []

        async def arrive(index: int) -> None:
            await barrier.wait()
            released.append(index)

        first_wave = [asyncio.create_task(arrive(i)) for i in range(2)]
        await asyncio.sleep(0)
        assert released == []
        first_wave.append(asyncio.create_task(arrive(2)))
        await asyncio.gather(*first_wave)
        assert sorted(released) == [0, 1, 2]

        released.clear()
        await asyncio.gather(*(arrive(i) for i in range(3, 6)))
        assert sorted(released) == [3, 4, 5]

    asyncio.run(run())


def test_remote_draft_admission_barrier_bypasses_probe() -> None:
    async def run() -> None:
        barrier = _RemoteDraftAdmissionBarrier(2, bypass_requests=1)
        for _ in range(2):
            assert await barrier.wait()

            first = asyncio.create_task(barrier.wait())
            second = asyncio.create_task(barrier.wait())
            await asyncio.sleep(0)
            assert not first.done()
            assert not second.done()

            await barrier.complete_bypass()
            assert not await first
            assert not await second

    asyncio.run(run())


def test_remote_dspark_uses_one_long_lived_raiden_request(monkeypatch) -> None:
    async def run() -> list[dict]:
        requests: list[dict] = []

        async def post(_client, _endpoint, body, _headers, _request_id, phase):
            del phase
            requests.append(body)
            return object()

        monkeypatch.setattr(_toy_proxy_server, "_post_with_retries", post)
        body = {"max_tokens": 32, "stream": True}
        await send_remote_dspark_request_to_prefill(
            {}, "/v1/completions", body, "request"
        )
        return requests

    (producer,) = asyncio.run(run())
    assert producer["max_tokens"] == 33
    assert producer["stream"] is False
    assert producer["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "_remote_dspark": True,
    }


def test_ascii_safe_stream_encoder_handles_split_utf8() -> None:
    encoder = _AsciiSafeStreamEncoder()
    text = "\u044f\U0001f600"
    message = f'data: {{"choices": [{{"text": "{text}"}}]}}\n\n'
    encoded = message.encode("utf-8")
    split_at = encoded.index(text.encode("utf-8")) + 1

    output = encoder.encode(encoded[:split_at])
    output += encoder.encode(encoded[split_at:])
    output += encoder.encode(final=True)

    assert output.isascii()
    assert b"\\u044f" in output
    assert b"\\ud83d\\ude00" in output

    payload = output.decode("ascii").removeprefix("data: ").strip()
    assert json.loads(payload)["choices"][0]["text"] == text


def test_ascii_safe_stream_encoder_leaves_ascii_unchanged() -> None:
    encoder = _AsciiSafeStreamEncoder()
    message = b'data: {"choices": [{"text": "ok"}]}\n\n'

    assert encoder.encode(message) == message
    assert encoder.encode(final=True) == b""


def test_completion_text_matches_streamed_and_unstreamed_responses() -> None:
    unstreamed = json.dumps({"choices": [{"text": "hello world"}]}).encode()
    streamed = (
        b'data: {"choices": [{"text": "hello "}]}\n\n'
        b'data: {"choices": [{"text": "world"}]}\n\n'
        b'data: {"choices": [], "usage": {"completion_tokens": 2}}\n\n'
        b"data: [DONE]\n\n"
    )

    assert _completion_text(unstreamed, streamed=False) == "hello world"
    assert _completion_text(streamed, streamed=True) == "hello world"


def test_remote_draft_verification_records_mismatch(tmp_path, monkeypatch) -> None:
    output = tmp_path / "verification.jsonl"
    monkeypatch.setenv("VLLM_TPU_REMOTE_DSPARK_VERIFICATION_PATH", str(output))

    _record_remote_draft_verification(
        "request-1", shadow_text="shadow", decode_text="decode"
    )

    record = json.loads(output.read_text())
    assert record["request_id"] == "request-1"
    assert record["status"] == "mismatch"
    assert record["shadow_length"] == 6
    assert record["decode_length"] == 6
    assert record["shadow_sha256"] != record["decode_sha256"]


def test_replace_prompt_with_rendered_token_ids_single_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
        "prompt_embeds": b"unused-after-render",
    }
    rendered = [{"token_ids": [1, 2, 3]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [1, 2, 3]
    assert "prompt_embeds" not in converted
    assert request["prompt"] == "hello"


def test_replace_prompt_with_rendered_token_ids_multi_prompt() -> None:
    request = {
        "model": "test-model",
        "prompt": ["hello", "world"],
    }
    rendered = [{"token_ids": [1, 2, 3]}, {"token_ids": [4, 5]}]

    converted = _replace_prompt_with_rendered_token_ids(request, rendered)

    assert converted["prompt"] == [[1, 2, 3], [4, 5]]


def test_replace_prompt_with_rendered_token_ids_rejects_bad_response() -> None:
    request = {
        "model": "test-model",
        "prompt": "hello",
    }

    try:
        _replace_prompt_with_rendered_token_ids(request, [{"token_ids": ["1"]}])
    except ValueError as exc:
        assert "token_ids" in str(exc)
    else:
        raise AssertionError("malformed render response should fail")
