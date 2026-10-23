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
"""The fused EP MoE routing exchange gives the same [ep, width] either way."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.v2 import layer


@pytest.mark.parametrize("width", [128, 1280])
def test_all_to_all_matches_all_gather(monkeypatch, width):
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("needs 4 devices (--xla_force_host_platform_device_count=4)")
    mesh = Mesh(np.asarray(devices[:4]), axis_names=("d",))
    ep = mesh.size
    payload = np.random.default_rng(0).integers(0, 1 << 20, (ep * width,), np.int32)
    x = jax.device_put(jnp.asarray(payload), NamedSharding(mesh, P("d")))

    def run(mode):
        monkeypatch.setattr(envs, "MOE_ROUTING_EXCHANGE", mode, raising=False)
        out = jax.jit(
            jax.shard_map(
                lambda p: layer._exchange_routing(p, "d", ep, width)[None],
                mesh=mesh,
                in_specs=P("d"),
                out_specs=P("d", None, None),
                check_vma=False,
            )
        )(x)
        return np.asarray(out)

    gathered, direct = run("all_gather"), run("all_to_all")
    np.testing.assert_array_equal(gathered, direct)
    # Every shard sees every shard's payload in mesh order.
    for shard in range(ep):
        np.testing.assert_array_equal(gathered[shard], payload.reshape(ep, width))
