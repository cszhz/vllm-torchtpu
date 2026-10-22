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
"""Composed multi-rank reference step along a resolved `Schedule`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from .blocks.gdn import GDNBlock
from .blocks.gqa import GQABlock
from .blocks.norm import NormBlock
from .decls import BlockBinding, _lookup_raw_key
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .geometry import FromHF, ModelGeometry
from .ops import BF16, F32, rnd, rup
from .pool.config import cache_exec
from .profile import Profile
from .schedule import (
    BlockOrBinding,
    FromHFLayerTypes,
    ResolvedSchedule,
    Schedule,
    _block_cls_of,
    _block_name_of,
)
from .spec import Block


def _model_label(cls: Any, model_name: str | None = None) -> str:
    if model_name:
        return model_name
    own = cls.__dict__.get("name") if isinstance(cls, type) else None
    return str(own) if own else getattr(cls, "__name__", str(cls))


def _describe_binding(
    b: BlockOrBinding | None, expected: BlockOrBinding | None = None
) -> str:
    if b is None:
        return "None"
    label = f"{_block_name_of(b)}/{_block_cls_of(b).__name__}"
    b_opts = {k: v for k, v in b.options.items() if v is not None}
    if expected is not None:
        exp_opts = {k: v for k, v in expected.options.items() if v is not None}
        diff = {
            k: (b_opts.get(k), exp_opts.get(k))
            for k in sorted(set(b_opts) | set(exp_opts))
            if b_opts.get(k) != exp_opts.get(k)
        }
        if diff:
            opts_str = ", ".join(
                f"{k}={got!r} (expected {exp!r})" for k, (got, exp) in diff.items()
            )
            return f"{label}(prefix={b.prefix!r}, {opts_str})"
    if b_opts:
        opts_str = ", ".join(f"{k}={v!r}" for k, v in sorted(b_opts.items()))
        return f"{label}(prefix={b.prefix!r}, {opts_str})"
    return f"{label}(prefix={b.prefix!r})"


def _norm_zc_and_eps(
    b: BlockOrBinding, merged: dict[str, Any] | None, default_eps: float
) -> tuple[bool, float]:
    if not issubclass(_block_cls_of(b), NormBlock):
        raise MegakernelConfigError(
            f"Expected NormBlock binding, got {_describe_binding(b)}"
        )
    zc = bool(b.options.get("zero_centered", False))
    if isinstance(b, BlockBinding) and "eps" in b.explicit_fields:
        if merged is not None:
            return zc, float(b.resolve(merged).eps)
        if isinstance(b.fields.get("eps"), (int, float)):
            return zc, float(b.fields["eps"])
    elif not isinstance(b, BlockBinding) and getattr(b, "eps", None) is not None:
        return zc, float(b.eps)
    return zc, default_eps


def _validate_norms(
    cls: Any,
    sched: ResolvedSchedule,
    merged: dict[str, Any] | None,
    eps: float,
    expected_zc: bool | None = None,
    *,
    model_name: str | None = None,
) -> bool:
    mname = _model_label(cls, model_name)
    entries: list[tuple[str, BlockOrBinding]] = []
    for slot_name, seq in (
        ("pre_norm", sched.pre_norm),
        ("post_norm", sched.post_norm),
    ):
        for i, b in enumerate(seq):
            if b is None:
                raise MegakernelConfigError(
                    f"ModelDef {mname!r}: layer {i} slot {slot_name!r} is None "
                    f"(kernel requires {slot_name} on every layer)"
                )
            entries.append((f"layer {i} slot {slot_name!r}", b))
    if cls.final_norm is None:
        last_lyr = max(0, sched.num_layers - 1)
        raise MegakernelConfigError(
            f"ModelDef {mname!r}: layer {last_lyr} slot 'final_norm' is None "
            "(kernel requires final_norm)"
        )
    entries.append(("slot 'final_norm'", cls.final_norm))
    first_loc, first_b = entries[0]
    first_zc, _ = _norm_zc_and_eps(first_b, merged, eps)
    target_zc = first_zc if expected_zc is None else expected_zc
    for loc, b in entries:
        zc_i, eps_i = _norm_zc_and_eps(b, merged, eps)
        if zc_i != target_zc:
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: {loc} ({_describe_binding(b)}) has "
                f"zero_centered={zc_i}, disagreeing with {first_loc} "
                f"zero_centered={target_zc}"
            )
        if eps_i != eps:
            raise MegakernelConfigError(
                f"ModelDef {mname!r}: {loc} ({_describe_binding(b)}) has "
                f"eps={eps_i}, disagreeing with model eps={eps}"
            )
    return target_zc


def _make_ref_block(b: BlockOrBinding | None, g: ModelGeometry) -> Block | None:
    if b is None or isinstance(b, Block):
        return b
    if not isinstance(b, BlockBinding):
        raise MegakernelConfigError(f"Expected Block or BlockBinding, got {b!r}")
    if issubclass(b.block_cls, NormBlock):
        zc, eps_val = _norm_zc_and_eps(b, None, g.eps)
        return b.block_cls(b.prefix, geometry=g, zero_centered=zc, eps=eps_val)
    opts = {k: v for k, v in b.options.items() if v is not None}
    return b.block_cls(geometry=g, prefix=b.prefix, **opts)


def composed_reference_step(
    sched: ResolvedSchedule,
    final_norm: BlockOrBinding | None,
    g: ModelGeometry,
    ex: ExecSpec,
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    *,
    paged_kv: bool | None = None,
    pool_io: bool = False,
) -> tuple[jax.Array, dict[str, Any]]:
    """Execute hybrid GQA/GDN + MoE block `reference()` along `sched`."""
    R, B, T, NS = ex.R, ex.B, ex.T, ex.NS
    L, ngdn = g.L, g.NGDN
    use_paged_kv = cache_exec(ex).paged_kv if paged_kv is None else bool(paged_kv)
    step_state: dict[str, Any] = {
        "kc": logical_states["kc"],
        "vc": logical_states["vc"],
    }
    CR, HR = rup(3 * B, 8), rup(3 + T, 8)
    vh = KD = VD = QKV = 0
    conv_in = rec_in = None
    if ngdn > 0:
        lay_gdn = GDNBlock.lin_in_layout(g, ex)
        vh, KD, VD, QKV = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD, lay_gdn.QKV
        conv_in = logical_states.get(
            "conv", jnp.zeros((R, ngdn, NS, 3, QKV), dtype=F32)
        )
        if pool_io:
            conv_in = conv_in.astype(BF16).astype(F32)
        rec_in = logical_states.get(
            "rec", jnp.zeros((R, ngdn, NS, vh, KD, VD), dtype=BF16)
        ).reshape(R, ngdn, NS, vh, KD, VD)
        step_state["conv"], step_state["rec"] = conv_in, rec_in
        step_state["conv_out"] = jnp.zeros(
            (R, ngdn, CR, QKV) if T == 1 else (R, ngdn, NS, HR, QKV), dtype=F32
        )
        step_state["rec_out"] = jnp.zeros((R, ngdn, B * vh, KD, VD), dtype=BF16)

    res = hidden[:B].astype(F32)
    for lyr in range(L):
        pre_blk = _make_ref_block(sched.pre_norm[lyr], g)
        if pre_blk is None:
            raise MegakernelConfigError(
                f"layer {lyr} slot 'pre_norm' is None (kernel requires pre_norm)"
            )
        x_mix = pre_blk.reference(logical_weights["in_norm"][lyr], res, g=g)[0]
        mix_blk = _make_ref_block(sched.mixer[lyr], g)
        if mix_blk is None:
            raise MegakernelConfigError(f"Layer {lyr} has no mixer block")
        b_idx = sched.mixer_branch_indices[lyr]
        if isinstance(mix_blk, GQABlock):
            mix_kw = {
                "pos": pos,
                "att_idx": b_idx,
                "layer_idx": lyr,
                "paged_kv": use_paged_kv,
            }
        elif isinstance(mix_blk, GDNBlock):
            mix_kw = {"gdn_idx": b_idx, "layer_idx": lyr, "multi_layer": True}
        else:
            mix_kw = {"branch_idx": b_idx, "layer_idx": lyr}
        mix_delta, step_state = mix_blk.reference(
            logical_weights, x_mix, step_state, g=g, ex=ex, **mix_kw
        )
        res = rnd(res + mix_delta)
        post_blk = _make_ref_block(sched.post_norm[lyr], g)
        if post_blk is None:
            raise MegakernelConfigError(
                f"layer {lyr} slot 'post_norm' is None (kernel requires post_norm)"
            )
        x_ffn = post_blk.reference(logical_weights["post_norm"][lyr], res, g=g)[0]
        ffn_blk = _make_ref_block(sched.ffn[lyr], g)
        if ffn_blk is None:
            raise MegakernelConfigError(f"Layer {lyr} has no ffn block")
        ffn_delta, step_state = ffn_blk.reference(
            logical_weights, x_ffn, step_state, g=g, ex=ex, layer_idx=lyr
        )
        res = rnd(res + ffn_delta)

    fn_blk = _make_ref_block(final_norm, g)
    if fn_blk is None:
        raise MegakernelConfigError(
            f"layer {max(0, L - 1)} slot 'final_norm' is None "
            "(kernel requires final_norm)"
        )
    out_b = fn_blk.reference(logical_weights["final_norm"], res, g=g, out_dtype=BF16)[0]
    out_states: dict[str, Any] = {"kc": step_state["kc"], "vc": step_state["vc"]}
    if ngdn > 0 and conv_in is not None and rec_in is not None:
        conv_out, rec_out = step_state["conv_out"], step_state["rec_out"]
        if pool_io:
            conv_out = conv_out.astype(BF16).astype(F32)
        if ex.VER:
            pad_spec = ((0, 0), (0, 0), (0, 0), (0, HR - conv_in.shape[-2]), (0, 0))
            conv_pad = (
                jnp.pad(conv_in, pad_spec)
                if conv_in.ndim == 5 and conv_in.shape[-2] < HR
                else conv_in
            )
            out_states["conv"] = conv_pad
            out_states["rec"] = rec_in.reshape(R, ngdn, NS * vh, KD, VD)
            out_states["conv_ckpt"], out_states["rec_ckpt"] = conv_out, rec_out
        else:
            out_states["conv"], out_states["rec"] = conv_out, rec_out
    return out_b, out_states


def _infer_static_layers(cls: Any, from_hf: FromHF) -> int | None:
    layers_spec = from_hf.fields.get("layers")
    if cls.default_hf_config is not None:
        key = layers_spec[0] if isinstance(layers_spec, tuple) else str(layers_spec)
        raw_l = _lookup_raw_key(cls.default_hf_config, key)
        if isinstance(raw_l, int) and not isinstance(raw_l, bool):
            return raw_l
    if isinstance(layers_spec, int) and not isinstance(layers_spec, bool):
        return layers_spec
    if (
        isinstance(layers_spec, tuple)
        and isinstance(layers_spec[1], int)
        and not isinstance(layers_spec[1], bool)
    ):
        return layers_spec[1]
    if (
        isinstance(cls.schedule, Schedule)
        and isinstance(cls.schedule.mixer, Sequence)
        and not isinstance(cls.schedule.mixer, (str, bytes))
    ):
        return len(cls.schedule.mixer)
    return None


def _validate_model_def(
    cls: Any,
    name: str,
    archs: tuple[str, ...],
    model_types: tuple[str, ...],
    base_cls: type,
) -> None:
    if not isinstance(name, str) or not name.strip() or name.strip() != name:
        raise MegakernelConfigError(f"register_model: invalid model name {name!r}")
    if not issubclass(cls, base_cls):
        raise MegakernelConfigError(
            f"register_model({name!r}): {cls.__name__} must subclass ModelDef"
        )
    from_hf = cls._from_hf_decl()
    for req_key in ("hidden", "layers", "eps"):
        if req_key not in from_hf.fields:
            raise MegakernelConfigError(
                f"register_model({name!r}): FromHF is missing required key {req_key!r}"
            )
    if not isinstance(cls.schedule, Schedule):
        raise MegakernelConfigError(
            f"register_model({name!r}): `schedule` must be a Schedule instance, "
            f"got {cls.schedule!r}"
        )
    if cls.pool_layout_override is not None and not callable(cls.pool_layout_override):
        raise MegakernelConfigError(
            f"register_model({name!r}): pool_layout_override must be callable"
        )
    for slot_name, val in (
        ("pre_norm", cls.schedule.pre_norm),
        ("post_norm", cls.schedule.post_norm),
        ("final_norm", cls.final_norm),
    ):
        if val is None:
            raise MegakernelConfigError(
                f"register_model({name!r}): layer 0 slot {slot_name!r} is None "
                f"(kernel requires {slot_name})"
            )

    static_l = _infer_static_layers(cls, from_hf)
    if static_l is not None:
        cls.schedule.validate_bounds(static_l)
        if cls.default_hf_config is not None or not isinstance(
            cls.schedule.mixer, FromHFLayerTypes
        ):
            sched = cls.schedule.resolve(cls.default_hf_config or {}, static_l)
            if cls.default_hf_config is None:
                _validate_norms(cls, sched, None, 1e-6, model_name=name)

    if cls.default_hf_config is not None:
        try:
            g_def = cls.derive_geometry(cls.default_hf_config)
            cls.pool_layout(g_def, R=8)
        except MegakernelConfigError as exc:
            if name not in str(exc):
                raise MegakernelConfigError(f"register_model({name!r}): {exc}") from exc
            raise
        except Exception as exc:
            raise MegakernelConfigError(
                f"register_model({name!r}): invalid default_hf_config: {exc}"
            ) from exc

    for alias, prof in (cls.profiles or {}).items():
        if not isinstance(alias, str) or not alias.strip() or alias.strip() != alias:
            raise MegakernelConfigError(
                f"register_model({name!r}): invalid profile alias key {alias!r}"
            )
        if not isinstance(prof, Profile):
            raise MegakernelConfigError(
                f"register_model({name!r}): profiles[{alias!r}] must be a Profile, "
                f"got {prof!r}"
            )
        if prof.model is not None and prof.model != name:
            raise MegakernelConfigError(
                f"register_model({name!r}): profile {prof.name!r} targets model "
                f"{prof.model!r}, expected {name!r}"
            )
    for kind, seq in (("hf_architectures", archs), ("hf_model_type", model_types)):
        for entry in seq:
            if not isinstance(entry, str) or not entry.strip():
                raise MegakernelConfigError(
                    f"register_model({name!r}): invalid {kind} entry {entry!r}"
                )


__all__ = ["composed_reference_step"]
