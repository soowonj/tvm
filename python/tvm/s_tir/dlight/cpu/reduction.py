# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""CPU reduction rule for operators including softmax, layer norm, RMS norm, etc."""

import logging

from tvm import DataType, s_tir, tirx
from tvm.target import Target
from tvm.target.codegen import llvm_get_vector_width, target_has_features

from ..analysis import normalize_prim_func
from ..base import get_extent
from .base import CPUScheduleRule

logger = logging.getLogger(__name__)


def _get_num_leading_s(dom_kind: str) -> int:
    """Count leading spatial ('S') axes in a dom_kind string."""
    return len(dom_kind) - len(dom_kind.lstrip("S"))


def _detect_reduction_op(sch: "s_tir.Schedule", block_rv) -> str | None:
    """Inspect a reduction block's body to identify its reduction operator.

    Returns one of ``"max"``, ``"sum"``, or ``None`` if the operator is not
    one we can map onto a single RVV reduction intrinsic.

    Works by post-order walking every ``BufferStore`` in the block and
    examining the right-hand side: a ``tir.max`` call indicates max
    reduction; an ``Add`` whose operands include the same buffer being
    written indicates sum reduction.
    """
    block_stmt = sch.get(block_rv)
    found = {"op": None}

    def _is_self_load(operand, buf):
        return isinstance(operand, tirx.BufferLoad) and operand.buffer.same_as(buf)

    def _visit(stmt):
        if found["op"] is not None:
            return
        if not isinstance(stmt, tirx.BufferStore):
            return
        value = stmt.value
        # Strip an outer Cast if present.
        if isinstance(value, tirx.Cast):
            value = value.value
        same_buf = stmt.buffer
        # T.max(self, ...) / T.min(self, ...) lower to tirx.Max/tirx.Min binops
        # rather than generic Call nodes. Check both forms.
        if isinstance(value, tirx.Max):
            if _is_self_load(value.a, same_buf) or _is_self_load(value.b, same_buf):
                found["op"] = "max"
            return
        if isinstance(value, tirx.Min):
            # vfredmin would handle this; not enabled yet.
            if _is_self_load(value.a, same_buf) or _is_self_load(value.b, same_buf):
                found["op"] = "min"
            return
        # Defensive: also accept the Call form in case some frontend emits it.
        if isinstance(value, tirx.Call):
            op_name = getattr(value.op, "name", "")
            if op_name == "tir.max":
                found["op"] = "max"
                return
            if op_name == "tir.min":
                found["op"] = "min"
                return
        # Add(load(C), load(A)) or Add(load(A), load(C)) — sum accumulator
        if isinstance(value, tirx.Add):
            for operand in (value.a, value.b):
                if _is_self_load(operand, same_buf):
                    found["op"] = "sum"
                    return

    tirx.stmt_functor.post_order_visit(block_stmt, _visit)
    return found["op"]


def _rvv_supported(target: Target) -> bool:
    """True iff we can emit vfredmax.vs / vfredusum.vs on this target."""
    try:
        return bool(target_has_features("v", target))
    except Exception:  # pylint: disable=broad-except
        return False


class Reduction(CPUScheduleRule):
    """CPU reduction rule for softmax, layer norm, RMS norm, and similar operators.

    Targets patterns with a mix of reduction (SR) and injective (SS) blocks,
    where all blocks share the same leading spatial axes.
    Example: softmax = maxelem(SR) -> exp(SS) -> expsum(SR) -> norm(SS).

    Schedule strategy:
      1. Parallelize leading spatial axes (batch dimension).
      2. Move all blocks under the spatial loop via compute_at.
      3. Vectorize injective blocks (exp, delta, norm) on their inner axis.
      4. Reduction blocks:
           - On RVV (`+v`) targets, tensorize the inner reduction chunk into
             a single ``vfredmax.vs`` / ``vfredusum.vs`` instruction via the
             intrinsics registered in ``s_tir.tensor_intrin.riscv_cpu``.
             This bypasses the rfactor first-child constraint discussed
             below.
           - Otherwise, split + annotate the inner axis for LLVM unrolling,
             preventing harmful full-unroll by the backend.

    Note: vectorized reduction via rfactor is not used here because TVM's
    rfactor primitive requires the reduction block to be the first child of
    its enclosing loop, which is incompatible with compute_at when multiple
    blocks share the same spatial loop. The RVV tensorize path side-steps
    this constraint by replacing the inner chunk with a hardware reduction
    intrinsic that consumes a vector and an initial scalar accumulator.
    """

    def apply(  # pylint: disable=too-many-locals,too-many-return-statements,too-many-branches
        self,
        func: tirx.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tirx.PrimFunc) or not self.is_target_available(target):
            return None

        sch = s_tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        if block_infos is None or len(block_infos) < 2:
            return None

        # Must have at least one reduction block and last block must be injective.
        if not any(not bi.is_injective() for bi in block_infos):
            return None
        if not block_infos[-1].is_injective():
            return None

        # Every block must start with at least one spatial axis, and all blocks
        # must agree on the minimum number of leading spatial axes.
        num_leading_s = None
        for bi in block_infos:
            dk = bi.dom_kind()
            if not dk or dk[0] != "S":
                return None
            n = _get_num_leading_s(dk)
            num_leading_s = n if num_leading_s is None else min(num_leading_s, n)
        if not num_leading_s:
            return None

        # Infer dtype from the last block's write buffer.
        last_block_stmt = sch.get(block_infos[-1].block_rv)
        dtype_bits = (
            DataType(last_block_stmt.writes[0].buffer.dtype).bits if last_block_stmt.writes else 32
        )

        # Determine vector lanes from target VLEN.
        vlen_bits = llvm_get_vector_width(target)
        if vlen_bits <= 0:
            vlen_bits = 128
        vec_lanes = max(vlen_bits // dtype_bits, 2)

        # --- Phase 1: Parallelize spatial on the last block ---
        last_block = block_infos[-1]
        loops = sch.get_loops(last_block.block_rv)
        if num_leading_s > 1:
            spatial = sch.fuse(*loops[:num_leading_s])
        else:
            spatial = loops[0]
        sch.parallel(spatial)

        # --- Phase 2: Vectorize the last (injective) block ---
        self._vectorize_inner(sch, last_block.block_rv, vec_lanes)

        # --- Phase 3: compute_at all preceding blocks under spatial ---
        for block_info in reversed(block_infos[:-1]):
            sch.compute_at(block_info.block_rv, spatial, preserve_unit_loops=True)

        # --- Phase 4: Vectorize injective, schedule reduction blocks ---
        rvv_ok = _rvv_supported(target)
        for block_info in block_infos[:-1]:
            if block_info.is_injective():
                self._vectorize_inner(sch, block_info.block_rv, vec_lanes)
            else:
                self._schedule_reduction_inner(
                    sch, block_info.block_rv, vec_lanes, dtype_bits, target, rvv_ok
                )

        return sch

    @staticmethod
    def _vectorize_inner(sch, block_rv, vec_lanes):
        """Split the innermost loop to vec_lanes and vectorize."""
        block_loops = sch.get_loops(block_rv)
        if len(block_loops) <= 1:
            return
        inner = block_loops[-1]
        extent = get_extent(sch, inner)
        if isinstance(extent, int):
            if extent > vec_lanes:
                _, vec_loop = sch.split(inner, factors=[None, vec_lanes])
                sch.vectorize(vec_loop)
            elif extent >= 2:
                sch.vectorize(inner)
        else:
            _, vec_loop = sch.split(inner, factors=[None, vec_lanes])
            sch.vectorize(vec_loop)

    @staticmethod
    def _schedule_reduction_inner(sch, block_rv, vec_lanes, dtype_bits, target, rvv_ok):
        """Schedule a reduction block's inner axis.

        Splits the inner axis once into ``[outer, vec_lanes-chunk]`` and then:
          * If an RVV reduction intrinsic is registered and matches this
            block's reduction op + dtype, ``tensorize`` the chunk.
          * Otherwise (or if tensorize raises), fall back to the existing
            ``pragma_auto_unroll_max_step`` annotation that prevents
            harmful LLVM full-unroll on RVV targets.

        Doing the split unconditionally keeps the schedule transactional —
        the chunk loop is always a valid loop to either tensorize or
        annotate, even if intrinsic resolution fails partway through.
        """
        block_loops = sch.get_loops(block_rv)
        if len(block_loops) <= 1:
            return
        inner = block_loops[-1]
        extent = get_extent(sch, inner)
        if isinstance(extent, int) and extent <= vec_lanes:
            # Whole reduction fits in one vector; nothing to chunk.
            return

        outer, vec_loop = sch.split(inner, factors=[None, vec_lanes])

        # Resolve a registered RVV reduction intrinsic for this block (read-only).
        intrin_name = (
            Reduction._resolve_rvv_reduction_intrin(
                sch, block_rv, vec_lanes, dtype_bits, target
            )
            if rvv_ok
            else None
        )

        if intrin_name is not None:
            try:
                # Peel off the init clause as a separate block so the update
                # block matches the update-only desc registered for the
                # intrinsic.  decompose_reduction at the chunk-outer loop
                # places the init just inside the spatial loop, so it runs
                # once per batch element.
                sch.decompose_reduction(block_rv, outer)
                sch.tensorize(vec_loop, intrin_name)
                return
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug(
                    "RVV reduction tensorize failed for %s: %s; falling back to unroll",
                    intrin_name,
                    exc,
                )
                # Fall through to annotation path on the same vec_loop.

        sch.annotate(vec_loop, ann_key="pragma_auto_unroll_max_step", ann_val=vec_lanes)
        sch.annotate(vec_loop, ann_key="pragma_unroll_explicit", ann_val=1)

    @staticmethod
    def _resolve_rvv_reduction_intrin(sch, block_rv, vec_lanes, dtype_bits, target):
        """Return a registered RVV reduction intrinsic name for this block, or None.

        Returns None if any precondition fails — non-FP dtype, dtype-bits
        mismatch with the schedule's chunk size, unrecognised reduction
        op, intrinsic registration failure, or kernel-not-found.
        """
        op = _detect_reduction_op(sch, block_rv)
        if op not in ("max", "sum"):
            return None

        block_stmt = sch.get(block_rv)
        if not block_stmt.writes:
            return None
        dtype_obj = DataType(block_stmt.writes[0].buffer.dtype)
        if not str(dtype_obj).startswith("float"):
            return None
        # Require the schedule's vec_lanes to match the registered chunk size
        # for this dtype (n_elems = VLEN/SEW under LMUL=1).
        if dtype_obj.bits != dtype_bits:
            return None

        try:
            from tvm.s_tir.tensor_intrin.riscv_cpu import (
                _rvv_reduce_kernel_name,
                register_riscv_intrinsics,
            )
        except ImportError:
            return None
        try:
            register_riscv_intrinsics(target)
        except Exception:  # pylint: disable=broad-except
            return None

        intrin_name = _rvv_reduce_kernel_name(vec_lanes, str(dtype_obj), op)
        if s_tir.TensorIntrin.get(intrin_name, allow_missing=True) is None:
            return None
        return intrin_name
