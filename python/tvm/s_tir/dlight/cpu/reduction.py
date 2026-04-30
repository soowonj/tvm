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
from tvm.target.codegen import llvm_get_vector_width
from tvm.target.codegen import target_has_features

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

    def _visit(stmt):
        if found["op"] is not None:
            return
        if not isinstance(stmt, tirx.BufferStore):
            return
        value = stmt.value
        # Strip an outer Cast if present.
        if isinstance(value, tirx.Cast):
            value = value.value
        # tir.max(...) call
        if isinstance(value, tirx.Call):
            op_name = getattr(value.op, "name", "")
            if op_name in ("tir.max",):
                found["op"] = "max"
            elif op_name in ("tir.min",):
                # vfredmin would handle this; not enabled yet.
                found["op"] = "min"
        # Add(load(C), load(A)) or Add(load(A), load(C)) — sum accumulator
        elif isinstance(value, tirx.Add):
            same_buf = stmt.buffer
            for operand in (value.a, value.b):
                if isinstance(operand, tirx.BufferLoad) and operand.buffer.same_as(same_buf):
                    found["op"] = "sum"
                    break

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
                continue
            # Try RVV-tensorized reduction first; fall back to split+unroll.
            tensorized = False
            if rvv_ok:
                tensorized = self._tensorize_rvv_reduction(
                    sch, block_info.block_rv, vec_lanes, dtype_bits, target
                )
            if not tensorized:
                self._unroll_reduction_inner(sch, block_info.block_rv, vec_lanes)

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
    def _tensorize_rvv_reduction(sch, block_rv, vec_lanes, dtype_bits, target):
        """Tensorize a reduction block's inner loop with an RVV reduction intrinsic.

        Returns True on success. Returns False (without modifying the schedule)
        if the block's reduction op cannot be mapped, the chunk size is too
        small, or the matching intrinsic is unavailable.
        """
        op = _detect_reduction_op(sch, block_rv)
        if op not in ("max", "sum"):
            return False

        block_loops = sch.get_loops(block_rv)
        if len(block_loops) <= 1:
            return False

        inner = block_loops[-1]
        extent = get_extent(sch, inner)
        # Chunk must be >= vec_lanes for tensorize to be meaningful;
        # vfred*.vs needs at least one vector worth of input.
        if isinstance(extent, int) and extent < vec_lanes:
            return False

        # Lazy-register kernels for the current target (no-op if already
        # registered, mirrors the pattern used in MetaSchedule's RISC-V rules).
        try:
            from tvm.s_tir.tensor_intrin.riscv_cpu import (
                _rvv_reduce_kernel_name,
                register_riscv_intrinsics,
            )
        except ImportError:
            return False
        try:
            register_riscv_intrinsics(target)
        except Exception:  # pylint: disable=broad-except
            return False

        # Resolve element dtype from the block's write buffer.
        block_stmt = sch.get(block_rv)
        if not block_stmt.writes:
            return False
        dtype_obj = DataType(block_stmt.writes[0].buffer.dtype)
        # Only FP supported by vfredmax/vfredusum.
        if not str(dtype_obj).startswith("float"):
            return False
        # Require the schedule's vec_lanes to match the registered chunk size
        # for this dtype (n_elems = VLEN/SEW under LMUL=1).
        if dtype_obj.bits != dtype_bits:
            return False

        intrin_name = _rvv_reduce_kernel_name(vec_lanes, str(dtype_obj), op)
        if s_tir.TensorIntrin.get(intrin_name, allow_missing=True) is None:
            return False

        # Split inner axis into [outer, vec_lanes-chunk] then tensorize the chunk.
        try:
            _, vec_loop = sch.split(inner, factors=[None, vec_lanes])
            sch.tensorize(vec_loop, intrin_name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("RVV reduction tensorize failed for %s: %s", intrin_name, exc)
            return False
        return True

    @staticmethod
    def _unroll_reduction_inner(sch, block_rv, vec_lanes):
        """Split the reduction inner loop and annotate for unrolling."""
        block_loops = sch.get_loops(block_rv)
        if len(block_loops) <= 1:
            return
        inner = block_loops[-1]
        extent = get_extent(sch, inner)
        if isinstance(extent, int) and extent <= vec_lanes:
            return
        _, inner_loop = sch.split(inner, factors=[None, vec_lanes])
        sch.annotate(inner_loop, ann_key="pragma_auto_unroll_max_step", ann_val=vec_lanes)
        sch.annotate(inner_loop, ann_key="pragma_unroll_explicit", ann_val=1)
