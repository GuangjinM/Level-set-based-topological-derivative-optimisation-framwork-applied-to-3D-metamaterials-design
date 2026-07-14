#debug 24/02/2026: determine the sign of g 
# ============================================================
# CHANGE LOG (2026-03)
# - MPI/JIT stability:
#   1) Set FENICS_JIT_CACHE_DIR before importing fenics.
#   2) Ensure shared cache dir exists in MPI and print active cache path on rank 0.
# - MPI consistency:
#   3) Use global volume fraction in evaluate() via assemble(chi*dx)/Vol
#      (avoid rank-dependent J and branch divergence/deadlock).
# - Line-search robustness/performance:
#   4) Recover kappa after accepted step:
#      first kappa_increase_early_count accepts use kappa_increase_factor_early, then kappa_increase_factor;
#      kappa <- min(kappa * factor, kappa0).
#   5) Project g once per iteration and reuse in line-search.
#   6) Reuse theta/pn/gn in line-search (avoid repeated angle assemble/JIT).
#   7) In [12], shrink kappa until the line-search floor kappa_min.
#   8) Convergence criteria:
#      - [13] uses only the global comparison-objective history rel_tol < eps_J check.
#      - [14] remains the final theta gate, so both J and theta must pass to exit.
#      - Remove line-search J_try history and kappa_min-triggered convergence.
# - L2 metric optimization:
#  9) Use preassembled L2 mass matrix for angle_between/norm evaluations.
# 10) Rebuild l2_mass after mesh refinement.
# - MPI-safe diagnostics/output:
# 11) Keep collective ops on all ranks; print only on rank 0.
# 12) XDMF writes are collective (all ranks participate).
# - Refinement transfer fix:
# 13) Before projecting old lsf to refined mesh, enable extrapolation:
#      lsf_old_mesh.set_allow_extrapolation(True).
# 14) Refinement schedule changed from multiplicative growth (N *= 2)
#      to linear growth (N += refine_step, default 20) for smoother stage transitions.
# ============================================================
# CHANGE LOG (2026-03-10, HPC solver robustness)
# - Periodic nullspace construction:
#   15) Replace interpolate(Constant(...), V_periodic) path for translation modes
#       with direct component-DOF vector fill (more robust at high MPI core counts).
#   16) Add nullspace cache keyed by FunctionSpace instance so repeated cell solves
#       reuse the same translation nullspace within a mesh stage.
# ============================================================
# CHANGE LOG (2026-03-11)
# - Symmetry projection update:
#   17) Replace DG0/chi-based symmetry rewrite with direct CG1(level-set) nodal
#       group averaging on the regular cube grid:
#         lsf -> symmetry-group average on Vls DOFs -> lsf_sym
#       to avoid binarization-induced distortion of optimization space.
# - Octant parameterization update:
#   18) Replace center+diagonal (cubic-like) grouping with mirror parameterization
#       on Vls nodal values (phase-1 octant):
#         key=(min(x,1-x), min(y,1-y), min(z,1-z))
# - Strict reduced-DOF minimal-wedge workflow:
#   19) Upgrade to cubic fundamental wedge parameterization:
#       on Vls nodal values (phase-1 octant):
#         key=sort_desc(min(x,1-x), min(y,1-y), min(z,1-z))
#         (equivalent to 0 <= z <= y <= x <= 0.5)
#       and enforce it at every trial *before* evaluate:
#         lsf_trial -> hard-vf -> wedge-representative expansion -> evaluate
#       so mirrored/permuted copies are derived, not independently updated.
# ============================================================
from __future__ import print_function
import os
import sys
import time

# Set JIT cache *before* importing fenics so MPI root uses a shared path (avoids "Compilation failed on root node")
if not os.environ.get("FENICS_JIT_CACHE_DIR"):
    _jit_cache = os.path.abspath(os.path.join(os.getcwd(), ".fenics-jit-cache"))
    os.environ["FENICS_JIT_CACHE_DIR"] = _jit_cache

from fenics import *
import numpy as np
from mpi4py import MPI as _pyMPI

# Ensure cache dir exists (all ranks; same path when cwd is shared)
if MPI.size(MPI.comm_world) > 1:
    _jit_cache = os.environ.get("FENICS_JIT_CACHE_DIR", "")
    if _jit_cache:
        try:
            os.makedirs(_jit_cache, exist_ok=True)
        except OSError:
            pass
        if "form_compiler" in parameters and "cache_dir" in parameters["form_compiler"]:
            parameters["form_compiler"]["cache_dir"] = _jit_cache
        if MPI.rank(MPI.comm_world) == 0:
            print("[mpi] FENICS_JIT_CACHE_DIR=%s" % _jit_cache, file=sys.stderr)

import definition_3d_UTI as _definition3d
from definition_3d_UTI import *
from init_3d_UTI import *

# `from init_3d_UTI import *` re-exports raw FEniCS symbols and can overwrite the
# safe projection wrapper imported from `definition3d`. Bind it explicitly here.
_project = _definition3d._project
project = _project

# This file intentionally follows the R03D optimization workflow.  In the UTI
# definition module, r0_material_term/grad_r0_material_term are compatibility
# aliases for the active UTI J_hb objective, so the optimizer mechanics match
# R03D while the cost function remains UTI-specific.

_MPI_COMM = _pyMPI.COMM_WORLD
_MPI_RANK = _MPI_COMM.Get_rank()
_MPI_SIZE = _MPI_COMM.Get_size()


class _Tee(object):
    """Write to multiple files (e.g. stdout + log file)."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


# ============================================================
# main_3d.py
# 3D TD + Level-set microstructure design (FEniCS 2019)
#
# Implements:
#   - 3D homogenisation (6 cell problems) => C_hom
#   - Objective Phi(C_hom) as provided by you
#   - DTJ via chain rule (type-3 / type-4 logic in your slides)
#   - Level-set update via "slerp" (Eq. 4.11-style)
#   - Optional backtracking line search and uniform refinement
# ============================================================

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

def volume_fraction_from_chi(chi, mesh, vol_total=None):
    """
    Global solid volume fraction from DG0 indicator chi (1=solid, 0=void).
    Uses collective assemble so all MPI ranks get the same scalar.
    """
    dxm = Measure("dx", domain=mesh)
    if vol_total is None:
        vol_total = assemble(Constant(1.0) * dxm)
    solid_vol = assemble(chi * dxm)
    return float(solid_vol) / max(1e-30, float(vol_total))


def copy_function_values(dst, src):
    """MPI-safe function copy for constrained/periodic spaces."""
    vals = src.vector().get_local().copy()
    dst.vector().set_local(vals)
    dst.vector().apply("insert")
    return dst


def scaled_function(src, scale):
    out = Function(src.function_space())
    vals = src.vector().get_local().copy()
    vals *= float(scale)
    out.vector().set_local(vals)
    out.vector().apply("insert")
    return out


def difference_function(a, b):
    out = Function(a.function_space())
    vals = a.vector().get_local().copy() - b.vector().get_local()
    out.vector().set_local(vals)
    out.vector().apply("insert")
    return out


def safe_l2_inner(f, g, dxm, M=None):
    try:
        if M is not None:
            return float(l2_inner_mass(f, g, M))
    except RuntimeError:
        pass
    return float(assemble(f * g * dxm))


def safe_angle_between(f, g, dxm, M=None):
    try:
        if M is not None:
            return angle_between(f, g, M=M)
    except RuntimeError:
        pass
    return angle_between(f, g, dx=dxm, M=None)


def print_direction_chain_diagnostics(it, dtj, d0, d1, d2, d3, dxm, l2_mass, kappa_probe, filter_active=False):
    neg_dtj = scaled_function(dtj, -1.0)
    th_grad, _, _ = safe_angle_between(neg_dtj, d0, dxm, M=l2_mass)
    th_01, n0, n1 = safe_angle_between(d0, d1, dxm, M=l2_mass)
    th_12, _, n2 = safe_angle_between(d1, d2, dxm, M=l2_mass)
    th_23, _, n3 = safe_angle_between(d2, d3, dxm, M=l2_mass)
    th_0f, _, _ = safe_angle_between(d0, d3, dxm, M=l2_mass)
    inner_dtj_d0 = safe_l2_inner(dtj, d0, dxm, M=l2_mass)
    if MPI.rank(MPI.comm_world) == 0:
        print("[it %03d-dir] kappa_probe=%.3e filter=%s  angle(-DTJ,d0)=%.2f deg  <DTJ,d0>_M=%.6e" %
              (it, float(kappa_probe), "active" if filter_active else "inactive", np.degrees(th_grad), inner_dtj_d0))
        print("[it %03d-dir] ||d0||=%.6e  ||d1||=%.6e  ||d2||=%.6e  ||d3||=%.6e" %
              (it, n0, n1, n2, n3))
        print("[it %03d-dir] angle(d0,d1)=%.2f deg  angle(d1,d2)=%.2f deg  angle(d2,d3)=%.2f deg  angle(d0,d3)=%.2f deg" %
              (it, np.degrees(th_01), np.degrees(th_12), np.degrees(th_23), np.degrees(th_0f)))


def _alpha_ti_bounds(cfg):
    """Return `(alpha_min, alpha_floor, alpha_max)` for TI continuation."""
    alpha_max = max(0.0, float(cfg.get("alpha", 0.0)))
    alpha_min_raw = cfg.get("alpha_ti_min", None)
    if alpha_min_raw is None:
        min_factor = float(cfg.get("alpha_ti_min_factor", 0.10))
        alpha_min = alpha_max * min_factor
    else:
        alpha_min = float(alpha_min_raw)
    alpha_min = min(alpha_max, max(0.0, alpha_min))

    alpha_floor_raw = cfg.get("alpha_ti_floor", alpha_min)
    alpha_floor = min(alpha_max, max(0.0, float(alpha_floor_raw)))
    return float(alpha_min), float(alpha_floor), float(alpha_max)


def _alpha_ti_schedule_cap(cfg, vf=None):
    """Volume-dependent upper bound for the next-stage TI penalty weight."""
    alpha_min, _, alpha_max = _alpha_ti_bounds(cfg)
    if alpha_max <= 0.0:
        return 0.0
    if vf is None:
        vf = cfg.get("_current_vf", None)
    if vf is None:
        return float(alpha_min)

    vf_now = float(vf)
    if not np.isfinite(vf_now):
        return float(alpha_min)

    vf_start = float(cfg.get("alpha_ti_ramp_start_vf", 0.35))
    vf_end = float(cfg.get("alpha_ti_ramp_end_vf", cfg.get("stage2_volume_end_vf", 0.10)))
    if vf_start <= vf_end + 1e-15:
        return float(alpha_max if vf_now <= vf_end else alpha_min)
    if vf_now >= vf_start:
        return float(alpha_min)
    if vf_now <= vf_end:
        return float(alpha_max)

    s = (vf_start - vf_now) / max(vf_start - vf_end, 1e-30)
    s = min(1.0, max(0.0, float(s)))
    power = max(1e-12, float(cfg.get("alpha_ti_ramp_power", 1.0)))
    w = s ** power
    return float(alpha_min + (alpha_max - alpha_min) * w)


def initialise_alpha_ti_state(cfg):
    """Initialise and return the stage-wise TI penalty weight."""
    alpha_min, _, alpha_max = _alpha_ti_bounds(cfg)
    if alpha_max <= 0.0:
        cfg["_alpha_ti_current"] = 0.0
        return 0.0
    if not bool(cfg.get("alpha_ti_continuation_enabled", False)):
        cfg["_alpha_ti_current"] = float(alpha_max)
        return float(alpha_max)

    current_raw = cfg.get("_alpha_ti_current", None)
    if current_raw is None:
        current = float(alpha_min)
    else:
        current = float(current_raw)
        if not np.isfinite(current):
            current = float(alpha_min)
    current = min(alpha_max, max(0.0, current))
    cfg["_alpha_ti_current"] = float(current)
    return float(current)


def alpha_value(cfg, it=None, vf=None):
    """Effective TI penalty weight.

    With TI continuation enabled, alpha is stage-wise constant and is updated
    by `update_alpha_ti_from_stage_state` at stage-2 entry and after a stage
    advances.
    """
    alpha_max = max(0.0, float(cfg.get("alpha", 0.0)))
    if alpha_max <= 0.0:
        return 0.0
    if not bool(cfg.get("alpha_ti_continuation_enabled", False)):
        return float(alpha_max)
    return initialise_alpha_ti_state(cfg)


def _alpha_ti_ratio_cap_from_state(cfg, res):
    """Upper bound alpha so J_ti stays secondary to the UTI J_hb objective."""
    ratio_cap_raw = cfg.get("alpha_ti_ratio_cap", None)
    if ratio_cap_raw is None:
        return float("inf")
    ratio_cap = float(ratio_cap_raw)
    if (not np.isfinite(ratio_cap)) or ratio_cap <= 0.0:
        return float("inf")

    eps = max(1e-30, float(cfg.get("eps_denom", 1e-12)))
    alpha_used = abs(float(res.get("alpha", cfg.get("_alpha_ti_current", cfg.get("alpha", 0.0)))))
    J_ti = abs(float(res.get("J_ti", 0.0)))
    base_ti = J_ti / max(alpha_used, eps)
    if (not np.isfinite(base_ti)) or base_ti <= eps:
        phi = abs(float(res.get("phi_TI", res.get("rti_over_cref_raw", 0.0))))
        base_ti = phi * phi
    if (not np.isfinite(base_ti)) or base_ti <= eps:
        return float("inf")

    J_ref = abs(float(res.get("J_hb", res.get("J_R0", 0.0))))
    J_ref = max(J_ref, max(0.0, float(cfg.get("alpha_ti_ratio_ref_floor", 0.0))))
    return float(ratio_cap * J_ref / max(base_ti, eps))


def update_alpha_ti_from_stage_state(cfg, res, it=None, reason="stage-advance"):
    """Update alpha for the next stage, with schedule/growth/ratio caps.

    The current stage keeps alpha fixed. This function is called at stage-2
    entry and after a stage target is accepted and the next target is created.
    """
    if not bool(cfg.get("alpha_ti_continuation_enabled", False)):
        return False

    current = float(initialise_alpha_ti_state(cfg))
    alpha_min, alpha_floor, alpha_max = _alpha_ti_bounds(cfg)
    if alpha_max <= 0.0:
        return False

    if bool(cfg.get("alpha_ti_stage2_only", True)):
        try:
            stage2_alpha_active = stage2_volume_continuation_active(
                cfg,
                vf=res.get("vf", None),
                vf_target=current_vf_target(cfg),
                hard_shift_only=bool(cfg.get("_postprocess_active", False)),
            )
        except NameError:
            stage2_alpha_active = bool(cfg.get("_stage2_vc_started_once", False)) or bool(cfg.get("_stage2_vc_force_started", False))
        if not stage2_alpha_active:
            if MPI.rank(MPI.comm_world) == 0:
                print("[alpha-ti] it=%s reason=%s hold %.6e: stage2 volume continuation inactive" %
                      ("n/a" if it is None else "%03d" % int(it), str(reason), float(current)))
            return False

    vf_now = float(res.get("vf", cfg.get("_current_vf", float("nan"))))
    if bool(cfg.get("alpha_ti_schedule_cap_enabled", True)):
        schedule_cap = min(alpha_max, _alpha_ti_schedule_cap(cfg, vf=vf_now))
    else:
        schedule_cap = float(alpha_max)
    growth = max(1.0, float(cfg.get("alpha_ti_growth_per_stage", 1.25)))
    growth_cap = max(alpha_min, current * growth)
    ratio_cap = _alpha_ti_ratio_cap_from_state(cfg, res)

    candidate = min(alpha_max, schedule_cap, growth_cap, ratio_cap)
    if bool(cfg.get("alpha_ti_ratio_cap_strict", True)):
        # The floor is weak: it must not force J_ti above the configured ratio cap.
        if ratio_cap >= alpha_floor:
            candidate = max(alpha_floor, candidate)
    else:
        candidate = max(alpha_floor, candidate)
    candidate = min(alpha_max, max(0.0, float(candidate)))

    if not bool(cfg.get("alpha_ti_allow_decrease", True)) and candidate < current:
        candidate = current

    abs_tol = max(0.0, float(cfg.get("alpha_ti_update_abs_tol", 1e-12)))
    rel_tol = max(0.0, float(cfg.get("alpha_ti_update_rel_tol", 0.0)))
    changed = abs(candidate - current) > max(abs_tol, rel_tol * max(abs(current), 1.0))
    if changed:
        cfg["_alpha_ti_current"] = float(candidate)
        cfg["_alpha_ti_last_update_it"] = int(it) if it is not None else None

    if MPI.rank(MPI.comm_world) == 0:
        J_hb = abs(float(res.get("J_hb", res.get("J_R0", 0.0))))
        J_ti = abs(float(res.get("J_ti", 0.0)))
        ratio_now = J_ti / max(J_hb, max(1e-30, float(cfg.get("alpha_ti_ratio_ref_floor", 0.0))))
        base_ti = J_ti / max(abs(float(res.get("alpha", current))), 1e-30)
        ratio_next = candidate * base_ti / max(J_hb, max(1e-30, float(cfg.get("alpha_ti_ratio_ref_floor", 0.0))))
        print("[alpha-ti] it=%s reason=%s alpha %s %.6e -> %.6e; caps(schedule=%.6e growth=%.6e ratio=%.6e) Jti/Jhb %.3e -> %.3e" %
              ("n/a" if it is None else "%03d" % int(it), str(reason),
               "update" if changed else "hold", current, candidate,
               float(schedule_cap), float(growth_cap), float(ratio_cap),
               float(ratio_now), float(ratio_next)))
    return bool(changed)


def beta_value(cfg):
    """Backward-compat: legacy single beta weight."""
    return float(cfg.get("beta", 1.0))


def beta_a_value(cfg):
    """Weight on hb^2 in the UTI J_hb objective."""
    return float(cfg.get("beta_a", cfg.get("beta", 1.0)))


def beta_b_value(cfg):
    """Weight on 1/(ha^2 + H^2 + eps) in the UTI J_hb objective."""
    return float(cfg.get("beta_b", 0.0))


def _lambda_v_cap_from_cfg(cfg, include_plateau_cap=False):
    """Return `(lambda_floor, lambda_cap)` from the active config."""
    lambda_floor = max(0.0, float(cfg.get("lambda_v_adapt_min_abs", 0.0)))
    cap_candidates = []
    lambda_cap_raw = cfg.get("lambda_v_adapt_max_abs", None)
    if lambda_cap_raw is not None:
        lambda_cap_val = float(lambda_cap_raw)
        if np.isfinite(lambda_cap_val):
            cap_candidates.append(max(lambda_floor, lambda_cap_val))
    if include_plateau_cap:
        plateau_cap_raw = cfg.get("lambda_v_plateau_boost_max_abs", None)
        if plateau_cap_raw is not None:
            plateau_cap_val = float(plateau_cap_raw)
            if np.isfinite(plateau_cap_val):
                cap_candidates.append(max(lambda_floor, plateau_cap_val))
    lambda_cap = min(cap_candidates) if len(cap_candidates) > 0 else None
    return float(lambda_floor), lambda_cap


def stage2_effective_lambda_ratio_cap(cfg, vf, base_ratio_cap):
    """Runtime-adjusted stage-2 lambda ratio cap.

    The static low-vf cap is deliberately conservative.  A slow-progress
    watchdog can raise only the effective low-vf cap when accepted steps keep
    reducing volume too slowly while the AL controller is already saturated.
    """
    ratio_cap = max(0.0, float(base_ratio_cap))
    if not bool(cfg.get("stage2_slow_progress_watchdog_enabled", False)):
        return float(ratio_cap)
    try:
        vf_val = float(vf)
    except (TypeError, ValueError):
        return float(ratio_cap)
    low_vf_threshold = float(cfg.get("stage2_lambda_ratio_low_vf_threshold", 0.35))
    if vf_val >= low_vf_threshold:
        return float(ratio_cap)
    base_low = max(0.0, float(cfg.get("stage2_lambda_ratio_cap_low_vf", ratio_cap)))
    eff_low = max(base_low, float(cfg.get("_stage2_lambda_ratio_cap_low_vf_eff", base_low)))
    eff_max = max(base_low, float(cfg.get("stage2_slow_progress_lambda_ratio_cap_low_vf_max", base_low)))
    return float(min(eff_max, max(ratio_cap, eff_low)))


def _clamp_lambda_v_to_cfg(cfg, lambda_val, include_plateau_cap=False):
    """Clamp a lambda_v value to the active configured bounds."""
    lambda_floor, lambda_cap = _lambda_v_cap_from_cfg(cfg, include_plateau_cap=include_plateau_cap)
    lambda_new = max(lambda_floor, float(lambda_val))
    if lambda_cap is not None:
        lambda_new = min(lambda_cap, lambda_new)
    return float(lambda_new)


def _format_lambda_v_cap_status(lambda_target, lambda_cap, lambda_floor=0.0):
    """Compact cap/floor status string for logging."""
    lambda_target = float(lambda_target)
    lambda_floor = float(lambda_floor)
    lambda_cap_str = "None" if (lambda_cap is None) else ("%.6e" % float(lambda_cap))
    capped_hi = (lambda_cap is not None) and (lambda_target >= float(lambda_cap) - 1e-16)
    capped_lo = lambda_target <= lambda_floor + 1e-16
    return "cap=%s capped_hi=%s capped_lo=%s" % (
        lambda_cap_str,
        str(bool(capped_hi)),
        str(bool(capped_lo)),
    )


def _stage_dv_levels_from_cfg(cfg):
    """Return the stage-dv ladder from large to small."""
    dv0 = max(1e-30, float(cfg.get("vf_stage_dv0", 0.03)))
    dv_min = max(1e-30, float(cfg.get("vf_stage_dv_min", 0.002)))
    levels_raw = cfg.get("vf_stage_dv_levels", None)
    levels = []
    if levels_raw is not None:
        try:
            for val in levels_raw:
                val_f = float(val)
                if np.isfinite(val_f) and (val_f > 0.0):
                    levels.append(val_f)
        except TypeError:
            levels = []
    if len(levels) == 0:
        shrink = float(cfg.get("vf_stage_shrink", 0.5))
        shrink = min(max(shrink, 1e-6), 1.0 - 1e-12)
        levels = [dv0]
        while levels[-1] > dv_min + 1e-15:
            next_dv = max(dv_min, levels[-1] * shrink)
            if abs(next_dv - levels[-1]) <= 1e-15:
                break
            levels.append(next_dv)
            if next_dv <= dv_min + 1e-15:
                break
    levels.extend([dv0, dv_min])
    levels = sorted(levels, reverse=True)
    unique_levels = []
    for val in levels:
        if (len(unique_levels) == 0) or (abs(val - unique_levels[-1]) > 1e-12 * max(1.0, abs(unique_levels[-1]))):
            unique_levels.append(float(val))
    return tuple(unique_levels)


def _stage_dv_level_index(cfg, dv_now_raw=None):
    """Return `(idx, levels, dv_effective)` for the closest stage-dv ladder level."""
    levels = _stage_dv_levels_from_cfg(cfg)
    if len(levels) == 0:
        return None, tuple(), None
    if dv_now_raw is None:
        dv_now_raw = cfg.get("_vf_stage_dv", cfg.get("vf_stage_dv0", levels[0]))
    dv_now = float(dv_now_raw)
    idx = min(range(len(levels)), key=lambda k: abs(float(levels[k]) - dv_now))
    return int(idx), levels, float(levels[idx])


def _adjacent_stage_dv(cfg, dv_now_raw=None, direction="smaller"):
    """Move one rung on the stage-dv ladder and return the resulting dv."""
    idx, levels, dv_effective = _stage_dv_level_index(cfg, dv_now_raw=dv_now_raw)
    if idx is None:
        return None
    if direction == "larger":
        next_idx = max(0, idx - 1)
    else:
        next_idx = min(len(levels) - 1, idx + 1)
    return float(levels[next_idx])


def _format_stage_dv_state(cfg, dv_now_raw=None, rebound_active=None):
    """Compact human-readable status for the current dv rung."""
    idx, levels, dv_effective = _stage_dv_level_index(cfg, dv_now_raw=dv_now_raw)
    if idx is None:
        return "dv=None"
    next_up = float(levels[max(0, idx - 1)])
    next_down = float(levels[min(len(levels) - 1, idx + 1)])
    mode = "rebound-up" if bool(cfg.get("_vf_stage_rebound_active", False) if rebound_active is None else rebound_active) else "normal-down"
    return "dv=%.6f (level %d/%d, next_up=%.6f, next_down=%.6f, mode=%s)" % (
        float(dv_effective), int(idx + 1), int(len(levels)), float(next_up), float(next_down), str(mode)
    )


def _stage2_dv_cap_for_vf(cfg, vf_reference):
    """Low-vf cap for the next stage target drop."""
    if not bool(cfg.get("stage2_volume_continuation_enabled", False)):
        return None
    caps_raw = cfg.get("stage2_vf_stage_dv_caps", ())
    vf_ref = float(vf_reference)
    cap_val = None
    try:
        caps_iter = list(caps_raw)
    except TypeError:
        caps_iter = []
    for item in caps_iter:
        try:
            vf_threshold, dv_cap = item
            if vf_ref <= float(vf_threshold):
                cap_val = float(dv_cap)
        except (TypeError, ValueError):
            continue
    if cap_val is None:
        return None
    return max(float(cfg.get("vf_stage_dv_min", 0.002)), float(cap_val))


def _cap_stage_dv_for_vf(cfg, dv, vf_reference):
    """Apply the optional low-vf stage-dv cap and keep the value on a valid scale."""
    dv_eff = max(1e-30, float(dv))
    cap_val = _stage2_dv_cap_for_vf(cfg, vf_reference)
    if cap_val is not None:
        dv_eff = min(dv_eff, float(cap_val))
    return max(float(cfg.get("vf_stage_dv_min", 0.002)), float(dv_eff))


def _optional_positive_float(value):
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if (not np.isfinite(value_f)) or (value_f <= 0.0):
        return None
    return value_f


def _stage2_settle_reset(cfg):
    cfg["_stage2_settle_active"] = False
    cfg["_stage2_settle_target"] = None
    cfg["_stage2_settle_start_it"] = None
    cfg["_stage2_settle_accepts"] = 0
    cfg["_stage2_settle_j_history"] = []
    cfg["_stage2_settle_j_mech_history"] = []
    cfg["_stage2_settle_theta_history"] = []
    cfg["_stage2_settle_final"] = False
    cfg["_stage2_settle_exhausted"] = False
    cfg["_stage2_settle_last_status"] = None
    cfg["_stage2_settle_overshoot_start_it"] = None
    cfg["_stage2_settle_overshoot_accepts_base"] = 0


def stage2_base_convergence_vf(cfg):
    return float(
        cfg.get(
            "stage2_convergence_vf",
            cfg.get("stage2_volume_end_vf", cfg.get("hard_shift_switch_to_shift_vf", 0.10)),
        )
    )


def stage2_current_convergence_vf(cfg):
    if bool(cfg.get("_stage2_loss_envelope_stop_active", False)):
        try:
            return float(cfg.get("_stage2_loss_envelope_convergence_vf"))
        except (TypeError, ValueError):
            pass
    return stage2_base_convergence_vf(cfg)


def current_postprocess_vf_final_target(cfg):
    """Post-processing vf target, with early-envelope stops shifted by a fixed extra drop."""
    base = float(cfg.get("vf_final_target", 0.0))
    if bool(cfg.get("_stage2_loss_envelope_stop_active", False)):
        raw = cfg.get("_stage2_loss_envelope_convergence_vf", None)
        if raw is not None:
            try:
                convergence_vf = float(raw)
                extra_drop = max(0.0, float(cfg.get("postprocess_loss_envelope_extra_drop", 0.10)))
                if np.isfinite(convergence_vf):
                    return max(base, convergence_vf - extra_drop)
            except (TypeError, ValueError):
                pass
    return base


def postprocess_target_status(cfg, res_eval):
    """Return whether the post-processing vf target is reached under the configured tolerance."""
    target = float(current_postprocess_vf_final_target(cfg))
    tol = float(stage_success_tolerance(cfg, dv=float(cfg.get("vf_stage_dv_min", 0.002))))
    try:
        vf_now = float(res_eval.get("vf", float("nan")))
    except Exception:
        vf_now = float("nan")
    reached = bool(np.isfinite(vf_now) and np.isfinite(target) and vf_now <= target + tol)
    return reached, vf_now, target, tol


def _stage2_is_final_settle_target(cfg, vf_target, vf_stage_tol):
    if not bool(cfg.get("stage2_stop_for_convergence_enabled", False)):
        return False
    if vf_target is None:
        return False
    conv_vf = float(stage2_current_convergence_vf(cfg))
    return float(vf_target) <= conv_vf + float(vf_stage_tol)


def _stage2_settle_theta_limit_deg(cfg, vf_target=None, final_stage=False):
    if bool(final_stage):
        return float(cfg.get("stage2_settle_theta_final_deg", cfg.get("stage2_settle_theta_deg", 60.0)))
    low_thresh = float(cfg.get("stage2_settle_low_vf_threshold", 0.20))
    if (vf_target is not None) and (float(vf_target) <= low_thresh):
        return float(cfg.get("stage2_settle_theta_low_vf_deg", cfg.get("stage2_settle_theta_deg", 60.0)))
    return float(cfg.get("stage2_settle_theta_deg", 60.0))


def _stage2_settle_update(
    cfg,
    target,
    it,
    res_eval,
    theta_update_deg,
    theta_update_sym_deg=None,
    stage_hit=False,
    accepted_normal_step=False,
    blocked_by_stall=False,
    final_stage=False,
    vf_stage_tol=None,
):
    """Update and evaluate the stage-2 settle gate for the current frozen target."""
    target = float(target)
    target_changed = (
        (not bool(cfg.get("_stage2_settle_active", False)))
        or (cfg.get("_stage2_settle_target", None) is None)
        or (abs(float(cfg.get("_stage2_settle_target", target)) - target) > 1e-12)
        or (bool(cfg.get("_stage2_settle_final", False)) != bool(final_stage))
    )
    if target_changed:
        cfg["_stage2_settle_active"] = True
        cfg["_stage2_settle_target"] = float(target)
        cfg["_stage2_settle_start_it"] = int(it)
        cfg["_stage2_settle_accepts"] = 0
        cfg["_stage2_settle_j_history"] = []
        cfg["_stage2_settle_j_mech_history"] = []
        cfg["_stage2_settle_theta_history"] = []
        cfg["_stage2_settle_final"] = bool(final_stage)
        cfg["_stage2_settle_exhausted"] = False
        cfg["_stage2_settle_overshoot_start_it"] = None
        cfg["_stage2_settle_overshoot_accepts_base"] = 0

    j_keep = max(2, int(cfg.get("stage2_settle_exhaust_window", cfg.get("stage2_settle_j_window", 6))))
    if accepted_normal_step:
        cfg["_stage2_settle_accepts"] = int(cfg.get("_stage2_settle_accepts", 0)) + 1

    record_eval_j = bool(cfg.get("stage2_settle_record_all_evals", True)) or bool(accepted_normal_step)
    if record_eval_j:
        j_hist = list(cfg.get("_stage2_settle_j_history", []))
        j_mech_hist = list(cfg.get("_stage2_settle_j_mech_history", []))
        j_sample = float(res_eval.get("J_compare", res_eval.get("J", float("nan"))))
        j_mech_sample = float(res_eval.get("J", float("nan")))
        if np.isfinite(j_sample) and np.isfinite(j_mech_sample):
            j_hist.append(float(j_sample))
            j_mech_hist.append(float(j_mech_sample))
        cfg["_stage2_settle_j_history"] = j_hist[-j_keep:]
        cfg["_stage2_settle_j_mech_history"] = j_mech_hist[-j_keep:]

    start_it = cfg.get("_stage2_settle_start_it", it)
    try:
        age = int(it) - int(start_it) + 1
    except Exception:
        age = 1
    accepts = int(cfg.get("_stage2_settle_accepts", 0))
    min_iters = max(1, int(cfg.get("stage2_settle_min_iters", 6)))
    min_accepts = max(0, int(cfg.get("stage2_settle_min_accepts", 4)))

    theta_limit = _stage2_settle_theta_limit_deg(cfg, vf_target=target, final_stage=final_stage)
    theta_update_val = float(theta_update_deg)
    theta_hist = list(cfg.get("_stage2_settle_theta_history", []))
    theta_hist.append(float(theta_update_val))
    theta_keep = max(2, int(cfg.get("stage2_settle_exhaust_window", cfg.get("stage2_settle_j_window", 6))))
    cfg["_stage2_settle_theta_history"] = theta_hist[-theta_keep:]
    theta_ok = bool(np.isfinite(theta_update_val) and (theta_update_val <= theta_limit))
    theta_sym_limit = theta_limit + float(cfg.get("stage2_settle_theta_sym_margin_deg", 5.0))
    theta_sym_ok = True
    if bool(cfg.get("stage2_settle_use_symmetry_theta", True)) and (theta_update_sym_deg is not None):
        theta_sym_val = float(theta_update_sym_deg)
        theta_sym_ok = bool(np.isfinite(theta_sym_val) and (theta_sym_val <= theta_sym_limit))
    else:
        theta_sym_val = None

    j_window = max(2, int(cfg.get("stage2_settle_j_window", 6)))
    j_rel_tol = max(0.0, float(cfg.get("stage2_settle_j_rel_tol", 5e-3)))
    j_hist = list(cfg.get("_stage2_settle_j_history", []))
    if len(j_hist) >= j_window:
        recent = np.asarray(j_hist[-j_window:], dtype=float)
        j_span = float(np.max(recent) - np.min(recent))
        j_scale = max(float(np.max(np.abs(recent))), abs(float(cfg.get("J_min", 1e-12))), 1e-30)
        j_rel = j_span / j_scale
        j_ok = bool(j_rel <= j_rel_tol)
    else:
        j_span = float("nan")
        j_rel = float("nan")
        j_ok = False

    hit_ok = bool(stage_hit)
    iter_ok = bool(age >= min_iters)
    accept_ok = bool(accepts >= min_accepts)
    stall_ok = not bool(blocked_by_stall)
    settle_strict_ok = bool(hit_ok and iter_ok and accept_ok and theta_ok and theta_sym_ok and j_ok and stall_ok)
    vf_val = float(res_eval.get("vf", float("nan")))
    if vf_stage_tol is None:
        vf_tol_val = float(cfg.get("vf_stage_tol", cfg.get("stage2_vf_stage_tol", 0.0015)))
    else:
        vf_tol_val = float(vf_stage_tol)
    overshoot_gap = float(target - vf_val) if np.isfinite(vf_val) else float("nan")
    overshoot_tol = max(0.0, float(cfg.get("stage2_settle_overshoot_tol_factor", 1.0))) * max(vf_tol_val, 0.0)
    overshoot_hit = bool(np.isfinite(overshoot_gap) and (overshoot_gap >= overshoot_tol))
    if overshoot_hit:
        if cfg.get("_stage2_settle_overshoot_start_it", None) is None:
            cfg["_stage2_settle_overshoot_start_it"] = int(it)
            accepts_base = accepts - (1 if bool(accepted_normal_step) else 0)
            cfg["_stage2_settle_overshoot_accepts_base"] = max(0, int(accepts_base))
    else:
        cfg["_stage2_settle_overshoot_start_it"] = None
        cfg["_stage2_settle_overshoot_accepts_base"] = int(accepts)
    overshoot_start_it = cfg.get("_stage2_settle_overshoot_start_it", None)
    if overshoot_start_it is None:
        overshoot_age = 0
    else:
        try:
            overshoot_age = int(it) - int(overshoot_start_it) + 1
        except Exception:
            overshoot_age = 1
    overshoot_accepts_base = int(cfg.get("_stage2_settle_overshoot_accepts_base", accepts))
    overshoot_accepts = max(0, int(accepts) - overshoot_accepts_base)
    overshoot_release_enabled = bool(cfg.get("stage2_settle_overshoot_release_enabled", True))
    overshoot_final_enabled = bool(cfg.get("stage2_settle_overshoot_release_final_enabled", False))
    overshoot_min_iters = max(1, int(cfg.get("stage2_settle_overshoot_min_iters", min_iters)))
    overshoot_min_accepts = max(0, int(cfg.get("stage2_settle_overshoot_min_accepts", 0)))
    overshoot_iter_ok = bool(overshoot_age >= overshoot_min_iters)
    overshoot_accept_ok = bool(overshoot_accepts >= overshoot_min_accepts)
    overshoot_require_stall = bool(cfg.get("stage2_settle_overshoot_require_stall", True))
    overshoot_require_j_ok = bool(cfg.get("stage2_settle_overshoot_require_j_ok", True))
    overshoot_stall_ok = bool(blocked_by_stall)
    overshoot_j_ok = bool(j_ok)
    if overshoot_require_stall or overshoot_require_j_ok:
        overshoot_plateau_ok = bool(
            (overshoot_require_stall and overshoot_stall_ok)
            or (overshoot_require_j_ok and overshoot_j_ok)
        )
        overshoot_gate_ok = bool(
            (overshoot_require_stall and overshoot_stall_ok)
            or (overshoot_require_j_ok and overshoot_j_ok and overshoot_accept_ok)
        )
    else:
        overshoot_plateau_ok = True
        overshoot_gate_ok = bool(overshoot_accept_ok)
    overshoot_force_max_iters = int(cfg.get("stage2_settle_overshoot_force_max_iters", 0))
    if overshoot_force_max_iters > 0:
        overshoot_force_max_iters = max(overshoot_min_iters, overshoot_force_max_iters)
    overshoot_force_age_ok = bool(overshoot_force_max_iters > 0 and overshoot_age >= overshoot_force_max_iters)
    overshoot_release_ok = bool(
        overshoot_release_enabled
        and (overshoot_final_enabled or (not bool(final_stage)))
        and hit_ok
        and overshoot_hit
        and overshoot_iter_ok
        and (overshoot_gate_ok or overshoot_force_age_ok)
    )
    hit_release_enabled = bool(cfg.get("stage2_settle_hit_release_enabled", True))
    hit_release_final_enabled = bool(cfg.get("stage2_settle_hit_release_final_enabled", False))
    hit_require_stall = bool(cfg.get("stage2_settle_hit_require_stall", True))
    hit_require_j_ok = bool(cfg.get("stage2_settle_hit_require_j_ok", True))
    hit_stall_ok = bool(blocked_by_stall)
    hit_j_ok = bool(j_ok)
    if hit_require_stall or hit_require_j_ok:
        hit_plateau_ok = bool(
            (hit_require_stall and hit_stall_ok)
            or (hit_require_j_ok and hit_j_ok)
        )
    else:
        hit_plateau_ok = True
    hit_release_ok = bool(
        hit_release_enabled
        and (hit_release_final_enabled or (not bool(final_stage)))
        and hit_ok
        and (not overshoot_hit)
        and iter_ok
        and hit_plateau_ok
    )
    settle_ok = bool(settle_strict_ok or overshoot_release_ok or hit_release_ok)
    exhaust_enabled = bool(cfg.get("stage2_settle_exhaust_enabled", True))
    max_iters_key = "stage2_settle_final_max_iters" if bool(final_stage) else "stage2_settle_max_iters"
    max_iters = max(min_iters, int(cfg.get(max_iters_key, cfg.get("stage2_settle_max_iters", 40))))
    exhaust_window = max(2, int(cfg.get("stage2_settle_exhaust_window", 12)))
    theta_drop_tol = max(0.0, float(cfg.get("stage2_settle_exhaust_theta_drop_tol_deg", 2.0)))
    j_improve_tol = max(0.0, float(cfg.get("stage2_settle_exhaust_j_improve_tol", 1e-3)))
    j_worsen_tol = max(0.0, float(cfg.get("stage2_settle_exhaust_j_worsen_tol", 5e-3)))
    theta_hist_ex = list(cfg.get("_stage2_settle_theta_history", []))
    j_merit_hist_ex = list(cfg.get("_stage2_settle_j_history", []))
    j_mech_hist_ex = list(cfg.get("_stage2_settle_j_mech_history", []))
    theta_improve = float("nan")
    theta_plateau_ok = False
    if len(theta_hist_ex) >= exhaust_window:
        theta_recent = np.asarray(theta_hist_ex[-exhaust_window:], dtype=float)
        finite_theta = theta_recent[np.isfinite(theta_recent)]
        if len(finite_theta) >= exhaust_window:
            theta_improve = float(finite_theta[0] - finite_theta[-1])
            theta_plateau_ok = bool(abs(theta_improve) <= theta_drop_tol)

    def _settle_rel_change_pair(hist):
        if len(hist) < exhaust_window:
            return float("nan"), False, False
        recent = np.asarray(hist[-exhaust_window:], dtype=float)
        if not np.all(np.isfinite(recent)):
            return float("nan"), False, False
        scale = max(float(np.max(np.abs(recent))), abs(float(cfg.get("J_min", 1e-12))), 1e-30)
        rel_change = float((recent[-1] - recent[0]) / scale)
        improve_plateau = bool((-rel_change) <= j_improve_tol)
        worsen_safe = bool(rel_change <= j_worsen_tol)
        return rel_change, improve_plateau, worsen_safe

    j_merit_rel_change, j_merit_improve_plateau, j_merit_worsen_safe = _settle_rel_change_pair(j_merit_hist_ex)
    j_mech_rel_change, j_mech_improve_plateau, j_mech_worsen_safe = _settle_rel_change_pair(j_mech_hist_ex)
    exhausted_iter_ok = bool(age >= max_iters)
    exhausted_stall_release_ok = bool(
        cfg.get("stage2_settle_exhaust_allow_stall_release", True)
        and bool(blocked_by_stall)
        and exhausted_iter_ok
    )
    exhausted_accept_ok = bool(accept_ok or exhausted_stall_release_ok)
    exhausted_history_ok = bool(
        len(theta_hist_ex) >= exhaust_window
        and len(j_merit_hist_ex) >= exhaust_window
        and len(j_mech_hist_ex) >= exhaust_window
    )
    settle_exhausted_ok = bool(
        exhaust_enabled
        and (not settle_ok)
        and hit_ok
        and exhausted_iter_ok
        and exhausted_accept_ok
        and exhausted_history_ok
        and theta_plateau_ok
        and j_merit_improve_plateau
        and j_mech_improve_plateau
        and j_merit_worsen_safe
        and j_mech_worsen_safe
    )
    if settle_exhausted_ok:
        cfg["_stage2_settle_exhausted"] = True
    status = dict(
        active=True,
        target=float(target),
        final_stage=bool(final_stage),
        age=int(age),
        min_iters=int(min_iters),
        accepts=int(accepts),
        min_accepts=int(min_accepts),
        theta_update_deg=float(theta_update_val),
        theta_limit_deg=float(theta_limit),
        theta_sym_deg=theta_sym_val,
        theta_sym_limit_deg=float(theta_sym_limit),
        theta_ok=bool(theta_ok),
        theta_sym_ok=bool(theta_sym_ok),
        j_rel=float(j_rel),
        j_rel_tol=float(j_rel_tol),
        j_window=int(j_window),
        j_ok=bool(j_ok),
        exhaust_enabled=bool(exhaust_enabled),
        max_iters=int(max_iters),
        exhaust_window=int(exhaust_window),
        theta_improve_deg=float(theta_improve),
        theta_drop_tol_deg=float(theta_drop_tol),
        theta_plateau_ok=bool(theta_plateau_ok),
        j_merit_rel_change=float(j_merit_rel_change),
        j_mech_rel_change=float(j_mech_rel_change),
        j_improve_tol=float(j_improve_tol),
        j_worsen_tol=float(j_worsen_tol),
        j_merit_improve_plateau=bool(j_merit_improve_plateau),
        j_mech_improve_plateau=bool(j_mech_improve_plateau),
        j_merit_worsen_safe=bool(j_merit_worsen_safe),
        j_mech_worsen_safe=bool(j_mech_worsen_safe),
        exhausted_iter_ok=bool(exhausted_iter_ok),
        exhausted_accept_ok=bool(exhausted_accept_ok),
        exhausted_stall_release_ok=bool(exhausted_stall_release_ok),
        exhausted_history_ok=bool(exhausted_history_ok),
        settle_exhausted_ok=bool(settle_exhausted_ok),
        stage_hit=bool(stage_hit),
        blocked_by_stall=bool(blocked_by_stall),
        settle_ok=bool(settle_ok),
        settle_strict_ok=bool(settle_strict_ok),
        hit_release_ok=bool(hit_release_ok),
        hit_release_enabled=bool(hit_release_enabled),
        hit_require_stall=bool(hit_require_stall),
        hit_require_j_ok=bool(hit_require_j_ok),
        hit_plateau_ok=bool(hit_plateau_ok),
        hit_stall_ok=bool(hit_stall_ok),
        hit_j_ok=bool(hit_j_ok),
        overshoot_release_ok=bool(overshoot_release_ok),
        overshoot_release_enabled=bool(overshoot_release_enabled),
        overshoot_gap=float(overshoot_gap),
        overshoot_tol=float(overshoot_tol),
        overshoot_hit=bool(overshoot_hit),
        overshoot_age=int(overshoot_age),
        overshoot_accepts=int(overshoot_accepts),
        overshoot_min_iters=int(overshoot_min_iters),
        overshoot_iter_ok=bool(overshoot_iter_ok),
        overshoot_min_accepts=int(overshoot_min_accepts),
        overshoot_accept_ok=bool(overshoot_accept_ok),
        overshoot_require_stall=bool(overshoot_require_stall),
        overshoot_require_j_ok=bool(overshoot_require_j_ok),
        overshoot_plateau_ok=bool(overshoot_plateau_ok),
        overshoot_gate_ok=bool(overshoot_gate_ok),
        overshoot_stall_ok=bool(overshoot_stall_ok),
        overshoot_j_ok=bool(overshoot_j_ok),
        overshoot_force_max_iters=int(overshoot_force_max_iters),
        overshoot_force_age_ok=bool(overshoot_force_age_ok),
        just_started=bool(target_changed),
    )
    cfg["_stage2_settle_last_status"] = status
    return settle_ok, status


def _format_stage2_settle_status(status):
    sym_msg = "n/a"
    if status.get("theta_sym_deg", None) is not None:
        sym_msg = "%.2f<=%.2f %s" % (
            float(status["theta_sym_deg"]),
            float(status["theta_sym_limit_deg"]),
            "ok" if bool(status["theta_sym_ok"]) else "wait",
        )
    j_rel = float(status.get("j_rel", float("nan")))
    j_msg = "nan" if not np.isfinite(j_rel) else ("%.3e" % j_rel)
    theta_imp = float(status.get("theta_improve_deg", float("nan")))
    theta_imp_msg = "nan" if not np.isfinite(theta_imp) else ("%.2f" % theta_imp)
    jm = float(status.get("j_merit_rel_change", float("nan")))
    jj = float(status.get("j_mech_rel_change", float("nan")))
    jm_msg = "nan" if not np.isfinite(jm) else ("%.3e" % jm)
    jj_msg = "nan" if not np.isfinite(jj) else ("%.3e" % jj)
    exhaust_msg = "exhaust=%s age=%d/%d win=%d dtheta=%s<=%.2f %s dJmerit=%s dJmech=%s" % (
        str(bool(status.get("settle_exhausted_ok", False))),
        int(status.get("age", 0)),
        int(status.get("max_iters", 0)),
        int(status.get("exhaust_window", 0)),
        theta_imp_msg,
        float(status.get("theta_drop_tol_deg", 0.0)),
        "flat" if bool(status.get("theta_plateau_ok", False)) else "wait",
        jm_msg,
        jj_msg,
    )
    overshoot_msg = ""
    if bool(status.get("overshoot_hit", False)) or bool(status.get("overshoot_release_ok", False)):
        overshoot_modes = []
        if bool(status.get("overshoot_require_stall", False)):
            overshoot_modes.append("stall")
        if bool(status.get("overshoot_require_j_ok", False)):
            overshoot_modes.append("J")
        overshoot_mode_msg = "any(%s)" % "|".join(overshoot_modes) if overshoot_modes else "none"
        force_age_max = int(status.get("overshoot_force_max_iters", 0))
        if force_age_max > 0:
            force_age_msg = "%s@%d" % (
                "ok" if bool(status.get("overshoot_force_age_ok", False)) else "wait",
                force_age_max,
            )
        else:
            force_age_msg = "off"
        overshoot_msg = (
            " overshoot_release=%s gap=%.3e>=%.3e overshoot_age=%d/%d overshoot_accepts=%d/%d plateau=%s gate=%s mode=%s force_age=%s stall=%s J=%s"
        ) % (
            str(bool(status.get("overshoot_release_ok", False))),
            float(status.get("overshoot_gap", float("nan"))),
            float(status.get("overshoot_tol", float("nan"))),
            int(status.get("overshoot_age", 0)),
            int(status.get("overshoot_min_iters", 0)),
            int(status.get("overshoot_accepts", 0)),
            int(status.get("overshoot_min_accepts", 0)),
            "ok" if bool(status.get("overshoot_plateau_ok", False)) else "wait",
            "ok" if bool(status.get("overshoot_gate_ok", False)) else "wait",
            overshoot_mode_msg,
            force_age_msg,
            str(bool(status.get("blocked_by_stall", False))),
            "ok" if bool(status.get("overshoot_j_ok", False)) else "wait",
        )
    hit_release_msg = ""
    if bool(status.get("stage_hit", False)) and (not bool(status.get("overshoot_hit", False))):
        hit_modes = []
        if bool(status.get("hit_require_stall", False)):
            hit_modes.append("stall")
        if bool(status.get("hit_require_j_ok", False)):
            hit_modes.append("J")
        hit_mode_msg = "any(%s)" % "|".join(hit_modes) if hit_modes else "none"
        hit_release_msg = (
            " hit_release=%s plateau=%s mode=%s stall=%s J=%s"
        ) % (
            str(bool(status.get("hit_release_ok", False))),
            "ok" if bool(status.get("hit_plateau_ok", False)) else "wait",
            hit_mode_msg,
            "ok" if bool(status.get("hit_stall_ok", False)) else "wait",
            "ok" if bool(status.get("hit_j_ok", False)) else "wait",
        )
    return (
        "target=%.6f final=%s age=%d/%d accepts=%d/%d "
        "theta=%.2f<=%.2f %s theta_sym=%s Jrel=%s<=%.3e %s %s%s%s hit=%s stall=%s"
    ) % (
        float(status["target"]),
        str(bool(status.get("final_stage", False))),
        int(status["age"]),
        int(status["min_iters"]),
        int(status["accepts"]),
        int(status["min_accepts"]),
        float(status["theta_update_deg"]),
        float(status["theta_limit_deg"]),
        "ok" if bool(status["theta_ok"]) else "wait",
        sym_msg,
        j_msg,
        float(status["j_rel_tol"]),
        "ok" if bool(status["j_ok"]) else "wait",
        exhaust_msg,
        overshoot_msg,
        hit_release_msg,
        str(bool(status["stage_hit"])),
        str(bool(status["blocked_by_stall"])),
    )


def _lambda_v_ratio_from_dv(cfg, dv_now_raw=None, ratio_override=None, allow_override_below_min=False, vf_now=None):
    """Map the current stage dv to a target volume-direction ratio."""
    ratio_min = max(0.0, float(cfg.get("lambda_v_direction_ratio_min", 0.15)))
    ratio_max = max(ratio_min, float(cfg.get("lambda_v_direction_ratio_max", 1.0)))
    dv_min = max(1e-30, float(cfg.get("vf_stage_dv_min", 0.002)))
    dv0 = max(dv_min, float(cfg.get("vf_stage_dv0", dv_min)))
    if dv_now_raw is None:
        dv_now_raw = current_stage_dv(cfg)
    dv_now = None if dv_now_raw is None else float(dv_now_raw)
    dv_effective = dv0 if dv_now is None else min(max(float(dv_now), dv_min), dv0)
    dv_span = max(dv0 - dv_min, 1e-30)
    if dv0 <= dv_min + 1e-30:
        dv_progress_frac = 0.0
    else:
        dv_progress_frac = (dv0 - dv_effective) / dv_span
    dv_progress_frac = min(max(float(dv_progress_frac), 0.0), 1.0)
    ratio_levels_raw = cfg.get("lambda_v_direction_ratio_levels", None)
    low_vf_threshold = cfg.get("lambda_v_low_vf_ratio_threshold", None)
    if (vf_now is not None) and (low_vf_threshold is not None):
        try:
            if float(vf_now) < float(low_vf_threshold):
                ratio_levels_raw = cfg.get("lambda_v_low_vf_direction_ratio_levels", ratio_levels_raw)
        except (TypeError, ValueError):
            pass
    ratio_levels = []
    if ratio_levels_raw is not None:
        try:
            ratio_levels = [float(val) for val in ratio_levels_raw]
        except TypeError:
            ratio_levels = []
    if ratio_levels:
        ratio_max = max(ratio_max, max(ratio_levels))
    if ratio_override is None:
        stage_idx, stage_levels, _ = _stage_dv_level_index(cfg, dv_now_raw=dv_effective)
        if (stage_idx is not None) and (len(ratio_levels) == len(stage_levels)):
            ratio_target = min(ratio_max, max(ratio_min, float(ratio_levels[stage_idx])))
            ratio_source = "dv-ladder"
        else:
            ratio_target = ratio_min + (ratio_max - ratio_min) * dv_progress_frac
            ratio_source = "dv-scale"
    else:
        ratio_override_val = max(0.0, float(ratio_override))
        if bool(allow_override_below_min):
            ratio_target = min(ratio_max, ratio_override_val)
        else:
            ratio_target = min(ratio_max, max(ratio_min, ratio_override_val))
        ratio_source = "override"
    return dict(
        ratio_min=float(ratio_min),
        ratio_max=float(ratio_max),
        ratio_target=float(ratio_target),
        ratio_source=str(ratio_source),
        dv_now=(None if dv_now is None else float(dv_now)),
        dv_min=float(dv_min),
        dv0=float(dv0),
        dv_effective=float(dv_effective),
        dv_progress_frac=float(dv_progress_frac),
    )


def stage2_vf_rate_gain(cfg):
    """Current closed-loop multiplier for stage-2 lambda_v authority."""
    if not bool(cfg.get("stage2_vf_rate_control_enabled", False)):
        return 1.0
    gain_min = max(0.0, float(cfg.get("stage2_vf_rate_gain_min", 0.25)))
    gain_max = max(gain_min, float(cfg.get("stage2_vf_rate_gain_max", 2.0)))
    gain0 = min(gain_max, max(gain_min, float(cfg.get("stage2_vf_rate_gain0", 1.0))))
    gain = min(gain_max, max(gain_min, float(cfg.get("_stage2_vf_rate_gain", gain0))))
    cfg["_stage2_vf_rate_gain"] = float(gain)
    return float(gain)


def _lambda_v_target_from_state(cfg, res_eval, psi, M=None, ratio_override=None, allow_override_below_min=False):
    """Direction-balance lambda_v target driven mainly by the current stage dv."""
    vf = float(res_eval.get("vf", 0.0))
    vf_target_cfg = current_vf_target(cfg)
    if stage2_volume_continuation_active(cfg, vf=vf, vf_target=vf_target_cfg):
        vf_target = vf_target_cfg
    else:
        vf_target = res_eval.get("vf_target", vf_target_cfg)
    vf_target = None if vf_target is None else float(vf_target)
    vf_violation = 0.0 if vf_target is None else max(vf - vf_target, 0.0)
    rel_violation = 0.0 if vf_target is None else vf_violation / max(abs(vf_target), 1e-12)
    ratio_info = _lambda_v_ratio_from_dv(
        cfg,
        dv_now_raw=current_stage_dv(cfg),
        ratio_override=ratio_override,
        allow_override_below_min=allow_override_below_min,
        vf_now=vf,
    )

    g_obj = res_eval.get("g_obj", res_eval.get("g", None))
    if g_obj is None:
        raise ValueError("res_eval must contain g_obj or g for lambda_v adaptation.")
    g_vol = res_eval.get("g_vol", None)
    if g_vol is None:
        g_vol = build_generalized_volume_gradient(psi.function_space())

    g_mech_tangent, _, _ = tangent_project_l2(g_obj, psi, M=M)
    g_vol_tangent, _, _ = tangent_project_l2(g_vol, psi, M=M)
    g_mech_orth, _, _ = remove_l2_component(g_mech_tangent, g_vol_tangent, M=M)

    mech_norm = float(l2_norm(g_mech_orth, M=M))
    vol_norm = float(l2_norm(g_vol_tangent, M=M))

    direction_eps = max(1e-30, float(cfg.get("lambda_v_direction_ratio_eps", 1e-12)))

    lambda_floor, lambda_cap = _lambda_v_cap_from_cfg(cfg, include_plateau_cap=False)

    use_stage2_al = stage2_volume_continuation_active(cfg, vf=vf, vf_target=vf_target)
    if use_stage2_al:
        mu = max(0.0, float(cfg.get("_vf_aug_lag_mu", cfg.get("mu_v0", 0.0))))
        rho = max(0.0, float(cfg.get("_vf_aug_lag_rho", cfg.get("rho_v", 0.0))))
        lambda_al_raw = max(0.0, mu + rho * vf_violation)
        ratio_cap_high = float(cfg.get("stage2_lambda_ratio_cap", 0.5))
        ratio_cap = float(ratio_cap_high)
        low_vf_threshold = float(cfg.get("stage2_lambda_ratio_low_vf_threshold", 0.35))
        ratio_cap_base = float(ratio_cap)
        if vf < low_vf_threshold:
            ratio_cap_low = float(cfg.get("stage2_lambda_ratio_cap_low_vf", ratio_cap))
            blend_width = max(0.0, float(cfg.get("stage2_lambda_ratio_low_vf_blend_width", 0.0)))
            if blend_width > 0.0:
                blend = min(1.0, max(0.0, (low_vf_threshold - float(vf)) / max(blend_width, 1e-30)))
                ratio_cap = (1.0 - blend) * float(ratio_cap_high) + blend * float(ratio_cap_low)
            else:
                ratio_cap = float(ratio_cap_low)
            ratio_cap_base = float(ratio_cap)
        ratio_cap = stage2_effective_lambda_ratio_cap(cfg, vf, ratio_cap)
        rate_gain = stage2_vf_rate_gain(cfg)
        ratio_cap_before_rate = float(ratio_cap)
        ratio_cap *= float(rate_gain)
        ratio_cap = max(0.0, ratio_cap)
        mech_norm_ref = float(mech_norm)
        mech_norm_for_cap = float(mech_norm)
        if bool(cfg.get("stage2_lambda_mech_ref_floor_enabled", False)):
            ref_old_raw = cfg.get("_stage2_lambda_mech_norm_ref", None)
            ref_decay = min(1.0, max(0.0, float(cfg.get("stage2_lambda_mech_ref_decay", 0.98))))
            if ref_old_raw is None:
                mech_norm_ref = float(mech_norm)
            else:
                mech_norm_ref = max(float(mech_norm), float(ref_old_raw) * ref_decay)
            cfg["_stage2_lambda_mech_norm_ref"] = float(mech_norm_ref)
            floor_frac = max(0.0, float(cfg.get("stage2_lambda_mech_ref_floor_frac", 0.0)))
            mech_norm_for_cap = max(float(mech_norm), floor_frac * float(mech_norm_ref))
        if (not np.isfinite(vol_norm)) or (vol_norm <= direction_eps):
            lambda_ratio_cap = float(lambda_floor)
        else:
            lambda_ratio_cap = ratio_cap * mech_norm_for_cap / max(vol_norm, direction_eps)
        lambda_target_raw = min(lambda_al_raw, lambda_ratio_cap)
        warmstart_iters = max(0, int(cfg.get("stage2_plateau_lambda_warmstart_iters", 0)))
        warmstart_start_it = cfg.get("_stage2_vc_force_start_it", None)
        warmstart_active = False
        if bool(cfg.get("_stage2_vc_force_started", False)) and (warmstart_iters > 0) and (warmstart_start_it is not None):
            try:
                it_now = int(cfg.get("_current_it", int(warmstart_start_it)))
                warmstart_active = (it_now - int(warmstart_start_it)) <= warmstart_iters
            except Exception:
                warmstart_active = True
        if warmstart_active and (np.isfinite(vol_norm)) and (vol_norm > direction_eps):
            legacy_raw = float(ratio_info["ratio_target"]) * mech_norm / max(vol_norm, direction_eps)
            legacy_floor_factor = max(
                0.0, float(cfg.get("stage2_plateau_lambda_warmstart_floor_factor", 1.0))
            )
            legacy_floor = min(lambda_ratio_cap, legacy_floor_factor * legacy_raw)
            if legacy_floor > lambda_target_raw:
                lambda_target_raw = float(legacy_floor)
                ratio_source = "stage2-al-merit+dv-warmstart"
            else:
                ratio_source = "stage2-al-merit"
        else:
            ratio_source = "stage2-al-merit"
        if ratio_cap_before_rate > ratio_cap_base + 1e-15:
            ratio_source = "%s+slow-progress-cap" % str(ratio_source)
        if abs(float(rate_gain) - 1.0) > 1e-12:
            ratio_source = "%s+vf-rate-gain" % str(ratio_source)
        if mech_norm_for_cap > mech_norm + direction_eps:
            ratio_source = "%s+mech-ref-floor" % str(ratio_source)
        if mech_norm <= direction_eps:
            ratio_target = 0.0
        else:
            ratio_target = lambda_target_raw * vol_norm / max(mech_norm, direction_eps)
    else:
        rate_gain = 1.0
        ratio_cap_before_rate = float(ratio_info["ratio_max"])
        mech_norm_ref = float(mech_norm)
        mech_norm_for_cap = float(mech_norm)
        ratio_target = float(ratio_info["ratio_target"])
        ratio_source = str(ratio_info["ratio_source"])
        if (not np.isfinite(vol_norm)) or (vol_norm <= direction_eps):
            lambda_target_raw = float(lambda_floor)
        else:
            lambda_target_raw = ratio_target * mech_norm / max(vol_norm, direction_eps)
    lambda_target = float(lambda_target_raw)
    if lambda_cap is None:
        lambda_target = max(lambda_floor, lambda_target)
    else:
        lambda_target = min(lambda_cap, max(lambda_floor, lambda_target))
    if use_stage2_al:
        smooth = min(1.0, max(0.0, float(cfg.get("stage2_lambda_smoothing", 1.0))))
        lambda_old = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
        lambda_target = (1.0 - smooth) * float(lambda_old) + smooth * float(lambda_target)
        lambda_target = _clamp_lambda_v_to_cfg(cfg, lambda_target, include_plateau_cap=False)
    return dict(
        vf=float(vf),
        vf_target=vf_target,
        vf_violation=float(vf_violation),
        rel_violation=float(rel_violation),
        dv_now=ratio_info["dv_now"],
        dv_min=float(ratio_info["dv_min"]),
        dv0=float(ratio_info["dv0"]),
        dv_effective=float(ratio_info["dv_effective"]),
        dv_progress_frac=float(ratio_info["dv_progress_frac"]),
        ratio_target=float(ratio_target),
        ratio_source=str(ratio_source),
        ratio_min=float(ratio_info["ratio_min"]),
        ratio_max=float(ratio_info["ratio_max"]),
        ratio_cap_base=float(ratio_cap_base if use_stage2_al else ratio_info["ratio_max"]),
        ratio_cap_effective=float(ratio_cap if use_stage2_al else ratio_info["ratio_max"]),
        ratio_cap_before_rate=float(ratio_cap_before_rate),
        vf_rate_gain=float(rate_gain),
        mech_norm=float(mech_norm),
        mech_norm_ref=float(mech_norm_ref),
        mech_norm_for_cap=float(mech_norm_for_cap),
        vol_norm=float(vol_norm),
        direction_eps=float(direction_eps),
        target_vol_component_norm=float(ratio_target * mech_norm),
        lambda_target_raw=float(lambda_target_raw),
        lambda_target=float(lambda_target),
        lambda_floor=float(lambda_floor),
        lambda_cap=lambda_cap,
        lambda_source=str(ratio_source),
        stage2_al_active=bool(use_stage2_al),
    )


def initialize_lambda_v_seed(cfg, res_eval, psi, M=None):
    """Initial nonzero lambda_v seed; later updates are free to move away from it."""
    if not bool(cfg.get("lambda_v_seed_enabled", True)):
        cfg["_lambda_v_early_freeze_active"] = bool(cfg.get("lambda_v_early_freeze_enabled", True))
        return float(cfg.get("lambda_v", 0.0)), None

    seed_ratio = max(0.0, float(cfg.get("lambda_v_seed_ratio", 0.12)))
    seed_info = _lambda_v_target_from_state(
        cfg, res_eval, psi=psi, M=M,
        ratio_override=seed_ratio,
        allow_override_below_min=True,
    )
    lambda_seed = _clamp_lambda_v_to_cfg(cfg, seed_info["lambda_target"], include_plateau_cap=False)
    lambda_old = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
    lambda_new = _clamp_lambda_v_to_cfg(cfg, max(lambda_old, lambda_seed), include_plateau_cap=False)
    cfg["lambda_v"] = float(lambda_new)

    seed_data = dict(
        lambda_old=float(lambda_old),
        lambda_seed=float(lambda_seed),
        lambda_new=float(lambda_new),
        lambda_target_raw=float(seed_info["lambda_target_raw"]),
        lambda_floor=float(seed_info["lambda_floor"]),
        lambda_cap=seed_info["lambda_cap"],
        ratio_target=float(seed_info["ratio_target"]),
        ratio_source=str(seed_info["ratio_source"]),
        dv_now=seed_info["dv_now"],
        dv_min=float(seed_info["dv_min"]),
        dv0=float(seed_info["dv0"]),
        dv_progress_frac=float(seed_info["dv_progress_frac"]),
        vf=float(seed_info["vf"]),
        vf_target=seed_info["vf_target"],
        mech_norm=float(seed_info["mech_norm"]),
        vol_norm=float(seed_info["vol_norm"]),
        target_vol_component_norm=float(seed_info["target_vol_component_norm"]),
    )
    cfg["_lambda_v_seed_last"] = seed_data
    cfg["_lambda_v_early_freeze_active"] = bool(cfg.get("lambda_v_early_freeze_enabled", True))
    if MPI.rank(MPI.comm_world) == 0:
        print("[lambda_v-seed] lambda_v %.6e -> %.6e (seed=%.6e, uncapped=%.6e, %s, ratio=%.3f, ratio_source=%s, dv_progress=%.3f, dv_state=%s, ||g_mech_perp||=%.6e, ||g_vol_t||=%.6e)" %
              (float(seed_data["lambda_old"]),
               float(seed_data["lambda_new"]),
               float(seed_data["lambda_seed"]),
               float(seed_data["lambda_target_raw"]),
               _format_lambda_v_cap_status(seed_data["lambda_new"], seed_data["lambda_cap"], lambda_floor=seed_data["lambda_floor"]),
               float(seed_data["ratio_target"]),
               str(seed_data["ratio_source"]),
               float(seed_data["dv_progress_frac"]),
               _format_stage_dv_state(cfg, dv_now_raw=seed_data["dv_now"]),
               float(seed_data["mech_norm"]),
               float(seed_data["vol_norm"])),
              flush=True)
        if bool(cfg.get("_lambda_v_early_freeze_active", False)):
            print("[lambda_v-freeze] activate early freeze after initial seed", flush=True)
    return float(lambda_new), seed_data


def _lambda_v_early_freeze_active(cfg):
    return bool(cfg.get("lambda_v_early_freeze_enabled", True)) and bool(cfg.get("_lambda_v_early_freeze_active", False))


def _release_lambda_v_early_freeze(cfg, trigger, it=None):
    if not _lambda_v_early_freeze_active(cfg):
        return False
    cfg["_lambda_v_early_freeze_active"] = False
    if MPI.rank(MPI.comm_world) == 0:
        if it is None:
            print("[lambda_v-freeze] release early freeze on first %s" % str(trigger), flush=True)
        else:
            print("[lambda_v-freeze] it=%03d: release early freeze on first %s" % (int(it), str(trigger)), flush=True)
    return True


def update_lambda_v_after_nucleation(cfg, res_eval, psi, M=None, it=None):
    """Adapt lambda_v after a successful hard nucleation move."""
    _release_lambda_v_early_freeze(cfg, "hard nucleation", it=it)
    update_enabled = cfg.get(
        "nucleation_update_lambda_v",
        cfg.get("lambda_v_adapt_on_nucleation", True)
    )
    if not bool(update_enabled):
        return float(cfg.get("lambda_v", 0.0)), None
    lambda_old = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
    seed_info = _lambda_v_target_from_state(cfg, res_eval, psi=psi, M=M)
    lambda_new = _clamp_lambda_v_to_cfg(cfg, seed_info["lambda_target"], include_plateau_cap=False)
    cfg["lambda_v"] = float(lambda_new)
    adapt_info = dict(
        lambda_old=float(lambda_old),
        lambda_target_raw=float(seed_info["lambda_target_raw"]),
        lambda_target=float(seed_info["lambda_target"]),
        lambda_new=float(lambda_new),
        lambda_floor=float(seed_info["lambda_floor"]),
        lambda_cap=seed_info["lambda_cap"],
        ratio_target=float(seed_info["ratio_target"]),
        ratio_source=str(seed_info["ratio_source"]),
        rel_vf_violation=float(seed_info["rel_violation"]),
        dv_now=seed_info["dv_now"],
        dv_min=float(seed_info["dv_min"]),
        dv0=float(seed_info["dv0"]),
        dv_progress_frac=float(seed_info["dv_progress_frac"]),
        vf=float(seed_info["vf"]),
        vf_target=seed_info["vf_target"],
        mech_norm=float(seed_info["mech_norm"]),
        vol_norm=float(seed_info["vol_norm"]),
        target_vol_component_norm=float(seed_info["target_vol_component_norm"]),
        seeded=False,
    )
    cfg["_lambda_v_adapt_last"] = adapt_info
    if MPI.rank(MPI.comm_world) == 0:
        if it is None:
            print("[lambda_v-adapt] lambda_v %.6e -> %.6e (target=%.6e, uncapped=%.6e, %s, ratio=%.3f, ratio_source=%s, dv_progress=%.3f, rel_vf_violation=%.3f, dv_state=%s, ||g_mech_perp||=%.6e, ||g_vol_t||=%.6e, target_vol_norm=%.6e)" %
                  (float(adapt_info["lambda_old"]),
                   float(adapt_info["lambda_new"]),
                   float(adapt_info["lambda_target"]),
                   float(adapt_info["lambda_target_raw"]),
                   _format_lambda_v_cap_status(adapt_info["lambda_new"], adapt_info["lambda_cap"], lambda_floor=adapt_info["lambda_floor"]),
                   float(adapt_info["ratio_target"]),
                   str(adapt_info["ratio_source"]),
                   float(adapt_info["dv_progress_frac"]),
                   float(adapt_info["rel_vf_violation"]),
                   _format_stage_dv_state(cfg, dv_now_raw=adapt_info["dv_now"]),
                   float(adapt_info["mech_norm"]),
                   float(adapt_info["vol_norm"]),
                   float(adapt_info["target_vol_component_norm"])),
                  flush=True)
        else:
            print("[lambda_v-adapt] it=%03d: lambda_v %.6e -> %.6e (target=%.6e, uncapped=%.6e, %s, ratio=%.3f, ratio_source=%s, dv_progress=%.3f, rel_vf_violation=%.3f, dv_state=%s, ||g_mech_perp||=%.6e, ||g_vol_t||=%.6e, target_vol_norm=%.6e)" %
                  (int(it),
                   float(adapt_info["lambda_old"]),
                   float(adapt_info["lambda_new"]),
                   float(adapt_info["lambda_target"]),
                   float(adapt_info["lambda_target_raw"]),
                   _format_lambda_v_cap_status(adapt_info["lambda_new"], adapt_info["lambda_cap"], lambda_floor=adapt_info["lambda_floor"]),
                   float(adapt_info["ratio_target"]),
                   str(adapt_info["ratio_source"]),
                   float(adapt_info["dv_progress_frac"]),
                   float(adapt_info["rel_vf_violation"]),
                   _format_stage_dv_state(cfg, dv_now_raw=adapt_info["dv_now"]),
                   float(adapt_info["mech_norm"]),
                   float(adapt_info["vol_norm"]),
                   float(adapt_info["target_vol_component_norm"])),
                  flush=True)
    return float(lambda_new), adapt_info


def update_lambda_v_from_stage_state(cfg, res_eval, psi, M=None, it=None, reason="stage"):
    """Recompute lambda_v from the current dv-controlled target ratio."""
    lambda_old = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
    if bool(cfg.get("_post_nucleation_lambda_zero_active", False)):
        cfg["lambda_v"] = 0.0
        if MPI.rank(MPI.comm_world) == 0:
            if it is None:
                print("[lambda_v-zero-freeze] skip stage update (%s), keep lambda_v 0.000000e+00" %
                      str(reason), flush=True)
            else:
                print("[lambda_v-zero-freeze] it=%03d: skip stage update (%s), keep lambda_v 0.000000e+00" %
                      (int(it), str(reason)), flush=True)
        return 0.0, None
    if _lambda_v_early_freeze_active(cfg):
        if MPI.rank(MPI.comm_world) == 0:
            if it is None:
                print("[lambda_v-freeze] skip stage update (%s), keep lambda_v %.6e" %
                      (str(reason), float(lambda_old)), flush=True)
            else:
                print("[lambda_v-freeze] it=%03d: skip stage update (%s), keep lambda_v %.6e" %
                      (int(it), str(reason), float(lambda_old)), flush=True)
        return float(lambda_old), None
    if (
        bool(cfg.get("stage2_volume_continuation_enabled", False))
        and (not stage2_volume_continuation_active(cfg, vf=res_eval.get("vf", None), vf_target=current_vf_target(cfg)))
    ):
        cfg["lambda_v"] = 0.0
        if MPI.rank(MPI.comm_world) == 0:
            if it is None:
                print("[lambda_v-stage] reason=%s: stage2 continuation inactive -> keep lambda_v 0.000000e+00" %
                      str(reason), flush=True)
            else:
                print("[lambda_v-stage] it=%03d reason=%s: stage2 continuation inactive -> keep lambda_v 0.000000e+00" %
                      (int(it), str(reason)), flush=True)
        return 0.0, None
    target_info = _lambda_v_target_from_state(cfg, res_eval, psi=psi, M=M)
    lambda_new = _clamp_lambda_v_to_cfg(cfg, target_info["lambda_target"], include_plateau_cap=False)
    if abs(lambda_new - lambda_old) <= 1e-16:
        return float(lambda_old), None
    cfg["lambda_v"] = float(lambda_new)
    adapt_info = dict(
        reason=str(reason),
        lambda_old=float(lambda_old),
        lambda_target_raw=float(target_info["lambda_target_raw"]),
        lambda_target=float(target_info["lambda_target"]),
        lambda_new=float(lambda_new),
        lambda_floor=float(target_info["lambda_floor"]),
        lambda_cap=target_info["lambda_cap"],
        ratio_target=float(target_info["ratio_target"]),
        ratio_source=str(target_info["ratio_source"]),
        ratio_cap_base=float(target_info["ratio_cap_base"]),
        ratio_cap_effective=float(target_info["ratio_cap_effective"]),
        ratio_cap_before_rate=float(target_info["ratio_cap_before_rate"]),
        vf_rate_gain=float(target_info["vf_rate_gain"]),
        dv_now=target_info["dv_now"],
        dv_min=float(target_info["dv_min"]),
        dv0=float(target_info["dv0"]),
        dv_progress_frac=float(target_info["dv_progress_frac"]),
        vf=float(target_info["vf"]),
        vf_target=target_info["vf_target"],
        mech_norm=float(target_info["mech_norm"]),
        mech_norm_ref=float(target_info["mech_norm_ref"]),
        mech_norm_for_cap=float(target_info["mech_norm_for_cap"]),
        vol_norm=float(target_info["vol_norm"]),
    )
    cfg["_lambda_v_stage_last"] = adapt_info
    if MPI.rank(MPI.comm_world) == 0:
        if it is None:
            print("[lambda_v-stage] reason=%s: lambda_v %.6e -> %.6e (target=%.6e, uncapped=%.6e, %s, ratio=%.3f, ratio_cap=%.3f->%.3f, vf_rate_gain=%.3f, mech_cap_norm=%.3e(ref=%.3e), ratio_source=%s, dv_progress=%.3f, dv_state=%s, vf=%.6f, vf_target=%s)" %
                  (str(adapt_info["reason"]),
                   float(adapt_info["lambda_old"]),
                   float(adapt_info["lambda_new"]),
                   float(adapt_info["lambda_target"]),
                   float(adapt_info["lambda_target_raw"]),
                   _format_lambda_v_cap_status(adapt_info["lambda_new"], adapt_info["lambda_cap"], lambda_floor=adapt_info["lambda_floor"]),
                   float(adapt_info["ratio_target"]),
                   float(adapt_info["ratio_cap_base"]),
                   float(adapt_info["ratio_cap_effective"]),
                   float(adapt_info["vf_rate_gain"]),
                   float(adapt_info["mech_norm_for_cap"]),
                   float(adapt_info["mech_norm_ref"]),
                   str(adapt_info["ratio_source"]),
                   float(adapt_info["dv_progress_frac"]),
                   _format_stage_dv_state(cfg, dv_now_raw=adapt_info["dv_now"]),
                   float(adapt_info["vf"]),
                   ("None" if adapt_info["vf_target"] is None else ("%.6f" % float(adapt_info["vf_target"])))),
                  flush=True)
        else:
            print("[lambda_v-stage] it=%03d reason=%s: lambda_v %.6e -> %.6e (target=%.6e, uncapped=%.6e, %s, ratio=%.3f, ratio_cap=%.3f->%.3f, vf_rate_gain=%.3f, mech_cap_norm=%.3e(ref=%.3e), ratio_source=%s, dv_progress=%.3f, dv_state=%s, vf=%.6f, vf_target=%s)" %
                  (int(it),
                   str(adapt_info["reason"]),
                   float(adapt_info["lambda_old"]),
                   float(adapt_info["lambda_new"]),
                   float(adapt_info["lambda_target"]),
                   float(adapt_info["lambda_target_raw"]),
                   _format_lambda_v_cap_status(adapt_info["lambda_new"], adapt_info["lambda_cap"], lambda_floor=adapt_info["lambda_floor"]),
                   float(adapt_info["ratio_target"]),
                   float(adapt_info["ratio_cap_base"]),
                   float(adapt_info["ratio_cap_effective"]),
                   float(adapt_info["vf_rate_gain"]),
                   float(adapt_info["mech_norm_for_cap"]),
                   float(adapt_info["mech_norm_ref"]),
                   str(adapt_info["ratio_source"]),
                   float(adapt_info["dv_progress_frac"]),
                   _format_stage_dv_state(cfg, dv_now_raw=adapt_info["dv_now"]),
                   float(adapt_info["vf"]),
                   ("None" if adapt_info["vf_target"] is None else ("%.6f" % float(adapt_info["vf_target"])))),
                  flush=True)
    return float(lambda_new), adapt_info


def update_lambda_v_on_vf_plateau(cfg, res_eval, vf_history, psi, M=None, it=None):
    """Boost lambda_v when vf decrease enters a long flat region."""
    if not bool(cfg.get("lambda_v_plateau_boost_enabled", True)):
        cfg["_lambda_v_plateau_flat_hits"] = 0
        cfg["_lambda_v_plateau_accept_cooldown_left"] = 0
        cfg["_lambda_v_plateau_force_nucleation"] = False
        cfg["_lambda_v_plateau_boost_count_since_nucleation"] = 0
        return float(cfg.get("lambda_v", 0.0)), None
    if bool(cfg.get("_post_nucleation_lambda_zero_active", False)):
        cfg["lambda_v"] = 0.0
        return 0.0, None

    window = max(2, int(cfg.get("lambda_v_plateau_window", 8)))
    if len(vf_history) < window + 1:
        cfg["_lambda_v_plateau_flat_hits"] = 0
        return float(cfg.get("lambda_v", 0.0)), None

    vf_tail = [float(v) for v in vf_history[-(window + 1):]]
    vf_now = float(vf_tail[-1])
    total_drop = max(float(vf_tail[0]) - float(vf_tail[-1]), 0.0)
    avg_drop_abs = float(total_drop) / float(window)
    avg_drop_rel = avg_drop_abs / max(abs(vf_now), 1e-12)

    abs_thresh = max(0.0, float(cfg.get("lambda_v_plateau_abs_drop_thresh", 5e-5)))
    rel_thresh = max(0.0, float(cfg.get("lambda_v_plateau_rel_drop_thresh", 2e-4)))
    flat_detected = (avg_drop_abs <= abs_thresh) or (avg_drop_rel <= rel_thresh)

    vf_target_cfg = current_vf_target(cfg)
    if stage2_volume_continuation_active(cfg, vf=vf_now, vf_target=vf_target_cfg):
        vf_target = vf_target_cfg
    else:
        vf_target = res_eval.get("vf_target", vf_target_cfg)
    vf_target = None if vf_target is None else float(vf_target)
    min_gap_abs = max(0.0, float(cfg.get("lambda_v_plateau_target_gap_abs", 1e-3)))
    min_gap_ratio = max(0.0, float(cfg.get("lambda_v_plateau_target_gap_ratio", 0.02)))
    if vf_target is None:
        gap_ok = True
    else:
        vf_gap = abs(float(vf_now) - float(vf_target))
        gap_need = max(min_gap_abs, min_gap_ratio * max(abs(vf_target), 1e-12))
        gap_ok = bool(vf_gap > gap_need)

    flat_hits = int(cfg.get("_lambda_v_plateau_flat_hits", 0))
    if flat_detected and gap_ok:
        flat_hits += 1
    else:
        flat_hits = 0
    cfg["_lambda_v_plateau_flat_hits"] = int(flat_hits)

    hit_need = max(1, int(cfg.get("lambda_v_plateau_consecutive_hits", 1)))
    cooldown = max(0, int(cfg.get("lambda_v_plateau_cooldown_iters", 3)))
    last_boost_it = cfg.get("_lambda_v_plateau_last_boost_it", None)
    in_cooldown = (last_boost_it is not None) and (it is not None) and (int(it) - int(last_boost_it) < cooldown)
    if (flat_hits < hit_need) or in_cooldown:
        return float(cfg.get("lambda_v", 0.0)), None

    _release_lambda_v_early_freeze(cfg, "lambda_v plateau", it=it)
    lambda_old = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=True)
    boosts_since_nucleation = max(0, int(cfg.get("_lambda_v_plateau_boost_count_since_nucleation", 0)))
    if bool(cfg.get("lambda_v_plateau_second_hit_force_nucleation", True)) and (boosts_since_nucleation >= 1):
        dv_now = float(current_stage_dv(cfg) if (current_stage_dv(cfg) is not None) else cfg.get("vf_stage_dv0", 0.03))
        dv_min = float(cfg.get("vf_stage_dv_min", 0.002))
        if dv_now > dv_min + 1e-15:
            return float(lambda_old), None
        cfg["_lambda_v_plateau_flat_hits"] = 0
        cfg["_lambda_v_plateau_force_nucleation"] = True
        force_info = dict(
            action="force_nucleation",
            boosts_since_nucleation=int(boosts_since_nucleation),
            avg_drop_abs=float(avg_drop_abs),
            avg_drop_rel=float(avg_drop_rel),
            vf=float(vf_now),
            vf_target=vf_target,
            dv_now=float(dv_now),
            dv_min=float(dv_min),
        )
        cfg["_lambda_v_plateau_boost_last"] = force_info
        if MPI.rank(MPI.comm_world) == 0:
            if it is None:
                print("[lambda_v-plateau] second plateau trigger after prior boost -> skip lambda_v boost and force nucleation", flush=True)
            else:
                print("[lambda_v-plateau] it=%03d: second plateau trigger after prior boost -> skip lambda_v boost and force nucleation" % int(it), flush=True)
        return float(lambda_old), force_info

    seed_info = _lambda_v_target_from_state(cfg, res_eval, psi=psi, M=M)
    boost_factor = max(1.0, float(cfg.get("lambda_v_plateau_boost_factor", 1.05)))
    ratio_step = max(0.0, float(cfg.get("lambda_v_plateau_ratio_step", 0.005)))
    seeded = bool(lambda_old <= 1e-16)
    lambda_floor, lambda_cap = _lambda_v_cap_from_cfg(cfg, include_plateau_cap=True)
    ratio_max = float(seed_info["ratio_max"])
    mech_norm = float(seed_info["mech_norm"])
    vol_norm = float(seed_info["vol_norm"])
    direction_eps = max(1e-30, float(seed_info["direction_eps"]))
    if seeded:
        lambda_base = float(seed_info["lambda_target"])
        current_ratio = 0.0
        ratio_new = min(ratio_max, float(seed_info["ratio_target"]) * boost_factor)
        lambda_new = max(lambda_floor, ratio_new * mech_norm / max(vol_norm, direction_eps)) if vol_norm > direction_eps else float(lambda_floor)
    else:
        if mech_norm <= direction_eps:
            current_ratio = 0.0
        else:
            current_ratio = float(lambda_old) * vol_norm / max(mech_norm, direction_eps)
        ratio_new = min(ratio_max, current_ratio + ratio_step)
        if vol_norm <= direction_eps:
            lambda_new = float(lambda_floor)
        else:
            lambda_new = max(lambda_floor, ratio_new * mech_norm / max(vol_norm, direction_eps))
    lambda_new = _clamp_lambda_v_to_cfg(cfg, lambda_new, include_plateau_cap=True)
    if (not seeded) and (lambda_new <= lambda_old + 1e-16):
        return float(cfg.get("lambda_v", 0.0)), None

    cfg["lambda_v"] = float(lambda_new)
    cfg["_lambda_v_plateau_flat_hits"] = 0
    cfg["_lambda_v_plateau_boost_count_since_nucleation"] = int(boosts_since_nucleation + 1)
    if bool(cfg.get("lambda_v_plateau_accept_cooldown_enabled", True)):
        cfg["_lambda_v_plateau_accept_cooldown_left"] = max(
            0, int(cfg.get("lambda_v_plateau_accept_cooldown_steps", 3))
        )
    else:
        cfg["_lambda_v_plateau_accept_cooldown_left"] = 0
    if it is not None:
        cfg["_lambda_v_plateau_last_boost_it"] = int(it)
    else:
        cfg["_lambda_v_plateau_last_boost_it"] = None

    adapt_info = dict(
        lambda_old=float(lambda_old),
        lambda_seed=float(seed_info["lambda_target"]),
        lambda_new=float(lambda_new),
        lambda_cap=lambda_cap,
        lambda_floor=float(lambda_floor),
        boost_factor=float(boost_factor),
        ratio_step=float(ratio_step),
        avg_drop_abs=float(avg_drop_abs),
        avg_drop_rel=float(avg_drop_rel),
        abs_thresh=float(abs_thresh),
        rel_thresh=float(rel_thresh),
        flat_hits=int(flat_hits),
        hit_need=int(hit_need),
        vf=float(vf_now),
        vf_target=vf_target,
        ratio_target=float(seed_info["ratio_target"]),
        rel_vf_violation=float(seed_info["rel_violation"]),
        dv_now=seed_info["dv_now"],
        dv_min=float(seed_info["dv_min"]),
        dv0=float(seed_info["dv0"]),
        dv_progress_frac=float(seed_info["dv_progress_frac"]),
        mech_norm=float(seed_info["mech_norm"]),
        vol_norm=float(seed_info["vol_norm"]),
        target_vol_component_norm=float(seed_info["target_vol_component_norm"]),
        ratio_current=float(current_ratio),
        ratio_new=float(ratio_new),
        seeded=bool(seeded),
    )
    cfg["_lambda_v_plateau_boost_last"] = adapt_info
    if MPI.rank(MPI.comm_world) == 0:
        if bool(adapt_info["seeded"]):
            mode_msg = " seed=%.6e x%.3f" % (float(adapt_info["lambda_seed"]), float(adapt_info["boost_factor"]))
        else:
            mode_msg = " ratio=%.3f->%.3f (+%.3f)" % (
                float(adapt_info["ratio_current"]),
                float(adapt_info["ratio_new"]),
                float(adapt_info["ratio_step"]),
            )
        if it is None:
            print("[lambda_v-plateau] lambda_v %.6e -> %.6e%s (%s, avg_drop_abs=%.3e, avg_drop_rel=%.3e, vf=%.6f, vf_target=%s, dv_progress=%.3f, dv=%s, dv0=%.6f, dv_min=%.6f)" %
                  (float(adapt_info["lambda_old"]),
                   float(adapt_info["lambda_new"]),
                   mode_msg,
                   _format_lambda_v_cap_status(adapt_info["lambda_new"], adapt_info["lambda_cap"], lambda_floor=adapt_info["lambda_floor"]),
                   float(adapt_info["avg_drop_abs"]),
                   float(adapt_info["avg_drop_rel"]),
                   float(adapt_info["vf"]),
                   ("None" if adapt_info["vf_target"] is None else ("%.6f" % float(adapt_info["vf_target"]))),
                   float(adapt_info["dv_progress_frac"]),
                   ("None" if adapt_info["dv_now"] is None else ("%.6f" % float(adapt_info["dv_now"]))),
                   float(adapt_info["dv0"]),
                   float(adapt_info["dv_min"])) ,
                  flush=True)
        else:
            print("[lambda_v-plateau] it=%03d: lambda_v %.6e -> %.6e%s (%s, avg_drop_abs=%.3e, avg_drop_rel=%.3e, vf=%.6f, vf_target=%s, dv_progress=%.3f, dv=%s, dv0=%.6f, dv_min=%.6f)" %
                  (int(it),
                   float(adapt_info["lambda_old"]),
                   float(adapt_info["lambda_new"]),
                   mode_msg,
                   _format_lambda_v_cap_status(adapt_info["lambda_new"], adapt_info["lambda_cap"], lambda_floor=adapt_info["lambda_floor"]),
                   float(adapt_info["avg_drop_abs"]),
                   float(adapt_info["avg_drop_rel"]),
                   float(adapt_info["vf"]),
                   ("None" if adapt_info["vf_target"] is None else ("%.6f" % float(adapt_info["vf_target"]))),
                   float(adapt_info["dv_progress_frac"]),
                   ("None" if adapt_info["dv_now"] is None else ("%.6f" % float(adapt_info["dv_now"]))),
                   float(adapt_info["dv0"]),
                   float(adapt_info["dv_min"])) ,
                  flush=True)
    return float(lambda_new), adapt_info


def _vf_from_lsf_state(lsf, threshold, dxm, vol_total):
    """DG0/material-consistent vf estimate from the current level-set state."""
    mesh = lsf.function_space().mesh()
    V0 = FunctionSpace(mesh, "DG", 0)
    lsf_dg0 = _project(lsf, V0)
    vf = assemble(conditional(lt(lsf_dg0, Constant(float(threshold))),
                              Constant(1.0), Constant(0.0)) * dxm)
    return float(vf) / max(1e-30, float(vol_total))


def current_vf_target(cfg):
    """Current stage volume target."""
    return cfg.get("_vf_stage_target", cfg.get("vf_target", None))

def current_stage_dv(cfg):
    return cfg.get("_vf_stage_dv", cfg.get("vf_stage_dv0", None))

def stage_success_tolerance(cfg, dv=None):
    vf_stage_tol_fixed = cfg.get("vf_stage_tol", None)
    if vf_stage_tol_fixed is not None:
        return float(vf_stage_tol_fixed)
    if dv is None:
        dv = current_stage_dv(cfg)
    if dv is None:
        return float(cfg.get("vf_stage_tol_min", 0.002))
    return max(0.2 * float(dv), float(cfg.get("vf_stage_tol_min", 0.002)))

def stage_controller_tolerance(cfg, dv=None):
    """
    Effective tolerance used only by the stage controller.

    Keep it strictly smaller than the current stage step so the controller
    cannot simultaneously regard the same vf as both:
      - a success for the current stage, and
      - a miss for the next shrunken stage.
    This avoids ping-pong when dv has already reached dv_min.
    """
    tol_raw = float(stage_success_tolerance(cfg, dv=dv))
    if dv is None:
        dv = current_stage_dv(cfg)
    if dv is None:
        return tol_raw
    dv = float(dv)
    if dv <= 0.0:
        return tol_raw
    hysteresis_factor = float(cfg.get("vf_stage_hysteresis_factor", 0.49))
    hysteresis_factor = min(max(hysteresis_factor, 0.0), 0.499999)
    tol_cap = hysteresis_factor * dv
    return min(tol_raw, tol_cap)

def initialise_stage_controller(cfg, vf_init):
    vf_init = float(vf_init)
    dv_levels = _stage_dv_levels_from_cfg(cfg)
    dv0 = float(dv_levels[0] if len(dv_levels) > 0 else cfg.get("vf_stage_dv0", 0.03))
    cfg["_vf_stage_dv"] = dv0
    advance_stage_target(cfg, vf_init, dv=dv0)
    cfg["_vf_stage_iter_count"] = 0
    cfg["_vf_stage_rebound_active"] = False
    # Reference vf for dual-band interior ramping.
    cfg["_vf_rank_ref_vf"] = vf_init


def advance_stage_target(cfg, stage_base, dv=None):
    stage_base = float(stage_base)
    if dv is None:
        dv = float(current_stage_dv(cfg))
    else:
        dv = float(dv)
    dv = _cap_stage_dv_for_vf(cfg, dv, stage_base)
    cfg["_vf_stage_dv"] = dv
    vf_final = float(cfg.get("vf_final_target", 0.0))
    if bool(cfg.get("stage2_volume_continuation_enabled", False)):
        vf_final = max(vf_final, float(cfg.get("stage2_volume_end_vf", cfg.get("hard_shift_switch_to_shift_vf", vf_final))))
    cfg["_vf_stage_target"] = max(vf_final, stage_base - dv)
    cfg["_vf_stage_iter_count"] = 0
    return dv


def shrink_stage_step(cfg, vf_now):
    dv = _adjacent_stage_dv(cfg, dv_now_raw=current_stage_dv(cfg), direction="smaller")
    if dv is None:
        dv = float(current_stage_dv(cfg))
    dv = _cap_stage_dv_for_vf(cfg, dv, float(vf_now))
    cfg["_vf_stage_dv"] = dv
    vf_final = float(cfg.get("vf_final_target", 0.0))
    if bool(cfg.get("stage2_volume_continuation_enabled", False)):
        vf_final = max(vf_final, float(cfg.get("stage2_volume_end_vf", cfg.get("hard_shift_switch_to_shift_vf", vf_final))))
    cfg["_vf_stage_target"] = max(vf_final, float(vf_now) - dv)
    cfg["_vf_stage_iter_count"] = 0
    return dv


def build_ranked_deletion_direction(lsf, g_obj, vf_now, vf_stage, cfg, Vls, strength=1.0):
    debug_progress = _debug_progress_enabled(cfg) and (MPI.rank(MPI.comm_world) == 0)
    g_used = Function(Vls)
    copy_function_values(g_used, g_obj)
    strength = max(0.0, min(1.0, float(strength)))
    rank_info = {
        "active": False,
        "band": 0.0,
        "beta_rank": 0.0,
        "p_select": 0.0,
        "n_selected": 0,
        "n_candidates": 0,
        "n_selected_local": 0,
        "n_candidates_local": 0,
        "strength": strength,
        "beta_rank_boundary": 0.0,
        "p_select_boundary": 0.0,
        "n_selected_boundary": 0,
        "n_candidates_boundary": 0,
        "beta_rank_interior": 0.0,
        "p_select_interior": 0.0,
        "n_selected_interior": 0,
        "n_candidates_interior": 0,
        "interior_strength": 0.0,
        "use_boundary_band": True,
        "use_interior_band": True,
    }
    if (not bool(cfg.get("use_ranked_volume_enhancement", True))) or (vf_stage is None):
        return g_used, rank_info
    use_boundary_channel = bool(cfg.get("vf_rank_use_boundary_band", True))
    use_interior_channel = bool(cfg.get("vf_rank_use_interior_band", True))
    rank_info["use_boundary_band"] = bool(use_boundary_channel)
    rank_info["use_interior_band"] = bool(use_interior_channel)
    if (not use_boundary_channel) and (not use_interior_channel):
        return g_used, rank_info
    vf_stage_tol = float(stage_controller_tolerance(cfg))
    e_stage_raw = float(vf_now) - float(vf_stage)
    e_stage = max(0.0, e_stage_raw)
    if e_stage <= 0.0:
        return g_used, rank_info
    if strength <= 0.0:
        return g_used, rank_info
    use_hmin_band = bool(cfg.get("vf_rank_use_hmin_band", False))
    if use_hmin_band:
        band_factor = float(cfg.get("vf_rank_band_hmin_factor", 2.4))
        band = band_factor * float(Vls.mesh().hmin())
    elif bool(cfg.get("vf_rank_band_stage_scaled", False)):
        # Use the stage-controller tolerance here too:
        # once vf enters [vf_stage, vf_stage + tol], ranked deletion should switch off.
        # band = bmin + (bmax-bmin) * min(1, e_stage_eff / d0)
        d0 = float(cfg.get("vf_rank_band_gap_scale_d0", cfg.get("vf_stage_dv0", 0.03)))
        if d0 <= 1e-30:
            d0 = 0.03
        bmin = float(cfg.get("vf_rank_band_min", 0.03))
        bmax = float(cfg.get("vf_rank_band_max", cfg.get("vf_rank_band", 0.2)))
        if bmax < bmin:
            bmin, bmax = bmax, bmin
        scale = min(1.0, e_stage / d0)
        band = bmin + (bmax - bmin) * scale
    else:
        band = float(cfg.get("vf_rank_band", 0.08))
    rank_info["band"] = float(band)
    kp = float(cfg.get("vf_rank_kp", 0.8))
    pmin = float(cfg.get("vf_rank_pmin", 0.01))
    pmax = float(cfg.get("vf_rank_pmax", 0.20))
    beta0 = float(cfg.get("vf_rank_beta0", 0.0))
    kbeta = float(cfg.get("vf_rank_kbeta", 1.0))
    if debug_progress:
        print("[progress] ranked: before lsf.vector().get_local()")
    lsf_vals = lsf.vector().get_local()
    if debug_progress:
        print("[progress] ranked: after lsf.vector().get_local()")
        print("[progress] ranked: before g_obj.vector().get_local()")
    g_vals = g_obj.vector().get_local()
    if debug_progress:
        print("[progress] ranked: after g_obj.vector().get_local()")
    cand_boundary = np.empty(0, dtype=np.intp)
    if use_boundary_channel:
        if debug_progress:
            print("[progress] ranked: before boundary candidate selection")
        cand_boundary = np.where((lsf_vals < 0.0) & (np.abs(lsf_vals) <= band))[0]
        if debug_progress:
            print("[progress] ranked: after boundary candidate selection (n=%d)" % int(cand_boundary.size))
        if cand_boundary.size == 0:
            if debug_progress:
                print("[progress] ranked: before fallback solid candidate selection")
            cand_boundary = np.where(lsf_vals < 0.0)[0]
            if debug_progress:
                print("[progress] ranked: after fallback solid candidate selection (n=%d)" % int(cand_boundary.size))

    p_select_base = min(max(kp * e_stage, pmin), pmax)
    select_all = bool(cfg.get("vf_rank_select_all_candidates", False))

    p_select_boundary = 0.0
    beta_rank_boundary = 0.0
    nsel_boundary = 0
    chosen_boundary = np.empty(0, dtype=np.intp)
    if use_boundary_channel and (cand_boundary.size > 0):
        p_select_boundary = p_select_base * strength
        if select_all:
            nsel_boundary = int(cand_boundary.size)
            chosen_boundary = cand_boundary
            p_select_boundary = 1.0
        else:
            nsel_boundary = max(1, int(np.ceil(p_select_boundary * cand_boundary.size)))
            # Main channel: keep existing boundary-ranked deletion behavior.
            if debug_progress:
                print("[progress] ranked: before argsort on boundary candidates (n_candidates=%d, n_select=%d)" %
                      (int(cand_boundary.size), int(nsel_boundary)))
            order = np.argsort(-g_vals[cand_boundary])  # descending by g
            if debug_progress:
                print("[progress] ranked: after argsort on boundary candidates")
            chosen_boundary = cand_boundary[order[:nsel_boundary]]
        beta_rank_boundary = strength * (beta0 + kbeta * e_stage)

    # Auxiliary interior channel:
    # starts from zero and increases gradually as vf decreases, with a hard cap.
    interior_strength = 0.0
    p_select_interior = 0.0
    beta_rank_interior = 0.0
    nsel_interior = 0
    chosen_interior = np.empty(0, dtype=np.intp)
    cand_interior = np.empty(0, dtype=np.intp)
    if use_interior_channel:
        vf_final = float(cfg.get("vf_final_target", 0.0))
        vf_ref = float(cfg.get("_vf_rank_ref_vf", vf_now))
        if vf_ref < float(vf_now):
            vf_ref = float(vf_now)
            cfg["_vf_rank_ref_vf"] = vf_ref
        denom = max(vf_ref - vf_final, 1e-12)
        vf_progress = min(1.0, max(0.0, (vf_ref - float(vf_now)) / denom))
        progress_start = float(cfg.get("vf_rank_interior_progress_start", 0.20))
        progress_full = float(cfg.get("vf_rank_interior_progress_full", 0.85))
        if progress_full < progress_start:
            progress_start, progress_full = progress_full, progress_start
        progress_power = max(0.1, float(cfg.get("vf_rank_interior_progress_power", 1.0)))
        interior_max_strength = min(max(float(cfg.get("vf_rank_interior_max_strength", 0.35)), 0.0), 1.0)
        if vf_progress <= progress_start:
            interior_strength = 0.0
        elif vf_progress >= progress_full:
            interior_strength = interior_max_strength
        else:
            progress_span = max(progress_full - progress_start, 1e-12)
            progress_unit = (vf_progress - progress_start) / progress_span
            interior_strength = interior_max_strength * (progress_unit ** progress_power)

        interior_depth_abs = max(0.0, float(cfg.get("vf_rank_interior_depth_abs", 0.25)))
        interior_depth_band_mult = max(1.0, float(cfg.get("vf_rank_interior_depth_band_mult", 2.0)))
        interior_depth_min = max(interior_depth_abs, interior_depth_band_mult * float(band))
        interior_mask = (lsf_vals < -interior_depth_min)
        if cand_boundary.size > 0:
            interior_mask[cand_boundary] = False
        cand_interior = np.where(interior_mask)[0]

        if (interior_strength > 0.0) and (cand_interior.size > 0):
            interior_p_scale = max(0.0, float(cfg.get("vf_rank_interior_p_scale", 0.50)))
            interior_pmax = min(max(float(cfg.get("vf_rank_interior_pmax", 0.08)), 0.0), 1.0)
            p_select_interior = min(
                interior_pmax,
                p_select_base * strength * interior_p_scale * interior_strength
            )
            if select_all:
                nsel_interior = int(cand_interior.size)
                chosen_interior = cand_interior
                p_select_interior = 1.0
            elif p_select_interior > 0.0:
                nsel_interior = max(1, int(np.ceil(p_select_interior * cand_interior.size)))
                if debug_progress:
                    print("[progress] ranked: before argsort on interior candidates (n_candidates=%d, n_select=%d)" %
                          (int(cand_interior.size), int(nsel_interior)))
                order_interior = np.argsort(-g_vals[cand_interior])  # descending by g
                if debug_progress:
                    print("[progress] ranked: after argsort on interior candidates")
                chosen_interior = cand_interior[order_interior[:nsel_interior]]
            interior_beta_scale = max(0.0, float(cfg.get("vf_rank_interior_beta_scale", 0.35)))
            beta_rank_interior = (
                strength
                * interior_strength
                * interior_beta_scale
                * (beta0 + kbeta * e_stage)
            )

    p_select = max(float(p_select_boundary), float(p_select_interior))
    beta_rank = float(beta_rank_boundary) + float(beta_rank_interior)
    g_new = g_vals.copy()
    if debug_progress:
        print("[progress] ranked: before writing selected bias")
    if chosen_boundary.size > 0:
        g_new[chosen_boundary] += beta_rank_boundary
    if (chosen_interior.size > 0) and (beta_rank_interior > 0.0):
        g_new[chosen_interior] += beta_rank_interior
    if debug_progress:
        print("[progress] ranked: before g_used.vector().set_local()")
    g_used.vector().set_local(g_new)
    if debug_progress:
        print("[progress] ranked: after g_used.vector().set_local()")
        print("[progress] ranked: before g_used.vector().apply('insert')")
    g_used.vector().apply("insert")
    if debug_progress:
        print("[progress] ranked: after g_used.vector().apply('insert')")
        print("[progress] ranked: after writing selected bias")
    n_candidates_boundary_global = _MPI_COMM.allreduce(int(cand_boundary.size), op=_pyMPI.SUM) if _MPI_SIZE > 1 else int(cand_boundary.size)
    n_selected_boundary_global = _MPI_COMM.allreduce(int(nsel_boundary), op=_pyMPI.SUM) if _MPI_SIZE > 1 else int(nsel_boundary)
    n_candidates_interior_global = _MPI_COMM.allreduce(int(cand_interior.size), op=_pyMPI.SUM) if _MPI_SIZE > 1 else int(cand_interior.size)
    n_selected_interior_global = _MPI_COMM.allreduce(int(nsel_interior), op=_pyMPI.SUM) if _MPI_SIZE > 1 else int(nsel_interior)
    n_candidates_global = int(n_candidates_boundary_global + n_candidates_interior_global)
    n_selected_global = int(n_selected_boundary_global + n_selected_interior_global)
    rank_info = {
        "active": bool((chosen_boundary.size > 0) or (chosen_interior.size > 0)),
        "band": float(band),
        "beta_rank": float(beta_rank),
        "p_select": float(p_select),
        "n_selected": n_selected_global,
        "n_candidates": n_candidates_global,
        "n_selected_local": int(nsel_boundary + nsel_interior),
        "n_candidates_local": int(cand_boundary.size + cand_interior.size),
        "strength": strength,
        "beta_rank_boundary": float(beta_rank_boundary),
        "p_select_boundary": float(p_select_boundary),
        "n_selected_boundary": int(n_selected_boundary_global),
        "n_candidates_boundary": int(n_candidates_boundary_global),
        "beta_rank_interior": float(beta_rank_interior),
        "p_select_interior": float(p_select_interior),
        "n_selected_interior": int(n_selected_interior_global),
        "n_candidates_interior": int(n_candidates_interior_global),
        "interior_strength": float(interior_strength),
        "use_boundary_band": bool(use_boundary_channel),
        "use_interior_band": bool(use_interior_channel),
        "vf_stage_tol": float(vf_stage_tol),
        "e_stage_raw": float(e_stage_raw),
        "e_stage_eff": float(e_stage),
    }
    return g_used, rank_info


def helmholtz_filter_lsf(lsf_in, mesh, radius, use_plain_cg_space=False):
    """
    Apply Helmholtz smoothing:
      (I - r^2 * Delta) psi_f = psi_in
    on a plain scalar CG space.
    """
    r = float(max(radius, 0.0))
    if r <= 0.0:
        out = Function(lsf_in.function_space())
        copy_function_values(out, lsf_in)
        return out
    V_target = lsf_in.function_space()
    elem = V_target.ufl_element()
    family = str(elem.family())
    degree = int(elem.degree())
    # Use a plain CG working space for every Helmholtz solve. This avoids the
    # DOLFIN MPI/subfunction vector-write failure seen on periodic-constrained
    # spaces while keeping the filtered field projected back to V_target.
    V_work = FunctionSpace(mesh, family, degree)
    rhs_in = _project(lsf_in, V_work)
    u = TrialFunction(V_work)
    v = TestFunction(V_work)
    a = (u * v + Constant(r * r) * dot(grad(u), grad(v))) * dx(domain=mesh)
    L = rhs_in * v * dx(domain=mesh)
    out_work = Function(V_work)
    solve(a == L, out_work)
    return _project(out_work, V_target)


def renormalize_lsf_inplace(lsf, dxm):
    """Normalize lsf to unit L2 norm in-place."""
    nrm = float(np.sqrt(max(assemble(lsf * lsf * dxm), 0.0)))
    if nrm > 1e-14:
        lsf.vector()[:] /= nrm
        lsf.vector().apply("insert")
    return nrm


def _make_nucleated_lsf(lsf_try, entity_map, selected_key_pos, psi_void):
    """Create a trial level-set where the selected global candidate keys are forced to void."""
    shifted = Function(lsf_try.function_space())
    vals = lsf_try.vector().get_local().copy()
    local_key_pos = np.asarray(entity_map["local_key_pos"], dtype=np.int64)
    selected_key_pos = np.asarray(selected_key_pos, dtype=np.int64)
    if (vals.size > 0) and (selected_key_pos.size > 0):
        mask = np.isin(local_key_pos, selected_key_pos, assume_unique=False)
        if np.any(mask):
            vals[mask] = np.maximum(vals[mask], float(psi_void))
    shifted.vector().set_local(vals)
    shifted.vector().apply("insert")
    return shifted


def _make_shifted_lsf(lsf_try, c_shift):
    """Create a trial level-set by uniform shift: psi_new = psi_old - c_shift."""
    shifted = Function(lsf_try.function_space())
    vals = lsf_try.vector().get_local().copy()
    vals -= float(c_shift)
    shifted.vector().set_local(vals)
    shifted.vector().apply("insert")
    return shifted


def _select_nucleation_ranking_field(lsf_try, ranking_field):
    """Fallback ranking if no objective field is available."""
    if ranking_field is not None:
        return ranking_field
    rank_fallback = Function(lsf_try.function_space())
    vals = -lsf_try.vector().get_local().copy()
    rank_fallback.vector().set_local(vals)
    rank_fallback.vector().apply("insert")
    return rank_fallback


def build_full_dof_index_map(Vls, Nx, Ny, Nz):
    """Build a one-to-one global DOF key map on the regular cube grid."""
    coords = np.asarray(Vls.tabulate_dof_coordinates(), dtype=float).reshape((-1, 3))
    ndof = int(coords.shape[0])
    if ndof == 0:
        return dict(
            local_key_pos=np.zeros((0,), dtype=np.int64),
            global_key_codes=np.zeros((0,), dtype=np.int64),
            global_key_counts=np.zeros((0,), dtype=np.int64),
            n_reps=0,
            selection_label="dof",
        )

    ix = _coord_to_grid_index(coords[:, 0], int(Nx))
    iy = _coord_to_grid_index(coords[:, 1], int(Ny))
    iz = _coord_to_grid_index(coords[:, 2], int(Nz))
    mx = np.int64(int(Nx) + 1)
    my = np.int64(int(Ny) + 1)
    local_key_codes = ix.astype(np.int64) + mx * (iy.astype(np.int64) + my * iz.astype(np.int64))
    global_key_codes, local_key_pos = _build_global_key_index(local_key_codes)
    n_keys = int(global_key_codes.size)
    local_cnt = np.bincount(local_key_pos, minlength=n_keys).astype(np.int64)
    global_key_counts = _allreduce_sum_array(local_cnt)
    return dict(
        local_key_pos=local_key_pos,
        global_key_codes=global_key_codes,
        global_key_counts=global_key_counts,
        n_reps=n_keys,
        selection_label="dof",
    )


def _apply_ranked_nucleation_to_target(
    lsf_try,
    ranking_field,
    mesh,
    cfg,
    vf_target_value,
    vol_total=None,
    entity_map=None,
    rank_offset_entities=0,
):
    """
    Delete material only inside a solid inner band:
      delta1 < threshold - psi < delta2
    and rank candidates by the supplied field (typically the symmetry-consistent search direction).
    """
    info = {
        "active": False,
        "q_lo": float(cfg.get("hard_shift_nucleation_quantile_lo", 0.10)),
        "q_hi": float(cfg.get("hard_shift_nucleation_quantile_hi", 0.35)),
        "d_lo": None,
        "d_hi": None,
        "d_expand_steps": 0,
        "psi_void": None,
        "n_candidates": 0,
        "n_selected": 0,
        "n_candidate_entities": 0,
        "n_selected_entities": 0,
        "vf_before": None,
        "vf_after": None,
        "vf_target": None,
        "target_reached": False,
        "selection_mode": "top-g-global",
        "used_all_candidates": False,
        "failure_reason": None,
        "rank_offset_entities": int(max(0, int(rank_offset_entities))),
        "rank_offset_applied": 0,
    }
    if vf_target_value is None:
        info["failure_reason"] = "missing-target"
        return lsf_try, info, None

    threshold = float(cfg["threshold"])
    dxm = Measure("dx", domain=mesh)
    Vol = float(assemble(Constant(1.0) * dxm)) if vol_total is None else float(vol_total)
    vf_now = _vf_from_lsf_state(lsf_try, threshold, dxm, Vol)
    target = float(vf_target_value)
    info["vf_before"] = float(vf_now)
    info["vf_target"] = float(target)

    if target >= vf_now:
        info["vf_after"] = float(vf_now)
        info["failure_reason"] = "target-not-below-current-vf"
        return lsf_try, info, float(vf_now)

    q_lo = min(max(float(info["q_lo"]), 0.0), 1.0)
    q_hi = min(max(float(info["q_hi"]), 0.0), 1.0)
    if q_hi < q_lo:
        q_lo, q_hi = q_hi, q_lo
    info["q_lo"] = float(q_lo)
    info["q_hi"] = float(q_hi)
    vf_tol = float(cfg.get("vf_bisect_tol", 5e-4))
    d_cap = max(0.0, float(cfg.get("hard_shift_nucleation_d_cap", 0.40)))
    d_expand_step = max(1e-12, float(cfg.get("hard_shift_nucleation_d_expand_step", 0.03)))

    if entity_map is None:
        entity_map = build_full_dof_index_map(lsf_try.function_space(), cfg["Nx"], cfg["Ny"], cfg["Nz"])

    lsf_vals = lsf_try.vector().get_local()
    rank_field = _select_nucleation_ranking_field(lsf_try, ranking_field)
    rank_vals = rank_field.vector().get_local()
    local_key_pos = np.asarray(entity_map["local_key_pos"], dtype=np.int64)
    n_keys = int(entity_map["n_reps"])
    key_rank, _ = _global_key_average(rank_vals, local_key_pos, n_keys)
    depth_vals = threshold - lsf_vals
    key_depth, _ = _global_key_average(depth_vals, local_key_pos, n_keys)
    global_key_counts = np.asarray(entity_map["global_key_counts"], dtype=np.int64)
    positive_keys = np.where(key_depth > 0.0)[0]
    info["selection_mode"] = "top-g-global-%s" % str(entity_map.get("selection_label", "dof"))
    if positive_keys.size == 0:
        info["vf_after"] = float(vf_now)
        info["failure_reason"] = "no-solid-keys"
        return lsf_try, info, float(vf_now)

    positive_depth = np.asarray(key_depth[positive_keys], dtype=float)
    global_key_codes = np.asarray(entity_map["global_key_codes"], dtype=np.int64)
    sorted_keys = None
    cache = None
    lsf_all = None
    vf_all = None
    candidate_keys = np.zeros((0,), dtype=np.int64)
    psi_void = None
    _eps_band = 1e-14
    max_depth = float(np.max(positive_depth))
    d_lo_seed = float(np.quantile(positive_depth, q_lo))
    d_hi_seed_raw = float(np.quantile(positive_depth, q_hi))
    d_hi_seed = min(float(d_hi_seed_raw), float(d_cap))
    expand_step_idx = 0
    while True:
        d_lo = max(0.0, float(d_lo_seed) - expand_step_idx * d_expand_step)
        d_hi = min(float(max_depth), float(d_hi_seed) + expand_step_idx * d_expand_step)
        info["d_expand_steps"] = int(expand_step_idx)
        candidate_mask = (positive_depth >= (d_lo - _eps_band)) & (positive_depth <= (d_hi + _eps_band))
        candidate_keys_try = positive_keys[candidate_mask]
        final_expansion = (d_lo <= _eps_band) and (d_hi >= float(max_depth) - _eps_band)
        if candidate_keys_try.size == 0:
            if final_expansion:
                break
            expand_step_idx += 1
            continue

        psi_void_default = threshold + max(float(d_hi), 1e-6)
        psi_void_cfg = cfg.get("hard_shift_nucleation_psi_value", None)
        psi_void_try = float(psi_void_default if psi_void_cfg is None else psi_void_cfg)
        order = np.lexsort((global_key_codes[candidate_keys_try], -key_rank[candidate_keys_try]))
        sorted_keys_full = candidate_keys_try[order]
        rank_offset = min(int(info["rank_offset_entities"]), int(sorted_keys_full.size))
        sorted_keys_try = sorted_keys_full[rank_offset:]
        cache_try = {}

        def evaluate_nsel_try(nsel):
            nsel = int(max(0, min(int(nsel), int(sorted_keys_try.size))))
            if nsel in cache_try:
                return cache_try[nsel]
            lsf_mid = _make_nucleated_lsf(lsf_try, entity_map, sorted_keys_try[:nsel], psi_void_try)
            vf_mid = _vf_from_lsf_state(lsf_mid, threshold, dxm, Vol)
            cache_try[nsel] = (lsf_mid, float(vf_mid))
            return cache_try[nsel]

        lsf_all_try, vf_all_try = evaluate_nsel_try(sorted_keys_try.size)
        use_this_band = (vf_all_try <= target + vf_tol) or final_expansion
        if use_this_band:
            info["d_lo"] = float(d_lo)
            info["d_hi"] = float(d_hi)
            psi_void = float(psi_void_try)
            info["psi_void"] = float(psi_void)
            info["rank_offset_applied"] = int(rank_offset)
            candidate_keys = sorted_keys_try
            sorted_keys = sorted_keys_try
            cache = cache_try
            lsf_all = lsf_all_try
            vf_all = float(vf_all_try)
            break
        expand_step_idx += 1

    info["n_candidate_entities"] = int(candidate_keys.size)
    info["n_candidates"] = int(np.sum(global_key_counts[candidate_keys])) if candidate_keys.size > 0 else 0
    if candidate_keys.size == 0 or sorted_keys is None or cache is None or psi_void is None:
        info["vf_after"] = float(vf_now)
        info["failure_reason"] = "no-candidates-after-fallback"
        return lsf_try, info, float(vf_now)

    def evaluate_nsel(nsel):
        nsel = int(max(0, min(int(nsel), int(sorted_keys.size))))
        if nsel in cache:
            return cache[nsel]
        lsf_mid = _make_nucleated_lsf(lsf_try, entity_map, sorted_keys[:nsel], psi_void)
        vf_mid = _vf_from_lsf_state(lsf_mid, threshold, dxm, Vol)
        cache[nsel] = (lsf_mid, float(vf_mid))
        return cache[nsel]

    lsf_all, vf_all = evaluate_nsel(sorted_keys.size)
    if vf_all > target + vf_tol:
        chosen_n = int(sorted_keys.size)
        lsf_best, vf_best = lsf_all, vf_all
        info["used_all_candidates"] = True
        info["failure_reason"] = "fallback-exhausted-target-not-reached"
    else:
        lo = 1
        hi = int(sorted_keys.size)
        first_feasible = int(sorted_keys.size)
        while lo <= hi:
            mid = (lo + hi) // 2
            _, vf_mid = evaluate_nsel(mid)
            if vf_mid <= target + vf_tol:
                first_feasible = mid
                hi = mid - 1
            else:
                lo = mid + 1
        candidates_to_compare = [first_feasible]
        if first_feasible > 1:
            candidates_to_compare.append(first_feasible - 1)
        best_abs = None
        chosen_n = first_feasible
        lsf_best = None
        vf_best = None
        for nsel in candidates_to_compare:
            lsf_mid, vf_mid = evaluate_nsel(nsel)
            abs_err = abs(float(vf_mid) - target)
            if (best_abs is None) or (abs_err < best_abs):
                best_abs = abs_err
                chosen_n = int(nsel)
                lsf_best = lsf_mid
                vf_best = vf_mid

    info["active"] = bool(chosen_n > 0)
    selected_keys = sorted_keys[:int(chosen_n)]
    info["n_selected_entities"] = int(chosen_n)
    info["n_selected"] = int(np.sum(global_key_counts[selected_keys])) if selected_keys.size > 0 else 0
    info["vf_after"] = float(vf_best)
    info["target_reached"] = bool(float(vf_best) <= target + vf_tol)
    if info["target_reached"]:
        info["failure_reason"] = None
    return lsf_best, info, float(vf_best)


def enforce_hard_volume_fraction(lsf_try, mesh, cfg, vol_total=None, ranking_field=None, entity_map=None):
    """Apply band-limited ranked nucleation to approach the current vf target."""
    use_hard_vf = bool(cfg.get("use_hard_vf", False))
    vf_target = current_vf_target(cfg)
    if (not use_hard_vf) or (vf_target is None):
        return lsf_try, 0.0, None
    shifted, nuc_info, vf_star = _apply_ranked_nucleation_to_target(
        lsf_try, ranking_field, mesh, cfg, vf_target, vol_total=vol_total, entity_map=entity_map
    )
    return shifted, nuc_info, vf_star


def enforce_specific_volume_fraction(
    lsf_try,
    mesh,
    cfg,
    vf_target_value,
    vol_total=None,
    ranking_field=None,
    entity_map=None,
    rank_offset_entities=0,
):
    """Apply band-limited ranked nucleation to approach a caller-provided vf target."""
    return _apply_ranked_nucleation_to_target(
        lsf_try,
        ranking_field,
        mesh,
        cfg,
        vf_target_value,
        vol_total=vol_total,
        entity_map=entity_map,
        rank_offset_entities=rank_offset_entities,
    )


def enforce_specific_volume_fraction_by_shift(lsf_try, mesh, cfg, vf_target_value, vol_total=None):
    """Apply global uniform shift (psi <- psi - c) to approach caller-provided vf target."""
    info = {
        "active": False,
        "selection_mode": "uniform-lsf-shift",
        "q_lo": float("nan"),
        "q_hi": float("nan"),
        "d_lo": float("nan"),
        "d_hi": float("nan"),
        "d_expand_steps": 0,
        "psi_void": float("nan"),
        "n_candidates": 0,
        "n_selected": 0,
        "n_candidate_entities": 0,
        "n_selected_entities": 0,
        "vf_before": None,
        "vf_after": None,
        "vf_target": None,
        "target_reached": False,
        "used_all_candidates": False,
        "failure_reason": None,
        "c_shift": 0.0,
    }
    threshold = float(cfg["threshold"])
    dxm = Measure("dx", domain=mesh)
    Vol = float(assemble(Constant(1.0) * dxm)) if vol_total is None else float(vol_total)
    vf_now = _vf_from_lsf_state(lsf_try, threshold, dxm, Vol)
    target = float(vf_target_value)
    info["vf_before"] = float(vf_now)
    info["vf_target"] = float(target)
    if target >= vf_now:
        info["vf_after"] = float(vf_now)
        info["failure_reason"] = "target-not-below-current-vf"
        return lsf_try, info, float(vf_now)

    vf_tol = float(cfg.get("vf_bisect_tol", 5e-4))
    max_iter = max(8, int(cfg.get("vf_bisect_max_iter", 40)))
    min_psi = float(lsf_try.vector().min())
    c_low = float(min_psi - threshold - 1e-8)  # very negative: tends to vf -> 0
    c_high = 0.0

    cache = {}

    def evaluate_c(c_val):
        c_key = float(c_val)
        if c_key in cache:
            return cache[c_key]
        lsf_mid = _make_shifted_lsf(lsf_try, c_key)
        vf_mid = _vf_from_lsf_state(lsf_mid, threshold, dxm, Vol)
        cache[c_key] = (lsf_mid, float(vf_mid))
        return cache[c_key]

    lsf_low, vf_low = evaluate_c(c_low)
    lsf_high, vf_high = evaluate_c(c_high)
    if vf_low > target + vf_tol:
        # Extremely unlikely; keep safest available shift.
        info["failure_reason"] = "shift-bracket-failed"
        info["c_shift"] = float(c_low)
        info["vf_after"] = float(vf_low)
        info["active"] = bool(abs(c_low) > 0.0)
        info["target_reached"] = bool(vf_low <= target + vf_tol)
        return lsf_low, info, float(vf_low)

    for _ in range(max_iter):
        c_mid = 0.5 * (c_low + c_high)
        lsf_mid, vf_mid = evaluate_c(c_mid)
        if vf_mid <= target + vf_tol:
            c_low = c_mid
            lsf_low, vf_low = lsf_mid, vf_mid
        else:
            c_high = c_mid
            lsf_high, vf_high = lsf_mid, vf_mid

    candidates = [(c_low, lsf_low, vf_low), (c_high, lsf_high, vf_high)]
    c_best, lsf_best, vf_best = min(candidates, key=lambda x: abs(float(x[2]) - target))
    info["c_shift"] = float(c_best)
    info["vf_after"] = float(vf_best)
    info["active"] = bool(abs(float(c_best)) > 0.0)
    info["target_reached"] = bool(float(vf_best) <= target + vf_tol)
    if info["target_reached"]:
        info["failure_reason"] = None
    else:
        info["failure_reason"] = "shift-target-not-reached"
    return lsf_best, info, float(vf_best)


def _coord_to_grid_index(coord, n_axis):
    """Map coordinate in [0,1] to integer grid index in [0, n_axis]."""
    idx = np.rint(np.asarray(coord, dtype=float) * float(n_axis)).astype(np.int64)
    return np.clip(idx, 0, int(n_axis))


def _allreduce_sum_array(arr):
    """MPI allreduce sum for numpy arrays (serial-safe)."""
    a = np.asarray(arr)
    if _MPI_SIZE == 1:
        return a.copy()
    out = np.zeros_like(a)
    _MPI_COMM.Allreduce(a, out, op=_pyMPI.SUM)
    return out


def _allreduce_max_scalar(val):
    """MPI allreduce max for scalar float (serial-safe)."""
    if _MPI_SIZE == 1:
        return float(val)
    return float(_MPI_COMM.allreduce(float(val), op=_pyMPI.MAX))


def psi_global_stats(lsf, dxm):
    """
    Global statistics of the current level-set values.

    Quantiles are taken over the full distributed DOF value vector, while
    `l2` matches the continuous L2 norm used by `renormalize_lsf_inplace`.
    """
    local_vals = np.asarray(lsf.vector().get_local(), dtype=float).ravel()
    gathered = _MPI_COMM.allgather(local_vals) if _MPI_SIZE > 1 else [local_vals]
    parts = [np.asarray(v, dtype=float).ravel() for v in gathered if np.asarray(v).size > 0]
    if len(parts) == 0:
        all_vals = np.zeros((0,), dtype=float)
    else:
        all_vals = np.concatenate(parts)

    if all_vals.size == 0:
        return dict(
            min=0.0, max=0.0, mean=0.0, std=0.0, l2=0.0,
            q01=0.0, q10=0.0, q50=0.0, q90=0.0, q99=0.0,
        )

    qs = np.quantile(all_vals, [0.01, 0.10, 0.50, 0.90, 0.99])
    l2_sq = float(assemble(lsf * lsf * dxm))
    return dict(
        min=float(np.min(all_vals)),
        max=float(np.max(all_vals)),
        mean=float(np.mean(all_vals)),
        std=float(np.std(all_vals)),
        l2=float(np.sqrt(max(l2_sq, 0.0))),
        q01=float(qs[0]),
        q10=float(qs[1]),
        q50=float(qs[2]),
        q90=float(qs[3]),
        q99=float(qs[4]),
    )


def _build_global_key_index(local_key_codes):
    """Build global unique key table and local key positions."""
    local_unique = np.unique(np.asarray(local_key_codes, dtype=np.int64))
    gathered = _MPI_COMM.allgather(local_unique)
    if len(gathered) == 0:
        global_keys = np.zeros((0,), dtype=np.int64)
    else:
        global_keys = np.unique(np.concatenate(gathered)).astype(np.int64)
    key_to_pos = {int(k): i for i, k in enumerate(global_keys.tolist())}
    local_pos = np.asarray([key_to_pos[int(k)] for k in local_key_codes], dtype=np.int64)
    return global_keys, local_pos


def _global_key_average(local_vals, local_key_pos, n_keys):
    """Compute global mean value for each symmetry key."""
    local_sum = np.bincount(local_key_pos, weights=np.asarray(local_vals, dtype=float), minlength=int(n_keys)).astype(float)
    local_cnt = np.bincount(local_key_pos, minlength=int(n_keys)).astype(np.int64)
    gsum = _allreduce_sum_array(local_sum)
    gcnt = _allreduce_sum_array(local_cnt)
    out = np.zeros((int(n_keys),), dtype=float)
    nz = gcnt > 0
    out[nz] = gsum[nz] / gcnt[nz].astype(float)
    return out, gcnt


def _global_key_sum(local_vals, local_key_pos, n_keys):
    """Compute global sum value for each symmetry key."""
    local_sum = np.bincount(local_key_pos, weights=np.asarray(local_vals, dtype=float), minlength=int(n_keys)).astype(float)
    return _allreduce_sum_array(local_sum)


def build_wedge_index_map(Vls, Nx, Ny, Nz):
    """
    Build strict reduced-DOF cubic wedge map using integer grid indices only.
    MPI-strict version:
      - each local DOF gets an integer cubic-orbit key
      - keys are globalized across ranks
      - local DOFs store key-position (global-consistent)
    """
    coords = np.asarray(Vls.tabulate_dof_coordinates(), dtype=float).reshape((-1, 3))
    ndof = int(coords.shape[0])
    if ndof == 0:
        return dict(
            local_key_pos=np.zeros((0,), dtype=np.int64),
            global_key_codes=np.zeros((0,), dtype=np.int64),
            global_key_counts=np.zeros((0,), dtype=np.int64),
            n_reps=0,
        )

    ix = _coord_to_grid_index(coords[:, 0], int(Nx))
    iy = _coord_to_grid_index(coords[:, 1], int(Ny))
    iz = _coord_to_grid_index(coords[:, 2], int(Nz))

    fx = np.minimum(ix, int(Nx) - ix)
    fy = np.minimum(iy, int(Ny) - iy)
    fz = np.minimum(iz, int(Nz) - iz)

    folded = np.column_stack((fx, fy, fz))
    # sort_desc to encode permutation invariance (minimal wedge)
    folded_sorted = np.sort(folded, axis=1)[:, ::-1]
    # Encode integer key uniquely: key = a + M*(b + M*c), with a>=b>=c.
    m = int(max(int(Nx), int(Ny), int(Nz)) // 2 + 1)
    a = folded_sorted[:, 0].astype(np.int64)
    b = folded_sorted[:, 1].astype(np.int64)
    c = folded_sorted[:, 2].astype(np.int64)
    local_key_codes = a + np.int64(m) * (b + np.int64(m) * c)

    global_key_codes, local_key_pos = _build_global_key_index(local_key_codes)
    n_keys = int(global_key_codes.size)
    local_cnt = np.bincount(local_key_pos, minlength=n_keys).astype(np.int64)
    global_key_counts = _allreduce_sum_array(local_cnt)

    return dict(
        local_key_pos=local_key_pos,
        global_key_codes=global_key_codes,
        global_key_counts=global_key_counts,
        n_reps=n_keys,
    )


def build_tetragonal_index_map(Vls, Nx, Ny, Nz):
    """
    Build reduced-DOF map for tetragonal symmetry with z as the unique axis.
    Symmetry group:
      - mirrors x -> 1-x, y -> 1-y, z -> 1-z
      - x/y swap (fourfold symmetry in the x-y plane)
      - no permutation involving z
    """
    coords = np.asarray(Vls.tabulate_dof_coordinates(), dtype=float).reshape((-1, 3))
    ndof = int(coords.shape[0])
    if ndof == 0:
        return dict(
            local_key_pos=np.zeros((0,), dtype=np.int64),
            global_key_codes=np.zeros((0,), dtype=np.int64),
            global_key_counts=np.zeros((0,), dtype=np.int64),
            n_reps=0,
        )

    ix = _coord_to_grid_index(coords[:, 0], int(Nx))
    iy = _coord_to_grid_index(coords[:, 1], int(Ny))
    iz = _coord_to_grid_index(coords[:, 2], int(Nz))

    fx = np.minimum(ix, int(Nx) - ix)
    fy = np.minimum(iy, int(Ny) - iy)
    fz = np.minimum(iz, int(Nz) - iz)

    a = np.maximum(fx, fy).astype(np.int64)
    b = np.minimum(fx, fy).astype(np.int64)
    c = fz.astype(np.int64)
    m = int(max(int(Nx), int(Ny), int(Nz)) // 2 + 1)
    local_key_codes = a + np.int64(m) * (b + np.int64(m) * c)

    global_key_codes, local_key_pos = _build_global_key_index(local_key_codes)
    n_keys = int(global_key_codes.size)
    local_cnt = np.bincount(local_key_pos, minlength=n_keys).astype(np.int64)
    global_key_counts = _allreduce_sum_array(local_cnt)

    return dict(
        local_key_pos=local_key_pos,
        global_key_codes=global_key_codes,
        global_key_counts=global_key_counts,
        n_reps=n_keys,
    )


def build_tetragonal_rot4_index_map(Vls, Nx, Ny, Nz):
    """
    Build reduced-DOF map for pure fourfold rotational symmetry about z.
    Symmetry group:
      - quarter turns about the cell centerline parallel to z
      - z remains the unique axis and is not mirrored
      - no x/y mid-plane mirrors are imposed beyond what C4 generates
    """
    coords = np.asarray(Vls.tabulate_dof_coordinates(), dtype=float).reshape((-1, 3))
    ndof = int(coords.shape[0])
    if ndof == 0:
        return dict(
            local_key_pos=np.zeros((0,), dtype=np.int64),
            global_key_codes=np.zeros((0,), dtype=np.int64),
            global_key_counts=np.zeros((0,), dtype=np.int64),
            n_reps=0,
        )

    ix = _coord_to_grid_index(coords[:, 0], int(Nx))
    iy = _coord_to_grid_index(coords[:, 1], int(Ny))
    iz = _coord_to_grid_index(coords[:, 2], int(Nz))

    if int(Nx) != int(Ny):
        raise ValueError("tetragonal_z_rot4 symmetry requires Nx == Ny so 90-degree rotations preserve the grid.")

    nxy = int(Nx)
    mxy = np.int64(nxy + 1)
    xy_codes = np.stack((
        ix.astype(np.int64) + mxy * iy.astype(np.int64),
        iy.astype(np.int64) + mxy * (nxy - ix).astype(np.int64),
        (nxy - ix).astype(np.int64) + mxy * (nxy - iy).astype(np.int64),
        (nxy - iy).astype(np.int64) + mxy * ix.astype(np.int64),
    ), axis=1)
    rep_xy_code = np.min(xy_codes, axis=1)
    local_key_codes = rep_xy_code + (mxy * mxy) * iz.astype(np.int64)

    global_key_codes, local_key_pos = _build_global_key_index(local_key_codes)
    n_keys = int(global_key_codes.size)
    local_cnt = np.bincount(local_key_pos, minlength=n_keys).astype(np.int64)
    global_key_counts = _allreduce_sum_array(local_cnt)

    return dict(
        local_key_pos=local_key_pos,
        global_key_codes=global_key_codes,
        global_key_counts=global_key_counts,
        n_reps=n_keys,
    )


def build_symmetry_index_map(Vls, Nx, Ny, Nz, symmetry_mode):
    """Dispatch symmetry reduced-DOF map by mode."""
    mode = str(symmetry_mode).strip().lower()
    if mode == "cubic":
        out = build_wedge_index_map(Vls, Nx, Ny, Nz)
        out["selection_label"] = "symmetry-orbit"
        return out
    if mode == "tetragonal_z":
        out = build_tetragonal_index_map(Vls, Nx, Ny, Nz)
        out["selection_label"] = "symmetry-orbit"
        return out
    if mode == "tetragonal_z_rot4":
        out = build_tetragonal_rot4_index_map(Vls, Nx, Ny, Nz)
        out["selection_label"] = "symmetry-orbit"
        return out
    return None


def expand_wedge_to_full(lsf_in, wedge_map):
    """
    Expand to globally strict wedge field by global key averaging:
      psi_key = mean_{all ranks, dof in key}(psi_dof)
      psi_out(local dof) = psi_key(local key)
    """
    vals = lsf_in.vector().get_local()
    key_pos = np.asarray(wedge_map["local_key_pos"], dtype=np.int64)
    n_keys = int(wedge_map["n_reps"])
    key_vals, _ = _global_key_average(vals, key_pos, n_keys)
    vals_out = key_vals[key_pos]
    lsf_out = Function(lsf_in.function_space())
    lsf_out.vector().set_local(vals_out)
    lsf_out.vector().apply("insert")
    return lsf_out


def expand_symmetry_to_full(lsf_in, symmetry_map):
    """Expand any supported symmetry-reduced field by global key averaging."""
    return expand_wedge_to_full(lsf_in, symmetry_map)


def wedge_residual_stats(lsf_in, wedge_map):
    """Global residual after enforcing the cubic wedge symmetry map."""
    vals = lsf_in.vector().get_local()
    key_pos = np.asarray(wedge_map["local_key_pos"], dtype=np.int64)
    n_keys = int(wedge_map["n_reps"])
    if vals.size == 0:
        return 0.0, 0.0
    key_vals, _ = _global_key_average(vals, key_pos, n_keys)
    diff = np.abs(vals - key_vals[key_pos])
    local_max = float(np.max(diff))
    local_sum = float(np.sum(diff))
    local_n = int(diff.size)
    gmax = _allreduce_max_scalar(local_max)
    gsum = _MPI_COMM.allreduce(local_sum, op=_pyMPI.SUM) if _MPI_SIZE > 1 else local_sum
    gn = _MPI_COMM.allreduce(local_n, op=_pyMPI.SUM) if _MPI_SIZE > 1 else local_n
    gmean = float(gsum) / max(1, int(gn))
    return float(gmax), float(gmean)


def symmetry_residual_stats(lsf_in, symmetry_map):
    """Global residual of the current field against the active symmetry subspace."""
    return wedge_residual_stats(lsf_in, symmetry_map)


def symmetry_reduced_angle_between(f, g, symmetry_map):
    """
    Angle in the symmetry-reduced basic domain:
      - average each field over every symmetry orbit/key
      - treat each orbit once (unweighted representative angle)
    This differs from the full/global theta, which uses the expanded field on the whole domain.
    """
    key_pos = np.asarray(symmetry_map["local_key_pos"], dtype=np.int64)
    n_keys = int(symmetry_map["n_reps"])
    if n_keys <= 0:
        return 0.0, 0.0, 0.0
    f_vals = f.vector().get_local()
    g_vals = g.vector().get_local()
    f_key, _ = _global_key_average(f_vals, key_pos, n_keys)
    g_key, _ = _global_key_average(g_vals, key_pos, n_keys)
    fn = float(np.sqrt(max(np.dot(f_key, f_key), 0.0)))
    gn = float(np.sqrt(max(np.dot(g_key, g_key), 0.0)))
    if fn < 1e-14 or gn < 1e-14:
        return 0.0, fn, gn
    c = float(np.dot(f_key, g_key)) / (fn * gn)
    c = max(-1.0, min(1.0, c))
    th = float(np.arccos(c))
    return th, fn, gn


def reduced_gradient_expand_to_full(g_in, wedge_map):
    """
    Exact reduced-DOF chain rule in strict wedge parameterization:
      w_r are wedge representatives, psi = E w.
      grad_w(r) = sum_{i in orbit(r)} grad_psi(i)
      expanded full search direction = E grad_w.
    This avoids full-domain independent updates.
    """
    vals = g_in.vector().get_local()
    if vals.size == 0:
        return Function(g_in.function_space())
    key_pos = np.asarray(wedge_map["local_key_pos"], dtype=np.int64)
    n_keys = int(wedge_map["n_reps"])
    # Exact chain rule in MPI: global sum over each orbit/key.
    grad_w = _global_key_sum(vals, key_pos, n_keys)
    vals_out = grad_w[key_pos]
    g_out = Function(g_in.function_space())
    g_out.vector().set_local(vals_out)
    g_out.vector().apply("insert")
    return g_out


def cubic_residual_metrics(Chom, abs_eps=1e-30):
    """
    Cubic residual diagnostics:
      rN: normal-block anisotropy (C11,C22,C33 and C12,C13,C23 mismatch)
      rS: shear-block anisotropy (C44,C55,C66 mismatch)
      rF: forbidden coupling magnitude relative to tensor scale
    """
    C = np.asarray(Chom, dtype=float)
    scale = max(abs_eps, float(np.max(np.abs(C))))

    c11, c22, c33 = float(C[0, 0]), float(C[1, 1]), float(C[2, 2])
    c12, c13, c23 = float(C[0, 1]), float(C[0, 2]), float(C[1, 2])
    c44, c55, c66 = float(C[3, 3]), float(C[4, 4]), float(C[5, 5])

    rN_diag = max(abs(c11 - c22), abs(c11 - c33), abs(c22 - c33)) / scale
    rN_offd = max(abs(c12 - c13), abs(c12 - c23), abs(c13 - c23)) / scale
    rN = max(rN_diag, rN_offd)

    rS = max(abs(c44 - c55), abs(c44 - c66), abs(c55 - c66)) / scale

    forbidden = []
    for i in range(6):
        for j in range(6):
            allowed = False
            if (i <= 2) and (j <= 2):
                allowed = True
            if (i == j) and (i >= 3):
                allowed = True
            if not allowed:
                forbidden.append(abs(float(C[i, j])))
    rF = (max(forbidden) / scale) if forbidden else 0.0

    return dict(rN=float(rN), rS=float(rS), rF=float(rF))

def write_xdmf(outdir, mesh, it, materials, lsf, tag=None):
    """Write chi and level-set snapshots to per-iteration XDMF files.
       All MPI ranks must participate in XDMFFile write (collective I/O).
    """
    chi = materials_to_chi(mesh, materials)
    tag_clean = None
    if tag is not None:
        tag_clean = "".join(
            ch if (ch.isalnum() or ch in ("_", "-")) else "_"
            for ch in str(tag)
        ).strip("_")
    if tag_clean:
        base_chi = "chi_%s_%03d" % (tag_clean, it)
        base_slf = "slf_%s_%03d" % (tag_clean, it)
    else:
        base_chi = "chi_%03d" % it
        base_slf = "slf_%03d" % it
    with XDMFFile(MPI.comm_world, os.path.join(outdir, base_chi + ".xdmf")) as xf:
        xf.parameters["flush_output"] = True
        xf.parameters["functions_share_mesh"] = True
        xf.write(chi, 0.0)
    slf_out = Function(lsf.function_space())
    copy_function_values(slf_out, lsf)
    slf_out.rename("slf", "level-set")
    with XDMFFile(MPI.comm_world, os.path.join(outdir, base_slf + ".xdmf")) as xf:
        xf.parameters["flush_output"] = True
        xf.parameters["functions_share_mesh"] = True
        xf.write(slf_out, 0.0)


def write_stage2_raw_xdmf_once(outdir, mesh, it, materials, lsf, cfg, reason=""):
    """Save the unfiltered stage-2 final state once, before final smoothing."""
    if bool(cfg.get("_stage2_raw_xdmf_written", False)):
        return False
    write_xdmf(outdir, mesh, it, materials, lsf, tag="stage2_raw")
    cfg["_stage2_raw_xdmf_written"] = True
    cfg["_stage2_raw_xdmf_it"] = int(it)
    if MPI.rank(MPI.comm_world) == 0:
        reason_msg = (" reason=%s" % str(reason)) if reason else ""
        print("[stage2-raw-xdmf] it=%03d: wrote chi_stage2_raw_%03d.xdmf and slf_stage2_raw_%03d.xdmf before final filter%s" %
              (int(it), int(it), int(it), reason_msg))
    return True


def should_print_chom_voigt(cfg, it, event=False, final_iteration=False):
    """Gate full 6x6 Chom logging to reduce HPC stdout volume."""
    if bool(cfg.get("print_chom_voigt_each_iter", False)):
        return True
    if bool(event) or bool(final_iteration):
        return True
    try:
        stride = int(cfg.get("print_chom_voigt_stride", 10))
    except Exception:
        stride = 10
    if stride <= 0:
        return False
    try:
        return int(it) % stride == 0
    except Exception:
        return False


def _debug_progress_enabled(cfg):
    return bool(cfg.get("debug_progress_markers", False))


def stage2_volume_continuation_active(cfg, vf=None, vf_target=None, hard_shift_only=False):
    """Whether the second-stage volume-continuation controller should own volume reduction."""
    if not bool(cfg.get("stage2_volume_continuation_enabled", False)):
        return False
    if bool(hard_shift_only):
        return False
    if vf_target is None:
        vf_target = current_vf_target(cfg)
    if vf_target is None:
        return False
    forced_start = bool(cfg.get("_stage2_vc_force_started", False))
    started_once = bool(cfg.get("_stage2_vc_started_once", False)) or forced_start
    if not started_once:
        if not bool(cfg.get("stage2_auto_start_enabled", False)):
            return False
        it_now = cfg.get("_current_it", None)
        start_iter = cfg.get("stage2_start_iter", None)
        if (it_now is not None) and (start_iter is not None):
            try:
                if int(it_now) < int(start_iter):
                    return False
            except Exception:
                pass
    if vf is not None:
        if not started_once:
            start_vf = cfg.get("stage2_start_vf", None)
            if start_vf is not None:
                try:
                    if float(vf) > float(start_vf):
                        return False
                except Exception:
                    pass
        stage2_end = float(cfg.get("stage2_volume_end_vf", cfg.get("hard_shift_switch_to_shift_vf", 0.10)))
        if bool(cfg.get("_hard_shift_switch_reached_once", False)) and (float(vf) <= stage2_end + 1e-15):
            return False
    return True


def _stage_target_matches(a, b, tol=1e-12):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= float(tol)
    except Exception:
        return False


def stage2_loss_envelope_goal_vf(cfg):
    raw = cfg.get("stage2_loss_envelope_vf_goal", None)
    if raw is None:
        return stage2_base_convergence_vf(cfg)
    return float(raw)


def update_stage2_loss_envelope_anchor(cfg, res_eval=None, it=None, reason="state"):
    """Track best J_mech after the loss envelope activation vf."""
    if not bool(cfg.get("stage2_loss_envelope_enabled", False)):
        return None
    if not isinstance(res_eval, dict):
        return None
    try:
        vf = float(res_eval.get("vf"))
        J = float(res_eval.get("J", res_eval.get("J_compare")))
    except (TypeError, ValueError):
        return None
    if (not np.isfinite(vf)) or (not np.isfinite(J)):
        return None
    vf_anchor = float(cfg.get("stage2_loss_envelope_vf_anchor", 0.30))
    if vf > vf_anchor + 1e-15 and "_stage2_loss_envelope_anchor_J" not in cfg:
        return None
    old_J_raw = cfg.get("_stage2_loss_envelope_anchor_J", None)
    improve_tol = max(0.0, float(cfg.get("stage2_loss_envelope_improve_rel_tol", 1e-4)))
    should_update = old_J_raw is None
    if old_J_raw is not None:
        old_J = float(old_J_raw)
        should_update = bool(
            cfg.get("stage2_loss_envelope_update_anchor_on_improve", True)
            and (J < old_J * (1.0 - improve_tol))
        )
    if not should_update:
        return dict(
            active=True,
            updated=False,
            vf_anchor=float(cfg.get("_stage2_loss_envelope_anchor_vf", vf_anchor)),
            J_anchor=float(old_J_raw),
        )
    cfg["_stage2_loss_envelope_anchor_J"] = float(J)
    cfg["_stage2_loss_envelope_anchor_vf"] = float(vf)
    cfg["_stage2_loss_envelope_anchor_it"] = None if it is None else int(it)
    if MPI.rank(MPI.comm_world) == 0:
        it_msg = "" if it is None else (" it=%03d" % int(it))
        print("[stage2-loss-envelope-anchor]%s reason=%s vf_anchor_eff=%.6f J_anchor=%.6e" %
              (it_msg, str(reason), float(vf), float(J)), flush=True)
    return dict(active=True, updated=True, vf_anchor=float(vf), J_anchor=float(J))


def stage2_loss_envelope_status(cfg, vf_target, J_current=None, vf_current=None):
    if not bool(cfg.get("stage2_loss_envelope_enabled", False)):
        return dict(active=False)
    if "_stage2_loss_envelope_anchor_J" not in cfg:
        return dict(active=False, reason="no-anchor")
    try:
        target = float(vf_target)
        J = float(J_current)
    except (TypeError, ValueError):
        return dict(active=False, reason="missing-state")
    if (not np.isfinite(target)) or (not np.isfinite(J)):
        return dict(active=False, reason="nonfinite-state")
    J_anchor = max(float(cfg.get("_stage2_loss_envelope_anchor_J")), 1e-30)
    vf_anchor = float(cfg.get("_stage2_loss_envelope_anchor_vf", cfg.get("stage2_loss_envelope_vf_anchor", 0.30)))
    vf_goal = float(stage2_loss_envelope_goal_vf(cfg))
    denom = max(float(vf_anchor) - float(vf_goal), float(cfg.get("stage2_merit_stage_budget_eps", 1e-12)), 1e-12)
    progress = (float(vf_anchor) - float(target)) / denom
    progress = min(1.0, max(0.0, progress))
    power = max(1e-12, float(cfg.get("stage2_loss_envelope_power", 1.0)))
    total_rel = max(0.0, float(cfg.get("stage2_loss_envelope_total_rel", 1.0)))
    allowed = J_anchor * (1.0 + total_rel * (progress ** power))
    remaining = float(allowed) - float(J)
    return dict(
        active=True,
        vf_anchor=float(vf_anchor),
        J_anchor=float(J_anchor),
        vf_goal=float(vf_goal),
        vf_target=float(target),
        vf_current=(None if vf_current is None else float(vf_current)),
        progress=float(progress),
        total_rel=float(total_rel),
        J_allowed=float(allowed),
        J_current=float(J),
        budget_remaining=float(remaining),
    )


def stage2_loss_envelope_existing_stop(cfg, res_eval=None, reason="already-active"):
    """Return the active loss-envelope stop without mutating settle state."""
    if not bool(cfg.get("_stage2_loss_envelope_stop_active", False)):
        return None
    conv_raw = cfg.get("_stage2_loss_envelope_convergence_vf", None)
    if conv_raw is None:
        return None
    try:
        conv_vf = float(conv_raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(conv_vf):
        return None
    J_now = None
    vf_now = None
    if isinstance(res_eval, dict):
        try:
            J_now = float(res_eval.get("J", res_eval.get("J_compare", cfg.get("_stage2_loss_envelope_stop_J", 0.0))))
        except (TypeError, ValueError):
            J_now = None
        try:
            vf_now = float(res_eval.get("vf", conv_vf))
        except (TypeError, ValueError):
            vf_now = None
    if J_now is None:
        try:
            J_now = float(cfg.get("_stage2_loss_envelope_stop_J", 0.0))
        except (TypeError, ValueError):
            J_now = 0.0
    if vf_now is None:
        vf_now = conv_vf
    try:
        J_stop = float(cfg.get("_stage2_loss_envelope_stop_J", J_now))
    except (TypeError, ValueError):
        J_stop = float(J_now)
    status = stage2_loss_envelope_status(
        cfg, conv_vf, J_current=J_now, vf_current=vf_now
    )
    out = dict(status) if isinstance(status, dict) else dict(active=True)
    out.update(
        active=True,
        stopped=True,
        already_active=True,
        reason=str(reason),
        stop_info=dict(
            vf_convergence=float(conv_vf),
            J=float(J_stop),
            rejected_target=cfg.get("_stage2_loss_envelope_rejected_target", None),
            stop_reason=cfg.get("_stage2_loss_envelope_stop_reason", None),
        ),
    )
    return out


def activate_stage2_loss_envelope_stop(cfg, res_eval, rejected_target, status, reason="budget"):
    if not isinstance(res_eval, dict):
        return None
    vf_now = float(res_eval.get("vf", current_vf_target(cfg)))
    J_now = float(res_eval.get("J", res_eval.get("J_compare", 0.0)))
    existing = stage2_loss_envelope_existing_stop(cfg, res_eval=res_eval, reason=reason)
    if existing is not None:
        conv_vf = float(existing.get("stop_info", {}).get("vf_convergence", existing.get("vf_target", vf_now)))
        cfg["_stage2_final_pending"] = True
        cfg["_vf_stage_target"] = float(conv_vf)
        return dict(
            vf_convergence=float(conv_vf),
            J=float(existing.get("stop_info", {}).get("J", J_now)),
            already_active=True,
        )
    cfg["_stage2_loss_envelope_stop_active"] = True
    cfg["_stage2_loss_envelope_convergence_vf"] = float(vf_now)
    cfg["_stage2_loss_envelope_rejected_target"] = float(rejected_target)
    cfg["_stage2_loss_envelope_stop_J"] = float(J_now)
    cfg["_stage2_loss_envelope_stop_reason"] = str(reason)
    cfg["_stage2_final_pending"] = True
    cfg["_stage2_final_convergence_announced"] = False
    cfg["_vf_stage_target"] = float(vf_now)
    _stage2_settle_reset(cfg)
    if MPI.rank(MPI.comm_world) == 0:
        print("[stage2-loss-envelope-stop] reason=%s rejected_target=%.6f -> convergence_vf=%.6f "
              "J_mech=%.6e J_allowed=%.6e remaining=%.3e anchor(vf=%.6f,J=%.6e,progress=%.3f,total_rel=%.3f)" %
              (str(reason),
               float(rejected_target),
               float(vf_now),
               float(J_now),
               float(status.get("J_allowed", float("nan"))),
               float(status.get("budget_remaining", float("nan"))),
               float(status.get("vf_anchor", float("nan"))),
               float(status.get("J_anchor", float("nan"))),
               float(status.get("progress", float("nan"))),
               float(status.get("total_rel", float("nan")))),
              flush=True)
    return dict(vf_convergence=float(vf_now), J=float(J_now))


def check_stage2_loss_envelope_target(cfg, res_eval, vf_target=None, reason="stage-target"):
    """Keep the mechanical-loss envelope active even without fixed-stage merit."""
    if not bool(cfg.get("stage2_loss_envelope_enabled", False)):
        return None
    existing = stage2_loss_envelope_existing_stop(cfg, res_eval=res_eval, reason=reason)
    if existing is not None:
        return existing
    if not isinstance(res_eval, dict):
        return None
    if vf_target is None:
        vf_target = current_vf_target(cfg)
    if vf_target is None:
        return None
    update_stage2_loss_envelope_anchor(cfg, res_eval=res_eval, reason=reason)
    try:
        J_now = float(res_eval.get("J", res_eval.get("J_compare", 0.0)))
        vf_now = float(res_eval.get("vf", 0.0))
    except (TypeError, ValueError):
        return None
    status = stage2_loss_envelope_status(
        cfg, vf_target, J_current=J_now, vf_current=vf_now
    )
    if not bool(status.get("active", False)):
        return status
    J_ref = max(
        abs(J_now),
        float(cfg.get("stage2_merit_stage_budget_j_floor", 1e-8)),
        1e-30,
    )
    stop_tol = max(
        float(cfg.get("stage2_loss_envelope_min_budget_abs", 0.0)),
        max(0.0, float(cfg.get("stage2_loss_envelope_stop_tol_rel", 1e-3))) * J_ref,
    )
    if float(status.get("budget_remaining", 0.0)) <= stop_tol:
        stop_info = activate_stage2_loss_envelope_stop(
            cfg, res_eval, vf_target, status, reason="%s:no-envelope-budget" % str(reason)
        )
        out = dict(status)
        out["stopped"] = True
        out["stop_info"] = stop_info
        return out
    out = dict(status)
    out["stopped"] = False
    return out


def projected_merit_coefficients_from_budget(cfg, p_stage, c0, lambda_ctrl, mu_fallback, rho_fallback):
    eps = max(1e-30, float(cfg.get("stage2_merit_stage_budget_eps", 1e-12)))
    p_stage = max(0.0, float(p_stage))
    c0 = max(0.0, float(c0))
    if c0 <= eps or p_stage <= eps:
        return 0.0, 0.0, 0.0, "zero-budget"
    if not bool(cfg.get("stage2_merit_stage_budget_lambda_projection_enabled", True)):
        return float(mu_fallback), float(rho_fallback), 1.0, "controller-scale"
    lam_ctrl = max(0.0, float(lambda_ctrl))
    lam_min = p_stage / max(c0, eps)
    lam_max = 2.0 * p_stage / max(c0, eps)
    lam_budget = min(max(lam_ctrl, lam_min), lam_max)
    rho = 2.0 * (lam_budget * c0 - p_stage) / max(c0 * c0, eps)
    mu = (2.0 * p_stage - lam_budget * c0) / max(c0, eps)
    return max(0.0, float(mu)), max(0.0, float(rho)), float(lam_budget), "budget-projection"


def _controller_mu_rho(cfg):
    mu = max(0.0, float(cfg.get("_vf_aug_lag_mu", cfg.get("mu_v0", 0.0))))
    rho = max(0.0, float(cfg.get("_vf_aug_lag_rho", cfg.get("rho_v", 0.0))))
    return float(mu), float(rho)


def _fixed_stage_merit_available(cfg, vf_target):
    if not bool(cfg.get("stage2_merit_freeze_enabled", False)):
        return False
    if vf_target is None:
        return False
    if "_vf_merit_mu" not in cfg or "_vf_merit_rho" not in cfg:
        return False
    return _stage_target_matches(cfg.get("_vf_merit_stage_target", None), vf_target)


def freeze_stage_merit_from_controller(cfg, res_eval=None, reason="stage-target"):
    """
    Freeze the acceptance merit coefficients for the current vf_stage.

    The AL controller variables (_vf_aug_lag_mu/rho) continue to adapt lambda_v.
    The frozen _vf_merit_mu/rho are used only by J_merit inside this vf_stage.
    """
    if not bool(cfg.get("stage2_merit_freeze_enabled", False)):
        return check_stage2_loss_envelope_target(
            cfg, res_eval, vf_target=current_vf_target(cfg), reason=reason
        )
    vf_target = current_vf_target(cfg)
    if vf_target is None:
        return None
    existing_stop = stage2_loss_envelope_existing_stop(cfg, res_eval=res_eval, reason=reason)
    if existing_stop is not None:
        return existing_stop
    if (
        _stage_target_matches(cfg.get("_vf_merit_stage_target", None), vf_target)
        and ("_vf_merit_mu" in cfg)
        and ("_vf_merit_rho" in cfg)
        and (not bool(cfg.get("stage2_merit_refreeze_same_target", False)))
    ):
        return dict(
            vf_target=float(vf_target),
            vf=float(res_eval.get("vf", cfg.get("_current_vf", vf_target)) if isinstance(res_eval, dict) else cfg.get("_current_vf", vf_target)),
            c0=float(cfg.get("_vf_merit_stage_c0", 0.0)),
            J_ref=float(cfg.get("_vf_merit_stage_j_ref", 0.0)),
            mu_ctrl=float(cfg.get("_vf_merit_mu_ctrl_snapshot", 0.0)),
            rho_ctrl=float(cfg.get("_vf_merit_rho_ctrl_snapshot", 0.0)),
            mu_merit=float(cfg.get("_vf_merit_mu", 0.0)),
            rho_merit=float(cfg.get("_vf_merit_rho", 0.0)),
            budget=float(cfg.get("_vf_merit_stage_budget", 0.0)),
            requested_budget=float(cfg.get("_vf_merit_stage_budget_requested", 0.0)),
            scale=1.0,
            reused=True,
        )
    vf = None
    J = None
    if isinstance(res_eval, dict):
        vf = res_eval.get("vf", None)
        J = res_eval.get("J", res_eval.get("J_compare", None))
    if vf is None:
        vf = cfg.get("_current_vf", vf_target)
    if J is None:
        J = cfg.get("_current_J_mech", 0.0)
    vf = float(vf)
    J = float(J)
    vf_target = float(vf_target)
    update_stage2_loss_envelope_anchor(cfg, res_eval=res_eval, reason=reason)
    c0 = max(vf - vf_target, 0.0)
    mu_ctrl, rho_ctrl = _controller_mu_rho(cfg)
    mu_max = max(0.0, float(cfg.get("stage2_merit_stage_mu_max", cfg.get("stage2_al_mu_max", max(mu_ctrl, 1.0)))))
    rho_max = max(0.0, float(cfg.get("stage2_merit_stage_rho_max", cfg.get("stage2_al_rho_max", max(rho_ctrl, 1.0)))))
    mu_merit = min(mu_ctrl, mu_max) if mu_max > 0.0 else mu_ctrl
    rho_merit = min(rho_ctrl, rho_max) if rho_max > 0.0 else rho_ctrl
    p_ctrl = mu_ctrl * c0 + 0.5 * rho_ctrl * c0 * c0
    p_stage = p_ctrl
    scale = 1.0
    lambda_budget = None
    budget_mode = "controller"
    envelope_status = stage2_loss_envelope_status(cfg, vf_target, J_current=J, vf_current=vf)
    eps = max(1e-30, float(cfg.get("stage2_merit_stage_budget_eps", 1e-12)))
    J_ref = max(abs(J), float(cfg.get("stage2_merit_stage_budget_j_floor", 1e-8)), 1e-30)
    if bool(cfg.get("stage2_merit_stage_budget_enabled", True)) and c0 > eps:
        eta = max(0.0, float(cfg.get("stage2_merit_stage_budget_eta", 0.05)))
        eta_min = max(0.0, float(cfg.get("stage2_merit_stage_budget_eta_min", 0.0)))
        p_min_base = eta_min * J_ref
        p_max = max(p_min_base, eta * J_ref)
        if bool(envelope_status.get("active", False)):
            envelope_remaining = float(envelope_status.get("budget_remaining", 0.0))
            stop_tol = max(
                float(cfg.get("stage2_loss_envelope_min_budget_abs", 0.0)),
                max(0.0, float(cfg.get("stage2_loss_envelope_stop_tol_rel", 1e-3))) * J_ref,
            )
            if envelope_remaining <= stop_tol:
                activate_stage2_loss_envelope_stop(
                    cfg, res_eval, vf_target, envelope_status,
                    reason="%s:no-envelope-budget" % str(reason),
                )
                vf_target = float(current_vf_target(cfg))
                c0 = max(vf - vf_target, 0.0)
                p_ctrl = mu_ctrl * c0 + 0.5 * rho_ctrl * c0 * c0
                envelope_status = stage2_loss_envelope_status(cfg, vf_target, J_current=J, vf_current=vf)
                p_stage = 0.0
                p_min_base = 0.0
                p_max = 0.0
            else:
                p_max = min(p_max, envelope_remaining)
                p_min_base = min(p_min_base, p_max)
        p_min = p_min_base
        p_max = max(0.0, p_max)
        p_stage = min(max(p_ctrl, p_min), p_max)
        if p_stage <= eps:
            mu_merit = 0.0
            rho_merit = 0.0
            lambda_budget = 0.0
            budget_mode = "zero-budget"
        else:
            if p_ctrl > eps:
                scale = p_stage / max(p_ctrl, eps)
                mu_scaled = mu_ctrl * scale
                rho_scaled = rho_ctrl * scale
            else:
                beta = min(1.0, max(0.0, float(cfg.get("stage2_merit_stage_budget_mu_share", 0.5))))
                mu_scaled = beta * p_stage / max(c0, eps)
                rho_scaled = 2.0 * (1.0 - beta) * p_stage / max(c0 * c0, eps)
            lambda_ctrl = mu_ctrl + rho_ctrl * c0
            mu_merit, rho_merit, lambda_budget, budget_mode = projected_merit_coefficients_from_budget(
                cfg, p_stage, c0, lambda_ctrl, mu_scaled, rho_scaled
            )
        if mu_max > 0.0:
            mu_merit = min(mu_merit, mu_max)
        if rho_max > 0.0:
            rho_merit = min(rho_merit, rho_max)
    p_stage_actual = mu_merit * c0 + 0.5 * rho_merit * c0 * c0
    cfg["_vf_merit_stage_target"] = float(vf_target)
    cfg["_vf_merit_mu"] = float(max(0.0, mu_merit))
    cfg["_vf_merit_rho"] = float(max(0.0, rho_merit))
    cfg["_vf_merit_mu_ctrl_snapshot"] = float(mu_ctrl)
    cfg["_vf_merit_rho_ctrl_snapshot"] = float(rho_ctrl)
    cfg["_vf_merit_stage_c0"] = float(c0)
    cfg["_vf_merit_stage_j_ref"] = float(J_ref)
    cfg["_vf_merit_stage_budget"] = float(p_stage_actual)
    cfg["_vf_merit_stage_budget_requested"] = float(p_stage)
    cfg["_vf_merit_stage_reason"] = str(reason)
    cfg["_vf_merit_stage_budget_mode"] = str(budget_mode)
    cfg["_vf_merit_stage_lambda_budget"] = float(0.0 if lambda_budget is None else lambda_budget)
    cfg["_vf_merit_loss_envelope_active"] = bool(envelope_status.get("active", False))
    cfg["_vf_merit_loss_envelope_allowed"] = float(envelope_status.get("J_allowed", float("nan")))
    cfg["_vf_merit_loss_envelope_remaining"] = float(envelope_status.get("budget_remaining", float("nan")))
    if MPI.rank(MPI.comm_world) == 0:
        print("[stage2-merit-freeze] reason=%s target=%.6f vf=%.6f c0=%.3e J_ref=%.6e "
              "mu_ctrl=%.3e rho_ctrl=%.3e -> mu_merit=%.3e rho_merit=%.3e "
              "P0=%.3e requested=%.3e scale=%.3e mode=%s lambda_budget=%.3e env=%s J_allowed=%.6e remaining=%.3e" %
              (str(reason), float(vf_target), float(vf), float(c0), float(J_ref),
               float(mu_ctrl), float(rho_ctrl), float(cfg["_vf_merit_mu"]),
               float(cfg["_vf_merit_rho"]), float(p_stage_actual), float(p_stage),
               float(scale), str(budget_mode), float(cfg["_vf_merit_stage_lambda_budget"]),
               ("on" if bool(envelope_status.get("active", False)) else "off"),
               float(cfg["_vf_merit_loss_envelope_allowed"]),
               float(cfg["_vf_merit_loss_envelope_remaining"])), flush=True)
    if isinstance(res_eval, dict):
        refresh_volume_merit_inplace(cfg, res_eval)
    return dict(
        vf_target=float(vf_target),
        vf=float(vf),
        c0=float(c0),
        J_ref=float(J_ref),
        mu_ctrl=float(mu_ctrl),
        rho_ctrl=float(rho_ctrl),
        mu_merit=float(cfg["_vf_merit_mu"]),
        rho_merit=float(cfg["_vf_merit_rho"]),
        budget=float(p_stage_actual),
        requested_budget=float(p_stage),
        scale=float(scale),
        budget_mode=str(budget_mode),
        loss_envelope=dict(envelope_status),
    )


def stage2_mechanical_step_cap_ok(cfg, res_old, res_try, stage2_active):
    """Optional legacy guard on accepted mechanical deterioration.

    The knob lives with the stage-2 merit settings because that is where the
    coupled volume/mechanical acceptance is used.  It is disabled by default;
    dynamic AL merit and the low-vf loss envelope normally set the
    mechanical/volume tradeoff rather than a separate J_mech cap.
    """
    if not bool(cfg.get("stage2_merit_step_jmech_cap_enabled", False)):
        return True, dict(active=False)
    if (not stage2_active) and bool(cfg.get("stage2_merit_step_jmech_cap_stage2_only", False)):
        return True, dict(active=False)
    J_old = float(res_old.get("J", res_old.get("J_compare", 0.0)))
    J_try = float(res_try.get("J", res_try.get("J_compare", 0.0)))
    J_scale = max(abs(J_old), float(cfg.get("stage2_merit_step_jmech_cap_j_floor", 1e-12)), 1e-30)
    below_target = stage2_merit_relax_below_target_active(cfg, res_eval=res_old)
    cap_rel = max(0.0, float(cfg.get("stage2_merit_step_jmech_cap_rel", 2e-3)))
    mode = "global-step"
    if below_target and bool(cfg.get("stage2_merit_below_target_mech_strict", False)):
        cap_rel = max(0.0, float(cfg.get("stage2_merit_below_target_mech_relax", 0.0)))
        mode = "below-target"
    J_allow = J_old + cap_rel * J_scale
    ok = bool(J_try <= J_allow + 1e-15 * J_scale)
    return ok, dict(
        active=True,
        mode=mode,
        below_target=bool(below_target),
        J_old=float(J_old),
        J_try=float(J_try),
        J_allow=float(J_allow),
        cap_rel=float(cap_rel),
        dJ=float(J_try - J_old),
        dJ_rel=float((J_try - J_old) / J_scale),
    )


def print_stage2_mechanical_step_cap_reject(context, info, action="reject"):
    if MPI.rank(MPI.comm_world) != 0:
        return
    print("[12-mech-cap] %s: J_mech_try=%.6e > J_mech_allow=%.6e "
          "(J_mech_old=%.6e, cap=%.3e, mode=%s) => %s" %
          (str(context),
           float(info.get("J_try", float("nan"))),
           float(info.get("J_allow", float("nan"))),
           float(info.get("J_old", float("nan"))),
           float(info.get("cap_rel", float("nan"))),
           str(info.get("mode", "n/a")),
           str(action)), flush=True)


def volume_merit_terms(cfg, J, vf, vf_target=None, hard_shift_only=False):
    """
    Internal acceptance merit for the constrained problem:
        min J_mech(phi)  subject to vf(phi) <= V_target.

    The reported objective remains the pure mechanical J_mech.  Stage 1 deliberately
    has no active volume acceptance target, so J_accept reduces exactly to J_mech.
    Active stage 2 switches the same acceptance field to the AL merit.
    """
    if vf_target is None:
        vf_target = current_vf_target(cfg)
    vf_target_eff = float(vf_target) if vf_target is not None else float(vf)
    residual = float(vf) - vf_target_eff
    violation = max(residual, 0.0)
    mu_ctrl, rho_ctrl = _controller_mu_rho(cfg)
    mu = mu_ctrl
    rho = rho_ctrl
    J_mech = float(J)
    J_compare_vf_weight = float(cfg.get("J_compare_vf_weight", 0.05))
    J_compare_legacy = J_mech + J_compare_vf_weight * float(vf)
    use_merit = (
        bool(cfg.get("stage2_merit_linesearch_enabled", False))
        and stage2_volume_continuation_active(cfg, vf=vf, vf_target=vf_target, hard_shift_only=hard_shift_only)
    )
    fixed_merit = bool(use_merit and _fixed_stage_merit_available(cfg, vf_target_eff))
    if fixed_merit:
        mu = max(0.0, float(cfg.get("_vf_merit_mu", 0.0)))
        rho = max(0.0, float(cfg.get("_vf_merit_rho", 0.0)))
    penalty_raw = mu * violation + 0.5 * rho * violation * violation
    penalty_scale = 1.0
    if (
        use_merit
        and bool(cfg.get("stage2_merit_penalty_ratio_cap_enabled", False))
        and ((not fixed_merit) or bool(cfg.get("stage2_merit_freeze_runtime_penalty_cap_enabled", False)))
        and penalty_raw > 0.0
        and float(vf) < float(cfg.get("stage2_merit_penalty_ratio_cap_vf_threshold", -1.0))
    ):
        eta = max(0.0, float(cfg.get("stage2_merit_penalty_ratio_cap", 0.0)))
        j_floor = max(0.0, float(cfg.get("stage2_merit_penalty_ratio_cap_j_floor", 0.0)))
        J_scale = max(abs(J_mech), j_floor, 1e-30)
        penalty_cap = eta * J_scale
        if eta > 0.0:
            penalty_scale = min(1.0, penalty_cap / max(penalty_raw, 1e-30))
    mu_merit = penalty_scale * mu
    rho_merit = penalty_scale * rho
    penalty_merit = mu_merit * violation + 0.5 * rho_merit * violation * violation
    J_merit = J_mech + penalty_merit
    J_accept = float(J_merit if use_merit else J_mech)
    return dict(
        vf_constraint_target=float(vf_target_eff),
        vf_constraint_residual=float(residual),
        vf_constraint_violation=float(violation),
        mu_v=float(mu_ctrl),
        rho_v=float(rho_ctrl),
        mu_v_ctrl=float(mu_ctrl),
        rho_v_ctrl=float(rho_ctrl),
        mu_v_merit=float(mu_merit),
        rho_v_merit=float(rho_merit),
        J_merit_fixed_stage=bool(fixed_merit),
        J_merit_stage_target=float(
            vf_target_eff if cfg.get("_vf_merit_stage_target", None) is None
            else cfg.get("_vf_merit_stage_target", vf_target_eff)
        ),
        J_merit_stage_budget=float(cfg.get("_vf_merit_stage_budget", 0.0)),
        J_merit_raw=float(J_mech + penalty_raw),
        J_merit_penalty_raw=float(penalty_raw),
        J_merit_penalty=float(penalty_merit),
        J_merit_penalty_scale=float(penalty_scale),
        J_merit=float(J_merit),
        J_accept=float(J_accept),
        J_compare_legacy=float(J_compare_legacy),
        J_compare=float(J_accept),
        J_compare_mode=("stage2-merit" if use_merit else "mechanical"),
        J_compare_vf_weight=float(J_compare_vf_weight),
    )


def refresh_volume_merit_inplace(cfg, res_eval, hard_shift_only=False):
    """Refresh residual/merit fields after the stage target or AL state changes."""
    terms = volume_merit_terms(
        cfg,
        J=float(res_eval.get("J", 0.0)),
        vf=float(res_eval.get("vf", 0.0)),
        vf_target=current_vf_target(cfg),
        hard_shift_only=hard_shift_only,
    )
    res_eval.update(terms)
    res_eval["vf_target"] = current_vf_target(cfg)
    return res_eval


def initialise_volume_continuation_state(cfg, res_eval):
    rho = max(float(cfg.get("vf_al_rho_min", 0.0)), float(cfg.get("rho_v", 0.0)))
    rho = min(rho, float(cfg.get("vf_al_rho_max", max(rho, 1.0))))
    mu = max(0.0, float(cfg.get("mu_v0", 0.0)))
    mu = min(mu, float(cfg.get("stage2_al_mu_max", max(mu, 1.0))))
    cfg["_vf_aug_lag_rho"] = float(rho)
    cfg["_vf_aug_lag_mu"] = float(mu)
    refresh_volume_merit_inplace(cfg, res_eval)
    cfg["_vf_aug_lag_prev_violation"] = float(res_eval.get("vf_constraint_violation", 0.0))
    cfg["_vf_aug_lag_last_target"] = current_vf_target(cfg)


def update_volume_continuation_state(cfg, res_eval, it=None, reason="iter", force_rho_grow=False, update_mu=True, adapt_rho=True):
    """Update the AL/KKT multiplier state from the current volume residual."""
    refresh_volume_merit_inplace(cfg, res_eval)
    if not stage2_volume_continuation_active(cfg, vf=res_eval.get("vf", None), vf_target=current_vf_target(cfg)):
        return None
    stage2_started_before = bool(cfg.get("_stage2_vc_started_once", False))
    cfg["_stage2_vc_started_once"] = True
    if not stage2_started_before:
        update_alpha_ti_from_stage_state(cfg, res_eval, it=it, reason="stage2-start")

    violation = float(res_eval.get("vf_constraint_violation", 0.0))
    residual = float(res_eval.get("vf_constraint_residual", 0.0))
    prev_violation = cfg.get("_vf_aug_lag_prev_violation", None)
    rho_old = max(0.0, float(cfg.get("_vf_aug_lag_rho", cfg.get("rho_v", 0.0))))
    mu_old = max(0.0, float(cfg.get("_vf_aug_lag_mu", cfg.get("mu_v0", 0.0))))
    rho_new = rho_old
    adapt_tag = "keep"
    ratio = None
    eps = max(1e-30, float(cfg.get("vf_al_adapt_eps", 1e-12)))
    if (not bool(adapt_rho)):
        adapt_tag = "hold-rho"
    elif prev_violation is not None:
        ratio = violation / max(float(prev_violation), eps)
        if bool(force_rho_grow) or ((violation > eps) and (ratio > float(cfg.get("vf_al_adapt_bad_ratio", 0.95)))):
            rho_new = min(rho_old * float(cfg.get("vf_al_rho_grow", 1.5)), float(cfg.get("vf_al_rho_max", 10.0)))
            adapt_tag = "grow"
        elif ratio < float(cfg.get("vf_al_adapt_good_ratio", 0.70)):
            rho_new = max(rho_old * float(cfg.get("vf_al_rho_shrink", 1.0)), float(cfg.get("vf_al_rho_min", 0.0)))
            adapt_tag = "shrink"
    elif bool(force_rho_grow):
        rho_new = min(rho_old * float(cfg.get("vf_al_rho_grow", 1.5)), float(cfg.get("vf_al_rho_max", 10.0)))
        adapt_tag = "grow"

    mu_max = float(cfg.get("stage2_al_mu_max", cfg.get("stage2_lambda_abs_cap", 1.0)))
    if bool(update_mu):
        mu_new = max(0.0, min(mu_max, mu_old + rho_new * residual))
    else:
        mu_new = mu_old
    mu_released = False
    mu_release_tol = max(
        0.0,
        float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0)))
        * float(cfg.get("stage2_al_mu_overshoot_tol_factor", 1.0)),
    )
    if bool(cfg.get("stage2_al_mu_overshoot_release_enabled", True)) and (residual < -mu_release_tol):
        mu_decay = min(1.0, max(0.0, float(cfg.get("stage2_al_mu_overshoot_decay", 0.0))))
        mu_release_candidate = max(0.0, min(mu_max, mu_old * mu_decay))
        if mu_release_candidate < mu_new:
            mu_new = float(mu_release_candidate)
            mu_released = True
            adapt_tag = "%s+mu-release" % str(adapt_tag)
    cfg["_vf_aug_lag_rho"] = float(rho_new)
    cfg["_vf_aug_lag_mu"] = float(mu_new)
    cfg["_vf_aug_lag_prev_violation"] = float(violation)
    cfg["_vf_aug_lag_last_target"] = current_vf_target(cfg)
    refresh_volume_merit_inplace(cfg, res_eval)
    update_stage2_loss_envelope_anchor(cfg, res_eval=res_eval, it=it, reason=reason)

    info = dict(
        reason=str(reason),
        violation=float(violation),
        residual=float(residual),
        prev_violation=(None if prev_violation is None else float(prev_violation)),
        ratio=(None if ratio is None else float(ratio)),
        rho_old=float(rho_old),
        rho_new=float(rho_new),
        mu_old=float(mu_old),
        mu_new=float(mu_new),
        mu_released=bool(mu_released),
        adapt_tag=str(adapt_tag),
        vf=float(res_eval.get("vf", 0.0)),
        vf_target=current_vf_target(cfg),
        J_merit=float(res_eval.get("J_merit", res_eval.get("J_compare", 0.0))),
        J_merit_raw=float(res_eval.get("J_merit_raw", res_eval.get("J_merit", res_eval.get("J_compare", 0.0)))),
        merit_penalty_scale=float(res_eval.get("J_merit_penalty_scale", 1.0)),
    )
    cfg["_stage2_volume_continuation_last"] = info
    if MPI.rank(MPI.comm_world) == 0:
        prefix = "[stage2-vc]" if it is None else "[stage2-vc] it=%03d" % int(it)
        ratio_msg = "None" if info["ratio"] is None else ("%.3f" % float(info["ratio"]))
        print("%s reason=%s vf=%.6f target=%s viol=%.3e ratio=%s mu %.3e->%.3e rho %.3e->%.3e action=%s merit_scale=%.3e J_merit=%.6e raw=%.6e" %
              (prefix, str(info["reason"]), float(info["vf"]),
               ("None" if info["vf_target"] is None else ("%.6f" % float(info["vf_target"]))),
               float(info["violation"]), ratio_msg,
               float(info["mu_old"]), float(info["mu_new"]),
               float(info["rho_old"]), float(info["rho_new"]),
               str(info["adapt_tag"]), float(info["merit_penalty_scale"]),
               float(info["J_merit"]), float(info["J_merit_raw"])),
              flush=True)
    return info


def reset_stage2_stall_watchdog(cfg):
    """Clear consecutive stall counters after a real state/target change."""
    for key in (
        "_stage2_stall_watchdog_target",
        "_stage2_stall_watchdog_total",
        "_stage2_stall_watchdog_nonweak",
        "_stage2_stall_watchdog_weak",
        "_stage2_stall_watchdog_last_action",
        "_stage2_stall_controller_handled_it",
        "_stage2_stall_wait_active",
        "_stage2_stall_wait_start_it",
        "_stage2_stall_wait_until_it",
        "_stage2_stall_wait_reason",
        "_stage2_stall_wait_vf",
        "_stage2_stall_wait_target",
        "_stage2_stall_wait_skipped_iters",
        "_stage2_stall_wait_same_state_snapshot",
        "_stage2_same_state_stall_snapshot",
        "_stage2_same_state_stall_streak",
    ):
        if key in cfg:
            del cfg[key]


def activate_stage2_stall_wait(cfg, it, release_it, reason, vf=None, vf_target=None):
    """Arm a cheap wait state while rare fallback is blocked by cooldown gates."""
    if not bool(cfg.get("stage2_stall_wait_enabled", True)):
        return False
    try:
        it_i = int(it)
        release_i = int(release_it)
    except Exception:
        return False
    if release_i < it_i + 1:
        release_i = it_i + 1
    max_skip = int(cfg.get("stage2_stall_wait_max_skip_iters", 120))
    if max_skip >= 0:
        release_i = min(release_i, it_i + max(1, max_skip))
    cfg["_stage2_stall_wait_active"] = True
    cfg["_stage2_stall_wait_start_it"] = it_i
    cfg["_stage2_stall_wait_until_it"] = int(release_i)
    cfg["_stage2_stall_wait_reason"] = str(reason)
    cfg["_stage2_stall_wait_skipped_iters"] = 0
    if vf is not None:
        try:
            cfg["_stage2_stall_wait_vf"] = float(vf)
        except Exception:
            pass
    if vf_target is not None:
        try:
            cfg["_stage2_stall_wait_target"] = float(vf_target)
        except Exception:
            pass
    return True


def _stage2_same_state_stall_snapshot(cfg, res_eval):
    """Compact state signature used to detect repeated identical line-search stalls."""
    vf_target = current_vf_target(cfg)
    dv_now = current_stage_dv(cfg)
    return dict(
        vf=float(res_eval.get("vf", float("nan"))),
        vf_target=(None if vf_target is None else float(vf_target)),
        J=float(res_eval.get("J", float("nan"))),
        J_compare=float(res_eval.get("J_compare", res_eval.get("J", float("nan")))),
        alpha=float(res_eval.get("alpha", cfg.get("alpha", float("nan")))),
        dv=(None if dv_now is None else float(dv_now)),
        lambda_v=float(cfg.get("lambda_v", 0.0)),
    )


def _finite_close(a, b, abs_tol=0.0, rel_tol=0.0):
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return False
    if not (np.isfinite(af) and np.isfinite(bf)):
        return False
    return abs(af - bf) <= max(float(abs_tol), float(rel_tol) * max(abs(af), abs(bf), 1.0))


def _same_state_stall_snapshot_matches(cfg, snapshot, res_eval):
    """Return True only when the current state is close enough to the stored stall state."""
    if not isinstance(snapshot, dict):
        return False
    current = _stage2_same_state_stall_snapshot(cfg, res_eval)
    vf_tol = max(
        float(cfg.get("hard_shift_plateau_vf_tol", 1e-12)),
        float(cfg.get("stage2_stall_wait_vf_tol", 1e-10)),
    )
    j_rel_tol = max(0.0, float(cfg.get("stage2_same_state_stall_wait_j_rel_tol", 1e-8)))
    lambda_rel_tol = max(0.0, float(cfg.get("stage2_same_state_stall_wait_lambda_rel_tol", 1e-8)))
    target_tol = max(1e-12, vf_tol)
    checks = (
        ("vf", vf_tol, 0.0),
        ("J", 0.0, j_rel_tol),
        ("J_compare", 0.0, j_rel_tol),
        ("alpha", 0.0, j_rel_tol),
        ("lambda_v", 0.0, lambda_rel_tol),
    )
    for key, abs_tol, rel_tol in checks:
        if not _finite_close(current.get(key), snapshot.get(key), abs_tol=abs_tol, rel_tol=rel_tol):
            return False
    for key in ("vf_target", "dv"):
        a = current.get(key)
        b = snapshot.get(key)
        if a is None or b is None:
            if a is not None or b is not None:
                return False
        elif not _finite_close(a, b, abs_tol=target_tol, rel_tol=0.0):
            return False
    return True


def maybe_activate_stage2_same_state_stall_wait(cfg, res_eval, it):
    """Arm lightweight stall-wait after repeated full line-search stalls on an unchanged state."""
    if not bool(cfg.get("stage2_same_state_stall_wait_enabled", False)):
        return None
    if not bool(cfg.get("stage2_stall_wait_enabled", True)):
        return None
    try:
        it_i = int(it)
    except Exception:
        return None
    snapshot = _stage2_same_state_stall_snapshot(cfg, res_eval)
    prev_snapshot = cfg.get("_stage2_same_state_stall_snapshot", None)
    if _same_state_stall_snapshot_matches(cfg, prev_snapshot, res_eval):
        streak = int(cfg.get("_stage2_same_state_stall_streak", 0)) + 1
    else:
        streak = 1
    cfg["_stage2_same_state_stall_snapshot"] = snapshot
    cfg["_stage2_same_state_stall_streak"] = int(streak)
    after = max(1, int(cfg.get("stage2_same_state_stall_wait_after", 2)))
    if streak < after:
        return dict(armed=False, streak=int(streak), after=int(after))
    max_skip = max(0, int(cfg.get("stage2_same_state_stall_wait_max_skip_iters", 30)))
    if max_skip <= 0:
        return dict(armed=False, streak=int(streak), after=int(after))
    release_it = it_i + max_skip
    if activate_stage2_stall_wait(
        cfg,
        it_i,
        release_it,
        reason="same-state-line-search-stall",
        vf=snapshot.get("vf", None),
        vf_target=snapshot.get("vf_target", None),
    ):
        cfg["_stage2_stall_wait_same_state_snapshot"] = dict(snapshot)
        info = dict(armed=True, streak=int(streak), after=int(after), release_it=int(cfg.get("_stage2_stall_wait_until_it", release_it)))
        if MPI.rank(MPI.comm_world) == 0:
            print("[stage2-stall-wait] it=%03d armed: same state full line-search stalled %d/%d times; skip repeated trials until it=%d unless state/target/lambda changes" %
                  (it_i, int(streak), int(after), int(info["release_it"])), flush=True)
        return info
    return dict(armed=False, streak=int(streak), after=int(after))


def stage2_stall_wait_should_skip(
    cfg,
    res_eval,
    it,
    cooldown_active=False,
    recovery_active=False,
    lambda_plateau_cooldown_active=False,
    hard_shift_only_active=False,
    skip_j_compare_this_iter=False,
):
    """Return True when the current iteration should bypass repeated full line-search."""
    if not bool(cfg.get("stage2_stall_wait_enabled", True)):
        return False
    if not bool(cfg.get("_stage2_stall_wait_active", False)):
        return False
    reason_wait = str(cfg.get("_stage2_stall_wait_reason", "cooldown"))
    same_state_wait = (reason_wait == "same-state-line-search-stall")
    if cooldown_active or recovery_active or lambda_plateau_cooldown_active:
        reset_stage2_stall_watchdog(cfg)
        return False
    if hard_shift_only_active or skip_j_compare_this_iter:
        reset_stage2_stall_watchdog(cfg)
        return False
    try:
        it_i = int(it)
        release_i = int(cfg.get("_stage2_stall_wait_until_it", -1))
    except Exception:
        reset_stage2_stall_watchdog(cfg)
        return False
    if release_i < it_i:
        reset_stage2_stall_watchdog(cfg)
        return False
    vf_target = current_vf_target(cfg)
    vc_active = stage2_volume_continuation_active(
        cfg,
        vf=res_eval.get("vf", None),
        vf_target=vf_target,
        hard_shift_only=hard_shift_only_active,
    )
    if not vc_active:
        if same_state_wait:
            if vf_target is None:
                reset_stage2_stall_watchdog(cfg)
                return False
            try:
                if float(res_eval.get("vf", 0.0)) <= float(vf_target) + stage_controller_tolerance(cfg):
                    reset_stage2_stall_watchdog(cfg)
                    return False
            except Exception:
                reset_stage2_stall_watchdog(cfg)
                return False
        else:
            reset_stage2_stall_watchdog(cfg)
            return False
    vf_wait = cfg.get("_stage2_stall_wait_vf", None)
    vf_now = float(res_eval.get("vf", float("nan")))
    vf_tol = max(
        float(cfg.get("hard_shift_plateau_vf_tol", 1e-12)),
        float(cfg.get("stage2_stall_wait_vf_tol", 1e-10)),
    )
    if vf_wait is not None:
        try:
            if abs(float(vf_wait) - vf_now) > vf_tol:
                reset_stage2_stall_watchdog(cfg)
                return False
        except Exception:
            reset_stage2_stall_watchdog(cfg)
            return False
    target_wait = cfg.get("_stage2_stall_wait_target", None)
    target_tol = max(
        1e-12,
        float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0))),
    )
    if target_wait is not None:
        if vf_target is None:
            reset_stage2_stall_watchdog(cfg)
            return False
        try:
            if abs(float(target_wait) - float(vf_target)) > target_tol:
                reset_stage2_stall_watchdog(cfg)
                return False
        except Exception:
            reset_stage2_stall_watchdog(cfg)
            return False
    if same_state_wait:
        same_snapshot = cfg.get("_stage2_stall_wait_same_state_snapshot", None)
        if not _same_state_stall_snapshot_matches(cfg, same_snapshot, res_eval):
            reset_stage2_stall_watchdog(cfg)
            return False
        return True
    diag = cfg.get("_stage2_last_stall_diagnosis", None)
    if isinstance(diag, dict) and not bool(diag.get("watchdog_allow_fallback", False)):
        reset_stage2_stall_watchdog(cfg)
        return False
    return True


def mark_stage2_stall_wait_iteration(cfg, res_eval, it, kappa_min):
    """Refresh stall diagnosis for a skipped wait iteration so existing fallback code runs."""
    diag_old = cfg.get("_stage2_last_stall_diagnosis", {})
    diag = dict(diag_old) if isinstance(diag_old, dict) else {}
    reason_wait = str(cfg.get("_stage2_stall_wait_reason", "cooldown"))
    same_state_wait = (reason_wait == "same-state-line-search-stall")
    vf_target = current_vf_target(cfg)
    vf_now = float(res_eval.get("vf", 0.0))
    c_now = max(vf_now - float(vf_target), 0.0) if vf_target is not None else 0.0
    skipped = int(cfg.get("_stage2_stall_wait_skipped_iters", 0)) + 1
    cfg["_stage2_stall_wait_skipped_iters"] = int(skipped)
    if same_state_wait:
        prev_action = str(diag.get("action", "ambiguous-hold"))
        if not stage2_volume_continuation_active(cfg, vf=res_eval.get("vf", None), vf_target=vf_target):
            prev_action = "inactive"
        allow_fallback = bool(diag.get("watchdog_allow_fallback", False))
    else:
        prev_action = "escape-soft-reset"
        allow_fallback = True
    diag.update(
        action=prev_action,
        reason="%s+stall-wait-skip" % str(diag.get("reason", reason_wait)),
        n_trials=0,
        c_old=float(c_now),
        c_best=float(c_now),
        vf_reduction_frac=0.0,
        mech_increase=0.0,
        penalty_decrease=0.0,
        merit_excess=0.0,
        best_kappa=float(kappa_min),
        best_merit=float(res_eval.get("J_compare", res_eval.get("J", 0.0))),
        best_merit_kappa=float(kappa_min),
        watchdog_total_streak=int(diag.get("watchdog_total_streak", cfg.get("_stage2_stall_watchdog_total", 0))),
        watchdog_nonweak_streak=int(diag.get("watchdog_nonweak_streak", cfg.get("_stage2_stall_watchdog_nonweak", 0))),
        watchdog_weak_streak=0,
        watchdog_allow_fallback=bool(allow_fallback),
        it=int(it),
    )
    cfg["_stage2_last_stall_diagnosis"] = diag
    return diag


def stage2_post_nucleation_takeover_active(cfg, res_eval=None, it=None, hard_shift_only=False):
    """Protected window where continuous volume control should recover after rare nucleation."""
    if not bool(cfg.get("stage2_post_nucleation_takeover_enabled", False)):
        return False
    if bool(hard_shift_only):
        return False
    last_nuc_it = cfg.get("_last_nucleation_it", None)
    if last_nuc_it is None or it is None:
        return False
    try:
        age = int(it) - int(last_nuc_it)
    except Exception:
        return False
    max_age = max(0, int(cfg.get("stage2_post_nucleation_takeover_iters", 0)))
    if age <= 0 or age > max_age:
        return False
    vf_target = current_vf_target(cfg)
    if vf_target is None:
        return False
    if not isinstance(res_eval, dict):
        return False
    try:
        vf_now = float(res_eval.get("vf", 0.0))
        vf_target_f = float(vf_target)
    except Exception:
        return False
    if not stage2_volume_continuation_active(
        cfg, vf=vf_now, vf_target=vf_target_f, hard_shift_only=hard_shift_only
    ):
        return False
    min_gap = max(
        0.0,
        float(cfg.get("stage2_post_nucleation_takeover_min_gap_abs", 0.0)),
        float(cfg.get("stage2_vf_rate_min_gap_abs", 0.0)),
        stage_controller_tolerance(cfg),
    )
    return bool(vf_now > vf_target_f + min_gap)


def apply_stage2_post_nucleation_takeover(cfg, res_eval, psi, M=None, it=None, reason="stall"):
    """Grow vf-rate authority during the post-nucleation takeover window."""
    if not stage2_post_nucleation_takeover_active(cfg, res_eval=res_eval, it=it):
        return None
    gain_old = stage2_vf_rate_gain(cfg)
    gain_min = max(0.0, float(cfg.get("stage2_vf_rate_gain_min", 0.25)))
    gain_max = max(gain_min, float(cfg.get("stage2_vf_rate_gain_max", 2.0)))
    grow = max(1.0, float(cfg.get("stage2_post_nucleation_takeover_gain_grow", 1.20)))
    gain_floor = max(gain_min, float(cfg.get("stage2_post_nucleation_takeover_gain_floor", gain_old)))
    gain_new = min(gain_max, max(gain_floor, gain_old * grow))
    cfg["_stage2_vf_rate_gain"] = float(gain_new)
    cfg["_stage2_vf_rate_last_update_it"] = None if it is None else int(it)
    cfg["_stage2_vf_rate_history"] = []

    lambda_old = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
    _, lambda_info = update_lambda_v_from_stage_state(
        cfg, res_eval, psi=psi, M=M, it=it, reason="%s-post-nucleation-takeover" % str(reason)
    )
    lambda_new = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
    cfg["_stage2_stall_wait_active"] = False
    cfg["_stage2_same_state_stall_streak"] = 0
    info = dict(
        reason=str(reason),
        action="post-nucleation-takeover-grow",
        gain_old=float(gain_old),
        gain_new=float(gain_new),
        lambda_old=float(lambda_old),
        lambda_new=float(lambda_new),
        vf=float(res_eval.get("vf", 0.0)),
        vf_target=current_vf_target(cfg),
        last_nucleation_it=cfg.get("_last_nucleation_it", None),
        age=(None if it is None or cfg.get("_last_nucleation_it", None) is None else int(it) - int(cfg.get("_last_nucleation_it"))),
        lambda_info=lambda_info,
    )
    cfg["_stage2_volume_continuation_last"] = info
    if MPI.rank(MPI.comm_world) == 0:
        print("[stage2-vc-stall] it=%s reason=%s action=post-nucleation-takeover-grow "
              "vf=%.6f target=%s age=%s gain %.3f->%.3f lambda_v %.3e->%.3e" %
              ("n/a" if it is None else "%03d" % int(it),
               str(reason),
               float(info["vf"]),
               ("None" if info["vf_target"] is None else "%.6f" % float(info["vf_target"])),
               ("n/a" if info["age"] is None else str(int(info["age"]))),
               float(gain_old), float(gain_new), float(lambda_old), float(lambda_new)),
              flush=True)
    return info


def stage2_merit_relax_below_target_active(cfg, res_eval=None, vf=None, vf_target=None):
    if cfg.get("stage2_merit_relax_below_target", None) is None:
        return False
    if vf is None and isinstance(res_eval, dict):
        vf = res_eval.get("vf", None)
    if vf_target is None and isinstance(res_eval, dict):
        vf_target = res_eval.get("vf_target", None)
    if vf_target is None:
        vf_target = current_vf_target(cfg)
    if vf is None or vf_target is None:
        return False
    try:
        tol = max(
            0.0,
            float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0))),
            float(cfg.get("vf_al_adapt_eps", 1e-12)),
        )
        return float(vf) <= float(vf_target) + float(tol)
    except Exception:
        return False


def stage2_base_merit_relax(cfg, res_eval=None, vf=None, vf_target=None):
    base = max(0.0, float(cfg.get("stage2_merit_relax", 0.0)))
    if stage2_merit_relax_below_target_active(cfg, res_eval=res_eval, vf=vf, vf_target=vf_target):
        base = max(0.0, float(cfg.get("stage2_merit_relax_below_target", base)))
    return base


def stage2_effective_merit_relax(cfg, res_eval=None, vf=None, vf_target=None):
    base = stage2_base_merit_relax(cfg, res_eval=res_eval, vf=vf, vf_target=vf_target)
    if (
        stage2_merit_relax_below_target_active(cfg, res_eval=res_eval, vf=vf, vf_target=vf_target)
        and bool(cfg.get("stage2_merit_relax_below_target_disable_auto", True))
    ):
        return base
    if not bool(cfg.get("stage2_merit_relax_auto_enabled", False)):
        return base
    cap = max(base, float(cfg.get("stage2_merit_relax_auto_max", base)))
    eff = max(base, float(cfg.get("_stage2_merit_relax_eff", base)))
    return min(cap, eff)


def stage2_raise_merit_relax_for_rescue(cfg, required_relax):
    base = stage2_base_merit_relax(cfg)
    if not bool(cfg.get("stage2_merit_relax_auto_enabled", False)):
        return base
    cap = max(base, float(cfg.get("stage2_merit_relax_auto_max", base)))
    safety = max(1.0, float(cfg.get("stage2_merit_relax_auto_safety", 1.10)))
    old_eff = stage2_effective_merit_relax(cfg)
    new_eff = min(cap, max(base, old_eff, max(0.0, float(required_relax)) * safety))
    cfg["_stage2_merit_relax_eff"] = float(new_eff)
    return float(new_eff)


def stage2_decay_merit_relax_after_accept(cfg):
    base = stage2_base_merit_relax(cfg)
    if not bool(cfg.get("stage2_merit_relax_auto_enabled", False)):
        cfg["_stage2_merit_relax_eff"] = float(base)
        return float(base)
    eff = stage2_effective_merit_relax(cfg)
    if eff <= base + 1e-15:
        cfg["_stage2_merit_relax_eff"] = float(base)
        return float(base)
    decay = min(1.0, max(0.0, float(cfg.get("stage2_merit_relax_auto_decay", 0.80))))
    new_eff = base + (eff - base) * decay
    if new_eff <= base + 1e-12:
        new_eff = base
    cfg["_stage2_merit_relax_eff"] = float(new_eff)
    return float(new_eff)


def stage2_update_vf_rate_controller(
    cfg,
    res_eval,
    it=None,
    cooldown_active=False,
    recovery_active=False,
    hard_shift_only=False,
):
    """Closed-loop controller for total stage-2 lambda_v authority.

    The controller compares the observed accepted-step vf drop rate with the
    desired rate for the current target.  It updates one multiplier,
    `_stage2_vf_rate_gain`, which is applied to the final ratio cap used by
    `_lambda_v_target_from_state`.
    """
    if not bool(cfg.get("stage2_vf_rate_control_enabled", False)):
        cfg["_stage2_vf_rate_history"] = []
        cfg["_stage2_vf_rate_gain"] = 1.0
        return None
    if bool(cooldown_active) or bool(recovery_active) or bool(hard_shift_only):
        return None

    vf_target = current_vf_target(cfg)
    vf_now = float(res_eval.get("vf", 0.0))
    if not stage2_volume_continuation_active(cfg, vf=vf_now, vf_target=vf_target, hard_shift_only=hard_shift_only):
        cfg["_stage2_vf_rate_history"] = []
        return None
    if vf_target is None:
        return None
    vf_target = float(vf_target)
    violation = max(float(vf_now) - vf_target, 0.0)
    tol = max(0.0, float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0))))
    min_gap = max(
        0.0,
        float(cfg.get("stage2_vf_rate_min_gap_abs", 0.0)),
        2.0 * tol,
    )

    gain_old = stage2_vf_rate_gain(cfg)
    gain_min = max(0.0, float(cfg.get("stage2_vf_rate_gain_min", 0.25)))
    gain_max = max(gain_min, float(cfg.get("stage2_vf_rate_gain_max", 2.0)))
    target_key = round(vf_target, 12)
    prev_target = cfg.get("_stage2_vf_rate_target", None)
    if prev_target is None or abs(float(prev_target) - target_key) > 1e-12:
        cfg["_stage2_vf_rate_target"] = float(target_key)
        cfg["_stage2_vf_rate_history"] = []

    def _set_gain(action, gain_new, avg_drop=None, desired_drop=None, rate_ratio=None):
        gain_new = min(gain_max, max(gain_min, float(gain_new)))
        cfg["_stage2_vf_rate_gain"] = float(gain_new)
        info = dict(
            action=str(action),
            gain_old=float(gain_old),
            gain_new=float(gain_new),
            vf=float(vf_now),
            vf_target=float(vf_target),
            violation=float(violation),
            avg_drop=(None if avg_drop is None else float(avg_drop)),
            desired_drop=(None if desired_drop is None else float(desired_drop)),
            rate_ratio=(None if rate_ratio is None else float(rate_ratio)),
        )
        cfg["_stage2_vf_rate_last"] = info
        if MPI.rank(MPI.comm_world) == 0:
            avg_msg = "None" if avg_drop is None else "%.3e" % float(avg_drop)
            des_msg = "None" if desired_drop is None else "%.3e" % float(desired_drop)
            ratio_msg = "None" if rate_ratio is None else "%.3f" % float(rate_ratio)
            print("[stage2-vf-rate] it=%s action=%s vf=%.6f target=%.6f gap=%.3e avg_drop=%s desired=%s rate_ratio=%s gain %.3f->%.3f" %
                  ("n/a" if it is None else "%03d" % int(it), str(action),
                   float(vf_now), float(vf_target), float(violation),
                   avg_msg, des_msg, ratio_msg, float(gain_old), float(gain_new)),
                  flush=True)
        return info

    if violation <= min_gap:
        cfg["_stage2_vf_rate_history"] = []
        if float(vf_now) < vf_target - max(tol, 1e-12):
            shrink = min(1.0, max(0.0, float(cfg.get("stage2_vf_rate_gain_shrink", 0.75))))
            return _set_gain("overshoot-shrink", gain_old * shrink)
        relax = min(1.0, max(0.0, float(cfg.get("stage2_vf_rate_gain_relax", 0.08))))
        if abs(gain_old - 1.0) > 1e-12 and relax > 0.0:
            return _set_gain("near-target-relax", gain_old + (1.0 - gain_old) * relax)
        return None

    hist = list(cfg.get("_stage2_vf_rate_history", []))
    hist.append(dict(it=(None if it is None else int(it)), vf=float(vf_now), violation=float(violation)))
    window = max(2, int(cfg.get("stage2_vf_rate_window", 12)))
    hist = hist[-(window + 1):]
    cfg["_stage2_vf_rate_history"] = hist
    if len(hist) < window + 1:
        return dict(action="collect", gain_old=float(gain_old), gain_new=float(gain_old))

    vf_seq = [float(h["vf"]) for h in hist]
    total_drop = max(vf_seq[0] - vf_seq[-1], 0.0)
    avg_drop = total_drop / float(max(1, len(vf_seq) - 1))
    dv_now = current_stage_dv(cfg)
    dv_scale = float(dv_now) if dv_now is not None else float(cfg.get("vf_stage_dv0", 0.03))
    target_iters = max(1.0, float(cfg.get("stage2_vf_rate_target_iters", 80)))
    desired_drop = min(max(float(violation), 0.0), max(float(dv_scale), 0.0)) / target_iters
    desired_drop = max(float(cfg.get("stage2_vf_rate_min_drop_abs", 0.0)), desired_drop)
    max_drop = float(cfg.get("stage2_vf_rate_max_drop_abs", desired_drop))
    if max_drop > 0.0:
        desired_drop = min(max_drop, desired_drop)
    desired_drop = max(desired_drop, 1e-30)
    rate_ratio = float(avg_drop) / desired_drop

    slow_factor = max(0.0, float(cfg.get("stage2_vf_rate_slow_factor", 0.55)))
    fast_factor = max(slow_factor, float(cfg.get("stage2_vf_rate_fast_factor", 1.60)))
    cooldown = max(0, int(cfg.get("stage2_vf_rate_update_cooldown_iters", 0)))
    last_update = cfg.get("_stage2_vf_rate_last_update_it", None)
    in_cooldown = (
        last_update is not None
        and it is not None
        and int(it) - int(last_update) < cooldown
    )
    if in_cooldown:
        return dict(
            action="cooldown",
            gain_old=float(gain_old),
            gain_new=float(gain_old),
            avg_drop=float(avg_drop),
            desired_drop=float(desired_drop),
            rate_ratio=float(rate_ratio),
        )

    if rate_ratio < slow_factor:
        grow = max(1.0, float(cfg.get("stage2_vf_rate_gain_grow", 1.15)))
        cfg["_stage2_vf_rate_last_update_it"] = None if it is None else int(it)
        return _set_gain("slow-grow", gain_old * grow, avg_drop, desired_drop, rate_ratio)
    if rate_ratio > fast_factor:
        shrink = min(1.0, max(0.0, float(cfg.get("stage2_vf_rate_gain_shrink", 0.75))))
        cfg["_stage2_vf_rate_last_update_it"] = None if it is None else int(it)
        return _set_gain("fast-shrink", gain_old * shrink, avg_drop, desired_drop, rate_ratio)

    relax = min(1.0, max(0.0, float(cfg.get("stage2_vf_rate_gain_relax", 0.08))))
    if abs(gain_old - 1.0) > 1e-12 and relax > 0.0:
        cfg["_stage2_vf_rate_last_update_it"] = None if it is None else int(it)
        return _set_gain("in-band-relax", gain_old + (1.0 - gain_old) * relax, avg_drop, desired_drop, rate_ratio)
    return dict(
        action="in-band",
        gain_old=float(gain_old),
        gain_new=float(gain_old),
        avg_drop=float(avg_drop),
        desired_drop=float(desired_drop),
        rate_ratio=float(rate_ratio),
    )


def stage2_update_slow_progress_watchdog(
    cfg,
    res_eval,
    it=None,
    kappa=None,
    cooldown_active=False,
    recovery_active=False,
    hard_shift_only=False,
):
    """Detect accepted-step slow volume progress and raise the effective low-vf cap.

    This handles the outputs8 failure mode: steps were accepted and `vf` did
    decrease, but only by about 7e-5 per iteration while `rho` and `mu` were
    already capped.  The ordinary stall watchdog never fired because this was
    not a hard line-search stall.
    """
    base_low = max(0.0, float(cfg.get("stage2_lambda_ratio_cap_low_vf", 0.0)))
    if not bool(cfg.get("stage2_slow_progress_watchdog_enabled", False)):
        cfg["_stage2_slow_progress_history"] = []
        cfg["_stage2_slow_progress_hits"] = 0
        cfg["_stage2_lambda_ratio_cap_low_vf_eff"] = float(base_low)
        return None
    if bool(cooldown_active) or bool(recovery_active) or bool(hard_shift_only):
        return None
    vf_target = current_vf_target(cfg)
    vf_now = float(res_eval.get("vf", 0.0))
    if not stage2_volume_continuation_active(cfg, vf=vf_now, vf_target=vf_target, hard_shift_only=hard_shift_only):
        cfg["_stage2_slow_progress_history"] = []
        cfg["_stage2_slow_progress_hits"] = 0
        return None
    if vf_target is None:
        return None
    vf_target = float(vf_target)
    low_vf_threshold = float(cfg.get("stage2_lambda_ratio_low_vf_threshold", 0.35))
    if vf_now >= low_vf_threshold:
        cfg["_stage2_slow_progress_history"] = []
        cfg["_stage2_slow_progress_hits"] = 0
        return None

    violation = max(vf_now - vf_target, 0.0)
    tol = max(0.0, float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0))))
    min_gap = max(
        0.0,
        float(cfg.get("stage2_slow_progress_min_gap_abs", 0.0)),
        tol * float(cfg.get("stage2_slow_progress_min_gap_tol_factor", 1.0)),
    )
    if violation <= min_gap:
        cfg["_stage2_slow_progress_history"] = []
        cfg["_stage2_slow_progress_hits"] = 0
        eff_old = max(base_low, float(cfg.get("_stage2_lambda_ratio_cap_low_vf_eff", base_low)))
        if eff_old > base_low + 1e-15:
            decay = min(1.0, max(0.0, float(cfg.get("stage2_slow_progress_lambda_ratio_decay", 0.75))))
            eff_new = base_low + (eff_old - base_low) * decay
            if eff_new <= base_low + 1e-12:
                eff_new = base_low
            cfg["_stage2_lambda_ratio_cap_low_vf_eff"] = float(eff_new)
            if MPI.rank(MPI.comm_world) == 0:
                print("[stage2-slow-progress] it=%s action=lambda-ratio-decay gap=%.3e<=%.3e low_vf_cap %.3f->%.3f" %
                      ("n/a" if it is None else "%03d" % int(it), float(violation), float(min_gap), float(eff_old), float(eff_new)),
                      flush=True)
        return None

    target_key = round(vf_target, 12)
    prev_target = cfg.get("_stage2_slow_progress_target", None)
    if prev_target is None or abs(float(prev_target) - target_key) > 1e-12:
        cfg["_stage2_slow_progress_target"] = float(target_key)
        cfg["_stage2_slow_progress_history"] = []
        cfg["_stage2_slow_progress_hits"] = 0

    hist = list(cfg.get("_stage2_slow_progress_history", []))
    hist.append(dict(it=(None if it is None else int(it)), vf=float(vf_now), violation=float(violation)))
    window = max(2, int(cfg.get("stage2_slow_progress_window", 20)))
    hist = hist[-(window + 1):]
    cfg["_stage2_slow_progress_history"] = hist
    if len(hist) < window + 1:
        return None

    vf_seq = [float(h["vf"]) for h in hist]
    drop_seq = [max(vf_seq[k] - vf_seq[k + 1], 0.0) for k in range(len(vf_seq) - 1)]
    avg_positive_drop_abs = float(sum(drop_seq)) / float(max(1, len(drop_seq)))
    total_drop = max(vf_seq[0] - vf_seq[-1], 0.0)
    avg_drop_abs = float(total_drop) / float(max(1, len(drop_seq)))
    slow_abs = max(0.0, float(cfg.get("stage2_slow_progress_min_avg_drop_abs", 1e-4)))
    slow_rel = max(0.0, float(cfg.get("stage2_slow_progress_min_avg_drop_rel", 0.0)))
    avg_drop_rel = avg_drop_abs / max(abs(vf_now), 1e-12)
    slow = bool((avg_drop_abs <= slow_abs) or (slow_rel > 0.0 and avg_drop_rel <= slow_rel))

    rho = max(0.0, float(cfg.get("_vf_aug_lag_rho", cfg.get("rho_v", 0.0))))
    rho_max = max(rho, float(cfg.get("vf_al_rho_max", cfg.get("stage2_al_rho_max", rho))))
    mu = max(0.0, float(cfg.get("_vf_aug_lag_mu", cfg.get("mu_v0", 0.0))))
    mu_max = max(mu, float(cfg.get("stage2_al_mu_max", cfg.get("stage2_lambda_abs_cap", mu))))
    cap_frac = min(1.0, max(0.0, float(cfg.get("stage2_slow_progress_controller_cap_fraction", 0.95))))
    rho_capped = bool(rho_max <= 0.0 or rho >= cap_frac * rho_max)
    mu_capped = bool(mu_max <= 0.0 or mu >= cap_frac * mu_max)
    require_cap = bool(cfg.get("stage2_slow_progress_require_controller_cap", True))
    controller_ok = (rho_capped or mu_capped) if require_cap else True

    kappa_ok = True
    if kappa is not None:
        kappa_max = float(cfg.get("stage2_slow_progress_kappa_max", float("inf")))
        if np.isfinite(kappa_max) and kappa_max > 0.0:
            kappa_ok = bool(float(kappa) <= kappa_max)

    hit = bool(slow and controller_ok and kappa_ok)
    hits = int(cfg.get("_stage2_slow_progress_hits", 0))
    hits = hits + 1 if hit else 0
    cfg["_stage2_slow_progress_hits"] = int(hits)
    hit_need = max(1, int(cfg.get("stage2_slow_progress_consecutive_hits", 1)))
    cooldown = max(0, int(cfg.get("stage2_slow_progress_cooldown_iters", 10)))
    last_it = cfg.get("_stage2_slow_progress_last_boost_it", None)
    in_cooldown = (last_it is not None) and (it is not None) and (int(it) - int(last_it) < cooldown)
    if (hits < hit_need) or in_cooldown:
        return dict(
            action="watch",
            slow=bool(slow),
            avg_drop_abs=float(avg_drop_abs),
            avg_drop_rel=float(avg_drop_rel),
            avg_positive_drop_abs=float(avg_positive_drop_abs),
            total_drop=float(total_drop),
            violation=float(violation),
            controller_ok=bool(controller_ok),
            kappa_ok=bool(kappa_ok),
            hits=int(hits),
        )

    eff_old = max(base_low, float(cfg.get("_stage2_lambda_ratio_cap_low_vf_eff", base_low)))
    eff_max = max(base_low, float(cfg.get("stage2_slow_progress_lambda_ratio_cap_low_vf_max", base_low)))
    grow_factor = max(1.0, float(cfg.get("stage2_slow_progress_lambda_ratio_grow", 1.0)))
    grow_step = max(0.0, float(cfg.get("stage2_slow_progress_lambda_ratio_step", 0.0)))
    eff_new = min(eff_max, max(eff_old * grow_factor, eff_old + grow_step))
    if eff_new <= eff_old + 1e-15:
        action = "lambda-ratio-maxed"
        eff_new = eff_old
    else:
        action = "lambda-ratio-grow"
        cfg["_stage2_lambda_ratio_cap_low_vf_eff"] = float(eff_new)
        if it is not None:
            cfg["_stage2_slow_progress_last_boost_it"] = int(it)
        cfg["_stage2_slow_progress_hits"] = 0

    info = dict(
        action=str(action),
        vf=float(vf_now),
        vf_target=float(vf_target),
        violation=float(violation),
        min_gap=float(min_gap),
        avg_drop_abs=float(avg_drop_abs),
        avg_drop_rel=float(avg_drop_rel),
        avg_positive_drop_abs=float(avg_positive_drop_abs),
        total_drop=float(total_drop),
        window=int(window),
        rho=float(rho),
        rho_max=float(rho_max),
        mu=float(mu),
        mu_max=float(mu_max),
        rho_capped=bool(rho_capped),
        mu_capped=bool(mu_capped),
        kappa=(None if kappa is None else float(kappa)),
        low_vf_cap_old=float(eff_old),
        low_vf_cap_new=float(eff_new),
        low_vf_cap_max=float(eff_max),
    )
    cfg["_stage2_slow_progress_last"] = info
    if MPI.rank(MPI.comm_world) == 0:
        print("[stage2-slow-progress] it=%s action=%s vf=%.6f target=%.6f gap=%.3e avg_drop=%.3e rel=%.3e window=%d rho=%.3e/%.3e mu=%.3e/%.3e kappa=%s low_vf_cap %.3f->%.3f max=%.3f" %
              ("n/a" if it is None else "%03d" % int(it),
               str(action),
               float(vf_now),
               float(vf_target),
               float(violation),
               float(avg_drop_abs),
               float(avg_drop_rel),
               int(window),
               float(rho),
               float(rho_max),
               float(mu),
               float(mu_max),
               ("None" if kappa is None else ("%.3e" % float(kappa))),
               float(eff_old),
               float(eff_new),
               float(eff_max)),
              flush=True)
    return info


def diagnose_stage2_line_search_stall(cfg, res_eval, rejected_trials, J_allow, it=None):
    """Classify a stage-2 line-search stall from the rejected trial scalars.

    A stall above the current volume target can mean either the volume
    controller is too weak, or that it is already too strong and the trial
    steps reduce volume only by damaging the mechanical objective.  This
    diagnostic keeps the AL update from treating every stall as a grow signal.
    """
    info = dict(
        enabled=bool(cfg.get("stage2_stall_diagnosis_enabled", False)),
        action="weak-grow",
        reason="disabled",
        n_trials=0,
        vf_reduction_frac=0.0,
        vf_reduction=0.0,
        mech_increase=0.0,
        penalty_decrease=0.0,
        merit_excess=0.0,
        best_kappa=float("nan"),
        c_old=0.0,
        c_best=0.0,
    )
    if not bool(info["enabled"]):
        cfg["_stage2_last_stall_diagnosis"] = info
        return info
    if not stage2_volume_continuation_active(cfg, vf=res_eval.get("vf", None), vf_target=current_vf_target(cfg)):
        info["action"] = "inactive"
        info["reason"] = "stage2-inactive"
        cfg["_stage2_last_stall_diagnosis"] = info
        return info
    valid_trials = []
    for trial in rejected_trials:
        try:
            J_try = float(trial.get("J", float("nan")))
            J_compare_try = float(trial.get("J_compare", float("nan")))
            vf_try = float(trial.get("vf", float("nan")))
            kappa_try = float(trial.get("kappa", float("nan")))
        except Exception:
            continue
        if np.isfinite(J_try) and np.isfinite(J_compare_try) and np.isfinite(vf_try):
            valid_trials.append(
                dict(
                    J=float(J_try),
                    J_compare=float(J_compare_try),
                    vf=float(vf_try),
                    kappa=float(kappa_try),
                )
            )
    info["n_trials"] = int(len(valid_trials))
    if len(valid_trials) == 0:
        info["action"] = "ambiguous-hold"
        info["reason"] = "no-valid-trials"
        cfg["_stage2_last_stall_diagnosis"] = info
        return info

    vf_target = current_vf_target(cfg)
    if vf_target is None:
        info["action"] = "ambiguous-hold"
        info["reason"] = "no-vf-target"
        cfg["_stage2_last_stall_diagnosis"] = info
        return info
    vf_target = float(vf_target)
    c_old = max(float(res_eval.get("vf", 0.0)) - vf_target, 0.0)
    info["c_old"] = float(c_old)
    tol = max(0.0, float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0))))
    eps = max(1e-30, float(cfg.get("vf_al_adapt_eps", 1e-12)))
    if c_old <= max(tol, eps):
        info["action"] = "ambiguous-hold"
        info["reason"] = "no-positive-violation"
        cfg["_stage2_last_stall_diagnosis"] = info
        return info

    J_old = float(res_eval.get("J", 0.0))
    J_compare_old = float(res_eval.get("J_compare", J_old))
    penalty_old = J_compare_old - J_old
    for trial in valid_trials:
        c_try = max(float(trial["vf"]) - vf_target, 0.0)
        penalty_try = float(trial["J_compare"]) - float(trial["J"])
        trial["c"] = float(c_try)
        trial["vf_reduction"] = float(c_old - c_try)
        trial["vf_reduction_frac"] = float((c_old - c_try) / max(c_old, eps))
        trial["mech_increase"] = float(trial["J"] - J_old)
        trial["penalty_decrease"] = float(penalty_old - penalty_try)
        trial["merit_excess"] = float(trial["J_compare"] - float(J_allow))

    best_vf_trial = max(valid_trials, key=lambda t: (t["vf_reduction"], -t["J_compare"]))
    best_merit_trial = min(valid_trials, key=lambda t: t["J_compare"])
    info.update(
        vf_reduction=float(best_vf_trial["vf_reduction"]),
        vf_reduction_frac=float(best_vf_trial["vf_reduction_frac"]),
        mech_increase=float(best_vf_trial["mech_increase"]),
        penalty_decrease=float(best_vf_trial["penalty_decrease"]),
        merit_excess=float(best_vf_trial["merit_excess"]),
        best_kappa=float(best_vf_trial["kappa"]),
        c_best=float(best_vf_trial["c"]),
        best_merit=float(best_merit_trial["J_compare"]),
        best_merit_kappa=float(best_merit_trial["kappa"]),
        best_merit_mech_increase=float(best_merit_trial["mech_increase"]),
    )

    j_scale = max(abs(J_old), abs(J_compare_old), 1e-12)
    overdrive_vf_frac = max(0.0, float(cfg.get("stage2_stall_overdrive_vf_reduction_frac", 0.20)))
    overdrive_vf_abs = max(
        0.0,
        float(cfg.get("stage2_stall_overdrive_vf_reduction_abs", 0.0)),
        tol * float(cfg.get("stage2_stall_overdrive_vf_tol_factor", 0.25)),
    )
    weak_vf_frac = max(0.0, float(cfg.get("stage2_stall_weak_vf_reduction_frac", 0.08)))
    weak_vf_abs = max(
        0.0,
        float(cfg.get("stage2_stall_weak_vf_reduction_abs", 0.0)),
        tol * float(cfg.get("stage2_stall_weak_vf_tol_factor", 0.10)),
    )
    mech_bad_abs = max(
        0.0,
        float(cfg.get("stage2_stall_overdrive_mech_increase_abs", 0.0)),
        float(cfg.get("stage2_stall_overdrive_mech_increase_rel", 5e-3)) * j_scale,
    )
    dominance = max(0.0, float(cfg.get("stage2_stall_overdrive_mech_penalty_dominance", 1.0)))
    merit_excess_tol = max(0.0, float(cfg.get("stage2_stall_merit_excess_rel", 1e-5)) * j_scale)

    volume_helped = (
        best_vf_trial["vf_reduction"] >= overdrive_vf_abs
        or best_vf_trial["vf_reduction_frac"] >= overdrive_vf_frac
    )
    no_volume_progress = (
        best_vf_trial["vf_reduction"] <= weak_vf_abs
        and best_vf_trial["vf_reduction_frac"] <= weak_vf_frac
    )
    mech_dominates = (
        best_vf_trial["mech_increase"] > mech_bad_abs
        and best_vf_trial["mech_increase"] > dominance * max(best_vf_trial["penalty_decrease"], 0.0)
    )
    merit_still_rejected = best_vf_trial["merit_excess"] > merit_excess_tol
    best_merit_mech_ok = best_merit_trial["mech_increase"] <= mech_bad_abs

    if volume_helped and mech_dominates and merit_still_rejected:
        info["action"] = "overdrive-shrink"
        info["reason"] = "vf-improves-but-mechanical-J-dominates"
    elif no_volume_progress and best_merit_mech_ok:
        info["action"] = "weak-grow"
        info["reason"] = "vf-does-not-move-and-mechanical-J-ok"
    elif no_volume_progress and (not best_merit_mech_ok):
        info["action"] = "conflict-hold"
        info["reason"] = "vf-does-not-move-and-mechanical-J-worsens"
    else:
        info["action"] = "ambiguous-hold"
        info["reason"] = "mixed-trial-signals"
    if bool(cfg.get("stage2_stall_watchdog_enabled", True)):
        target_key = None if vf_target is None else float(vf_target)
        prev_target = cfg.get("_stage2_stall_watchdog_target", None)
        target_tol = max(
            1e-12,
            float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0))),
        )
        same_target = (
            prev_target is not None
            and target_key is not None
            and abs(float(prev_target) - float(target_key)) <= target_tol
        )
        if not same_target:
            cfg["_stage2_stall_watchdog_total"] = 0
            cfg["_stage2_stall_watchdog_nonweak"] = 0
            cfg["_stage2_stall_watchdog_weak"] = 0
        cfg["_stage2_stall_watchdog_target"] = target_key
        total_streak = int(cfg.get("_stage2_stall_watchdog_total", 0)) + 1
        cfg["_stage2_stall_watchdog_total"] = int(total_streak)
        if str(info["action"]) == "weak-grow":
            weak_streak = int(cfg.get("_stage2_stall_watchdog_weak", 0)) + 1
            nonweak_streak = 0
        else:
            nonweak_streak = int(cfg.get("_stage2_stall_watchdog_nonweak", 0)) + 1
            weak_streak = 0
        cfg["_stage2_stall_watchdog_nonweak"] = int(nonweak_streak)
        cfg["_stage2_stall_watchdog_weak"] = int(weak_streak)
        cfg["_stage2_stall_watchdog_last_action"] = str(info["action"])
        soft_reset_after = max(1, int(cfg.get("stage2_stall_nonweak_soft_reset_after", 3)))
        fallback_after = max(soft_reset_after, int(cfg.get("stage2_stall_nonweak_allow_fallback_after", 6)))
        info["watchdog_total_streak"] = int(total_streak)
        info["watchdog_nonweak_streak"] = int(nonweak_streak)
        info["watchdog_weak_streak"] = int(weak_streak)
        info["watchdog_allow_fallback"] = False
        if str(info["action"]) != "weak-grow" and nonweak_streak >= soft_reset_after:
            info["action"] = "escape-soft-reset"
            info["reason"] = "%s+watchdog-soft-reset" % str(info["reason"])
        if str(info["action"]) != "weak-grow" and nonweak_streak >= fallback_after:
            info["watchdog_allow_fallback"] = True
            info["reason"] = "%s+watchdog-fallback-ok" % str(info["reason"])
    else:
        info["watchdog_total_streak"] = 0
        info["watchdog_nonweak_streak"] = 0
        info["watchdog_weak_streak"] = 0
        info["watchdog_allow_fallback"] = False
    if it is not None:
        info["it"] = int(it)
    cfg["_stage2_last_stall_diagnosis"] = info
    if MPI.rank(MPI.comm_world) == 0:
        prefix = "[stage2-stall-diagnosis]" if it is None else "[stage2-stall-diagnosis] it=%03d" % int(it)
        print("%s action=%s reason=%s trials=%d c %.3e->%.3e dcf=%.3f dJ_mech=%.3e dPenalty=%.3e merit_excess=%.3e best_kappa=%.4e best_merit=%.6e@%.4e streak=%d/nonweak=%d fallback=%s" %
              (prefix, str(info["action"]), str(info["reason"]), int(info["n_trials"]),
               float(info["c_old"]), float(info["c_best"]), float(info["vf_reduction_frac"]),
               float(info["mech_increase"]), float(info["penalty_decrease"]),
               float(info["merit_excess"]), float(info["best_kappa"]),
               float(info["best_merit"]), float(info["best_merit_kappa"]),
               int(info.get("watchdog_total_streak", 0)),
               int(info.get("watchdog_nonweak_streak", 0)),
               str(bool(info.get("watchdog_allow_fallback", False)))),
              flush=True)
    return info


def handle_stage2_stall_volume_controller(cfg, res_eval, psi, M=None, it=None, reason="stall"):
    """Apply AL/lambda_v response chosen by the latest stall diagnosis."""
    if it is not None:
        last_handled_it = cfg.get("_stage2_stall_controller_handled_it", None)
        if last_handled_it is not None and int(last_handled_it) == int(it):
            info = cfg.get("_stage2_volume_continuation_last", None)
            if isinstance(info, dict):
                if MPI.rank(MPI.comm_world) == 0:
                    print("[stage2-vc-stall] it=%03d reason=%s action=already-handled-this-iteration previous=%s" %
                          (int(it), str(reason), str(info.get("action", info.get("adapt_tag", "unknown")))),
                          flush=True)
                return info
        cfg["_stage2_stall_controller_handled_it"] = int(it)
    diag = cfg.get("_stage2_last_stall_diagnosis", None)
    action = "weak-grow"
    if bool(cfg.get("stage2_stall_diagnosis_enabled", False)) and isinstance(diag, dict):
        if (it is not None) and ("it" in diag) and (int(diag.get("it")) != int(it)):
            action = "ambiguous-hold"
        else:
            action = str(diag.get("action", "weak-grow"))
    if action in ("overdrive-shrink", "escape-soft-reset", "conflict-hold", "ambiguous-hold"):
        takeover_info = apply_stage2_post_nucleation_takeover(
            cfg, res_eval, psi=psi, M=M, it=it, reason=reason
        )
        if takeover_info is not None:
            return takeover_info
    if action in ("overdrive-shrink", "escape-soft-reset"):
        refresh_volume_merit_inplace(cfg, res_eval)
        rho_old = max(0.0, float(cfg.get("_vf_aug_lag_rho", cfg.get("rho_v", 0.0))))
        mu_old = max(0.0, float(cfg.get("_vf_aug_lag_mu", cfg.get("mu_v0", 0.0))))
        lambda_old = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
        rho_min = max(0.0, float(cfg.get("vf_al_rho_min", 0.0)))
        mu_max = float(cfg.get("stage2_al_mu_max", cfg.get("stage2_lambda_abs_cap", max(mu_old, 1.0))))
        action_for_log = str(action)
        hold_for_fallback = (
            action == "escape-soft-reset"
            and bool(cfg.get("stage2_stall_escape_hold_when_fallback_allowed", False))
            and isinstance(diag, dict)
            and bool(diag.get("watchdog_allow_fallback", False))
        )
        if action == "escape-soft-reset":
            if hold_for_fallback:
                rho_factor = 1.0
                mu_factor = 1.0
                lambda_factor = 1.0
                action_for_log = "escape-hold-for-fallback"
            else:
                rho_factor = max(0.0, float(cfg.get("stage2_stall_escape_rho_shrink", 0.50)))
                mu_factor = max(0.0, float(cfg.get("stage2_stall_escape_mu_decay", 0.35)))
                lambda_factor = max(0.0, float(cfg.get("stage2_stall_escape_lambda_decay", 0.0)))
        else:
            rho_factor = max(0.0, float(cfg.get("stage2_stall_overdrive_rho_shrink", 0.75)))
            mu_factor = max(0.0, float(cfg.get("stage2_stall_overdrive_mu_decay", 0.65)))
            lambda_factor = max(0.0, float(cfg.get("stage2_stall_overdrive_lambda_decay", 0.70)))
        rho_new = max(rho_min, rho_old * rho_factor)
        mu_new = max(0.0, min(mu_max, mu_old * mu_factor))
        lambda_new = _clamp_lambda_v_to_cfg(
            cfg,
            lambda_old * lambda_factor,
            include_plateau_cap=False,
        )
        cfg["_vf_aug_lag_rho"] = float(rho_new)
        cfg["_vf_aug_lag_mu"] = float(mu_new)
        cfg["lambda_v"] = float(lambda_new)
        cfg["_vf_aug_lag_prev_violation"] = float(res_eval.get("vf_constraint_violation", 0.0))
        cfg["_vf_aug_lag_last_target"] = current_vf_target(cfg)
        refresh_volume_merit_inplace(cfg, res_eval)
        info = dict(
            reason=str(reason),
            action=str(action_for_log),
            rho_old=float(rho_old),
            rho_new=float(rho_new),
            mu_old=float(mu_old),
            mu_new=float(mu_new),
            lambda_old=float(lambda_old),
            lambda_new=float(lambda_new),
            violation=float(res_eval.get("vf_constraint_violation", 0.0)),
            vf=float(res_eval.get("vf", 0.0)),
            vf_target=current_vf_target(cfg),
        )
        cfg["_stage2_volume_continuation_last"] = info
        if MPI.rank(MPI.comm_world) == 0:
            prefix = "[stage2-vc-stall]" if it is None else "[stage2-vc-stall] it=%03d" % int(it)
            print("%s reason=%s action=%s vf=%.6f target=%s viol=%.3e rho %.3e->%.3e mu %.3e->%.3e lambda_v %.3e->%.3e" %
                  (prefix, str(reason), str(action_for_log), float(info["vf"]),
                   ("None" if info["vf_target"] is None else ("%.6f" % float(info["vf_target"]))),
                   float(info["violation"]), float(rho_old), float(rho_new),
                   float(mu_old), float(mu_new), float(lambda_old), float(lambda_new)),
                  flush=True)
        return info
    if action == "conflict-hold":
        info = update_volume_continuation_state(
            cfg, res_eval, it=it, reason="%s-conflict-hold" % str(reason),
            update_mu=False, adapt_rho=False
        )
        lambda_old = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
        lambda_new = _clamp_lambda_v_to_cfg(
            cfg,
            lambda_old * max(0.0, float(cfg.get("stage2_stall_conflict_lambda_decay", 0.85))),
            include_plateau_cap=False,
        )
        cfg["lambda_v"] = float(lambda_new)
        if MPI.rank(MPI.comm_world) == 0:
            prefix = "[stage2-vc-stall]" if it is None else "[stage2-vc-stall] it=%03d" % int(it)
            print("%s reason=%s action=conflict-hold lambda_v %.3e->%.3e (rho/mu held)" %
                  (prefix, str(reason), float(lambda_old), float(lambda_new)), flush=True)
        return info
    if action in ("ambiguous-hold", "inactive"):
        return update_volume_continuation_state(
            cfg, res_eval, it=it, reason="%s-ambiguous-hold" % str(reason),
            update_mu=False, adapt_rho=False
        )

    info = update_volume_continuation_state(
        cfg, res_eval, it=it, reason="%s-weak-grow" % str(reason), force_rho_grow=True
    )
    update_lambda_v_from_stage_state(cfg, res_eval, psi=psi, M=M, it=it, reason="%s-weak-grow" % str(reason))
    return info


def force_stage2_volume_continuation_start(cfg, res_eval, it=None, reason="plateau-nucleation-trigger"):
    """Start stage-2 volume continuation from a detected mechanical plateau."""
    if not bool(cfg.get("stage2_volume_continuation_enabled", False)):
        return None
    if current_vf_target(cfg) is None:
        return None
    if bool(cfg.get("_stage2_vc_started_once", False)):
        return None
    cfg["_stage2_vc_force_started"] = True
    if it is not None:
        cfg["_stage2_vc_force_start_it"] = int(it)
    refresh_volume_merit_inplace(cfg, res_eval)
    info = update_volume_continuation_state(
        cfg,
        res_eval,
        it=it,
        reason=reason,
        update_mu=False,
        adapt_rho=False,
    )
    if MPI.rank(MPI.comm_world) == 0:
        if info is None:
            print("[stage2-trigger] it=%03d: requested by %s but stage2 did not activate" %
                  (int(it), str(reason)), flush=True)
        else:
            print("[stage2-trigger] it=%03d: stage2 volume continuation started by %s" %
                  (int(it), str(reason)), flush=True)
    return info


def stage2_nucleation_target_vf(cfg, vf_ref):
    """Small, target-aware volume drop for rare stage-2 nucleation fallback."""
    vf_ref = float(vf_ref)
    vf_stage = current_vf_target(cfg)
    if vf_stage is None:
        return max(0.0, float(cfg.get("hard_shift_factor", 0.99)) * vf_ref)
    gap = max(0.0, vf_ref - float(vf_stage))
    if gap <= 0.0:
        return max(0.0, float(cfg.get("hard_shift_factor", 0.99)) * vf_ref)
    drop_abs = max(0.0, float(cfg.get("stage2_nucleation_max_abs_drop", 0.004)))
    drop_frac = max(0.0, float(cfg.get("stage2_nucleation_max_gap_fraction", 0.25))) * gap
    drop = min(drop_abs, drop_frac, gap)
    return max(float(vf_stage), vf_ref - drop)


def postprocess_hard_shift_target_vf(cfg, vf_ref):
    """Target-aware uniform hard-shift drop for the post-processing phase."""
    vf_ref = float(vf_ref)
    vf_final = float(current_postprocess_vf_final_target(cfg))
    gap = max(0.0, vf_ref - vf_final)
    if gap <= 0.0:
        return vf_final
    drop_abs = max(0.0, float(cfg.get("postprocess_hard_shift_max_abs_drop", 0.020)))
    drop_frac = max(0.0, float(cfg.get("postprocess_hard_shift_gap_fraction", 0.45))) * gap
    drop_min = max(0.0, float(cfg.get("postprocess_hard_shift_min_abs_drop", 0.0)))
    drop = min(drop_abs, max(drop_min, drop_frac), gap)
    return max(vf_final, vf_ref - drop)


def postprocess_history_plateau(values, window, rel_tol, j_floor=1e-12):
    """Return (ok, rel_span) for a short post-processing objective history."""
    try:
        window = max(2, int(window))
    except Exception:
        window = 2
    hist = list(values or [])
    if len(hist) < window:
        return False, float("nan")
    recent = np.asarray(hist[-window:], dtype=float)
    finite = recent[np.isfinite(recent)]
    if len(finite) < window:
        return False, float("nan")
    span = float(np.max(finite) - np.min(finite))
    try:
        floor = abs(float(j_floor))
    except Exception:
        floor = 1e-12
    scale = max(float(np.max(np.abs(finite))), floor, 1e-30)
    rel_span = span / scale
    return bool(rel_span <= max(0.0, float(rel_tol))), float(rel_span)


def evaluate(mesh, W, lsf, materials, cfg, alpha=None):
    debug_progress = _debug_progress_enabled(cfg) and (MPI.rank(MPI.comm_world) == 0)
    # mark materials
    mark_materials_from_lsf(mesh, lsf, materials, threshold=cfg["threshold"])

    # homogenisation
    if debug_progress:
        print("[progress] evaluate: enter compute_homogenized_C")
    t_hom0 = time.time()
    Chom, sig_cache, Vol, E_expr = compute_homogenized_C(
        mesh, W, materials, E0=cfg["E0"], gamma_star=cfg["gamma_star"], nu=cfg["nu"],
        threshold=cfg["threshold"], solver_cfg=cfg["cell_solver"]
    )
    if debug_progress:
        print("[progress] evaluate: exit compute_homogenized_C (dt=%.2fs)" % (time.time() - t_hom0))
    chi = materials_to_chi(mesh, materials)
    vf = volume_fraction_from_chi(chi, mesh, vol_total=Vol)

    if alpha is None:
        alpha = 0.0
    eps_denom = float(cfg.get("eps_denom", 1e-12))
    ti_penalty_mode = cfg.get("ti_penalty_normalization_mode", None)
    ti_norm_cref = bool(cfg.get("ti_penalty_normalize_by_cref", True))
    ti_penalty_mode_eff = (
        str(ti_penalty_mode).strip().lower()
        if ti_penalty_mode is not None
        else ("cref" if ti_norm_cref else "none")
    )

    beta_a = beta_a_value(cfg)
    beta_b = beta_b_value(cfg)
    hb = hb_value(Chom)
    ha = ha_value(Chom)
    HH = H_value(Chom)
    R_ti = ti_residual_value(Chom)
    den_obj = denominator_value(Chom, eps_denom=eps_denom)
    cref = cref_value(Chom, eps_denom=eps_denom)
    hb_raw = hb
    hb_over_cref_raw = hb / cref
    rti_over_cref_raw = R_ti / cref
    rti_ratio_raw = R_ti * R_ti
    phi_TI = abs(rti_over_cref_raw)
    J_hb = uti_ratio_term(Chom, beta_a=beta_a, beta_b=beta_b, eps_denom=eps_denom)
    J_ti = ti_penalty_term(
        Chom, alpha=float(alpha), eps_denom=eps_denom,
        normalize_by_cref=ti_norm_cref, normalization_mode=ti_penalty_mode
    )
    # Hilbertian volume control is handled outside Phi(C); keep J_vol only as a diagnostic scale.
    lambda_v = _clamp_lambda_v_to_cfg(cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False)
    cfg["lambda_v"] = float(lambda_v)
    J_vol = lambda_v * float(vf)
    J_phi = float(J_hb) + float(J_ti)
    J = float(J_phi)
    merit_terms = volume_merit_terms(cfg, J=J, vf=vf, vf_target=current_vf_target(cfg))
    vf_target = current_vf_target(cfg)
    vf_constraint_target = float(merit_terms["vf_constraint_target"])
    vf_constraint_residual = float(merit_terms["vf_constraint_residual"])
    vf_constraint_violation = float(merit_terms["vf_constraint_violation"])
    J_compare_vf_weight = float(merit_terms["J_compare_vf_weight"])
    J_compare = float(merit_terms["J_compare"])
    J_merit = float(merit_terms["J_merit"])
    J_compare_mode = str(merit_terms["J_compare_mode"])
    rho_v = float(merit_terms["rho_v"])
    mu_v = float(merit_terms["mu_v"])
    rho_v_merit = float(merit_terms.get("rho_v_merit", rho_v))
    mu_v_merit = float(merit_terms.get("mu_v_merit", mu_v))

    dPhi = grad_uti_ratio_term(Chom, beta_a=beta_a, beta_b=beta_b, eps_denom=eps_denom) + grad_ti_penalty_term(
        Chom, alpha=float(alpha), eps_denom=eps_denom,
        normalize_by_cref=ti_norm_cref, normalization_mode=ti_penalty_mode
    )
    if debug_progress:
        print("[progress] evaluate: enter compute_DTJ")
    t_dtj0 = time.time()
    DTJ_CG1, DTJ_DG0 = compute_DTJ(
        mesh, sig_cache, chi, Vol, dPhi,
        E0=cfg["E0"], gamma_star=cfg["gamma_star"], nu=cfg["nu"]
    )
    if debug_progress:
        print("[progress] evaluate: exit compute_DTJ (dt=%.2fs)" % (time.time() - t_dtj0))

    Vls = lsf.function_space()
    chi_on_Vls = _project(chi, Vls)
    s_expr = 2.0 * chi_on_Vls - 1.0
    DTJ_on_Vls = DTJ_CG1 if DTJ_CG1.function_space().ufl_element() == Vls.ufl_element() else _project(DTJ_CG1, Vls)
    g_obj = _project(-s_expr * DTJ_on_Vls, Vls)
    g_vol = build_generalized_volume_gradient(Vls, scale=1.0 / max(float(Vol), 1e-30))

    return dict(
        J=J, J_compare=J_compare, J_merit=J_merit,
        J_compare_mode=J_compare_mode,
        J_compare_vf_weight=J_compare_vf_weight,
        J_phi=J_phi, J_R0=J_hb, J_hb=J_hb, J_ratio_hb=J_hb, J_ti=J_ti, J_vol=J_vol, vf=vf,
        hb=hb, ha=ha, H=HH, cref=cref, R_ti=R_ti, den_obj=den_obj, alpha=float(alpha), beta_a=float(beta_a), beta_b=float(beta_b), rho_v=rho_v, mu_v=mu_v,
        rho_v_merit=rho_v_merit, mu_v_merit=mu_v_merit,
        J_merit_fixed_stage=bool(merit_terms.get("J_merit_fixed_stage", False)),
        J_merit_stage_budget=float(merit_terms.get("J_merit_stage_budget", 0.0)),
        ti_penalty_mode=ti_penalty_mode_eff,
        hb_raw=hb_raw, hb_ratio_raw=hb_over_cref_raw, hb_over_cref_raw=hb_over_cref_raw,
        rti_ratio_raw=rti_ratio_raw, rti_over_cref_raw=rti_over_cref_raw, phi_TI=phi_TI,
        vf_target=vf_target,
        vf_constraint_target=vf_constraint_target,
        vf_constraint_residual=vf_constraint_residual,
        vf_constraint_violation=vf_constraint_violation,
        Chom=Chom, g=g_obj, g_obj=g_obj, g_vol=g_vol, DTJ=DTJ_on_Vls
    )

def evaluate_safe(mesh, W, lsf, materials, cfg, alpha=None, context="", fail_tag="[solver-fail]"):
    """Evaluate objective/gradient and return (res, solver_fail)."""
    debug_progress = _debug_progress_enabled(cfg) and (MPI.rank(MPI.comm_world) == 0)
    ctx = (" " + context) if context else ""
    t_eval0 = time.time()
    if debug_progress:
        print("[progress] enter evaluate%s" % ctx)
    try:
        res = evaluate(mesh, W, lsf, materials, cfg, alpha=alpha)
        if debug_progress:
            print("[progress] exit evaluate%s ok (dt=%.2fs)" % (ctx, time.time() - t_eval0))
        return res, False
    except RuntimeError as err:
        if MPI.rank(MPI.comm_world) == 0:
            if debug_progress:
                print("[progress] exit evaluate%s fail (dt=%.2fs)" % (ctx, time.time() - t_eval0))
            print("%s%s %s" % (str(fail_tag), ctx, str(err)))
        return None, True

def preview_vf_aug_lag_update(cfg, res_eval):
    violation_now = float(res_eval["vf_constraint_violation"])
    mu_old = float(cfg.get("_vf_aug_lag_mu", cfg.get("mu_v0", 0.0)))
    rho_old = float(cfg.get("_vf_aug_lag_rho", cfg.get("rho_v", 0.0)))
    prev_violation = cfg.get("_vf_aug_lag_prev_violation", None)
    adapt_eps = float(cfg.get("vf_al_adapt_eps", 1e-12))
    good_ratio = float(cfg.get("vf_al_adapt_good_ratio", 0.70))
    bad_ratio = float(cfg.get("vf_al_adapt_bad_ratio", 0.95))
    rho_grow = float(cfg.get("vf_al_rho_grow", 1.5))
    rho_shrink = float(cfg.get("vf_al_rho_shrink", 0.8))
    rho_min = float(cfg.get("vf_al_rho_min", 1e-3))
    rho_max = float(cfg.get("vf_al_rho_max", 10.0))
    ratio = None
    rho_new = rho_old
    adapt_tag = "init"
    if prev_violation is not None:
        ratio = violation_now / max(float(prev_violation), adapt_eps)
        if ratio > bad_ratio:
            rho_new = min(rho_old * rho_grow, rho_max)
            adapt_tag = "grow"
        elif ratio < good_ratio:
            rho_new = max(rho_old * rho_shrink, rho_min)
            adapt_tag = "shrink"
        else:
            adapt_tag = "keep"
    residual_now = float(res_eval["vf_constraint_residual"])
    mu_max = float(cfg.get("stage2_al_mu_max", cfg.get("stage2_lambda_abs_cap", 1.0)))
    mu_new = max(0.0, min(mu_max, mu_old + rho_new * residual_now))
    mu_released = False
    mu_release_tol = max(
        0.0,
        float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0)))
        * float(cfg.get("stage2_al_mu_overshoot_tol_factor", 1.0)),
    )
    if bool(cfg.get("stage2_al_mu_overshoot_release_enabled", True)) and (residual_now < -mu_release_tol):
        mu_decay = min(1.0, max(0.0, float(cfg.get("stage2_al_mu_overshoot_decay", 0.0))))
        mu_release_candidate = max(0.0, min(mu_max, mu_old * mu_decay))
        if mu_release_candidate < mu_new:
            mu_new = float(mu_release_candidate)
            mu_released = True
            adapt_tag = "%s+mu-release" % str(adapt_tag)
    return dict(
        violation_now=violation_now,
        mu_old=mu_old,
        rho_old=rho_old,
        prev_violation=prev_violation,
        ratio=ratio,
        rho_new=rho_new,
        mu_new=mu_new,
        mu_released=mu_released,
        adapt_tag=adapt_tag,
    )

def run():
    cfg = init_3d()
    outdir = "out_3D_UTI"
    _stdout, log_file = None, None

    if MPI.rank(MPI.comm_world) == 0:
        os.makedirs(outdir, exist_ok=True)
        log_path = os.path.join(outdir, "run.log")
        _stdout = sys.stdout
        log_file = open(log_path, 'w', encoding='utf-8')
        sys.stdout = _Tee(_stdout, log_file)

    eps_theta_rad = np.radians(cfg["eps_theta_deg"])
    if MPI.rank(MPI.comm_world) == 0:
        _cell = cfg["cell_solver"]
        print("[solver] cell periodic PDE: ksp=%s pc=%s hypre=%s rtol=%s atol=%s max_it=%s near_nullspace=%s" % (
            str(_cell["ksp_type"]),
            str(_cell["pc_type"]),
            str(_cell["pc_hypre_type"]),
            str(_cell["ksp_rtol"]),
            str(_cell["ksp_atol"]),
            str(_cell["ksp_max_it"]),
            str(bool(_cell["set_near_nullspace"])),
        ))

    # mesh + spaces (track Nx,Ny,Nz for refinement N += refine_step)
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]
    symmetry_mode = str(cfg.get("symmetry_mode", "")).strip().lower()
    if symmetry_mode == "":
        use_legacy_cubic = bool(
            cfg.get("use_strict_wedge_parameterization",
                    cfg.get("use_minimal_wedge_parameterization",
                            cfg.get("use_octant_parameterization", cfg.get("use_centerdiag_symmetry", False))))
        )
        symmetry_mode = "cubic" if use_legacy_cubic else "none"
    if symmetry_mode not in ("none", "cubic", "tetragonal_z", "tetragonal_z_rot4"):
        raise ValueError("symmetry_mode must be one of: none, cubic, tetragonal_z, tetragonal_z_rot4")
    if (symmetry_mode == "cubic") and (not (int(Nx) == int(Ny) == int(Nz))):
        raise ValueError("cubic symmetry requires Nx == Ny == Nz on the structured cube grid.")
    if (symmetry_mode in ("tetragonal_z", "tetragonal_z_rot4")) and (int(Nx) != int(Ny)):
        raise ValueError("%s symmetry requires Nx == Ny so the x-y plane supports fourfold symmetry." % symmetry_mode)
    mesh = UnitCubeMesh(Nx, Ny, Nz)
    W, Vls, VtDG, VsDG, VDG0 = build_spaces(mesh, deg=cfg["deg"], tol=1e-10)
    # Preassembled L2 mass matrix for angle/norm evaluation (stable in MPI, no repeated assemble in line-search)
    l2_mass = build_l2_mass_matrix(Vls)

    # fields
    lsf = _project(cfg["lsf_expr"], Vls)
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    materials.set_all(0)
    use_symmetry_parameterization = (symmetry_mode != "none")
    symmetry_map = (
        build_symmetry_index_map(Vls, Nx, Ny, Nz, symmetry_mode) if use_symmetry_parameterization else None
    )
    nucleation_map = symmetry_map if use_symmetry_parameterization else build_full_dof_index_map(Vls, Nx, Ny, Nz)
    symmetry_assert_tol = float(cfg.get("symmetry_assert_tol", cfg.get("wedge_assert_tol", 1e-12)))
    symmetry_diag_stride = int(cfg.get("symmetry_diag_stride", cfg.get("wedge_diag_stride", 1)))
    dx = Measure("dx", domain=mesh)
    domain_volume = float(assemble(Constant(1.0) * dx))

    def prepare_trial_lsf(lsf_candidate, apply_hard_vf=False):
        """Trial chain: symmetry expansion -> renormalisation."""
        lsf_prepared = Function(Vls)
        copy_function_values(lsf_prepared, lsf_candidate)
        c_hard = 0.0
        vf_hard = None
        if use_symmetry_parameterization and (symmetry_map is not None):
            lsf_prepared = expand_symmetry_to_full(lsf_prepared, symmetry_map)
        renormalize_lsf_inplace(lsf_prepared, dx)
        return lsf_prepared, c_hard, vf_hard

    # initial symmetry expansion + renormalisation
    lsf, init_c_shift, init_vf_shift = prepare_trial_lsf(lsf, apply_hard_vf=False)

    alpha_now = alpha_value(cfg, it=0, vf=None)
    cfg["_current_it"] = 0
    res, solver_fail_init = evaluate_safe(mesh, W, lsf, materials, cfg, alpha=alpha_now, context="[init]")
    if solver_fail_init:
        raise RuntimeError("Initial evaluate failed before entering iterations; please check cell solver settings.")
    cfg["_current_vf"] = float(res["vf"])
    J_old = res["J"]
    J_compare_old = res["J_compare"]
    J_history = [J_compare_old]
    initialise_stage_controller(cfg, res["vf"])
    initialise_volume_continuation_state(cfg, res)
    freeze_stage_merit_from_controller(cfg, res, reason="initial-stage")
    initialize_lambda_v_seed(cfg, res, psi=lsf, M=l2_mass)
    refresh_volume_merit_inplace(cfg, res)
    J_compare_old = res["J_compare"]
    J_history = [J_compare_old]
    sym_max_init = None
    sym_mean_init = None
    if use_symmetry_parameterization and (symmetry_map is not None):
        # Collective: all ranks must participate.
        sym_max_init, sym_mean_init = symmetry_residual_stats(lsf, symmetry_map)

    if MPI.rank(MPI.comm_world) == 0:
        print("[init] beta_a=%.3e beta_b=%.3e alpha=%.3e alpha_max=%.3e ti_mode=%s lambda_v=%.3e rho_v=%.2f mu_v=%.2f vf=%.4f vf_target=%.4f mesh=%dx%dx%d symmetry=%s" %
              (float(cfg.get("beta_a", cfg.get("beta", 1.0))), float(cfg.get("beta_b", 0.0)), res["alpha"], float(cfg.get("alpha", res["alpha"])), str(res.get("ti_penalty_mode", "n/a")), float(cfg.get("lambda_v", 0.0)), float(cfg.get("rho_v", 0.0)), res["mu_v"], res["vf"], res["vf_constraint_target"], Nx, Ny, Nz, symmetry_mode))
        print("[init] J_mech=%.6e = j_hb(%.6e) + j_ti(%.6e) ; J_merit=%.6e ; diag_j_vol(%.6e)" %
              (J_old, res["J_hb"], res["J_ti"], res["J_compare"], res["J_vol"]))
        print("[init-metrics] ha=%.3e  H=%.3e  hb=%.3e  R_TI=%.3e  cref=%.3e  phi_TI=%.3e  vf=%.4f  vf_stage=%.4f  vf-vf_stage=%.6f  %s" %
              (float(res["ha"]), float(res["H"]), float(res["hb"]), float(res["R_ti"]), float(res["cref"]), float(res["phi_TI"]), float(res["vf"]), float(current_vf_target(cfg)), float(res["vf_constraint_residual"]), _format_stage_dv_state(cfg)))
        print("[init-Chom] Voigt6x6=\n%s" %
              (np.array2string(res["Chom"], precision=4, suppress_small=True)))
        if symmetry_mode == "cubic":
            cubic_init = cubic_residual_metrics(res["Chom"])
            print("[init-cubic] rN=%.3e rS=%.3e rF=%.3e" %
                  (cubic_init["rN"], cubic_init["rS"], cubic_init["rF"]))
        if use_symmetry_parameterization and (symmetry_map is not None):
            sym_max, sym_mean = sym_max_init, sym_mean_init
            print("[init-symmetry] mode=%s reps=%d residual_max=%.3e residual_mean=%.3e (tol=%.3e)" %
                  (symmetry_mode, int(symmetry_map["n_reps"]), sym_max, sym_mean, symmetry_assert_tol))
            if sym_max > symmetry_assert_tol:
                print("[symmetry-warning] init residual %.3e exceeds tol %.3e" % (sym_max, symmetry_assert_tol))
    if _debug_progress_enabled(cfg) and (MPI.rank(MPI.comm_world) == 0):
        print("[progress] before write_xdmf(it=0)")
    write_xdmf(outdir, mesh, 0, materials, lsf)
    if _debug_progress_enabled(cfg) and (MPI.rank(MPI.comm_world) == 0):
        print("[progress] after write_xdmf(it=0)")

    kappa = cfg["kappa0"]
    kappa_min = float(cfg.get("kappa_min", 1e-4))
    j_relax = 0.0  # strict descent: no acceptance margin (when stage relax inactive)
    stage_relax_tau = float(cfg.get("stage_relax_tau", 0.0))
    stage_relax_steps_total = int(cfg.get("stage_relax_steps_total", 0))
    stage_relax_steps_left = 0
    xdmf_stride = int(cfg.get("xdmf_stride", 10))  # write XDMF every N steps
    refine_cap_blocked = False   # once cap is reached, stop re-entering refinement branch
    # Optional debug hook: force entering fail-recover branch at chosen iterations.
    force_fail_recover_enabled = bool(cfg.get("debug_force_fail_recover", False))
    force_fail_recover_iters = set()
    _force_iters_cfg = cfg.get("debug_force_fail_recover_iters", ())
    if _force_iters_cfg is None:
        _force_iters_cfg = ()
    if isinstance(_force_iters_cfg, (int, float)):
        _force_iters_cfg = (_force_iters_cfg,)
    for _v in _force_iters_cfg:
        try:
            force_fail_recover_iters.add(int(_v))
        except Exception:
            continue
    force_fail_recover_all = force_fail_recover_enabled and (len(force_fail_recover_iters) == 0)
    # Count accepted line-search steps that apply kappa increase (first N use kappa_increase_factor_early).
    kappa_increase_apply_count = 0
    cfg["_lambda_v_adapt_max_abs_base"] = cfg.get("lambda_v_adapt_max_abs", None)
    converged = False
    stage_target_just_changed = False
    vf_history = [float(res["vf"])]
    lambda_plateau_vf_history = [float(res["vf"])]
    plateau_kappa_min_hits = 0
    plateau_last_vf = None
    hard_shift_cooldown_left = 0
    refine_post_cooldown_left = 0
    hard_shift_recovery_left = 0
    final_filter_post_cooldown_left = 0
    final_filter_post_pending_terminate = False
    final_filter_post_action = "terminate"
    lambda_plateau_cooldown_left = int(cfg.get("_lambda_v_plateau_accept_cooldown_left", 0))
    cfg.setdefault("_hard_shift_force_uniform_early", False)
    cfg.setdefault("_hard_shift_force_uniform_early_announced", False)
    cfg.setdefault("_hard_shift_switch_reached_once", False)
    cfg.setdefault("_stage2_final_convergence_active", False)
    cfg.setdefault("_stage2_final_convergence_announced", False)
    cfg.setdefault("_stage2_final_filter_done", False)
    cfg.setdefault("_stage2_final_settle_confirmed", False)
    cfg.setdefault("_stage2_final_settle_exhausted", False)
    cfg.setdefault("_stage2_loss_envelope_stop_active", False)
    cfg.setdefault("_stage2_loss_envelope_convergence_vf", None)
    cfg.setdefault("_stage2_loss_envelope_rejected_target", None)
    cfg.setdefault("_stage2_loss_envelope_stop_J", None)
    _stage2_settle_reset(cfg)
    cfg.setdefault("_stage2_optimization_converged", False)
    cfg.setdefault("_postprocess_active", False)
    cfg.setdefault("_postprocess_start_it", None)
    cfg.setdefault("_last_postprocess_shift_it", None)
    cfg.setdefault("_postprocess_started_announced", False)
    cfg.setdefault("_vf_stage_frozen_in_hard_shift_announced", False)
    cfg.setdefault("_vf_stage_rebound_active", False)
    cfg.setdefault("_nucleation_rebound_hits", 0)
    cfg.setdefault("_last_nucleation_it", None)
    cfg.setdefault("_post_nucleation_lambda_zero_active", False)
    cfg.setdefault("_post_nucleation_lambda_zero_start_it", None)
    cfg.setdefault("_last_nucleation_post_vf", None)
    cfg.setdefault("_last_uniform_hard_shift_pre_vf", None)
    cfg.setdefault("_last_uniform_hard_shift_post_vf", None)
    cfg.setdefault("_last_uniform_hard_shift_it", None)
    cfg.setdefault("_uniform_hard_shift_aggressive_active", False)
    cfg.setdefault("_lambda_v_early_freeze_active", False)
    cfg.setdefault("_uniform_hard_shift_exit_hits", 0)
    cfg.setdefault("_uniform_hard_shift_exit_override", False)
    vf_prev_for_milestone = float(res["vf"])
    _milestone_targets_raw = cfg.get(
        "vf_milestone_filter_targets",
        (cfg.get("hard_shift_switch_to_shift_vf", 0.20), cfg.get("vf_final_target", 0.10))
    )
    if isinstance(_milestone_targets_raw, (int, float)):
        _milestone_targets_raw = (_milestone_targets_raw,)
    milestone_filter_targets = []
    for _v in _milestone_targets_raw:
        try:
            _vf_target_val = float(_v)
        except Exception:
            continue
        if np.isfinite(_vf_target_val) and (_vf_target_val >= 0.0):
            milestone_filter_targets.append(_vf_target_val)
    milestone_filter_targets = sorted(set(milestone_filter_targets), reverse=True)
    milestone_filter_done = dict((v, False) for v in milestone_filter_targets)
    milestone_filter_radius_factor = max(0.0, float(cfg.get("vf_milestone_filter_radius_factor", 0.5)))
    milestone_filter_vf_rel_change_max = max(
        0.0, float(cfg.get("milestone_filter_vf_rel_change_max",
                           cfg.get("final_filter_vf_rel_change_max", 0.10)))
    )
    filter_radius_search_max_iter = max(1, int(cfg.get("final_filter_radius_search_max_iter", 10)))
    fail_recover_vf_rel_start = max(0.0, float(cfg.get("fail_recover_filter_vf_rel_change_start", 0.10)))
    fail_recover_vf_rel_step = max(1e-12, float(cfg.get("fail_recover_filter_vf_rel_change_step", 0.05)))
    fail_recover_vf_rel_max = max(fail_recover_vf_rel_start, float(cfg.get("fail_recover_filter_vf_rel_change_max", 1.0)))
    milestone_filter_tol = 1e-15

    def adaptive_helmholtz_filter_with_vf_guard(
        lsf_input, vf_before, radius_cap, rel_limit, search_max_iter, context_prefix,
        alpha_eval, apply_hard_vf=False, base_res=None, use_plain_cg_helmholtz=False,
        eval_cfg=None, fail_tag="[solver-fail]"
    ):
        """
        Find the largest feasible Helmholtz radius in [0, radius_cap] such that:
          1) evaluate_safe succeeds
          2) |vf_after-vf_before|/max(|vf_before|,1e-12) <= rel_limit (if rel_limit finite)
        """
        info = {
            "success": False,
            "mode": "none",
            "radius_star": 0.0,
            "radius_cap": float(max(0.0, radius_cap)),
            "rel_limit": float(rel_limit) if np.isfinite(rel_limit) else float("inf"),
            "rel_change": 0.0,
            "last_rel_change": float("nan"),
            "solver_fail_count": 0,
            "vf_before": float(vf_before),
            "vf_after": float(vf_before),
        }
        r_cap = float(max(0.0, radius_cap))
        vf_scale = max(abs(float(vf_before)), 1e-12)
        rel_limit_eff = float(rel_limit)
        rel_limit_is_finite = bool(np.isfinite(rel_limit_eff))
        n_search = max(1, int(search_max_iter))

        best_lsf = None
        best_res = None
        best_r = 0.0
        best_rel = float("inf")

        # If caller already has a valid state at r=0, use it as baseline feasible anchor.
        if base_res is not None:
            best_lsf = Function(Vls)
            copy_function_values(best_lsf, lsf_input)
            best_res = base_res
            best_r = 0.0
            best_rel = 0.0
            info["mode"] = "baseline-r0"

        def eval_radius(r_try):
            lsf_try = helmholtz_filter_lsf(
                lsf_input, mesh, float(r_try),
                use_plain_cg_space=bool(use_plain_cg_helmholtz)
            )
            lsf_try, _, _ = prepare_trial_lsf(lsf_try, apply_hard_vf=apply_hard_vf)
            res_try, fail_try = evaluate_safe(
                mesh, W, lsf_try, materials, (cfg if eval_cfg is None else eval_cfg), alpha=alpha_eval,
                context="%s r=%.3e" % (str(context_prefix), float(r_try)),
                fail_tag=fail_tag
            )
            if fail_try:
                info["solver_fail_count"] += 1
                return None, None, float("nan"), False
            vf_try = float(res_try.get("vf", float("nan")))
            rel_try = abs(vf_try - float(vf_before)) / vf_scale
            info["last_rel_change"] = float(rel_try)
            ok_rel = (not rel_limit_is_finite) or (rel_try <= rel_limit_eff + 1e-15)
            return lsf_try, res_try, float(rel_try), bool(ok_rel)

        if r_cap <= 0.0:
            if best_res is None:
                lsf0, res0, rel0, ok0 = eval_radius(0.0)
                if ok0:
                    best_lsf, best_res, best_r, best_rel = lsf0, res0, 0.0, rel0
                    info["mode"] = "r0-eval"
            if best_res is not None:
                info["success"] = True
                info["radius_star"] = float(best_r)
                info["rel_change"] = float(best_rel)
                info["vf_after"] = float(best_res.get("vf", float(vf_before)))
            return best_lsf, best_res, info

        # 1) Probe upper cap first.
        lsf_cap, res_cap, rel_cap, ok_cap = eval_radius(r_cap)
        if ok_cap:
            best_lsf, best_res, best_r, best_rel = lsf_cap, res_cap, r_cap, rel_cap
            info["success"] = True
            info["mode"] = "cap-feasible"
            info["radius_star"] = float(best_r)
            info["rel_change"] = float(best_rel)
            info["vf_after"] = float(best_res.get("vf", float(vf_before)))
            return best_lsf, best_res, info

        # 2) Find at least one feasible point by halving the radius from cap.
        r_feas = float(best_r) if (best_res is not None) else None
        r_infeas = float(r_cap)
        r_probe = 0.5 * float(r_cap)
        n_halve = max(4, n_search + 2)
        for _ in range(n_halve):
            if r_probe <= 1e-16:
                break
            lsf_probe, res_probe, rel_probe, ok_probe = eval_radius(r_probe)
            if ok_probe:
                r_feas = float(r_probe)
                best_lsf, best_res, best_r, best_rel = lsf_probe, res_probe, float(r_probe), float(rel_probe)
                break
            r_infeas = float(r_probe)
            r_probe *= 0.5

        # 3) If feasible anchor exists, expand to the largest feasible radius by bisection.
        if r_feas is not None:
            info["mode"] = "bisection"
            r_lo = float(r_feas)
            r_hi = float(r_infeas)
            for _ in range(n_search):
                if (r_hi - r_lo) <= 1e-16:
                    break
                r_mid = 0.5 * (r_lo + r_hi)
                lsf_mid, res_mid, rel_mid, ok_mid = eval_radius(r_mid)
                if ok_mid:
                    r_lo = float(r_mid)
                    best_lsf, best_res, best_r, best_rel = lsf_mid, res_mid, float(r_mid), float(rel_mid)
                else:
                    r_hi = float(r_mid)

        if best_res is not None:
            info["success"] = True
            info["radius_star"] = float(best_r)
            info["rel_change"] = float(best_rel)
            info["vf_after"] = float(best_res.get("vf", float(vf_before)))
        return best_lsf, best_res, info

    def pre_jacobi_filter_active():
        if not bool(cfg.get("pre_jacobi_filter_enabled", False)):
            return False
        if not bool(cfg.get("use_helmholtz_filter", False)):
            return False
        if str(cfg.get("pc_type", "")).lower() != "gamg":
            return False
        return True

    def evaluate_trial_with_pre_jacobi_filter(lsf_input, context, alpha_eval, vf_ref=None, apply_hard_vf=False):
        if not pre_jacobi_filter_active():
            res_eval, fail_eval = evaluate_safe(
                mesh, W, lsf_input, materials, cfg, alpha=alpha_eval, context=context
            )
            return lsf_input, res_eval, fail_eval

        primary_cfg = dict(cfg)
        primary_cfg["ksp_type"] = "gmres"
        primary_cfg["pc_type"] = "gamg"
        primary_cfg["enable_ksp_fallback"] = False
        primary_cfg["fallback_ksp_types"] = ()
        primary_cfg["fallback_pc_types"] = ()

        res_primary, fail_primary = evaluate_safe(
            mesh, W, lsf_input, materials, primary_cfg, alpha=alpha_eval,
            context="%s primary-gamg" % str(context),
            fail_tag="[pde-primary-fail]"
        )
        if not fail_primary:
            return lsf_input, res_primary, False

        try:
            vf_before_filter = _vf_from_lsf_state(
                lsf_input, float(cfg["threshold"]), dx, domain_volume
            )
        except Exception:
            vf_before_filter = float(vf_ref) if vf_ref is not None else float(cfg.get("_current_vf", 0.0))

        radius_raw = cfg.get("pre_jacobi_filter_radius_factors", (0.25, 0.50))
        if isinstance(radius_raw, (int, float)):
            radius_factors = [float(radius_raw)]
        else:
            radius_factors = [float(x) for x in radius_raw]
        vf_rel_limit = max(0.0, float(cfg.get("pre_jacobi_filter_vf_rel_change_max", 0.002)))
        search_iters = max(1, int(cfg.get(
            "pre_jacobi_filter_search_max_iter",
            cfg.get("final_filter_radius_search_max_iter", 10),
        )))

        for i_filter, radius_factor in enumerate(radius_factors):
            radius_cap = max(0.0, float(radius_factor)) * float(mesh.hmin())
            lsf_filtered, res_filtered, filter_info = adaptive_helmholtz_filter_with_vf_guard(
                lsf_input, vf_before_filter, radius_cap, vf_rel_limit, search_iters,
                context_prefix="%s pre-jacobi-filter #%d rel=%.4f" % (
                    str(context), int(i_filter + 1), float(vf_rel_limit)
                ),
                alpha_eval=alpha_eval,
                apply_hard_vf=apply_hard_vf,
                base_res=None,
                eval_cfg=primary_cfg,
                fail_tag="[pde-filter-primary-fail]",
            )
            if bool(filter_info.get("success", False)):
                if MPI.rank(MPI.comm_world) == 0:
                    print("[pde-filter-rescue] %s: primary gmres/gamg recovered after Helmholtz "
                          "(try=%d, r*=%.3e, cap=%.3e, vf_rel_limit=%.4f, vf_rel_change=%.3e); skip jacobi" %
                          (str(context), int(i_filter + 1),
                           float(filter_info.get("radius_star", 0.0)),
                           float(filter_info.get("radius_cap", radius_cap)),
                           float(vf_rel_limit),
                           float(filter_info.get("rel_change", float("nan")))))
                return lsf_filtered, res_filtered, False

        jacobi_cfg = dict(cfg)
        jacobi_cfg["ksp_type"] = "gmres"
        jacobi_cfg["pc_type"] = "jacobi"
        jacobi_cfg["enable_ksp_fallback"] = False
        jacobi_cfg["fallback_ksp_types"] = ()
        jacobi_cfg["fallback_pc_types"] = ()
        if MPI.rank(MPI.comm_world) == 0:
            print("[pde-filter-rescue] %s: Helmholtz did not recover primary gmres/gamg; "
                  "try direct gmres/jacobi fallback" % str(context))
        res_jacobi, fail_jacobi = evaluate_safe(
            mesh, W, lsf_input, materials, jacobi_cfg, alpha=alpha_eval,
            context="%s jacobi-fallback" % str(context)
        )
        return lsf_input, res_jacobi, fail_jacobi

    for it in range(1, cfg["it_max"]+1):
        cfg["_current_it"] = int(it)
        debug_progress_iter = _debug_progress_enabled(cfg) and (MPI.rank(MPI.comm_world) == 0)
        if debug_progress_iter:
            print("[progress] begin iteration %03d" % int(it))
        lambda_cap_after_start = int(cfg.get("lambda_v_adapt_max_abs_after_iter_start", 10**12))
        lambda_cap_after_raw = cfg.get("lambda_v_adapt_max_abs_after_iter", None)
        if (lambda_cap_after_raw is not None) and (int(it) > lambda_cap_after_start):
            cfg["lambda_v_adapt_max_abs"] = float(lambda_cap_after_raw)
        else:
            cfg["lambda_v_adapt_max_abs"] = cfg.get("_lambda_v_adapt_max_abs_base", cfg.get("lambda_v_adapt_max_abs", None))
        t0 = time.time()
        alpha_desired = alpha_value(cfg, it=it, vf=float(res.get("vf", cfg.get("_current_vf", float("nan")))))
        alpha_current = float(res.get("alpha", alpha_desired))
        if bool(cfg.get("alpha_ti_continuation_enabled", False)):
            alpha_abs_tol = max(0.0, float(cfg.get("alpha_ti_refresh_abs_tol", 0.0)))
            alpha_rel_tol = max(0.0, float(cfg.get("alpha_ti_refresh_rel_tol", 0.0)))
            alpha_scale = max(abs(alpha_current), abs(alpha_desired), 1e-12)
            alpha_tol = max(alpha_abs_tol, alpha_rel_tol * alpha_scale)
            if abs(alpha_desired - alpha_current) > alpha_tol:
                alpha_before_refresh = float(alpha_current)
                res_alpha, solver_fail_alpha = evaluate_safe(
                    mesh, W, lsf, materials, cfg,
                    alpha=alpha_desired,
                    context="[it %03d alpha-refresh]" % int(it)
                )
                if solver_fail_alpha:
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[alpha-ti-refresh] it=%03d: refresh failed for alpha %.6e -> %.6e; keep previous alpha %.6e" %
                              (int(it), float(alpha_current), float(alpha_desired), float(alpha_current)))
                else:
                    res = res_alpha
                    cfg["_current_vf"] = float(res["vf"])
                    refresh_volume_merit_inplace(cfg, res)
                    J_old = res["J"]
                    J_compare_old = res["J_compare"]
                    J_history = [J_compare_old]
                    alpha_current = float(res.get("alpha", alpha_desired))
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[alpha-ti-refresh] it=%03d: alpha %.6e -> %.6e at vf=%.6f; reset comparison history" %
                              (int(it), float(alpha_before_refresh), float(alpha_current), float(res.get("vf", float("nan")))))
        alpha_now = float(alpha_current)
        solver_fail = False
        stage2_final_filter_just_done = False
        vf_hard_target_try = None
        skip_j_compare_this_iter = bool(stage_target_just_changed)
        stage_target_just_changed = False
        use_stage_relaxed_accept = (stage_relax_steps_left > 0) and (not skip_j_compare_this_iter)
        hard_cooldown_active = (hard_shift_cooldown_left > 0)
        final_filter_cooldown_active = (final_filter_post_cooldown_left > 0)
        refine_cooldown_active = (
            (not hard_cooldown_active)
            and (not final_filter_cooldown_active)
            and (refine_post_cooldown_left > 0)
        )
        cooldown_active = hard_cooldown_active or final_filter_cooldown_active or refine_cooldown_active
        recovery_active = (not cooldown_active) and (hard_shift_recovery_left > 0)
        postprocess_active = bool(cfg.get("_postprocess_active", False))
        stage2_stop_for_convergence = bool(cfg.get("stage2_stop_for_convergence_enabled", True))
        stage2_convergence_vf = float(stage2_current_convergence_vf(cfg))
        stage2_convergence_tol = stage_success_tolerance(
            cfg, dv=float(cfg.get("vf_stage_dv_min", 0.002))
        )
        postprocess_target_reached_early = False
        if (
            postprocess_active
            and bool(cfg.get("postprocess_terminate_on_target", True))
            and bool(cfg.get("postprocess_terminate_immediate_on_target", True))
        ):
            _pp_hit, _pp_vf, _pp_target, _pp_tol = postprocess_target_status(cfg, res)
            if _pp_hit:
                postprocess_target_reached_early = True
                cfg["_postprocess_target_stop_reason"] = "target-reached-pre-line-search"
                hard_shift_cooldown_left = 0
                hard_shift_recovery_left = 0
                lambda_plateau_cooldown_left = 0
                cfg["_lambda_v_plateau_accept_cooldown_left"] = 0
                hard_cooldown_active = False
                cooldown_active = bool(final_filter_cooldown_active or refine_cooldown_active)
                recovery_active = False
                lambda_plateau_cooldown_active = False
                final_filter_post_action = "terminate"
                final_filter_post_pending_terminate = True
                if MPI.rank(MPI.comm_world) == 0:
                    print("[postprocess-target] it=%03d: vf=%.6f <= target %.6f + tol %.3e; skip further line-search and terminate" %
                          (int(it), float(_pp_vf), float(_pp_target), float(_pp_tol)),
                          flush=True)
        if (
            stage2_stop_for_convergence
            and (not postprocess_active)
            and (float(res.get("vf", 1.0)) <= stage2_convergence_vf + stage2_convergence_tol)
        ):
            if bool(cfg.get("stage2_settle_enabled", True)) and (not bool(cfg.get("_stage2_final_settle_confirmed", False))):
                cfg["_stage2_final_pending"] = True
                cfg["_vf_stage_target"] = float(stage2_convergence_vf)
                freeze_stage_merit_from_controller(cfg, res, reason="stage2-final-pending")
                refresh_volume_merit_inplace(cfg, res)
                if (
                    (not bool(cfg.get("_stage2_final_convergence_announced", False)))
                    and (MPI.rank(MPI.comm_world) == 0)
                ):
                    print("[stage2-final] it=%03d: vf=%.6f reached convergence target %.6f (tol %.3e); enter stage2 settle gate before filtering/post-processing" %
                          (int(it), float(res.get("vf", 0.0)), float(stage2_convergence_vf), float(stage2_convergence_tol)))
                    cfg["_stage2_final_convergence_announced"] = True
            else:
                cfg["_stage2_final_convergence_active"] = True
                if (
                    (not bool(cfg.get("_stage2_final_convergence_announced", False)))
                    and (MPI.rank(MPI.comm_world) == 0)
                ):
                    print("[stage2-final] it=%03d: vf=%.6f reached convergence target %.6f (tol %.3e); freeze target and judge optimization convergence before post-processing" %
                          (int(it), float(res.get("vf", 0.0)), float(stage2_convergence_vf), float(stage2_convergence_tol)))
                    cfg["_stage2_final_convergence_announced"] = True
        stage2_final_convergence_active = (
            bool(cfg.get("_stage2_final_convergence_active", False))
            and (not postprocess_active)
        )
        # Low-vf hard-shift-only mode is now a post-processing mode.  Reaching
        # vf ~= stage2_convergence_vf first opens the stage-2 convergence gate; uniform hard
        # shifts are enabled only after that gate has accepted the optimized
        # stage-2 state.
        hard_shift_only_vf = float(cfg.get("hard_shift_switch_to_shift_vf", 0.20))
        allow_hard_shift_only_by_vf = (not stage2_stop_for_convergence) or postprocess_active
        if allow_hard_shift_only_by_vf and (float(res.get("vf", 1.0)) <= hard_shift_only_vf + 1e-15):
            cfg["_hard_shift_switch_reached_once"] = True
        hard_shift_force_uniform_early = bool(cfg.get("_hard_shift_force_uniform_early", False))
        hard_shift_switch_reached_once = bool(cfg.get("_hard_shift_switch_reached_once", False))
        hard_shift_only_reached = (
            allow_hard_shift_only_by_vf
            and (hard_shift_switch_reached_once or hard_shift_force_uniform_early)
        )
        hard_shift_only_active = bool(hard_shift_only_reached)
        if (
            hard_shift_force_uniform_early
            and (not bool(cfg.get("_hard_shift_force_uniform_early_announced", False)))
            and (MPI.rank(MPI.comm_world) == 0)
        ):
            print("[vf-hard-shift-mode] early uniform-shift path activated by nucleation-rebound rule")
            cfg["_hard_shift_force_uniform_early_announced"] = True
        lambda_plateau_master_enabled = bool(cfg.get("lambda_v_plateau_boost_enabled", True))
        lambda_plateau_cooldown_enabled = (
            lambda_plateau_master_enabled
            and bool(cfg.get("lambda_v_plateau_accept_cooldown_enabled", True))
        )
        if not lambda_plateau_cooldown_enabled:
            lambda_plateau_cooldown_left = 0
            cfg["_lambda_v_plateau_accept_cooldown_left"] = 0
        lambda_plateau_cooldown_active = (
            lambda_plateau_cooldown_enabled
            and (not cooldown_active)
            and (not recovery_active)
            and (lambda_plateau_cooldown_left > 0)
        )
        cooldown_kappa = max(
            float(kappa_min),
            float(cfg.get("hard_shift_cooldown_kappa_factor", 0.1)) * float(cfg["kappa0"])
        )
        final_filter_cooldown_kappa = max(
            float(kappa_min),
            float(cfg.get("final_filter_post_cooldown_kappa_factor", 0.1)) * float(cfg["kappa0"])
        )
        refine_cooldown_kappa = max(
            float(kappa_min),
            float(cfg.get("refine_cooldown_kappa_factor", 0.1)) * float(cfg["kappa0"])
        )
        recovery_steps_total = max(1, int(cfg.get("hard_shift_recovery_steps", 3)))
        recovery_strength = 1.0
        recovery_step_idx = 0
        recovery_kappa_target = float(cfg["kappa0"])
        if recovery_active:
            recovery_step_idx = recovery_steps_total - int(hard_shift_recovery_left) + 1
            recovery_strength = min(1.0, max(0.0, float(recovery_step_idx) / float(recovery_steps_total)))
            # During recovery, ramp kappa back to kappa0 in a fixed schedule.
            recovery_kappa_target = cooldown_kappa + recovery_strength * (float(cfg["kappa0"]) - cooldown_kappa)

        lambda_zero_steps = max(0, int(cfg.get("post_nucleation_lambda_zero_steps", 0)))
        lambda_zero_active = bool(cfg.get("_post_nucleation_lambda_zero_active", False))
        lambda_zero_start_raw = cfg.get("_post_nucleation_lambda_zero_start_it", None)
        if lambda_zero_active and lambda_zero_steps > 0 and lambda_zero_start_raw is not None:
            try:
                lambda_zero_age = int(it) - int(lambda_zero_start_raw)
            except Exception:
                lambda_zero_age = lambda_zero_steps + 1
            if 1 <= lambda_zero_age <= lambda_zero_steps:
                if abs(float(cfg.get("lambda_v", 0.0))) > 1e-16:
                    cfg["lambda_v"] = 0.0
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[lambda_v-zero-freeze] it=%03d: force lambda_v=0 during post-nucleation relaxation window (age=%d/%d)" %
                              (int(it), int(lambda_zero_age), int(lambda_zero_steps)))
                else:
                    cfg["lambda_v"] = 0.0
            elif lambda_zero_age > lambda_zero_steps:
                cfg["_post_nucleation_lambda_zero_active"] = False
                _, lambda_zero_info = update_lambda_v_from_stage_state(
                    cfg, res, psi=lsf, M=l2_mass, it=it, reason="post-nucleation-zero-complete"
                )
                if MPI.rank(MPI.comm_world) == 0:
                    print("[lambda_v-zero-freeze] it=%03d: completed %d-step zero-lambda relaxation; lambda_v=%.6e" %
                          (int(it), int(lambda_zero_steps), float(cfg.get("lambda_v", 0.0))))
        elif lambda_zero_active:
            cfg["_post_nucleation_lambda_zero_active"] = False

        if (
            (not cooldown_active)
            and (not final_filter_cooldown_active)
            and (not hard_shift_only_active)
            and stage2_volume_continuation_active(cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg))
        ):
            update_volume_continuation_state(cfg, res, it=it, reason="pre-direction", update_mu=False, adapt_rho=False)
            update_lambda_v_from_stage_state(
                cfg, res, psi=lsf, M=l2_mass, it=it, reason="stage2-volume-continuation"
            )
            refresh_volume_merit_inplace(cfg, res, hard_shift_only=hard_shift_only_active)
            J_compare_old = res["J_compare"]

        # Continuous search direction:
        #   - plain objective direction only for final-filter cooldown / hard-shift-only mode
        #   - keep the Hilbertian direction (with the updated lambda_v) during
        #     hard-nucleation cooldown and recovery
        g_obj = res.get("g_obj", res["g"])
        g_vol = res.get("g_vol", None)
        if g_vol is None:
            g_vol = build_generalized_volume_gradient(Vls)
        g_raw = Function(Vls)
        if debug_progress_iter:
            print("[progress] iteration %03d: before copy g_raw" % int(it))
        copy_function_values(g_raw, g_obj)
        if debug_progress_iter:
            print("[progress] iteration %03d: after copy g_raw" % int(it))
        if final_filter_cooldown_active or hard_shift_only_reached:
            direction_info = {
                "mode": "plain-objective",
                "lambda_v_active": 0.0,
                "radial_coeff": 0.0,
                "mech_tangent_coeff": 0.0,
                "vol_tangent_coeff": 0.0,
                "mech_vol_coeff": 0.0,
                "vol_tangent_norm_sq": 0.0,
                "mech_tangent_norm": float("nan"),
                "mech_perp_norm": float("nan"),
                "vol_tangent_norm": float("nan"),
                "lambda_ratio_eff": float("nan"),
                "recovery_strength": 0.0,
            }
            if debug_progress_iter:
                print("[progress] iteration %03d: before project plain g_obj" % int(it))
            g_drive = _project(g_obj, Vls)
            if debug_progress_iter:
                print("[progress] iteration %03d: after project plain g_obj" % int(it))
        else:
            if debug_progress_iter:
                print("[progress] iteration %03d: before Hilbertian projection chain" % int(it))
            g_mech_tangent, mech_tangent_coeff, _ = tangent_project_l2(g_obj, lsf, M=l2_mass)
            g_vol_tangent, vol_tangent_coeff, vol_tangent_norm_sq = tangent_project_l2(g_vol, lsf, M=l2_mass)
            g_mech_orth, mech_vol_coeff, _ = remove_l2_component(g_mech_tangent, g_vol_tangent, M=l2_mass)
            lambda_v_active = float(cfg.get("lambda_v", 0.0))
            if not stage2_volume_continuation_active(
                cfg,
                vf=res.get("vf", None),
                vf_target=current_vf_target(cfg),
                hard_shift_only=hard_shift_only_active,
            ):
                lambda_v_active = 0.0
            radial_coeff = float(mech_tangent_coeff + lambda_v_active * vol_tangent_coeff)
            mech_tangent_norm = float(l2_norm(g_mech_tangent, M=l2_mass))
            mech_perp_norm = float(l2_norm(g_mech_orth, M=l2_mass))
            vol_tangent_norm = float(np.sqrt(max(float(vol_tangent_norm_sq), 0.0)))
            direction_eps = max(1e-30, float(cfg.get("lambda_v_direction_ratio_eps", 1e-12)))
            lambda_ratio_eff = (
                float(lambda_v_active) * vol_tangent_norm / max(mech_perp_norm, direction_eps)
            )
            g_drive = combine_l2_search_direction(g_mech_orth, g_vol_tangent, lambda_v_active)
            direction_info = {
                "mode": "hilbertian",
                "lambda_v_active": float(lambda_v_active),
                "radial_coeff": float(radial_coeff),
                "mech_tangent_coeff": float(mech_tangent_coeff),
                "vol_tangent_coeff": float(vol_tangent_coeff),
                "mech_vol_coeff": float(mech_vol_coeff),
                "vol_tangent_norm_sq": float(vol_tangent_norm_sq),
                "mech_tangent_norm": float(mech_tangent_norm),
                "mech_perp_norm": float(mech_perp_norm),
                "vol_tangent_norm": float(vol_tangent_norm),
                "lambda_ratio_eff": float(lambda_ratio_eff),
                "recovery_strength": 1.0,
            }
            if debug_progress_iter:
                print("[progress] iteration %03d: after Hilbertian projection chain" % int(it))
            if recovery_active and MPI.rank(MPI.comm_world) == 0:
                print("[12-recovery] hard-shift recovery step %d/%d: keep updated lambda_v active while ramping kappa (progress=%.3f)" %
                      (int(recovery_step_idx), int(recovery_steps_total), float(recovery_strength)))
        if use_symmetry_parameterization and (symmetry_map is not None):
            if debug_progress_iter:
                print("[progress] iteration %03d: before reduced_gradient_expand_to_full" % int(it))
            g_drive = reduced_gradient_expand_to_full(g_drive, symmetry_map)
            if debug_progress_iter:
                print("[progress] iteration %03d: after reduced_gradient_expand_to_full" % int(it))
        if str(direction_info.get("mode", "")) == "hilbertian":
            g_proj = combine_l2_search_direction(g_drive, lsf, float(direction_info.get("radial_coeff", 0.0)))
        else:
            g_proj = Function(Vls)
            copy_function_values(g_proj, g_drive)
        g_rank_field = Function(Vls)
        copy_function_values(g_rank_field, g_drive)
        g_sym = Function(Vls)
        copy_function_values(g_sym, g_rank_field)
        g_filt = Function(Vls)
        copy_function_values(g_filt, g_sym)

        if debug_progress_iter:
            print("[progress] iteration %03d: before angle_between" % int(it))
        theta_update, pn_update, gn_update = angle_between(lsf, g_proj, M=l2_mass)
        theta_update_sym = None
        if use_symmetry_parameterization and (symmetry_map is not None):
            theta_update_sym, _, _ = symmetry_reduced_angle_between(lsf, g_proj, symmetry_map)
        theta_opt, _, _ = angle_between(lsf, g_obj, M=l2_mass)
        gn_drive = l2_norm(g_drive, M=l2_mass)
        if debug_progress_iter:
            print("[progress] iteration %03d: after angle_between" % int(it))
        apply_hard_vf_this_iter = False
        g = g_drive
        stall_wait_skip_this_iter = stage2_stall_wait_should_skip(
            cfg,
            res,
            it,
            cooldown_active=cooldown_active,
            recovery_active=recovery_active,
            lambda_plateau_cooldown_active=lambda_plateau_cooldown_active,
            hard_shift_only_active=hard_shift_only_active,
            skip_j_compare_this_iter=skip_j_compare_this_iter,
        )

        if stall_wait_skip_this_iter:
            lsf_trial_base = Function(Vls)
            copy_function_values(lsf_trial_base, lsf)
            vf_hard_target_try = current_vf_target(cfg)
            lsf_try, c_hard_try, vf_hard_try = prepare_trial_lsf(
                lsf_trial_base, apply_hard_vf=apply_hard_vf_this_iter
            )
        elif skip_j_compare_this_iter:
            # Stage-change warmup: do not compare with previous-stage J.
            # Keep lsf search step at kappa=0 equivalent this iter.
            lsf_trial_base = Function(Vls)
            copy_function_values(lsf_trial_base, lsf)
            vf_hard_target_try = current_vf_target(cfg)
            lsf_try, c_hard_try, vf_hard_try = prepare_trial_lsf(
                lsf_trial_base, apply_hard_vf=apply_hard_vf_this_iter
            )
        else:
            psi_try_expr, theta_update, gn_drive = slerp_update(lsf, g, Vls, dx, kappa, deg=cfg["deg"], it=it, g_in_V=g_drive, psi_in_V=lsf, theta_pn_gn=(theta_update, pn_update, gn_drive), l2_mass=l2_mass)
            lsf_trial_base = _project(psi_try_expr, Vls)
            d_final_probe = difference_function(lsf_trial_base, lsf)
            if bool(cfg.get("print_direction_chain_diagnostics", False)):
                print_direction_chain_diagnostics(
                    it, res["DTJ"], g_raw, g_sym, g_filt, d_final_probe,
                    dxm=dx, l2_mass=l2_mass, kappa_probe=kappa, filter_active=False
                )
            vf_hard_target_try = current_vf_target(cfg)
            lsf_try, c_hard_try, vf_hard_try = prepare_trial_lsf(
                lsf_trial_base, apply_hard_vf=apply_hard_vf_this_iter
            )

        accept = True
        J_try = None
        line_search_stalled = False
        line_search_rejected_trials = []

        if postprocess_target_reached_early:
            accept = False
            solver_fail = False
            line_search_stalled = False
            lsf_try = Function(Vls)
            copy_function_values(lsf_try, lsf)
        elif cooldown_active:
            cooldown_context_name = "hard-shift-cooldown"
            cooldown_step_idx = int(cfg.get("hard_shift_cooldown_steps", 5)) - int(hard_shift_cooldown_left) + 1
            cooldown_step_total = int(cfg.get("hard_shift_cooldown_steps", 5))
            cooldown_step_kappa = float(cooldown_kappa)
            if final_filter_cooldown_active:
                cooldown_context_name = "final-filter-cooldown"
                cooldown_step_idx = int(cfg.get("final_filter_post_cooldown_steps", 5)) - int(final_filter_post_cooldown_left) + 1
                cooldown_step_total = int(cfg.get("final_filter_post_cooldown_steps", 5))
                cooldown_step_kappa = float(final_filter_cooldown_kappa)
            elif refine_cooldown_active:
                cooldown_context_name = "refine-cooldown"
                cooldown_step_idx = int(cfg.get("refine_cooldown_steps", 2)) - int(refine_post_cooldown_left) + 1
                cooldown_step_total = int(cfg.get("refine_cooldown_steps", 2))
                cooldown_step_kappa = float(refine_cooldown_kappa)
            if debug_progress_iter:
                print("[progress] iteration %03d: before slerp_update (cooldown)" % int(it))
            psi_try_expr, theta_update, gn_drive = slerp_update(
                lsf, g, Vls, dx, cooldown_step_kappa, deg=cfg["deg"], it=it,
                g_in_V=g_drive, psi_in_V=lsf, theta_pn_gn=(theta_update, pn_update, gn_drive), l2_mass=l2_mass
            )
            if debug_progress_iter:
                print("[progress] iteration %03d: after slerp_update (cooldown)" % int(it))
                print("[progress] iteration %03d: before project psi_try_expr (cooldown)" % int(it))
            lsf_trial_base = _project(psi_try_expr, Vls)
            if debug_progress_iter:
                print("[progress] iteration %03d: after project psi_try_expr (cooldown)" % int(it))
            vf_hard_target_try = current_vf_target(cfg)
            if debug_progress_iter:
                print("[progress] iteration %03d: before prepare_trial_lsf (cooldown)" % int(it))
            lsf_try, c_hard_try, vf_hard_try = prepare_trial_lsf(
                lsf_trial_base, apply_hard_vf=apply_hard_vf_this_iter
            )
            if debug_progress_iter:
                print("[progress] iteration %03d: after prepare_trial_lsf (cooldown)" % int(it))
            lsf_try, res_try, solver_fail = evaluate_trial_with_pre_jacobi_filter(
                lsf_try,
                context="[it %03d %s %d/%d]" % (
                    it,
                    cooldown_context_name,
                    int(cooldown_step_idx),
                    int(cooldown_step_total)
                ),
                alpha_eval=alpha_now,
                vf_ref=res.get("vf", None),
                apply_hard_vf=apply_hard_vf_this_iter,
            )
            if not solver_fail:
                cooldown_stage_active = stage2_volume_continuation_active(
                    cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                    hard_shift_only=hard_shift_only_active
                )
                mech_cap_ok, mech_cap_info = stage2_mechanical_step_cap_ok(
                    cfg, res, res_try, cooldown_stage_active
                )
                if mech_cap_ok:
                    res = res_try
                    kappa = cooldown_step_kappa
                    if MPI.rank(MPI.comm_world) == 0:
                        if final_filter_cooldown_active:
                            print("[12-cooldown] final-filter cooldown step %d/%d: continuous volume correction off, accept without J_merit check (kappa=%.4e)" %
                                  (int(cooldown_step_idx), int(cooldown_step_total), float(cooldown_step_kappa)))
                        elif refine_cooldown_active:
                            print("[12-cooldown] refine cooldown step %d/%d: accept without J_merit check on refined mesh (kappa=%.4e)" %
                                  (int(cooldown_step_idx), int(cooldown_step_total), float(cooldown_step_kappa)))
                        else:
                            print("[12-cooldown] hard-shift cooldown step %d/%d: keep updated lambda_v active, accept without J_merit check (kappa=%.4e)" %
                                  (int(cooldown_step_idx), int(cooldown_step_total), float(cooldown_step_kappa)))
                else:
                    print_stage2_mechanical_step_cap_reject(
                        "%s step %d/%d" % (cooldown_context_name, int(cooldown_step_idx), int(cooldown_step_total)),
                        mech_cap_info,
                        action="mark stalled",
                    )
                    line_search_stalled = True
                    lsf_try = Function(Vls)
                    copy_function_values(lsf_try, lsf)
        elif skip_j_compare_this_iter:
            lsf_try, res_try, solver_fail = evaluate_trial_with_pre_jacobi_filter(
                lsf_try,
                context="[it %03d stage-warmup]" % it,
                alpha_eval=alpha_now,
                vf_ref=res.get("vf", None),
                apply_hard_vf=apply_hard_vf_this_iter,
            )
            if not solver_fail:
                warmup_stage_active = stage2_volume_continuation_active(
                    cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                    hard_shift_only=hard_shift_only_active
                )
                mech_cap_ok, mech_cap_info = stage2_mechanical_step_cap_ok(
                    cfg, res, res_try, warmup_stage_active
                )
                if mech_cap_ok:
                    res = res_try
                else:
                    print_stage2_mechanical_step_cap_reject(
                        "stage-warmup", mech_cap_info, action="mark stalled"
                    )
                    line_search_stalled = True
                    lsf_try = Function(Vls)
                    copy_function_values(lsf_try, lsf)
        elif lambda_plateau_cooldown_active:
            # Plateau-lambda cooldown: keep advancing with current kappa,
            # but skip J_merit comparison and accept whenever solver succeeds.
            lsf_try, res_try, solver_fail = evaluate_trial_with_pre_jacobi_filter(
                lsf_try,
                context="[it %03d lambda-v-plateau-cooldown %d/%d]" % (
                    it,
                    int(cfg.get("lambda_v_plateau_accept_cooldown_steps", 3)) - int(lambda_plateau_cooldown_left) + 1,
                    int(cfg.get("lambda_v_plateau_accept_cooldown_steps", 3)),
                ),
                alpha_eval=alpha_now,
                vf_ref=res.get("vf", None),
                apply_hard_vf=apply_hard_vf_this_iter,
            )
            if not solver_fail:
                plateau_stage_active = stage2_volume_continuation_active(
                    cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                    hard_shift_only=hard_shift_only_active
                )
                mech_cap_ok, mech_cap_info = stage2_mechanical_step_cap_ok(
                    cfg, res, res_try, plateau_stage_active
                )
                if mech_cap_ok:
                    res = res_try
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[lambda_v-plateau-cooldown] step %d/%d: accept without J_merit comparison (kappa=%.4e)" %
                              (int(cfg.get("lambda_v_plateau_accept_cooldown_steps", 3)) - int(lambda_plateau_cooldown_left) + 1,
                               int(cfg.get("lambda_v_plateau_accept_cooldown_steps", 3)),
                               float(kappa)))
                else:
                    print_stage2_mechanical_step_cap_reject(
                        "lambda-v-plateau-cooldown", mech_cap_info, action="mark stalled"
                    )
                    line_search_stalled = True
                    lsf_try = Function(Vls)
                    copy_function_values(lsf_try, lsf)
        elif stall_wait_skip_this_iter:
            line_search_stalled = True
            accept = False
            lsf_try = Function(Vls)
            copy_function_values(lsf_try, lsf)
            stall_wait_diag = mark_stage2_stall_wait_iteration(cfg, res, it, kappa_min)
            if MPI.rank(MPI.comm_world) == 0:
                release_it = int(cfg.get("_stage2_stall_wait_until_it", int(it)))
                reason_wait = str(cfg.get("_stage2_stall_wait_reason", "cooldown"))
                skipped_wait = int(cfg.get("_stage2_stall_wait_skipped_iters", 0))
                release_mode = "release-probe" if int(it) >= release_it else "cooldown-wait"
                print("[stage2-stall-wait] it=%03d %s: skip repeated full line-search "
                      "(reason=%s, skipped=%d, release_it=%d, vf=%.6f, target=%s, action=%s)" %
                      (int(it), release_mode, reason_wait, int(skipped_wait), int(release_it),
                       float(res.get("vf", 0.0)),
                       ("None" if current_vf_target(cfg) is None else ("%.6f" % float(current_vf_target(cfg)))),
                       str(stall_wait_diag.get("action", "unknown"))),
                      flush=True)
        elif cfg["do_full_linesearch"]:
            # [12] relaxed acceptance (sign-robust): accept if
            # J_merit_try <= J_merit_old + j_relax * abs(J_merit_old).
            # The optimization direction still comes from the pure mechanical J_mech.
            accept = False
            kappa_try = kappa
            ls_shrink_step = 1
            stage2_ls_active = stage2_volume_continuation_active(
                cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                hard_shift_only=hard_shift_only_active
            )
            stage2_rescue_candidate = None
            stage2_rescue_enabled = False
            stage2_rescue_target = None
            stage2_rescue_c_old = 0.0
            stage2_rescue_j_scale = max(
                abs(float(J_compare_old)),
                abs(float(res.get("J", J_compare_old))),
                1e-12,
            )
            if stage2_ls_active and bool(cfg.get("stage2_stall_rescue_accept_enabled", False)):
                stage2_rescue_target = current_vf_target(cfg)
                if stage2_rescue_target is not None:
                    stage2_rescue_target = float(stage2_rescue_target)
                    stage2_rescue_c_old = max(float(res.get("vf", 0.0)) - stage2_rescue_target, 0.0)
                    _stage2_rescue_tol = max(
                        0.0,
                        float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0))),
                        float(cfg.get("vf_al_adapt_eps", 1e-12)),
                    )
                    stage2_rescue_enabled = bool(stage2_rescue_c_old > _stage2_rescue_tol)
            if use_stage_relaxed_accept:
                J_allow = J_compare_old + stage_relax_tau * abs(J_compare_old)
            else:
                if stage2_ls_active:
                    merit_relax = stage2_effective_merit_relax(cfg, res_eval=res)
                    J_allow = J_compare_old + merit_relax * max(abs(J_compare_old), 1e-12)
                else:
                    J_allow = J_compare_old + j_relax * abs(J_compare_old)
            accepted_in_linesearch = False
            while True:
                if debug_progress_iter:
                    print("[progress] iteration %03d: before slerp_update (line-search kappa=%.4e)" % (int(it), float(kappa_try)))
                psi_try_expr, theta_update, gn_drive = slerp_update(lsf, g, Vls, dx, kappa_try, deg=cfg["deg"], it=it, g_in_V=g_drive, psi_in_V=lsf, theta_pn_gn=(theta_update, pn_update, gn_drive), l2_mass=l2_mass)
                if debug_progress_iter:
                    print("[progress] iteration %03d: after slerp_update (line-search kappa=%.4e)" % (int(it), float(kappa_try)))
                    print("[progress] iteration %03d: before project psi_try_expr (line-search kappa=%.4e)" % (int(it), float(kappa_try)))
                lsf_trial_base = _project(psi_try_expr, Vls)
                if debug_progress_iter:
                    print("[progress] iteration %03d: after project psi_try_expr (line-search kappa=%.4e)" % (int(it), float(kappa_try)))
                vf_hard_target_try = current_vf_target(cfg)
                if debug_progress_iter:
                    print("[progress] iteration %03d: before prepare_trial_lsf (line-search kappa=%.4e)" % (int(it), float(kappa_try)))
                lsf_try, c_hard_try, vf_hard_try = prepare_trial_lsf(
                    lsf_trial_base, apply_hard_vf=apply_hard_vf_this_iter
                )
                if debug_progress_iter:
                    print("[progress] iteration %03d: after prepare_trial_lsf (line-search kappa=%.4e)" % (int(it), float(kappa_try)))
                lsf_try, res_try, solver_fail = evaluate_trial_with_pre_jacobi_filter(
                    lsf_try,
                    context="[it %03d line-search kappa=%.4e]" % (it, float(kappa_try)),
                    alpha_eval=alpha_now,
                    vf_ref=res.get("vf", None),
                    apply_hard_vf=apply_hard_vf_this_iter,
                )
                if solver_fail:
                    shrink_factor = (
                        float(cfg["delta"]) if ls_shrink_step <= 2 else float(cfg.get("delta_ls_tail", 0.6))
                    )
                    if kappa_try <= kappa_min + 1e-15:
                        break
                    next_kappa_try = max(kappa_try * shrink_factor, kappa_min)
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[12] solver failed at trial kappa %.4f => line search: kappa %.4f -> %.4f (min=%.4f)" %
                              (kappa_try, kappa_try, next_kappa_try, kappa_min))
                    kappa_try = next_kappa_try
                    ls_shrink_step += 1
                    solver_fail = False
                    continue
                J_try = float(res_try["J_compare"])
                J_try_mech = float(res_try["J"])
                J_allow_trial = float(J_allow)
                if (not use_stage_relaxed_accept) and stage2_ls_active:
                    merit_relax_eff = stage2_effective_merit_relax(cfg, res_eval=res)
                    J_allow_trial = J_compare_old + float(merit_relax_eff) * max(abs(J_compare_old), 1e-12)
                mech_cap_ok, mech_cap_info = stage2_mechanical_step_cap_ok(
                    cfg, res, res_try, stage2_ls_active
                )

                if (J_try <= J_allow_trial) and mech_cap_ok:
                    if MPI.rank(MPI.comm_world) == 0:
                        if use_stage_relaxed_accept:
                            print("[12-stage-relax] J_merit_try=%.6e <= J_merit_allow=%.6e (J_merit_old=%.6e, J_mech_try=%.6e, tau=%.2e, steps_left=%d) => accept (kappa=%.4f, alpha=%.3e)" %
                                  (J_try, J_allow_trial, J_compare_old, J_try_mech, stage_relax_tau, stage_relax_steps_left, kappa_try, alpha_now))
                        else:
                            print("[12] J_merit_try=%.6e <= J_merit_allow=%.6e (J_merit_old=%.6e, J_mech_try=%.6e) => accept (kappa=%.4f, alpha=%.3e)" %
                                  (J_try, J_allow_trial, J_compare_old, J_try_mech, kappa_try, alpha_now))
                    accept = True
                    res = res_try
                    kappa = kappa_try
                    # recover kappa towards kappa0 so topology keeps moving (cap at kappa0)
                    _n_early = int(cfg.get("kappa_increase_early_count", 2))
                    if kappa_increase_apply_count < _n_early:
                        _inc = float(cfg.get("kappa_increase_factor_early", 1.1))
                    else:
                        _inc = float(cfg.get("kappa_increase_factor", 1.2))
                    kappa = min(kappa * _inc, cfg["kappa0"])
                    kappa_increase_apply_count += 1
                    accepted_in_linesearch = True
                    if stage2_ls_active:
                        stage2_decay_merit_relax_after_accept(cfg)
                    break
                if (J_try <= J_allow_trial) and (not mech_cap_ok) and MPI.rank(MPI.comm_world) == 0:
                    print("[12-mech-cap] J_merit_try=%.6e <= J_merit_allow=%.6e but "
                          "J_mech_try=%.6e > J_mech_allow=%.6e "
                          "(J_mech_old=%.6e, cap=%.3e, mode=%s) => reject and shrink kappa" %
                          (float(J_try), float(J_allow_trial), float(J_try_mech),
                           float(mech_cap_info.get("J_allow", float("nan"))),
                           float(mech_cap_info.get("J_old", res.get("J", float("nan")))),
                           float(mech_cap_info.get("cap_rel", float("nan"))),
                           str(mech_cap_info.get("mode", "n/a"))), flush=True)

                if stage2_ls_active:
                    line_search_rejected_trials.append(
                        dict(
                            kappa=float(kappa_try),
                            J_compare=float(J_try),
                            J=float(J_try_mech),
                            J_allow=float(J_allow_trial),
                            J_mech_allow=float(mech_cap_info.get("J_allow", float("nan"))),
                            J_mech_cap_ok=bool(mech_cap_ok),
                            vf=float(res_try.get("vf", float("nan"))),
                            vf_constraint_violation=float(res_try.get("vf_constraint_violation", float("nan"))),
                        )
                    )
                    if stage2_rescue_enabled:
                        vf_try_rescue = float(res_try.get("vf", float("nan")))
                        if np.isfinite(vf_try_rescue):
                            c_try_rescue = max(vf_try_rescue - float(stage2_rescue_target), 0.0)
                            vf_reduction = float(stage2_rescue_c_old - c_try_rescue)
                            vf_reduction_frac = float(vf_reduction / max(stage2_rescue_c_old, 1e-30))
                            min_vf_reduction_abs = max(
                                0.0,
                                float(cfg.get("stage2_stall_rescue_vf_reduction_abs", 0.0)),
                                float(cfg.get("stage2_vf_stage_tol", cfg.get("vf_stage_tol", 0.0)))
                                * float(cfg.get("stage2_stall_rescue_vf_tol_factor", 0.0)),
                            )
                            min_vf_reduction_frac = max(
                                0.0,
                                float(cfg.get("stage2_stall_rescue_vf_reduction_frac", 0.0)),
                            )
                            required_relax = max(
                                0.0,
                                float(J_try - J_compare_old) / max(stage2_rescue_j_scale, 1e-12),
                            )
                            rescue_relax_cap = max(
                                stage2_base_merit_relax(cfg),
                                float(cfg.get("stage2_merit_relax_auto_max", stage2_base_merit_relax(cfg))),
                            )
                            volume_rescue_ok = (
                                vf_reduction >= min_vf_reduction_abs
                                or vf_reduction_frac >= min_vf_reduction_frac
                            )
                            merit_rescue_ok = bool(required_relax <= rescue_relax_cap + 1e-15)
                            if volume_rescue_ok and merit_rescue_ok:
                                rescue_better = (
                                    stage2_rescue_candidate is None
                                    or float(J_try) < float(stage2_rescue_candidate["J_compare"])
                                )
                                if rescue_better and mech_cap_ok:
                                    lsf_rescue = Function(Vls)
                                    copy_function_values(lsf_rescue, lsf_try)
                                    stage2_rescue_candidate = dict(
                                        lsf=lsf_rescue,
                                        res=res_try,
                                        kappa=float(kappa_try),
                                        J_compare=float(J_try),
                                        J=float(J_try_mech),
                                        vf=float(vf_try_rescue),
                                        vf_reduction=float(vf_reduction),
                                        vf_reduction_frac=float(vf_reduction_frac),
                                        required_relax=float(required_relax),
                                        relax_cap=float(rescue_relax_cap),
                                    )
                shrink_factor = (
                    float(cfg["delta"]) if ls_shrink_step <= 2 else float(cfg.get("delta_ls_tail", 0.6))
                )
                if kappa_try <= kappa_min + 1e-15:
                    break
                if MPI.rank(MPI.comm_world) == 0:
                    next_kappa_try = max(kappa_try * shrink_factor, kappa_min)
                    if use_stage_relaxed_accept:
                        fail_reason = "J_mech-cap" if (J_try <= J_allow_trial and not mech_cap_ok) else "J_merit"
                        print("[12-stage-relax] reject=%s J_merit_try=%.6e J_merit_allow=%.6e (J_merit_old=%.6e, tau=%.2e, steps_left=%d) => line search: kappa %.4f -> %.4f (min=%.4f)" %
                              (fail_reason, J_try, J_allow_trial, J_compare_old, stage_relax_tau, stage_relax_steps_left, kappa_try, next_kappa_try, kappa_min))
                    else:
                        fail_reason = "J_mech-cap" if (J_try <= J_allow_trial and not mech_cap_ok) else "J_merit"
                        print("[12] reject=%s J_merit_try=%.6e J_merit_allow=%.6e (J_merit_old=%.6e) => line search: kappa %.4f -> %.4f (min=%.4f)" %
                              (fail_reason, J_try, J_allow_trial, J_compare_old, kappa_try, next_kappa_try, kappa_min))
                kappa_try = max(kappa_try * shrink_factor, kappa_min)
                ls_shrink_step += 1
            if (not solver_fail) and (not accepted_in_linesearch):
                if stage2_rescue_candidate is not None:
                    rescue_eff = stage2_raise_merit_relax_for_rescue(
                        cfg, stage2_rescue_candidate["required_relax"]
                    )
                    res = stage2_rescue_candidate["res"]
                    lsf_try = stage2_rescue_candidate["lsf"]
                    kappa = float(stage2_rescue_candidate["kappa"])
                    kappa = min(
                        kappa * max(1.0, float(cfg.get("stage2_stall_rescue_kappa_increase_factor", 1.0))),
                        cfg["kappa0"],
                    )
                    accepted_in_linesearch = True
                    accept = True
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[12-stage2-adapt] stalled line search rescued a volume-improving near-miss: J_merit_try=%.6e (old=%.6e, required_relax=%.3e, eff_relax=%.3e, cap=%.3e), vf %.6f -> %.6f (dc=%.3e, frac=%.3f), kappa=%.4e" %
                              (float(stage2_rescue_candidate["J_compare"]),
                               float(J_compare_old),
                               float(stage2_rescue_candidate["required_relax"]),
                               float(rescue_eff),
                               float(stage2_rescue_candidate["relax_cap"]),
                               float(res.get("vf", float("nan"))) + float(stage2_rescue_candidate["vf_reduction"]),
                               float(stage2_rescue_candidate["vf"]),
                               float(stage2_rescue_candidate["vf_reduction"]),
                               float(stage2_rescue_candidate["vf_reduction_frac"]),
                               float(stage2_rescue_candidate["kappa"])))
                else:
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[12] no acceptable trial found down to kappa_min=%.4e => mark stalled (let stage controller react)" %
                              kappa_min)
                    line_search_stalled = True
                    diagnose_stage2_line_search_stall(
                        cfg, res, line_search_rejected_trials, J_allow, it=it
                    )
                    maybe_activate_stage2_same_state_stall_wait(cfg, res, it)
                    lsf_try = Function(Vls)
                    copy_function_values(lsf_try, lsf)
        else:
            lsf_try, res_try, solver_fail = evaluate_trial_with_pre_jacobi_filter(
                lsf_try,
                context="[it %03d single-step]" % it,
                alpha_eval=alpha_now,
                vf_ref=res.get("vf", None),
                apply_hard_vf=apply_hard_vf_this_iter,
            )
            if not solver_fail:
                single_stage_active = stage2_volume_continuation_active(
                    cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                    hard_shift_only=hard_shift_only_active
                )
                mech_cap_ok, mech_cap_info = stage2_mechanical_step_cap_ok(
                    cfg, res, res_try, single_stage_active
                )
                if mech_cap_ok:
                    res = res_try
                else:
                    print_stage2_mechanical_step_cap_reject(
                        "single-step", mech_cap_info, action="mark stalled"
                    )
                    line_search_stalled = True
                    lsf_try = Function(Vls)
                    copy_function_values(lsf_try, lsf)

        if use_stage_relaxed_accept and stage_relax_steps_left > 0:
            stage_relax_steps_left -= 1

        force_fail_recover_now = False
        if (
            force_fail_recover_enabled
            and (not solver_fail)
            and bool(cfg.get("use_helmholtz_filter", False))
            and (force_fail_recover_all or (int(it) in force_fail_recover_iters))
        ):
            # Debug mode: route this iteration through the fail-recover path.
            solver_fail = True
            force_fail_recover_now = True
            if MPI.rank(MPI.comm_world) == 0:
                print("[debug-force-fail-recover] forcing fail-recover branch at it=%03d" % int(it))
        if solver_fail and bool(cfg.get("use_helmholtz_filter", False)):
            # Solver-fail recovery policy:
            #   1) try emergency Helmholtz smoothing with adaptive radius search under
            #      progressively relaxed vf-change limits (10%, 15%, 20%, ...).
            #   2) if still failing, then refine
            _default_factors = (1.0, 1.5)
            _ft = cfg.get("helmholtz_fail_radius_factors", _default_factors)
            _fl = [float(x) for x in _ft]
            _mt = int(cfg.get("helmholtz_fail_max_tries", len(_fl)))
            _n = min(max(1, _mt), len(_fl))
            fail_radius_factor_list = _fl[:_n]
            vf_before_fail = _vf_from_lsf_state(
                lsf_try, float(cfg["threshold"]), dx, float(assemble(Constant(1.0) * dx))
            )
            rel_limits = []
            rel_now = float(fail_recover_vf_rel_start)
            while rel_now <= float(fail_recover_vf_rel_max) + 1e-15:
                rel_limits.append(float(rel_now))
                rel_now += float(fail_recover_vf_rel_step)
            if len(rel_limits) == 0:
                rel_limits.append(float(fail_recover_vf_rel_start))
            # Last round: if still not recovered, drop vf guard and only require solver recovery.
            rel_limits.append(float("inf"))

            for rel_limit_try in rel_limits:
                for i_try, fail_radius_factor in enumerate(fail_radius_factor_list):
                    radius_fail_cap = fail_radius_factor * float(mesh.hmin())
                    lsf_fail, res_fail, fail_adapt_info = adaptive_helmholtz_filter_with_vf_guard(
                        lsf_try, vf_before_fail, radius_fail_cap, rel_limit_try,
                        filter_radius_search_max_iter,
                        context_prefix="[it %03d fail-recover helmholtz #%d rel=%s" % (
                            it,
                            i_try + 1,
                            ("inf" if not np.isfinite(rel_limit_try) else ("%.3f" % float(rel_limit_try))),
                        ),
                        alpha_eval=alpha_now,
                        apply_hard_vf=apply_hard_vf_this_iter,
                        base_res=None,
                    )
                    if bool(fail_adapt_info.get("success", False)):
                        fail_stage_active = stage2_volume_continuation_active(
                            cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                            hard_shift_only=hard_shift_only_active
                        )
                        mech_cap_ok, mech_cap_info = stage2_mechanical_step_cap_ok(
                            cfg, res, res_fail, fail_stage_active
                        )
                        if not mech_cap_ok:
                            print_stage2_mechanical_step_cap_reject(
                                "fail-recover helmholtz #%d rel=%s" % (
                                    i_try + 1,
                                    "inf" if not np.isfinite(rel_limit_try) else ("%.3f" % float(rel_limit_try)),
                                ),
                                mech_cap_info,
                                action="continue recovery search",
                            )
                            continue
                        lsf_try = lsf_fail
                        res = res_fail
                        solver_fail = False
                        if MPI.rank(MPI.comm_world) == 0:
                            if force_fail_recover_now:
                                print("[13-fail-recover] debug-forced recover by adaptive Helmholtz (try=%d, r*=%.3e, cap=%.3e, vf_rel_limit=%s, vf_rel_change=%.3e) => skip refinement" %
                                      (i_try + 1,
                                       float(fail_adapt_info.get("radius_star", 0.0)),
                                       float(fail_adapt_info.get("radius_cap", radius_fail_cap)),
                                       ("inf" if not np.isfinite(rel_limit_try) else ("%.3f" % float(rel_limit_try))),
                                       float(fail_adapt_info.get("rel_change", float("nan")))))
                            else:
                                print("[13-fail-recover] solver recovered by adaptive Helmholtz (try=%d, r*=%.3e, cap=%.3e, vf_rel_limit=%s, vf_rel_change=%.3e) => skip refinement" %
                                      (i_try + 1,
                                       float(fail_adapt_info.get("radius_star", 0.0)),
                                       float(fail_adapt_info.get("radius_cap", radius_fail_cap)),
                                       ("inf" if not np.isfinite(rel_limit_try) else ("%.3f" % float(rel_limit_try))),
                                       float(fail_adapt_info.get("rel_change", float("nan")))))
                        break
                if not solver_fail:
                    break

        outer_al_handled = False
        hard_shift_done = False
        hard_shift_attempted = False
        if solver_fail:
            step13_ok = False
            step13_refine_ok = False
            step14_ok = False
            rel_tol = float("nan")
            eps_J = cfg["eps_J"]
            if MPI.rank(MPI.comm_world) == 0:
                print("[13] solver_fail persists after fail-recovery => force mesh refinement this iteration")
        elif line_search_stalled:
            step13_ok = False
            step13_refine_ok = False
            step14_ok = False
            rel_tol = float("nan")
            eps_J = cfg["eps_J"]
            if MPI.rank(MPI.comm_world) == 0:
                print("[13] line-search stalled at kappa_min with no acceptable step (no refinement unless final vf target is satisfied)")
            if recovery_active and bool(cfg.get("hard_shift_recovery_stall_counts", True)):
                hard_shift_recovery_left = max(0, int(hard_shift_recovery_left) - 1)
                if MPI.rank(MPI.comm_world) == 0:
                    print("[12-recovery-stall] line-search stalled during recovery -> consume one recovery step, remaining=%d" %
                          int(hard_shift_recovery_left))
            if bool(cfg.get("_post_nucleation_lambda_zero_active", False)):
                cfg["_post_nucleation_lambda_zero_active"] = False
                _, lambda_zero_info = update_lambda_v_from_stage_state(
                    cfg, res, psi=lsf, M=l2_mass, it=it, reason="post-nucleation-zero-kappa-min"
                )
                if MPI.rank(MPI.comm_world) == 0:
                    print("[lambda_v-zero-freeze] it=%03d: kappa_min reached during zero-lambda window -> release freeze; lambda_v=%.6e" %
                          (int(it), float(cfg.get("lambda_v", 0.0))))
            if bool(cfg.get("post_nucleation_freeze_early_release_enabled", True)):
                last_nucleation_it_raw = cfg.get("_last_nucleation_it", None)
                freeze_steps = max(0, int(cfg.get("post_nucleation_freeze_steps", 0)))
                freeze_active_for_release = False
                freeze_age_for_release = None
                if (last_nucleation_it_raw is not None) and (freeze_steps > 0):
                    try:
                        freeze_age_for_release = int(it) - int(last_nucleation_it_raw)
                        freeze_active_for_release = (
                            freeze_age_for_release > 0
                            and freeze_age_for_release <= freeze_steps
                        )
                    except Exception:
                        freeze_active_for_release = False
                lambda_floor_rel, lambda_cap_rel = _lambda_v_cap_from_cfg(cfg, include_plateau_cap=False)
                lambda_at_cap = (
                    lambda_cap_rel is not None
                    and float(cfg.get("lambda_v", 0.0)) >= float(lambda_cap_rel) - 1e-16
                )
                plat_win_rel = int(cfg.get("hard_shift_plateau_window", 5))
                plat_tol_rel = float(cfg.get("hard_shift_plateau_vf_tol", 1e-12))
                vf_plateau_rel = False
                if len(vf_history) >= plat_win_rel:
                    vf_tail_rel = [float(v) for v in vf_history[-plat_win_rel:]]
                    vf_plateau_rel = max(abs(v - vf_tail_rel[-1]) for v in vf_tail_rel) <= plat_tol_rel
                if freeze_active_for_release and lambda_at_cap and vf_plateau_rel:
                    cfg["_last_nucleation_it"] = int(it) - freeze_steps - 1
                    cfg["_post_nucleation_lambda_zero_active"] = False
                    if MPI.rank(MPI.comm_world) == 0:
                        age_msg = "unknown" if freeze_age_for_release is None else str(int(freeze_age_for_release))
                        cap_msg = "None" if lambda_cap_rel is None else ("%.6e" % float(lambda_cap_rel))
                        print("[vf-hard-nucleation-freeze] it=%03d: early release post-nucleation freeze (age=%s/%d, lambda_v at cap=%s, kappa_min reached, vf plateau over %d steps)" %
                              (int(it), age_msg, int(freeze_steps), cap_msg, int(plat_win_rel)))
        else:
            reset_stage2_stall_watchdog(cfg)
            # (A) Diagnostic: did psi actually change this step?
            # All ranks must run assemble/vector.min/max (collective); only rank 0 prints.
            lsf_old = Function(Vls)
            copy_function_values(lsf_old, lsf)
            # lsf_try is already in Vls; avoid extra projection that breaks strict wedge consistency.
            copy_function_values(lsf, lsf_try)
            # Fix-2: re-normalise lsf to unit L2 norm after each accept so slerp N=1 assumption holds.
            _lsf_norm_iter = renormalize_lsf_inplace(lsf, dx)

            # Strict minimal-wedge parameterization is already enforced on each trial before evaluate.
            diff_psi = Function(Vls)
            diff_psi.vector().set_local(lsf.vector().get_local() - lsf_old.vector().get_local())
            diff_psi.vector().apply("insert")
            norm_psi_change = np.sqrt(assemble(diff_psi * diff_psi * dx))
            psi_min = lsf.vector().min()
            psi_max = lsf.vector().max()
            # Volume fraction where |psi|<0.1 (collective, safe in parallel; dof-fraction would need Allreduce)
            vol_psi_near = assemble(conditional(lt(abs(lsf), 0.1), Constant(1.0), Constant(0.0)) * dx)
            vol_total = assemble(Constant(1.0) * dx)
            frac_near_interface = float(vol_psi_near) / max(1e-30, float(vol_total))
            if MPI.rank(MPI.comm_world) == 0:
                print("[diag psi] ||psi_new-psi_old||_L2=%.4e  min(psi)=%.4e  max(psi)=%.4e  frac(|psi|<0.1)=%.4f" %
                      (norm_psi_change, psi_min, psi_max, frac_near_interface))

            if stage2_volume_continuation_active(cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg), hard_shift_only=hard_shift_only_active):
                update_volume_continuation_state(cfg, res, it=it, reason="post-accept", update_mu=True)
                rate_info = stage2_update_vf_rate_controller(
                    cfg,
                    res,
                    it=it,
                    cooldown_active=cooldown_active or lambda_plateau_cooldown_active,
                    recovery_active=recovery_active,
                    hard_shift_only=hard_shift_only_active,
                )
                if isinstance(rate_info, dict) and str(rate_info.get("action", "")) in (
                    "slow-grow",
                    "fast-shrink",
                    "overshoot-shrink",
                    "near-target-relax",
                    "in-band-relax",
                ):
                    update_lambda_v_from_stage_state(
                        cfg, res, psi=lsf, M=l2_mass, it=it, reason="stage2-vf-rate"
                    )
                slow_info = stage2_update_slow_progress_watchdog(
                    cfg,
                    res,
                    it=it,
                    kappa=kappa,
                    cooldown_active=cooldown_active or lambda_plateau_cooldown_active,
                    recovery_active=recovery_active,
                    hard_shift_only=hard_shift_only_active,
                )
                if isinstance(slow_info, dict) and str(slow_info.get("action", "")) == "lambda-ratio-grow":
                    update_lambda_v_from_stage_state(
                        cfg, res, psi=lsf, M=l2_mass, it=it, reason="stage2-slow-progress"
                    )
            J_old = res["J"]
            J_compare_old = res["J_compare"]
            if recovery_active:
                # Enforce kappa recovery: by the 3rd recovery step, kappa reaches kappa0.
                kappa = max(float(kappa), float(recovery_kappa_target))
                if recovery_step_idx >= recovery_steps_total:
                    kappa = float(cfg["kappa0"])
                if MPI.rank(MPI.comm_world) == 0:
                    print("[12-recovery-kappa] step %d/%d: enforce kappa=%.4e (target=%.4e, kappa0=%.4e)" %
                          (int(recovery_step_idx), int(recovery_steps_total),
                           float(kappa), float(recovery_kappa_target), float(cfg["kappa0"])))
            J_history.append(J_compare_old)
            if postprocess_active:
                _pp_j_hist = list(cfg.get("_postprocess_j_mech_history", []))
                _pp_j_hist.append(float(res.get("J", J_compare_old)))
                _pp_hist_keep = max(
                    2,
                    int(cfg.get("postprocess_hard_shift_max_gap_iters", cfg.get("postprocess_hard_shift_gap_iters", 20))) + 5,
                )
                cfg["_postprocess_j_mech_history"] = _pp_j_hist[-_pp_hist_keep:]
            else:
                cfg["_postprocess_j_mech_history"] = []
            m_window = int(cfg.get("J_window", 5))
            J_min = cfg["J_min"]
            eps_J = cfg["eps_J"]
            rel_tol = float("nan")
            step13_ok = False
            step13_refine_ok = False
            if len(J_history) >= max(2, m_window):
                recent = np.asarray(J_history[-max(2, m_window):], dtype=float)
                j_span = float(np.max(recent) - np.min(recent))
                j_scale = max(float(np.max(np.abs(recent))), abs(float(J_min)), 1e-30)
                rel_tol = j_span / j_scale
                step13_ok = bool(rel_tol < float(eps_J))
            if hard_cooldown_active:
                hard_shift_cooldown_left = max(0, int(hard_shift_cooldown_left) - 1)
            elif final_filter_cooldown_active:
                final_filter_post_cooldown_left = max(0, int(final_filter_post_cooldown_left) - 1)
                if final_filter_post_cooldown_left <= 0:
                    final_filter_post_pending_terminate = True
            elif refine_cooldown_active:
                refine_post_cooldown_left = max(0, int(refine_post_cooldown_left) - 1)
            elif recovery_active:
                hard_shift_recovery_left = max(0, int(hard_shift_recovery_left) - 1)
            elif lambda_plateau_cooldown_active:
                lambda_plateau_cooldown_left = max(0, int(lambda_plateau_cooldown_left) - 1)
                cfg["_lambda_v_plateau_accept_cooldown_left"] = int(lambda_plateau_cooldown_left)
        stage2_stall_diag_action = None
        stage2_stall_diag_allow_fallback = False
        stage2_stall_allows_fallback = True
        stage2_inactive_startup_fallback = False
        stage2_nonweak_stall_controller_handled = False
        stage2_weak_stall_controller_handled = False
        if not solver_fail:
            if line_search_stalled and isinstance(cfg.get("_stage2_last_stall_diagnosis", None), dict):
                _stall_diag = cfg.get("_stage2_last_stall_diagnosis", {})
                if ("it" not in _stall_diag) or (int(_stall_diag.get("it")) == int(it)):
                    stage2_stall_diag_action = str(_stall_diag.get("action", "ambiguous-hold"))
                    stage2_stall_diag_allow_fallback = bool(_stall_diag.get("watchdog_allow_fallback", False))
            post_nuc_takeover_active = stage2_post_nucleation_takeover_active(
                cfg, res_eval=res, it=it, hard_shift_only=hard_shift_only_active
            )
            if (
                bool(post_nuc_takeover_active)
                and bool(cfg.get("stage2_post_nucleation_takeover_suppress_fallback", True))
            ):
                if bool(stage2_stall_diag_allow_fallback) and MPI.rank(MPI.comm_world) == 0:
                    print("[stage2-post-nucleation-takeover] it=%03d: suppress rare fallback release; "
                          "continuous volume control still has takeover priority (vf=%.6f, target=%s)" %
                          (int(it), float(res.get("vf", 0.0)),
                           ("None" if current_vf_target(cfg) is None else "%.6f" % float(current_vf_target(cfg)))),
                          flush=True)
                stage2_stall_diag_allow_fallback = False
            if (
                line_search_stalled
                and (stage2_stall_diag_action == "inactive")
                and bool(cfg.get("stage2_enter_on_plateau_nucleation", True))
                and bool(cfg.get("stage2_volume_continuation_enabled", False))
                and (not hard_shift_only_active)
            ):
                _target_startup = current_vf_target(cfg)
                if _target_startup is not None:
                    stage2_inactive_startup_fallback = bool(
                        float(res.get("vf", 0.0))
                        > float(_target_startup) + stage_controller_tolerance(cfg)
                    )
            stage2_stall_allows_fallback = (
                (not line_search_stalled)
                or (stage2_stall_diag_action in (None, "weak-grow"))
                or bool(stage2_inactive_startup_fallback)
                or bool(stage2_stall_diag_allow_fallback)
            )
            include_in_plateau_windows = (
                (not cooldown_active)
                and (not recovery_active)
                and (not lambda_plateau_cooldown_active)
                and bool(stage2_stall_allows_fallback)
            )
            if include_in_plateau_windows:
                vf_history.append(float(res["vf"]))
            _vf_hist_keep = max(
                10,
                int(cfg.get("hard_shift_plateau_window", 5)) + 5,
                int(cfg.get("lambda_v_plateau_window", 8)) + 5,
            )
            if len(vf_history) > _vf_hist_keep:
                vf_history = vf_history[-_vf_hist_keep:]
            # Both plateau histories only track normal continuous-optimization steps.
            if include_in_plateau_windows:
                lambda_plateau_vf_history.append(float(res["vf"]))
            _lambda_vf_hist_keep = max(
                10,
                int(cfg.get("lambda_v_plateau_window", 8)) + 5,
            )
            if len(lambda_plateau_vf_history) > _lambda_vf_hist_keep:
                lambda_plateau_vf_history = lambda_plateau_vf_history[-_lambda_vf_hist_keep:]
            if (
                (not cooldown_active)
                and (not recovery_active)
                and (not hard_shift_only_reached)
                and (not lambda_plateau_cooldown_active)
                and bool(stage2_stall_allows_fallback)
            ):
                update_lambda_v_on_vf_plateau(cfg, res, lambda_plateau_vf_history, psi=lsf, M=l2_mass, it=it)
                lambda_plateau_cooldown_left = int(cfg.get("_lambda_v_plateau_accept_cooldown_left", lambda_plateau_cooldown_left))
            if (
                line_search_stalled
                and (stage2_stall_diag_action not in (None, "weak-grow", "inactive"))
                and stage2_volume_continuation_active(
                    cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                    hard_shift_only=hard_shift_only_active
                )
            ):
                handle_stage2_stall_volume_controller(
                    cfg, res, psi=lsf, M=l2_mass, it=it, reason="line-search-stall-nonweak"
                )
                stage2_nonweak_stall_controller_handled = True
            if (
                line_search_stalled
                and (stage2_stall_diag_action == "weak-grow")
                and isinstance(cfg.get("_stage2_last_stall_diagnosis", None), dict)
                and stage2_volume_continuation_active(
                    cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                    hard_shift_only=hard_shift_only_active
                )
            ):
                weak_streak = int(cfg.get("_stage2_last_stall_diagnosis", {}).get("watchdog_weak_streak", 1))
                weak_grow_before_dv = max(0, int(cfg.get("stage2_stall_weak_grow_before_dv_shrink", 2)))
                if weak_streak <= weak_grow_before_dv:
                    handle_stage2_stall_volume_controller(
                        cfg, res, psi=lsf, M=l2_mass, it=it, reason="line-search-stall-weak"
                    )
                    stage2_weak_stall_controller_handled = True
        force_lambda_plateau_nucleation = bool(cfg.get("_lambda_v_plateau_force_nucleation", False))
        if line_search_stalled and isinstance(cfg.get("_stage2_last_stall_diagnosis", None), dict):
            _stall_diag = cfg.get("_stage2_last_stall_diagnosis", {})
            _stall_diag_current = ("it" not in _stall_diag) or (int(_stall_diag.get("it")) == int(it))
            if (
                _stall_diag_current
                and str(_stall_diag.get("action", "ambiguous-hold")) != "weak-grow"
                and (not bool(_stall_diag.get("watchdog_allow_fallback", False)))
            ):
                force_lambda_plateau_nucleation = False
                cfg["_lambda_v_plateau_force_nucleation"] = False
        if hard_shift_only_reached and force_lambda_plateau_nucleation:
            force_lambda_plateau_nucleation = False
            cfg["_lambda_v_plateau_force_nucleation"] = False
        if force_lambda_plateau_nucleation:
            cfg["_lambda_v_plateau_force_nucleation"] = False
        hard_shift_final_target = float(current_postprocess_vf_final_target(cfg))
        hard_shift_final_tol = stage_success_tolerance(
            cfg, dv=float(cfg.get("vf_stage_dv_min", 0.002))
        )
        hard_shift_final_target_satisfied = (
            float(res.get("vf", 0.0)) <= hard_shift_final_target + hard_shift_final_tol
        )
        aggressive_uniform_active = bool(cfg.get("_uniform_hard_shift_aggressive_active", False))
        aggressive_uniform_max_steps = max(1, int(cfg.get("hard_shift_aggressive_max_steps", 50)))
        force_aggressive_uniform_shift = False
        postprocess_gap_default = max(1, int(cfg.get("postprocess_hard_shift_gap_iters", 20)))
        postprocess_min_gap_iters = max(
            1, int(cfg.get("postprocess_hard_shift_min_gap_iters", postprocess_gap_default))
        )
        postprocess_max_gap_iters = max(
            postprocess_min_gap_iters,
            int(cfg.get("postprocess_hard_shift_max_gap_iters", postprocess_gap_default)),
        )
        postprocess_plateau_window = max(
            2, int(cfg.get("postprocess_hard_shift_plateau_window", cfg.get("J_window", 5)))
        )
        postprocess_plateau_rel_tol = max(
            0.0, float(cfg.get("postprocess_hard_shift_plateau_rel_tol", cfg.get("eps_J", 2e-4)))
        )
        force_postprocess_uniform_shift = False
        postprocess_shift_gate_reason = None
        force_watchdog_fallback = (
            bool(line_search_stalled)
            and bool(stage2_stall_diag_allow_fallback)
            and bool(cfg.get("use_plateau_hard_shift", True))
        )
        if (
            postprocess_active
            and hard_shift_only_active
            and (not hard_shift_final_target_satisfied)
            and (not cooldown_active)
            and (not recovery_active)
            and (not lambda_plateau_cooldown_active)
        ):
            last_post_shift_it_raw = cfg.get("_last_postprocess_shift_it", None)
            if last_post_shift_it_raw is None:
                last_post_shift_it_raw = cfg.get("_postprocess_start_it", cfg.get("_last_uniform_hard_shift_it", None))
            try:
                postprocess_wait_age = int(it) - int(last_post_shift_it_raw)
            except Exception:
                postprocess_wait_age = postprocess_max_gap_iters
            jcmp_plateau_ok, jcmp_plateau_rel = postprocess_history_plateau(
                J_history,
                postprocess_plateau_window,
                postprocess_plateau_rel_tol,
                j_floor=cfg.get("J_min", 1e-12),
            )
            jmech_plateau_ok, jmech_plateau_rel = postprocess_history_plateau(
                cfg.get("_postprocess_j_mech_history", []),
                postprocess_plateau_window,
                postprocess_plateau_rel_tol,
                j_floor=cfg.get("J_min", 1e-12),
            )
            if postprocess_wait_age >= postprocess_max_gap_iters:
                force_postprocess_uniform_shift = True
                postprocess_shift_gate_reason = "max-wait"
            elif postprocess_wait_age >= postprocess_min_gap_iters:
                if bool(line_search_stalled):
                    force_postprocess_uniform_shift = True
                    postprocess_shift_gate_reason = "line-search-stall"
                elif bool(jmech_plateau_ok):
                    force_postprocess_uniform_shift = True
                    postprocess_shift_gate_reason = "J_mech-plateau"
                elif bool(jcmp_plateau_ok):
                    force_postprocess_uniform_shift = True
                    postprocess_shift_gate_reason = "J_compare-plateau"
                elif MPI.rank(MPI.comm_world) == 0:
                    print("[postprocess-hard-shift-gate] it=%03d: wait repair age=%d/%d (min=%d), J_mech_rel=%s, J_compare_rel=%s, stall=%s" %
                          (int(it), int(postprocess_wait_age), int(postprocess_max_gap_iters),
                           int(postprocess_min_gap_iters),
                           "nan" if not np.isfinite(jmech_plateau_rel) else "%.3e" % float(jmech_plateau_rel),
                           "nan" if not np.isfinite(jcmp_plateau_rel) else "%.3e" % float(jcmp_plateau_rel),
                           str(bool(line_search_stalled))))
            if force_postprocess_uniform_shift and MPI.rank(MPI.comm_world) == 0:
                print("[postprocess-hard-shift-gate] it=%03d: release hard shift by %s after age=%d (min=%d, max=%d, J_mech_rel=%s, J_compare_rel=%s, stall=%s)" %
                      (int(it), str(postprocess_shift_gate_reason), int(postprocess_wait_age),
                       int(postprocess_min_gap_iters), int(postprocess_max_gap_iters),
                       "nan" if not np.isfinite(jmech_plateau_rel) else "%.3e" % float(jmech_plateau_rel),
                       "nan" if not np.isfinite(jcmp_plateau_rel) else "%.3e" % float(jcmp_plateau_rel),
                       str(bool(line_search_stalled))))
        if aggressive_uniform_active and hard_shift_only_active and (not cooldown_active) and (not recovery_active) and (not lambda_plateau_cooldown_active):
            if not hard_shift_final_target_satisfied:
                last_uniform_shift_it_raw = cfg.get("_last_uniform_hard_shift_it", None)
                if last_uniform_shift_it_raw is None:
                    force_aggressive_uniform_shift = True
                else:
                    try:
                        force_aggressive_uniform_shift = (
                            int(it) - int(last_uniform_shift_it_raw) >= aggressive_uniform_max_steps
                        )
                    except Exception:
                        force_aggressive_uniform_shift = True
        generic_stall_hard_shift = bool(
            (not postprocess_active)
            and line_search_stalled
            and bool(cfg.get("use_plateau_hard_shift", True))
            and bool(stage2_stall_allows_fallback)
        )
        generic_watchdog_hard_shift = bool((not postprocess_active) and force_watchdog_fallback)
        if (not solver_fail) and (not hard_shift_final_target_satisfied) and (
            generic_stall_hard_shift
            or force_lambda_plateau_nucleation
            or generic_watchdog_hard_shift
            or force_aggressive_uniform_shift
            or force_postprocess_uniform_shift
        ):
            _plat_win = int(cfg.get("hard_shift_plateau_window", 5))
            _plat_tol = float(cfg.get("hard_shift_plateau_vf_tol", 1e-12))
            _hits_need = int(cfg.get("hard_shift_kappa_min_hits", 1))
            if (len(vf_history) >= _plat_win) or force_lambda_plateau_nucleation or generic_watchdog_hard_shift or force_postprocess_uniform_shift:
                if force_lambda_plateau_nucleation:
                    _vf_ref = float(res["vf"])
                    _same_plateau = True
                    plateau_last_vf = _vf_ref
                    plateau_kappa_min_hits = _hits_need
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[vf-hard-shift-check] plateau lambda second-hit -> force nucleation at vf=%.6f, %s" %
                              (float(_vf_ref), _format_stage_dv_state(cfg)))
                elif generic_watchdog_hard_shift:
                    _vf_ref = float(res["vf"])
                    _same_plateau = True
                    plateau_last_vf = _vf_ref
                    plateau_kappa_min_hits = _hits_need
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[vf-hard-shift-check] watchdog fallback: repeated non-weak stalls -> allow merit-gated hard fallback at vf=%.6f, %s" %
                              (float(_vf_ref), _format_stage_dv_state(cfg)))
                elif force_postprocess_uniform_shift:
                    _vf_ref = float(res["vf"])
                    _same_plateau = True
                    plateau_last_vf = _vf_ref
                    plateau_kappa_min_hits = _hits_need
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[postprocess-hard-shift] it=%03d: force uniform hard shift after %d relaxation iterations (vf=%.6f, final_target=%.6f)" %
                              (int(it), int(postprocess_wait_age), float(_vf_ref), float(hard_shift_final_target)))
                elif force_aggressive_uniform_shift:
                    _vf_ref = float(res["vf"])
                    _same_plateau = True
                    plateau_last_vf = _vf_ref
                    plateau_kappa_min_hits = _hits_need
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[vf-hard-shift-aggressive] it=%03d: max %d steps since last aggressive shift reached -> force uniform hard shift (vf=%.6f)" %
                              (int(it), int(aggressive_uniform_max_steps), float(_vf_ref)))
                else:
                    _vf_tail = vf_history[-_plat_win:]
                    _vf_ref = float(_vf_tail[-1])
                    _same_plateau = max(abs(v - _vf_ref) for v in _vf_tail) <= _plat_tol
                if _same_plateau:
                    dv_now = float(current_stage_dv(cfg) if (current_stage_dv(cfg) is not None) else cfg.get("vf_stage_dv0", 0.03))
                    dv_min = float(cfg.get("vf_stage_dv_min", 0.002))
                    # Keep the dv_min gate for the regular stage-controlled regime,
                    # but once low-vf hard-shift-only mode is active, allow plateau
                    # hits to accumulate again so uniform hard shift can still fire.
                    dv_ready_for_nucleation = (
                        force_lambda_plateau_nucleation
                        or generic_watchdog_hard_shift
                        or force_aggressive_uniform_shift
                        or force_postprocess_uniform_shift
                        or hard_shift_only_active
                        or (dv_now <= dv_min + 1e-15)
                    )
                    if (
                        dv_ready_for_nucleation
                        and (not hard_shift_only_active)
                        and bool(cfg.get("stage2_rare_nucleation_enabled", True))
                        and stage2_volume_continuation_active(cfg, vf=_vf_ref, vf_target=current_vf_target(cfg), hard_shift_only=hard_shift_only_active)
                    ):
                        last_nuc_for_gap = cfg.get("_last_nucleation_it", None)
                        min_gap_iters = max(0, int(cfg.get("stage2_min_nucleation_gap_iters", 60)))
                        block_until = int(cfg.get("_stage2_nucleation_block_until_it", -1))
                        gap_ok = True
                        if last_nuc_for_gap is not None:
                            try:
                                gap_ok = (int(it) - int(last_nuc_for_gap)) >= min_gap_iters
                            except Exception:
                                gap_ok = True
                        block_ok = int(it) >= block_until
                        if (not gap_ok) or (not block_ok):
                            dv_ready_for_nucleation = False
                            if MPI.rank(MPI.comm_world) == 0:
                                print("[stage2-nucleation-defer] it=%03d: rare-nucleation cooldown active (gap_ok=%s, block_ok=%s, last=%s, min_gap=%d, block_until=%d)" %
                                      (int(it), str(bool(gap_ok)), str(bool(block_ok)), str(last_nuc_for_gap), int(min_gap_iters), int(block_until)))
                            if bool(generic_watchdog_hard_shift):
                                release_it = int(it) + 1
                                if last_nuc_for_gap is not None:
                                    try:
                                        last_nuc_i = int(last_nuc_for_gap)
                                        release_it = max(release_it, last_nuc_i + int(min_gap_iters))
                                        freeze_steps_i = max(0, int(cfg.get("post_nucleation_freeze_steps", 0)))
                                        if freeze_steps_i > 0:
                                            release_it = max(release_it, last_nuc_i + freeze_steps_i + 1)
                                    except Exception:
                                        pass
                                if int(block_until) >= 0:
                                    release_it = max(release_it, int(block_until))
                                if activate_stage2_stall_wait(
                                    cfg,
                                    it,
                                    release_it,
                                    reason="rare-nucleation-cooldown",
                                    vf=_vf_ref,
                                    vf_target=current_vf_target(cfg),
                                ) and MPI.rank(MPI.comm_world) == 0:
                                    print("[stage2-stall-wait] it=%03d armed: rare fallback already allowed but cooldown blocks nucleation until it=%d; future repeated stalls will skip full line-search" %
                                          (int(it), int(cfg.get("_stage2_stall_wait_until_it", release_it))),
                                          flush=True)
                    if not (force_lambda_plateau_nucleation or generic_watchdog_hard_shift or force_aggressive_uniform_shift or force_postprocess_uniform_shift):
                        if (not hard_shift_only_active) and (not dv_ready_for_nucleation):
                            plateau_kappa_min_hits = 0
                        elif (plateau_last_vf is None) or (abs(float(plateau_last_vf) - _vf_ref) > _plat_tol):
                            plateau_kappa_min_hits = 0
                    plateau_last_vf = _vf_ref
                    if not (force_lambda_plateau_nucleation or generic_watchdog_hard_shift or force_aggressive_uniform_shift or force_postprocess_uniform_shift) and dv_ready_for_nucleation:
                        plateau_kappa_min_hits += 1
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[vf-hard-shift-check] plateau detected: vf≈%.6f over last %d iterations, kappa_min_hits=%d/%d, %s" %
                              (_vf_ref, _plat_win, plateau_kappa_min_hits, _hits_need, _format_stage_dv_state(cfg, dv_now_raw=dv_now)))
                    if plateau_kappa_min_hits >= _hits_need:
                        if (not dv_ready_for_nucleation):
                            if MPI.rank(MPI.comm_world) == 0:
                                print("[vf-hard-shift-check] plateau detected but defer nucleation: %s -> let stage controller shrink dv first" %
                                      _format_stage_dv_state(cfg, dv_now_raw=dv_now))
                        else:
                            hard_shift_attempted = True
                            hard_shift_switch_vf = float(cfg.get("hard_shift_switch_to_shift_vf", 0.20))
                            if (
                                (not hard_shift_only_active)
                                and bool(cfg.get("stage2_enter_on_plateau_nucleation", True))
                                and bool(cfg.get("stage2_volume_continuation_enabled", False))
                                and (float(_vf_ref) > float(cfg.get("stage2_volume_end_vf", hard_shift_switch_vf)) + 1e-15)
                            ):
                                force_stage2_volume_continuation_start(
                                    cfg, res, it=it, reason="mechanical-plateau-hard-nucleation"
                                )
                                update_lambda_v_from_stage_state(
                                    cfg, res, psi=lsf, M=l2_mass, it=it, reason="stage2-plateau-trigger"
                                )
                            force_uniform_by_rebound = False
                            if (
                                (not hard_shift_only_active)
                                and bool(cfg.get("hard_shift_rebound_early_shift_enabled", True))
                            ):
                                last_nucleation_post_vf_raw = cfg.get("_last_nucleation_post_vf", None)
                                if last_nucleation_post_vf_raw is not None:
                                    try:
                                        last_nucleation_post_vf = float(last_nucleation_post_vf_raw)
                                    except Exception:
                                        last_nucleation_post_vf = None
                                    if (last_nucleation_post_vf is not None) and np.isfinite(last_nucleation_post_vf):
                                        rebound_hits = max(0, int(cfg.get("_nucleation_rebound_hits", 0)))
                                        rebound_hits_needed = max(1, int(cfg.get("hard_shift_rebound_hits_needed", 2)))
                                        rebound_trigger_vf = float(cfg.get("hard_shift_rebound_trigger_vf", 0.4))
                                        rebound_gate_open = (float(_vf_ref) <= rebound_trigger_vf + 1e-15)
                                        if rebound_gate_open and (float(_vf_ref) > last_nucleation_post_vf + 1e-15):
                                            rebound_hits = max(0, int(cfg.get("_nucleation_rebound_hits", 0))) + 1
                                            cfg["_nucleation_rebound_hits"] = int(rebound_hits)
                                            if MPI.rank(MPI.comm_world) == 0:
                                                print("[vf-hard-shift-rebound] it=%03d: pre-nucleation vf rebound detected (vf_pre=%.6f > last_nucleation_vf=%.6f, trigger_vf=%.6f), consecutive_hits=%d/%d" %
                                                      (int(it), float(_vf_ref), float(last_nucleation_post_vf), float(rebound_trigger_vf), int(rebound_hits), int(rebound_hits_needed)))
                                            if rebound_hits >= rebound_hits_needed:
                                                force_uniform_by_rebound = True
                                                cfg["_hard_shift_force_uniform_early"] = True
                                                cfg["_hard_shift_force_uniform_early_announced"] = False
                                                if MPI.rank(MPI.comm_world) == 0:
                                                    print("[vf-hard-shift-rebound] it=%03d: rebound rule reached %d hits -> force early uniform-shift path" %
                                                          (int(it), int(rebound_hits)))
                                        else:
                                            if rebound_hits != 0:
                                                cfg["_nucleation_rebound_hits"] = 0
                                                if MPI.rank(MPI.comm_world) == 0:
                                                    if not rebound_gate_open:
                                                        print("[vf-hard-shift-rebound] it=%03d: vf=%.6f above trigger_vf=%.6f -> reset consecutive rebound hits" %
                                                              (int(it), float(_vf_ref), float(rebound_trigger_vf)))
                                                    else:
                                                        print("[vf-hard-shift-rebound] it=%03d: vf=%.6f <= last_nucleation_vf=%.6f -> reset consecutive rebound hits" %
                                                              (int(it), float(_vf_ref), float(last_nucleation_post_vf)))
                            last_nucleation_it_raw = cfg.get("_last_nucleation_it", None)
                            post_nucleation_freeze_steps = max(0, int(cfg.get("post_nucleation_freeze_steps", 0)))
                            post_nucleation_freeze_active = False
                            post_nucleation_freeze_age = None
                            if (post_nucleation_freeze_steps > 0) and (last_nucleation_it_raw is not None):
                                try:
                                    post_nucleation_freeze_age = int(it) - int(last_nucleation_it_raw)
                                    post_nucleation_freeze_active = (
                                        post_nucleation_freeze_age > 0
                                        and post_nucleation_freeze_age <= post_nucleation_freeze_steps
                                    )
                                except Exception:
                                    post_nucleation_freeze_age = None
                                    post_nucleation_freeze_active = False
                            aggressive_uniform_active = bool(cfg.get("_uniform_hard_shift_aggressive_active", False))
                            uniform_shift_factor = float(
                                cfg.get("hard_shift_aggressive_shift_factor", cfg.get("hard_shift_shift_factor", 0.95))
                                if aggressive_uniform_active
                                else cfg.get("hard_shift_shift_factor", 0.95)
                            )
                            if hard_shift_only_active or force_uniform_by_rebound:
                                skip_uniform_shift_due_to_exit = False
                                uniform_exit_override = bool(cfg.get("_uniform_hard_shift_exit_override", False))
                                if uniform_exit_override:
                                    plateau_kappa_min_hits = 0
                                    plateau_last_vf = _vf_ref
                                    if MPI.rank(MPI.comm_world) == 0:
                                        print("[vf-hard-shift-safety] it=%03d: uniform hard-shift exit override already active -> skip further uniform hard shifts and use J_merit/theta convergence gates" % int(it))
                                    skip_uniform_shift_due_to_exit = True
                                if bool(cfg.get("hard_shift_uniform_exit_enabled", True)) and (not aggressive_uniform_active):
                                    last_uniform_post_vf_raw = cfg.get("_last_uniform_hard_shift_post_vf", None)
                                    if last_uniform_post_vf_raw is not None:
                                        try:
                                            last_uniform_post_vf = float(last_uniform_post_vf_raw)
                                        except Exception:
                                            last_uniform_post_vf = None
                                        if (last_uniform_post_vf is not None) and np.isfinite(last_uniform_post_vf):
                                            uniform_exit_hits = max(0, int(cfg.get("_uniform_hard_shift_exit_hits", 0)))
                                            uniform_exit_need = max(1, int(cfg.get("hard_shift_uniform_exit_consecutive_hits", 3)))
                                            uniform_exit_tol = float(cfg.get("hard_shift_plateau_vf_tol", 2e-4))
                                            if float(_vf_ref) > float(last_uniform_post_vf) + uniform_exit_tol:
                                                uniform_exit_hits += 1
                                                cfg["_uniform_hard_shift_exit_hits"] = int(uniform_exit_hits)
                                                if MPI.rank(MPI.comm_world) == 0:
                                                    print("[vf-hard-shift-rebound] it=%03d: pre-shift vf rebound detected (vf_pre=%.6f > last_uniform_post_vf=%.6f, tol=%.3e), consecutive_hits=%d/%d" %
                                                          (int(it), float(_vf_ref), float(last_uniform_post_vf), float(uniform_exit_tol), int(uniform_exit_hits), int(uniform_exit_need)))
                                                if uniform_exit_hits >= uniform_exit_need:
                                                    cfg["_uniform_hard_shift_aggressive_active"] = True
                                                    aggressive_uniform_active = True
                                                    uniform_shift_factor = float(cfg.get("hard_shift_aggressive_shift_factor", 0.8))
                                                    cfg["_uniform_hard_shift_exit_override"] = False
                                                    plateau_kappa_min_hits = 0
                                                    plateau_last_vf = _vf_ref
                                                    if MPI.rank(MPI.comm_world) == 0:
                                                        print("[vf-hard-shift-aggressive] it=%03d: reached %d consecutive rebound hits -> enter aggressive uniform hard shift (shift_factor=%.3f, max_steps=%d)" %
                                                              (int(it), int(uniform_exit_need), float(uniform_shift_factor), int(cfg.get("hard_shift_aggressive_max_steps", 50))))
                                            else:
                                                if uniform_exit_hits != 0:
                                                    cfg["_uniform_hard_shift_exit_hits"] = 0
                                                    if MPI.rank(MPI.comm_world) == 0:
                                                        print("[vf-hard-shift-rebound] it=%03d: pre-shift vf=%.6f <= last_uniform_post_vf=%.6f (+tol %.3e) -> reset uniform hard-shift rebound hits" %
                                                              (int(it), float(_vf_ref), float(last_uniform_post_vf), float(uniform_exit_tol)))
                                if not skip_uniform_shift_due_to_exit:
                                    hard_shift_mode = "uniform-shift"
                                    if postprocess_active:
                                        _vf_target_shift = postprocess_hard_shift_target_vf(cfg, _vf_ref)
                                    else:
                                        _vf_target_shift = max(0.0, float(uniform_shift_factor) * _vf_ref)
                                    lsf_shifted, hard_shift_info, vf_shift_exact = enforce_specific_volume_fraction_by_shift(
                                        lsf, mesh, cfg, _vf_target_shift
                                    )
                                else:
                                    hard_shift_attempted = False
                                    hard_shift_mode = "uniform-shift-skipped"
                                    hard_shift_info = {"active": False, "target_reached": False}
                                    vf_shift_exact = float(_vf_ref)
                            elif force_lambda_plateau_nucleation:
                                if post_nucleation_freeze_active:
                                    hard_shift_mode = "nucleation-freeze-lambda-boost"
                                    _vf_target_shift = float(_vf_ref)
                                    vf_shift_exact = float(_vf_ref)
                                    hard_shift_info = {
                                        "active": False,
                                        "target_reached": False,
                                        "selection_mode": "post-nucleation-freeze-lambda-boost",
                                        "vf_before": float(_vf_ref),
                                        "vf_target": float(_vf_ref),
                                        "vf_after": float(_vf_ref),
                                    }
                                else:
                                    hard_shift_mode = "nucleation"
                                    _hard_shift_factor = float(cfg.get("hard_shift_factor", 0.98))
                                    if stage2_volume_continuation_active(cfg, vf=_vf_ref, vf_target=current_vf_target(cfg), hard_shift_only=hard_shift_only_active):
                                        _vf_target_shift = stage2_nucleation_target_vf(cfg, _vf_ref)
                                    else:
                                        _vf_target_shift = max(0.0, _hard_shift_factor * _vf_ref)
                                    lsf_shifted, hard_shift_info, vf_shift_exact = enforce_specific_volume_fraction(
                                        lsf, mesh, cfg, _vf_target_shift, ranking_field=g_rank_field, entity_map=nucleation_map
                                    )
                            else:
                                if float(_vf_ref) < hard_shift_switch_vf:
                                    hard_shift_mode = "uniform-shift"
                                    if postprocess_active:
                                        _vf_target_shift = postprocess_hard_shift_target_vf(cfg, _vf_ref)
                                    else:
                                        _vf_target_shift = max(0.0, float(uniform_shift_factor) * _vf_ref)
                                    lsf_shifted, hard_shift_info, vf_shift_exact = enforce_specific_volume_fraction_by_shift(
                                        lsf, mesh, cfg, _vf_target_shift
                                    )
                                else:
                                    if post_nucleation_freeze_active:
                                        hard_shift_mode = "nucleation-freeze-lambda-boost"
                                        _vf_target_shift = float(_vf_ref)
                                        vf_shift_exact = float(_vf_ref)
                                        hard_shift_info = {
                                            "active": False,
                                            "target_reached": False,
                                            "selection_mode": "post-nucleation-freeze-lambda-boost",
                                            "vf_before": float(_vf_ref),
                                            "vf_target": float(_vf_ref),
                                            "vf_after": float(_vf_ref),
                                        }
                                    else:
                                        hard_shift_mode = "nucleation"
                                        _hard_shift_factor = float(cfg.get("hard_shift_factor", 0.98))
                                        if stage2_volume_continuation_active(cfg, vf=_vf_ref, vf_target=current_vf_target(cfg), hard_shift_only=hard_shift_only_active):
                                            _vf_target_shift = stage2_nucleation_target_vf(cfg, _vf_ref)
                                        else:
                                            _vf_target_shift = max(0.0, _hard_shift_factor * _vf_ref)
                                        lsf_shifted, hard_shift_info, vf_shift_exact = enforce_specific_volume_fraction(
                                            lsf, mesh, cfg, _vf_target_shift, ranking_field=g_rank_field, entity_map=nucleation_map
                                        )
                            if MPI.rank(MPI.comm_world) == 0:
                                print("[vf-hard-shift-mode] it=%03d: mode=%s (vf=%.6f, switch_vf=%.6f)" %
                                      (it, hard_shift_mode, float(_vf_ref), hard_shift_switch_vf))
                            if str(hard_shift_mode) == "uniform-shift-skipped":
                                if MPI.rank(MPI.comm_world) == 0:
                                    print("[vf-hard-shift-safety] it=%03d: skip current uniform hard shift and proceed with final-target override based convergence checks" % int(it))
                                plateau_kappa_min_hits = 0
                                plateau_last_vf = _vf_ref
                                hard_shift_attempted = False
                            elif str(hard_shift_mode) == "nucleation-freeze-lambda-boost":
                                lambda_old_freeze = _clamp_lambda_v_to_cfg(
                                    cfg, cfg.get("lambda_v", 0.0), include_plateau_cap=False
                                )
                                boost_factor_freeze = max(
                                    1.0, float(cfg.get("post_nucleation_lambda_v_boost_factor", 1.2))
                                )
                                lambda_new_freeze = _clamp_lambda_v_to_cfg(
                                    cfg, lambda_old_freeze * boost_factor_freeze, include_plateau_cap=False
                                )
                                cfg["lambda_v"] = float(lambda_new_freeze)
                                plateau_kappa_min_hits = 0
                                plateau_last_vf = _vf_ref
                                hard_shift_attempted = False
                                if MPI.rank(MPI.comm_world) == 0:
                                    age_msg = "unknown" if post_nucleation_freeze_age is None else str(int(post_nucleation_freeze_age))
                                    print("[vf-hard-nucleation-freeze] it=%03d: suppress nucleation during post-nucleation freeze window (age=%s/%d); lambda_v %.6e -> %.6e (x%.3f)" %
                                          (int(it), age_msg, int(post_nucleation_freeze_steps),
                                           float(lambda_old_freeze), float(lambda_new_freeze), float(boost_factor_freeze)))
                            else:
                                if MPI.rank(MPI.comm_world) == 0:
                                    print("[vf-hard-nucleation] it=%03d: mode=%s, q_band=[%.2f, %.2f], d_band=[%.4f, %.4f], expand_steps=%d, candidate_points=%d, selected_points=%d, candidate_entities=%d, selected_entities=%d, vf %.6f -> target %.6f, predicted_vf %.6f, reached_target=%s, used_all_candidates=%s" %
                                          (it,
                                           str(hard_shift_info.get("selection_mode", "top-g-global")),
                                           float(hard_shift_info.get("q_lo", 0.0)),
                                           float(hard_shift_info.get("q_hi", 0.0)),
                                           float(hard_shift_info.get("d_lo", 0.0)),
                                           float(hard_shift_info.get("d_hi", 0.0)),
                                           int(hard_shift_info.get("d_expand_steps", 0)),
                                           int(hard_shift_info.get("n_candidates", 0)),
                                           int(hard_shift_info.get("n_selected", 0)),
                                           int(hard_shift_info.get("n_candidate_entities", 0)),
                                           int(hard_shift_info.get("n_selected_entities", 0)),
                                           float(hard_shift_info.get("vf_before", _vf_ref)),
                                           float(hard_shift_info.get("vf_target", _vf_target_shift)),
                                           float(hard_shift_info.get("vf_after", _vf_ref)),
                                           "yes" if bool(hard_shift_info.get("target_reached", False)) else "no",
                                           "yes" if bool(hard_shift_info.get("used_all_candidates", False)) else "no"))
                                hard_shift_failure_reason = str(hard_shift_info.get("failure_reason", "") or "")
                                if not bool(hard_shift_info.get("active", False)):
                                    err_msg = ("[vf-hard-nucleation-error] it=%03d: no viable hard nucleation move remains "
                                               "(reason=%s, vf=%.6f, target=%.6f, q_band=[%.2f, %.2f], "
                                               "candidate_entities=%d, candidate_points=%d)") % (
                                                   it,
                                                   hard_shift_failure_reason or "inactive",
                                                   float(hard_shift_info.get("vf_before", _vf_ref)),
                                                   float(hard_shift_info.get("vf_target", _vf_target_shift)),
                                                   float(hard_shift_info.get("q_lo", 0.0)),
                                                   float(hard_shift_info.get("q_hi", 0.0)),
                                                   int(hard_shift_info.get("n_candidate_entities", 0)),
                                                   int(hard_shift_info.get("n_candidates", 0)),
                                               )
                                    if MPI.rank(MPI.comm_world) == 0:
                                        print(err_msg)
                                    raise RuntimeError(err_msg)
                                if not bool(hard_shift_info.get("target_reached", False)):
                                    miss_msg = ("[vf-hard-nucleation-miss] it=%03d: hard nucleation exhausted fallback bands "
                                                "and missed vf target (reason=%s, vf=%.6f, target=%.6f, predicted_vf=%.6f, "
                                                "q_band=[%.2f, %.2f], candidate_entities=%d, candidate_points=%d, "
                                                "selected_entities=%d, selected_points=%d)") % (
                                                   it,
                                                   hard_shift_failure_reason or "target-not-reached",
                                                   float(hard_shift_info.get("vf_before", _vf_ref)),
                                                   float(hard_shift_info.get("vf_target", _vf_target_shift)),
                                                   float(hard_shift_info.get("vf_after", _vf_ref)),
                                                   float(hard_shift_info.get("q_lo", 0.0)),
                                                   float(hard_shift_info.get("q_hi", 0.0)),
                                                   int(hard_shift_info.get("n_candidate_entities", 0)),
                                                   int(hard_shift_info.get("n_candidates", 0)),
                                                   int(hard_shift_info.get("n_selected_entities", 0)),
                                                   int(hard_shift_info.get("n_selected", 0)),
                                               )
                                    allow_partial_target = bool(cfg.get("hard_shift_allow_partial_target", True))
                                    if not allow_partial_target:
                                        if MPI.rank(MPI.comm_world) == 0:
                                            print(miss_msg.replace("[vf-hard-nucleation-miss]", "[vf-hard-nucleation-error]"))
                                        raise RuntimeError(miss_msg)
                                    if MPI.rank(MPI.comm_world) == 0:
                                        print(miss_msg)
                                        print("[vf-hard-nucleation-warning] it=%03d: accept best reachable hard-nucleation move and continue" % it)
                                if bool(hard_shift_info.get("active", False)):
                                    if use_symmetry_parameterization and (symmetry_map is not None):
                                        lsf_shifted = expand_symmetry_to_full(lsf_shifted, symmetry_map)
                                    _lsf_norm_hard = renormalize_lsf_inplace(lsf_shifted, dx)
                                    hard_reject_res = None
                                    lsf_shifted, res_hard, solver_fail_hard = evaluate_trial_with_pre_jacobi_filter(
                                        lsf_shifted,
                                        context="[it %03d hard-vf-nucleation]" % it,
                                        alpha_eval=alpha_now,
                                        vf_ref=_vf_ref,
                                        apply_hard_vf=False,
                                    )
                                    if not solver_fail_hard:
                                        hard_accept_ok = True
                                        if (
                                            str(hard_shift_mode) == "nucleation"
                                            and stage2_volume_continuation_active(cfg, vf=_vf_ref, vf_target=current_vf_target(cfg), hard_shift_only=hard_shift_only_active)
                                        ):
                                            refresh_volume_merit_inplace(cfg, res, hard_shift_only=hard_shift_only_active)
                                            refresh_volume_merit_inplace(cfg, res_hard, hard_shift_only=hard_shift_only_active)
                                            hard_merit_old = float(res.get("J_compare", res.get("J", 0.0)))
                                            hard_merit_new = float(res_hard.get("J_compare", res_hard.get("J", 0.0)))
                                            hard_merit_scale = max(abs(hard_merit_old), 1e-12)
                                            hard_merit_tol_base = max(0.0, float(cfg.get("stage2_nucleation_merit_relax", 0.0)))
                                            auto_small_drop = max(
                                                0.0,
                                                float(cfg.get("stage2_nucleation_merit_relax_auto_small_drop", 0.0)),
                                            )
                                            auto_merit_tol = max(
                                                hard_merit_tol_base,
                                                float(cfg.get("stage2_nucleation_merit_relax_auto_max", hard_merit_tol_base)),
                                            )
                                            auto_merit_enabled = (
                                                bool(cfg.get("stage2_nucleation_merit_relax_auto_enabled", False))
                                                and bool(generic_watchdog_hard_shift)
                                            )

                                            def _nucleation_merit_tol(drop_abs):
                                                tol_now = hard_merit_tol_base
                                                if auto_merit_enabled and float(drop_abs) <= auto_small_drop + 1e-15:
                                                    tol_now = max(tol_now, auto_merit_tol)
                                                return tol_now

                                            hard_drop_abs_eval = max(
                                                0.0,
                                                float(_vf_ref) - float(res_hard.get("vf", vf_shift_exact)),
                                            )
                                            hard_merit_tol = _nucleation_merit_tol(hard_drop_abs_eval)
                                            hard_merit_allow = hard_merit_old + hard_merit_tol * hard_merit_scale
                                            hard_stage_active = stage2_volume_continuation_active(
                                                cfg, vf=_vf_ref, vf_target=current_vf_target(cfg),
                                                hard_shift_only=hard_shift_only_active
                                            )
                                            hard_mech_cap_ok, hard_mech_cap_info = stage2_mechanical_step_cap_ok(
                                                cfg, res, res_hard, hard_stage_active
                                            )
                                            if (hard_merit_new > hard_merit_allow) or (not hard_mech_cap_ok):
                                                hard_accept_ok = False
                                                hard_reject_res = res_hard
                                                retry_max_trials = max(1, int(cfg.get("stage2_nucleation_max_trials", 1)))
                                                retry_shrink = min(1.0, max(1e-6, float(cfg.get("stage2_nucleation_retry_drop_factor", 0.5))))
                                                retry_next_batch = bool(cfg.get("stage2_nucleation_retry_next_batch", True))
                                                retry_min_drop = max(0.0, float(cfg.get("stage2_nucleation_retry_min_abs_drop", 0.0)))
                                                retry_rank_offset = 0
                                                if retry_next_batch:
                                                    retry_rank_offset = max(
                                                        0, int(hard_shift_info.get("n_selected_entities", 0))
                                                    )
                                                retry_base_drop = max(0.0, float(_vf_ref) - float(_vf_target_shift))
                                                if MPI.rank(MPI.comm_world) == 0:
                                                    if hard_merit_new > hard_merit_allow:
                                                        print("[stage2-nucleation-retry] it=%03d trial=1/%d rejected by merit: %.6e > allow %.6e (old=%.6e, vf %.6f->%.6f)" %
                                                              (int(it), int(retry_max_trials), float(hard_merit_new), float(hard_merit_allow),
                                                               float(hard_merit_old), float(_vf_ref), float(res_hard.get("vf", vf_shift_exact))),
                                                              flush=True)
                                                    else:
                                                        print_stage2_mechanical_step_cap_reject(
                                                            "hard-vf-nucleation trial=1/%d" % int(retry_max_trials),
                                                            hard_mech_cap_info,
                                                            action="retry smaller hard move",
                                                        )
                                                for retry_trial in range(2, retry_max_trials + 1):
                                                    retry_drop = retry_base_drop * (retry_shrink ** (retry_trial - 1))
                                                    if retry_drop < retry_min_drop or retry_drop <= 0.0:
                                                        if MPI.rank(MPI.comm_world) == 0:
                                                            print("[stage2-nucleation-retry] it=%03d trial=%d/%d skipped: retry_drop=%.6e below min %.6e" %
                                                                  (int(it), int(retry_trial), int(retry_max_trials),
                                                                   float(retry_drop), float(retry_min_drop)), flush=True)
                                                        break
                                                    vf_stage_now = current_vf_target(cfg)
                                                    retry_target = float(_vf_ref) - float(retry_drop)
                                                    if vf_stage_now is not None:
                                                        retry_target = max(float(vf_stage_now), retry_target)
                                                    if retry_target >= float(_vf_ref) - 1e-15:
                                                        continue
                                                    lsf_retry, info_retry, vf_retry_exact = enforce_specific_volume_fraction(
                                                        lsf,
                                                        mesh,
                                                        cfg,
                                                        retry_target,
                                                        ranking_field=g_rank_field,
                                                        entity_map=nucleation_map,
                                                        rank_offset_entities=retry_rank_offset,
                                                    )
                                                    if MPI.rank(MPI.comm_world) == 0:
                                                        print("[stage2-nucleation-retry] it=%03d trial=%d/%d: target=%.6f, rank_offset=%d(applied=%d), selected_entities=%d, selected_points=%d, predicted_vf=%.6f, reached_target=%s" %
                                                              (int(it), int(retry_trial), int(retry_max_trials), float(retry_target),
                                                               int(retry_rank_offset), int(info_retry.get("rank_offset_applied", 0)),
                                                               int(info_retry.get("n_selected_entities", 0)),
                                                               int(info_retry.get("n_selected", 0)),
                                                               float(info_retry.get("vf_after", _vf_ref)),
                                                               "yes" if bool(info_retry.get("target_reached", False)) else "no"),
                                                              flush=True)
                                                    if not bool(info_retry.get("active", False)):
                                                        if retry_next_batch:
                                                            retry_rank_offset += max(
                                                                1, int(info_retry.get("rank_offset_applied", 0))
                                                            )
                                                        continue
                                                    if (not bool(info_retry.get("target_reached", False))) and (
                                                        not bool(cfg.get("hard_shift_allow_partial_target", True))
                                                    ):
                                                        if retry_next_batch:
                                                            retry_rank_offset += max(
                                                                1, int(info_retry.get("n_selected_entities", 0))
                                                            )
                                                        continue
                                                    if use_symmetry_parameterization and (symmetry_map is not None):
                                                        lsf_retry = expand_symmetry_to_full(lsf_retry, symmetry_map)
                                                    lsf_norm_retry = renormalize_lsf_inplace(lsf_retry, dx)
                                                    lsf_retry, res_retry, solver_fail_retry = evaluate_trial_with_pre_jacobi_filter(
                                                        lsf_retry,
                                                        context="[it %03d hard-vf-nucleation-retry-%d]" % (it, retry_trial),
                                                        alpha_eval=alpha_now,
                                                        vf_ref=_vf_ref,
                                                        apply_hard_vf=False,
                                                    )
                                                    if solver_fail_retry:
                                                        if MPI.rank(MPI.comm_world) == 0:
                                                            print("[stage2-nucleation-retry] it=%03d trial=%d/%d solver failed; continue to next retry if available" %
                                                                  (int(it), int(retry_trial), int(retry_max_trials)), flush=True)
                                                        if retry_next_batch:
                                                            retry_rank_offset += max(
                                                                1, int(info_retry.get("n_selected_entities", 0))
                                                            )
                                                        continue
                                                    refresh_volume_merit_inplace(cfg, res_retry, hard_shift_only=hard_shift_only_active)
                                                    retry_merit_new = float(res_retry.get("J_compare", res_retry.get("J", 0.0)))
                                                    retry_drop_abs_eval = max(
                                                        0.0,
                                                        float(_vf_ref) - float(res_retry.get("vf", vf_retry_exact)),
                                                    )
                                                    retry_merit_tol = _nucleation_merit_tol(retry_drop_abs_eval)
                                                    retry_merit_allow = hard_merit_old + retry_merit_tol * hard_merit_scale
                                                    retry_stage_active = stage2_volume_continuation_active(
                                                        cfg, vf=_vf_ref, vf_target=current_vf_target(cfg),
                                                        hard_shift_only=hard_shift_only_active
                                                    )
                                                    retry_mech_cap_ok, retry_mech_cap_info = stage2_mechanical_step_cap_ok(
                                                        cfg, res, res_retry, retry_stage_active
                                                    )
                                                    if retry_merit_new <= retry_merit_allow and retry_mech_cap_ok:
                                                        lsf_shifted = lsf_retry
                                                        hard_shift_info = info_retry
                                                        vf_shift_exact = float(vf_retry_exact)
                                                        res_hard = res_retry
                                                        _lsf_norm_hard = float(lsf_norm_retry)
                                                        hard_merit_new = float(retry_merit_new)
                                                        hard_merit_allow = float(retry_merit_allow)
                                                        hard_merit_tol = float(retry_merit_tol)
                                                        hard_accept_ok = True
                                                        if MPI.rank(MPI.comm_world) == 0:
                                                            print("[stage2-nucleation-retry] it=%03d trial=%d/%d accepted by merit: %.6e <= allow %.6e" %
                                                                  (int(it), int(retry_trial), int(retry_max_trials),
                                                                   float(retry_merit_new), float(retry_merit_allow)), flush=True)
                                                        break
                                                    hard_merit_new = float(retry_merit_new)
                                                    hard_merit_allow = float(retry_merit_allow)
                                                    hard_merit_tol = float(retry_merit_tol)
                                                    hard_reject_res = res_retry
                                                    if MPI.rank(MPI.comm_world) == 0:
                                                        if retry_merit_new > retry_merit_allow:
                                                            print("[stage2-nucleation-retry] it=%03d trial=%d/%d rejected by merit: %.6e > allow %.6e" %
                                                                  (int(it), int(retry_trial), int(retry_max_trials),
                                                                   float(retry_merit_new), float(retry_merit_allow)), flush=True)
                                                        else:
                                                            print_stage2_mechanical_step_cap_reject(
                                                                "hard-vf-nucleation-retry trial=%d/%d" % (int(retry_trial), int(retry_max_trials)),
                                                                retry_mech_cap_info,
                                                                action="continue retry",
                                                            )
                                                    if retry_next_batch:
                                                        retry_rank_offset += max(
                                                            1, int(info_retry.get("n_selected_entities", 0))
                                                        )
                                                if not hard_accept_ok:
                                                    cfg["_stage2_nucleation_block_until_it"] = int(it) + max(
                                                        0, int(cfg.get("stage2_nucleation_reject_cooldown_iters", 30))
                                                    )
                                                    hard_diag_res = hard_reject_res if hard_reject_res is not None else res_hard
                                                    diagnose_stage2_line_search_stall(
                                                        cfg,
                                                        res,
                                                        [dict(
                                                            kappa=0.0,
                                                            J_compare=float(hard_diag_res.get("J_compare", hard_diag_res.get("J", 0.0))),
                                                            J=float(hard_diag_res.get("J", 0.0)),
                                                            vf=float(hard_diag_res.get("vf", _vf_ref)),
                                                        )],
                                                        hard_merit_allow,
                                                        it=it,
                                                    )
                                                    handle_stage2_stall_volume_controller(
                                                        cfg, res, psi=lsf, M=l2_mass, it=it, reason="reject-nucleation-merit"
                                                    )
                                                    if MPI.rank(MPI.comm_world) == 0:
                                                        print("[stage2-nucleation-reject] it=%03d: all hard nucleation trials rejected; last J_accept/merit %.6e > allow %.6e (old=%.6e, vf %.6f->%.6f); block until it=%d" %
                                                              (int(it), float(hard_merit_new), float(hard_merit_allow), float(hard_merit_old),
                                                               float(_vf_ref), float(hard_diag_res.get("vf", vf_shift_exact)),
                                                               int(cfg.get("_stage2_nucleation_block_until_it", int(it)))), flush=True)
                                        if hard_accept_ok:
                                            reset_stage2_stall_watchdog(cfg)
                                            lsf = Function(Vls)
                                            copy_function_values(lsf, lsf_shifted)
                                            res = res_hard
                                            lambda_adapt_info = None
                                            if (not hard_shift_only_reached) and (str(hard_shift_mode) == "nucleation"):
                                                cfg["_lambda_v_plateau_boost_count_since_nucleation"] = 0
                                                cfg["_last_nucleation_it"] = int(it)
                                                cfg["_post_nucleation_lambda_zero_active"] = bool(
                                                    int(cfg.get("post_nucleation_lambda_zero_steps", 0)) > 0
                                                )
                                                cfg["_post_nucleation_lambda_zero_start_it"] = int(it)
                                                cfg["_last_nucleation_post_vf"] = float(res.get("vf", _vf_ref))
                                                cfg["_vf_stage_rebound_active"] = True
                                                _, lambda_adapt_info = update_lambda_v_after_nucleation(cfg, res, psi=lsf, M=l2_mass, it=it)
                                                if bool(cfg.get("_post_nucleation_lambda_zero_active", False)):
                                                    cfg["lambda_v"] = 0.0
                                                    if MPI.rank(MPI.comm_world) == 0:
                                                        print("[lambda_v-zero-freeze] it=%03d: accepted nucleation -> start %d-step lambda_v=0 relaxation window" %
                                                              (int(it), int(cfg.get("post_nucleation_lambda_zero_steps", 0))))
                                            elif str(hard_shift_mode) == "uniform-shift":
                                                cfg["_last_uniform_hard_shift_pre_vf"] = float(_vf_ref)
                                                cfg["_last_uniform_hard_shift_post_vf"] = float(res.get("vf", _vf_ref))
                                                cfg["_last_uniform_hard_shift_it"] = int(it)
                                                if postprocess_active:
                                                    cfg["_last_postprocess_shift_it"] = int(it)
                                            refresh_volume_merit_inplace(cfg, res, hard_shift_only=hard_shift_only_active)
                                            J_old = res["J"]
                                            J_compare_old = res["J_compare"]
                                            J_history = [J_compare_old]
                                            if postprocess_active:
                                                cfg["_postprocess_j_mech_history"] = [float(res.get("J", J_compare_old))]
                                            if len(vf_history) == 0:
                                                vf_history = [float(res["vf"])]
                                            else:
                                                vf_history[-1] = float(res["vf"])
                                            if len(lambda_plateau_vf_history) == 0:
                                                lambda_plateau_vf_history = [float(res["vf"])]
                                            else:
                                                lambda_plateau_vf_history[-1] = float(res["vf"])
                                            plateau_kappa_min_hits = 0
                                            plateau_last_vf = float(res["vf"])
                                            kappa = float(cfg.get("hard_shift_kappa_reset", 0.5 * float(cfg["kappa0"])))
                                            kappa_increase_apply_count = 0
                                            stage_relax_steps_left = int(cfg.get("hard_shift_relax_steps", cfg.get("stage_relax_steps_total", 0)))
                                            hard_shift_cooldown_left = int(cfg.get("hard_shift_cooldown_steps", 5))
                                            hard_shift_recovery_left = int(cfg.get("hard_shift_recovery_steps", 3))
                                            hard_shift_done = True
                                            if (
                                                postprocess_active
                                                and bool(cfg.get("postprocess_terminate_on_target", True))
                                                and bool(cfg.get("postprocess_terminate_immediate_on_target", True))
                                            ):
                                                _pp_hit, _pp_vf, _pp_target, _pp_tol = postprocess_target_status(cfg, res)
                                                if _pp_hit:
                                                    hard_shift_cooldown_left = 0
                                                    hard_shift_recovery_left = 0
                                                    lambda_plateau_cooldown_left = 0
                                                    cfg["_lambda_v_plateau_accept_cooldown_left"] = 0
                                                    final_filter_post_action = "terminate"
                                                    final_filter_post_pending_terminate = True
                                                    cfg["_postprocess_target_stop_reason"] = "target-reached-after-hard-shift"
                                                    if MPI.rank(MPI.comm_world) == 0:
                                                        print("[postprocess-target] it=%03d: hard shift reached vf=%.6f <= target %.6f + tol %.3e; cancel cooldown/recovery and terminate" %
                                                              (int(it), float(_pp_vf), float(_pp_target), float(_pp_tol)),
                                                              flush=True)
                                            if MPI.rank(MPI.comm_world) == 0:
                                                print("[vf-hard-nucleation] accepted at it=%03d: vf %.6f -> %.6f, ||lsf||_L2 was %.4f, psi_void=%.4f, c_shift=%.4e, restart_kappa=%.3e, cooldown_steps=%d, cooldown_kappa=%.3e, recovery_steps=%d, rebound_active=%s, %s" %
                                                      (it, _vf_ref, float(vf_shift_exact), float(_lsf_norm_hard),
                                                       float(hard_shift_info.get("psi_void", 0.0)), float(hard_shift_info.get("c_shift", 0.0)), float(kappa),
                                                       int(hard_shift_cooldown_left),
                                                       max(float(kappa_min), float(cfg.get("hard_shift_cooldown_kappa_factor", 0.1)) * float(cfg["kappa0"])),
                                                       int(hard_shift_recovery_left),
                                                       str(bool(cfg.get("_vf_stage_rebound_active", False))),
                                                       _format_stage_dv_state(cfg)))
                                        else:
                                            mark_materials_from_lsf(mesh, lsf, materials, threshold=cfg["threshold"])
                                            plateau_kappa_min_hits = 0
                                            plateau_last_vf = _vf_ref
                                            hard_shift_attempted = False
                                            hard_shift_done = False
                                else:
                                    lsf = Function(Vls)
                                    copy_function_values(lsf, lsf_shifted)
                                    solver_fail = True
                                    err_msg = "[vf-hard-nucleation] it=%03d: evaluate failed after accepted hard nucleation move => route to existing solver-fail recovery/refinement path" % it
                                    if MPI.rank(MPI.comm_world) == 0:
                                        print(err_msg)
                else:
                    plateau_kappa_min_hits = 0
                    plateau_last_vf = None
        # Stage volume controller (Deng-style spirit):
        # success -> next stage.
        vf_stage_target = current_vf_target(cfg)
        vf_stage_tol = stage_controller_tolerance(cfg)
        if hard_shift_only_active or stage2_final_convergence_active:
            if (
                (not bool(cfg.get("_vf_stage_frozen_in_hard_shift_announced", False)))
                and (MPI.rank(MPI.comm_world) == 0)
            ):
                if stage2_final_convergence_active:
                    print("[vf-stage] stage2-final convergence active -> freeze stage controller at vf_target=%.6f (no further target lowering before post-processing)" %
                          float(stage2_convergence_vf))
                else:
                    print("[vf-stage] hard-shift-only active -> freeze stage controller (no stage advance/shrink)")
                cfg["_vf_stage_frozen_in_hard_shift_announced"] = True
        else:
            cfg["_vf_stage_frozen_in_hard_shift_announced"] = False
            stage_hit = (vf_stage_target is not None) and (float(res["vf"]) <= float(vf_stage_target) + vf_stage_tol)
            stage2_active_for_settle = (
                bool(cfg.get("stage2_settle_enabled", True))
                and stage2_volume_continuation_active(
                    cfg, vf=res.get("vf", None), vf_target=vf_stage_target,
                    hard_shift_only=hard_shift_only_active
                )
            )
            final_stage_target = _stage2_is_final_settle_target(cfg, vf_stage_target, vf_stage_tol)
            if (
                stage2_active_for_settle
                and bool(cfg.get("stage2_stop_for_convergence_enabled", False))
                and (float(res["vf"]) <= float(stage2_convergence_vf) + vf_stage_tol)
            ):
                final_stage_target = True
            if stage2_active_for_settle and final_stage_target:
                vf_stage_target = float(stage2_convergence_vf)
                cfg["_vf_stage_target"] = float(vf_stage_target)
                stage_hit = float(res["vf"]) <= float(vf_stage_target) + vf_stage_tol
                freeze_stage_merit_from_controller(cfg, res, reason="stage2-final-target")
                refresh_volume_merit_inplace(cfg, res, hard_shift_only=hard_shift_only_active)

            accepted_normal_for_settle = (
                (not solver_fail)
                and (not line_search_stalled)
                and (not hard_shift_done)
                and (not cooldown_active)
                and (not recovery_active)
                and (not lambda_plateau_cooldown_active)
            )

            if stage2_active_for_settle and (stage_hit or bool(cfg.get("_stage2_settle_active", False))):
                settle_ok, settle_status = _stage2_settle_update(
                    cfg,
                    vf_stage_target,
                    it,
                    res,
                    np.degrees(theta_update),
                    theta_update_sym_deg=(None if theta_update_sym is None else np.degrees(theta_update_sym)),
                    stage_hit=stage_hit,
                    accepted_normal_step=accepted_normal_for_settle,
                    blocked_by_stall=line_search_stalled,
                    final_stage=final_stage_target,
                    vf_stage_tol=vf_stage_tol,
                )
                settle_exhausted_ok = bool(settle_status.get("settle_exhausted_ok", False))
                settle_overshoot_ok = bool(settle_status.get("overshoot_release_ok", False))
                settle_hit_ok = bool(settle_status.get("hit_release_ok", False))
                if settle_ok or settle_exhausted_ok:
                    old_stage_target = vf_stage_target
                    old_dv = float(current_stage_dv(cfg))
                    if final_stage_target:
                        cfg["_stage2_final_convergence_active"] = True
                        cfg["_stage2_final_settle_confirmed"] = True
                        cfg["_stage2_final_settle_exhausted"] = bool(settle_exhausted_ok)
                        cfg["_vf_stage_target"] = float(stage2_convergence_vf)
                        freeze_stage_merit_from_controller(cfg, res, reason="stage2-final-settle-confirmed")
                        refresh_volume_merit_inplace(cfg, res, hard_shift_only=hard_shift_only_active)
                        cfg["_vf_aug_lag_prev_violation"] = float(res.get("vf_constraint_violation", 0.0))
                        cfg["_vf_aug_lag_last_target"] = current_vf_target(cfg)
                        _stage2_settle_reset(cfg)
                        stage2_final_filter_just_done = False
                        if MPI.rank(MPI.comm_world) == 0:
                            final_tag = "[stage2-settle-exhausted-final]" if settle_exhausted_ok else "[stage2-settle]"
                            final_msg = "exhausted-safe final settle" if settle_exhausted_ok else "ok-final"
                            print("%s %s: %s -> freeze vf_target=%.6f and enter stage2 final filter/post-processing gate" %
                                  (final_tag, final_msg, _format_stage2_settle_status(settle_status), float(stage2_convergence_vf)))
                    else:
                        rebound_active = bool(cfg.get("_vf_stage_rebound_active", False))
                        if rebound_active:
                            new_dv = _adjacent_stage_dv(cfg, dv_now_raw=old_dv, direction="larger")
                            if new_dv is None:
                                new_dv = old_dv
                        else:
                            dv_levels = _stage_dv_levels_from_cfg(cfg)
                            new_dv = float(dv_levels[0] if len(dv_levels) > 0 else cfg.get("vf_stage_dv0", old_dv))
                        advance_dv_capped = False
                        if settle_exhausted_ok:
                            cap_val = _optional_positive_float(
                                cfg.get("stage2_settle_exhaust_max_advance_dv", cfg.get("vf_stage_dv_min", 0.003))
                            )
                            if cap_val is not None:
                                new_dv = min(new_dv, cap_val)
                                advance_dv_capped = True
                        elif settle_overshoot_ok:
                            cap_val = _optional_positive_float(cfg.get("stage2_settle_overshoot_max_advance_dv", None))
                            if cap_val is not None:
                                new_dv = min(new_dv, cap_val)
                                advance_dv_capped = True
                        # On settle success, advance from the more conservative of:
                        #   - the achieved current vf
                        #   - the old stage target
                        # so overshooting a stage does not make the next target drop too aggressively.
                        advance_stage_target(cfg, min(float(res["vf"]), float(old_stage_target)), dv=new_dv)
                        reset_stage2_stall_watchdog(cfg)
                        update_alpha_ti_from_stage_state(cfg, res, it=it, reason="stage-settle-success")
                        update_lambda_v_from_stage_state(cfg, res, psi=lsf, M=l2_mass, it=it, reason="stage-settle-success")
                        freeze_stage_merit_from_controller(cfg, res, reason="stage-settle-success")
                        refresh_volume_merit_inplace(cfg, res, hard_shift_only=hard_shift_only_active)
                        cfg["_vf_aug_lag_prev_violation"] = float(res.get("vf_constraint_violation", 0.0))
                        cfg["_vf_aug_lag_last_target"] = current_vf_target(cfg)
                        stage_target_just_changed = True
                        _stage2_settle_reset(cfg)
                        # Reset step length after stage change (avoid inheriting a shrunken kappa from previous stage).
                        kappa = float(cfg["kappa0"])
                        J_old = res["J"]
                        J_compare_old = res["J_compare"]
                        J_history = [J_compare_old]
                        vf_history = [float(res["vf"])]
                        lambda_plateau_vf_history = [float(res["vf"])]
                        plateau_kappa_min_hits = 0
                        plateau_last_vf = None
                        if not hard_shift_done:
                            hard_shift_cooldown_left = 0
                            hard_shift_recovery_left = 0
                        if MPI.rank(MPI.comm_world) == 0:
                            if settle_exhausted_ok:
                                stage_tag = "[stage2-settle-exhausted]"
                                success_msg = "exhausted-safe advance"
                                dv_msg = "capped" if advance_dv_capped else ("step-up" if rebound_active else "reset")
                            elif settle_overshoot_ok:
                                stage_tag = "[stage2-settle-overshoot]"
                                success_msg = "overshoot-safe advance"
                                dv_msg = "capped" if advance_dv_capped else ("step-up" if rebound_active else "reset")
                            elif settle_hit_ok:
                                stage_tag = "[stage2-settle-hit]"
                                success_msg = "hit-safe advance"
                                dv_msg = "step-up" if rebound_active else "reset"
                            else:
                                stage_tag = "[vf-stage]"
                                success_msg = "settle success"
                                dv_msg = "step-up" if rebound_active else "reset"
                            print("%s %s: vf=%.6f reached target %.6f (tol %.3e, %s) -> dv %s %.6f -> %.6f, next target %.6f" %
                                  (stage_tag, success_msg, float(res["vf"]), float(old_stage_target), vf_stage_tol,
                                   _format_stage2_settle_status(settle_status),
                                   dv_msg,
                                   old_dv, float(current_stage_dv(cfg)), float(current_vf_target(cfg))))
                else:
                    if (
                        line_search_stalled
                        and (not hard_shift_done)
                        and (vf_stage_target is not None)
                        and (float(res["vf"]) > float(vf_stage_target) + vf_stage_tol)
                    ):
                        handle_stage2_stall_volume_controller(
                            cfg, res, psi=lsf, M=l2_mass, it=it, reason="settle-stall-above-target"
                        )
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[stage2-settle] wait: %s -> keep vf_stage frozen (no advance)" %
                              _format_stage2_settle_status(settle_status))
            else:
                stage_stable = bool(step13_ok) or bool(hard_shift_done)
                if not stage2_volume_continuation_active(
                    cfg, vf=res.get("vf", None), vf_target=vf_stage_target,
                    hard_shift_only=hard_shift_only_active
                ):
                    stage_stable = stage_stable or bool(line_search_stalled)
                if stage_hit and stage_stable:
                    old_stage_target = vf_stage_target
                    old_dv = float(current_stage_dv(cfg))
                    rebound_active = bool(cfg.get("_vf_stage_rebound_active", False))
                    if rebound_active:
                        new_dv = _adjacent_stage_dv(cfg, dv_now_raw=old_dv, direction="larger")
                        if new_dv is None:
                            new_dv = old_dv
                    else:
                        dv_levels = _stage_dv_levels_from_cfg(cfg)
                        new_dv = float(dv_levels[0] if len(dv_levels) > 0 else cfg.get("vf_stage_dv0", old_dv))
                    advance_stage_target(cfg, min(float(res["vf"]), float(old_stage_target)), dv=new_dv)
                    reset_stage2_stall_watchdog(cfg)
                    update_alpha_ti_from_stage_state(cfg, res, it=it, reason="stage-success")
                    update_lambda_v_from_stage_state(cfg, res, psi=lsf, M=l2_mass, it=it, reason="stage-success")
                    freeze_stage_merit_from_controller(cfg, res, reason="stage-success")
                    refresh_volume_merit_inplace(cfg, res, hard_shift_only=hard_shift_only_active)
                    cfg["_vf_aug_lag_prev_violation"] = float(res.get("vf_constraint_violation", 0.0))
                    cfg["_vf_aug_lag_last_target"] = current_vf_target(cfg)
                    stage_target_just_changed = True
                    kappa = float(cfg["kappa0"])
                    J_old = res["J"]
                    J_compare_old = res["J_compare"]
                    J_history = [J_compare_old]
                    vf_history = [float(res["vf"])]
                    lambda_plateau_vf_history = [float(res["vf"])]
                    plateau_kappa_min_hits = 0
                    plateau_last_vf = None
                    if not hard_shift_done:
                        hard_shift_cooldown_left = 0
                        hard_shift_recovery_left = 0
                    if MPI.rank(MPI.comm_world) == 0:
                        if hard_shift_done:
                            print("[vf-stage] success after hard shift: vf=%.6f reached target %.6f (tol %.3e) -> dv step-up %.6f -> %.6f, next target %.6f, rebound_active=%s" %
                                  (float(res["vf"]), float(old_stage_target), vf_stage_tol, old_dv, float(current_stage_dv(cfg)), float(current_vf_target(cfg)), str(rebound_active)))
                        else:
                            print("[vf-stage] success: vf=%.6f reached target %.6f (tol %.3e) -> dv %s %.6f -> %.6f, next target %.6f" %
                                  (float(res["vf"]), float(old_stage_target), vf_stage_tol, ("step-up" if rebound_active else "reset"), old_dv, float(current_stage_dv(cfg)), float(current_vf_target(cfg))))
            if (
                (not solver_fail)
                and line_search_stalled
                and (vf_stage_target is not None)
                and (float(res["vf"]) > float(vf_stage_target) + vf_stage_tol)
                and (not hard_shift_attempted)
                and (not bool(cfg.get("_stage2_settle_active", False)))
            ):
                old_dv = float(current_stage_dv(cfg))
                dv_min = float(cfg.get("vf_stage_dv_min", 0.002))
                nonweak_diag = cfg.get("_stage2_last_stall_diagnosis", None)
                force_nonweak_dv_shrink = (
                    stage2_nonweak_stall_controller_handled
                    and bool(cfg.get("stage2_stall_nonweak_shrink_dv_after_fallback", False))
                    and isinstance(nonweak_diag, dict)
                    and bool(nonweak_diag.get("watchdog_allow_fallback", False))
                    and old_dv > dv_min + 1e-15
                )
                if stage2_nonweak_stall_controller_handled and (not force_nonweak_dv_shrink):
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[vf-stage] stall above target: vf=%.6f target=%.6f tol=%.3e and %s -> keep target/dv; non-weak stage2 controller response already applied" %
                              (float(res["vf"]), float(vf_stage_target), vf_stage_tol, _format_stage_dv_state(cfg, dv_now_raw=old_dv)))
                elif stage2_weak_stall_controller_handled:
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[vf-stage] stall above target: vf=%.6f target=%.6f tol=%.3e and %s -> keep target/dv; weak stage2 controller grow response applied before dv shrink" %
                              (float(res["vf"]), float(vf_stage_target), vf_stage_tol, _format_stage_dv_state(cfg, dv_now_raw=old_dv)))
                elif old_dv <= dv_min + 1e-15:
                    stall_vc_active = stage2_volume_continuation_active(
                        cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                        hard_shift_only=hard_shift_only_active
                    )
                    if stall_vc_active:
                        stall_controller_info = handle_stage2_stall_volume_controller(
                            cfg, res, psi=lsf, M=l2_mass, it=it, reason="stall-above-target-dv-min"
                        )
                    if MPI.rank(MPI.comm_world) == 0:
                        if stall_vc_active:
                            stall_action = "unknown"
                            try:
                                stall_action = str(stall_controller_info.get("action", stall_controller_info.get("adapt_tag", "unknown")))
                            except Exception:
                                stall_action = "unknown"
                            print("[vf-stage] stall above target: vf=%.6f target=%.6f tol=%.3e and %s -> keep target; stage2 controller action=%s before any rare nucleation fallback" %
                                  (float(res["vf"]), float(vf_stage_target), vf_stage_tol, _format_stage_dv_state(cfg, dv_now_raw=old_dv), stall_action))
                        else:
                            print("[vf-stage] stall above target: vf=%.6f target=%.6f tol=%.3e and %s -> keep target; stage2 inactive, wait for plateau/nucleation trigger" %
                                  (float(res["vf"]), float(vf_stage_target), vf_stage_tol, _format_stage_dv_state(cfg, dv_now_raw=old_dv)))
                else:
                    new_dv = shrink_stage_step(cfg, float(res["vf"]))
                    reset_stage2_stall_watchdog(cfg)
                    update_lambda_v_from_stage_state(cfg, res, psi=lsf, M=l2_mass, it=it, reason="dv-shrink")
                    freeze_stage_merit_from_controller(cfg, res, reason="dv-shrink")
                    refresh_volume_merit_inplace(cfg, res, hard_shift_only=hard_shift_only_active)
                    cfg["_vf_aug_lag_prev_violation"] = float(res.get("vf_constraint_violation", 0.0))
                    cfg["_vf_aug_lag_last_target"] = current_vf_target(cfg)
                    stage_target_just_changed = True
                    # Reset step length after stage change (dv shrink implies new local regime).
                    kappa = float(cfg["kappa0"])
                    J_old = res["J"]
                    J_compare_old = res["J_compare"]
                    J_history = [J_compare_old]
                    # Preserve vf plateau history here so the hard-shift detector can
                    # continue accumulating across stage-target shrink operations.
                    hard_shift_cooldown_left = 0
                    hard_shift_recovery_left = 0
                    if MPI.rank(MPI.comm_world) == 0:
                        reason_suffix = " after watchdog fallback" if force_nonweak_dv_shrink else ""
                        print("[vf-stage] stall above target%s: vf=%.6f target=%.6f tol=%.3e -> dv step-down %.6f -> %.6f, new target %.6f" %
                              (reason_suffix, float(res["vf"]), float(vf_stage_target), vf_stage_tol, old_dv, new_dv, float(current_vf_target(cfg))))

        if (not solver_fail) and (not hard_shift_done) and (len(milestone_filter_targets) > 0) and (milestone_filter_radius_factor > 0.0):
            vf_now_milestone = float(res["vf"])
            for vf_milestone in milestone_filter_targets:
                if bool(milestone_filter_done.get(vf_milestone, False)):
                    continue
                crossed = (
                    (vf_prev_for_milestone > float(vf_milestone) + milestone_filter_tol)
                    and (vf_now_milestone <= float(vf_milestone) + milestone_filter_tol)
                )
                if not crossed:
                    continue
                is_stage2_final_filter = (
                    stage2_stop_for_convergence
                    and (abs(float(vf_milestone) - float(stage2_convergence_vf)) <= 1e-12)
                )
                if (
                    is_stage2_final_filter
                    and bool(cfg.get("stage2_settle_enabled", True))
                    and (not bool(cfg.get("_stage2_final_settle_confirmed", False)))
                ):
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[stage2-final-filter] it=%03d: vf crossed %.3f, but stage2 settle gate is not confirmed yet; postpone final filter/post-processing" %
                              (int(it), float(vf_milestone)))
                    continue
                if is_stage2_final_filter and bool(cfg.get("_stage2_final_filter_done", False)):
                    milestone_filter_done[vf_milestone] = True
                    continue
                if is_stage2_final_filter and (not bool(cfg.get("stage2_final_filter_enabled", True))):
                    cfg["_stage2_final_convergence_active"] = True
                    cfg["_stage2_final_filter_done"] = True
                    milestone_filter_done[vf_milestone] = True
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[stage2-final-filter] it=%03d: filter disabled; enter stage2 convergence gate at vf=%.6f" %
                              (int(it), float(vf_now_milestone)))
                    continue
                radius_factor_eff = (
                    float(cfg.get("stage2_final_filter_radius_factor", milestone_filter_radius_factor))
                    if is_stage2_final_filter else float(milestone_filter_radius_factor)
                )
                vf_rel_limit_eff = (
                    float(cfg.get("stage2_final_filter_vf_rel_change_max", milestone_filter_vf_rel_change_max))
                    if is_stage2_final_filter else float(milestone_filter_vf_rel_change_max)
                )
                radius_milestone_cap = float(radius_factor_eff) * float(mesh.hmin())
                lsf_before_milestone = Function(Vls)
                copy_function_values(lsf_before_milestone, lsf)
                res_before_milestone = res
                vf_before_milestone = float(res_before_milestone.get("vf", vf_now_milestone))
                if is_stage2_final_filter:
                    write_stage2_raw_xdmf_once(
                        outdir, mesh, it, materials, lsf_before_milestone, cfg,
                        reason="stage2-final-milestone-filter",
                    )
                lsf_milestone, res_milestone, milestone_info = adaptive_helmholtz_filter_with_vf_guard(
                    lsf, float(vf_now_milestone), radius_milestone_cap, vf_rel_limit_eff,
                    filter_radius_search_max_iter,
                    context_prefix="[it %03d vf-milestone-filter vf<=%.3f]" % (it, float(vf_milestone)),
                    alpha_eval=alpha_now,
                    apply_hard_vf=False,
                    base_res=res,
                )
                if (not bool(milestone_info.get("success", False))) or (res_milestone is None):
                    mark_materials_from_lsf(mesh, lsf_before_milestone, materials, threshold=cfg["threshold"])
                    if MPI.rank(MPI.comm_world) == 0:
                        print("[vf-milestone-filter-warning] it=%03d: adaptive Helmholtz failed at vf<=%.3f (cap=%.3e, vf_rel_limit=%.3f), keep unfiltered state" %
                              (it, float(vf_milestone), float(radius_milestone_cap), float(vf_rel_limit_eff)))
                    continue
                if is_stage2_final_filter:
                    vf_inc_allow = max(0.0, float(cfg.get("stage2_final_filter_vf_increase_allow_abs", 0.0)))
                    vf_abs_max = max(
                        float(cfg.get("stage2_final_filter_vf_max", 0.11)),
                        float(stage2_convergence_vf) + vf_inc_allow,
                    )
                    if bool(cfg.get("stage2_final_filter_delete_bias", True)):
                        vf_cap = min(vf_abs_max, max(vf_before_milestone, stage2_convergence_vf) + vf_inc_allow)
                    else:
                        vf_cap = vf_abs_max
                    if float(res_milestone.get("vf", 0.0)) > vf_cap + 1e-15:
                        lsf_capped, cap_info, vf_cap_exact = enforce_specific_volume_fraction_by_shift(
                            lsf_milestone, mesh, cfg, vf_cap
                        )
                        if bool(cap_info.get("active", False)):
                            if use_symmetry_parameterization and (symmetry_map is not None):
                                lsf_capped = expand_symmetry_to_full(lsf_capped, symmetry_map)
                            _lsf_norm_cap = renormalize_lsf_inplace(lsf_capped, dx)
                            res_capped, solver_fail_cap = evaluate_safe(
                                mesh, W, lsf_capped, materials, cfg, alpha=alpha_now,
                                context="[it %03d stage2-final-filter-vf-cap]" % int(it)
                            )
                            if (not solver_fail_cap) and (res_capped is not None):
                                lsf_milestone = lsf_capped
                                res_milestone = res_capped
                                milestone_info["vf_cap_applied"] = True
                                milestone_info["vf_cap"] = float(vf_cap)
                                milestone_info["vf_cap_exact"] = float(vf_cap_exact)
                                milestone_info["vf_cap_lsf_norm_before"] = float(_lsf_norm_cap)
                            else:
                                milestone_info["vf_cap_applied"] = False
                                milestone_info["vf_cap_failed"] = True
                    if float(res_milestone.get("vf", 0.0)) > vf_abs_max + 1e-15:
                        mark_materials_from_lsf(mesh, lsf_before_milestone, materials, threshold=cfg["threshold"])
                        if MPI.rank(MPI.comm_world) == 0:
                            print("[stage2-final-filter-warning] it=%03d: filtered vf=%.6f exceeds vf_max=%.6f; keep unfiltered state" %
                                  (int(it), float(res_milestone.get("vf", 0.0)), float(vf_abs_max)))
                        continue
                    milestone_stage_active = stage2_volume_continuation_active(
                        cfg, vf=res.get("vf", None), vf_target=current_vf_target(cfg),
                        hard_shift_only=hard_shift_only_active
                    )
                    mech_cap_ok, mech_cap_info = stage2_mechanical_step_cap_ok(
                        cfg, res, res_milestone, milestone_stage_active
                    )
                    if not mech_cap_ok:
                        mark_materials_from_lsf(mesh, lsf_before_milestone, materials, threshold=cfg["threshold"])
                        print_stage2_mechanical_step_cap_reject(
                            "stage2-final-filter milestone",
                            mech_cap_info,
                            action="keep unfiltered state",
                        )
                        continue
                lsf = Function(Vls)
                copy_function_values(lsf, lsf_milestone)
                mark_materials_from_lsf(mesh, lsf, materials, threshold=cfg["threshold"])
                res = res_milestone
                J_old = res["J"]
                J_compare_old = res["J_compare"]
                if is_stage2_final_filter:
                    J_history = [J_compare_old]
                elif len(J_history) == 0:
                    J_history = [J_compare_old]
                else:
                    J_history[-1] = J_compare_old
                if is_stage2_final_filter:
                    vf_history = [float(res["vf"])]
                elif len(vf_history) == 0:
                    vf_history = [float(res["vf"])]
                else:
                    vf_history[-1] = float(res["vf"])
                if is_stage2_final_filter:
                    lambda_plateau_vf_history = [float(res["vf"])]
                elif len(lambda_plateau_vf_history) == 0:
                    lambda_plateau_vf_history = [float(res["vf"])]
                else:
                    lambda_plateau_vf_history[-1] = float(res["vf"])
                vf_now_milestone = float(res["vf"])
                milestone_filter_done[vf_milestone] = True
                if is_stage2_final_filter:
                    cfg["_stage2_final_convergence_active"] = True
                    cfg["_stage2_final_filter_done"] = True
                    stage2_final_filter_just_done = True
                    cfg["_vf_aug_lag_prev_violation"] = float(res.get("vf_constraint_violation", 0.0))
                    cfg["_vf_aug_lag_last_target"] = current_vf_target(cfg)
                if MPI.rank(MPI.comm_world) == 0:
                    filter_tag = "[stage2-final-filter]" if is_stage2_final_filter else "[vf-milestone-filter]"
                    cap_msg = ""
                    if is_stage2_final_filter:
                        cap_msg = ", vf_max=%.3f, cap_applied=%s" % (
                            float(vf_abs_max),
                            str(bool(milestone_info.get("vf_cap_applied", False))),
                        )
                    print("%s it=%03d: adaptive filter at vf<=%.3f with r*=%.3e (cap=%.3e, vf_rel_limit=%.3f, vf_rel_change=%.3e%s), new vf=%.6f" %
                          (filter_tag,
                           it,
                           float(vf_milestone),
                           float(milestone_info.get("radius_star", 0.0)),
                           float(milestone_info.get("radius_cap", radius_milestone_cap)),
                           float(vf_rel_limit_eff),
                           float(milestone_info.get("rel_change", float("nan"))),
                           cap_msg,
                           float(vf_now_milestone)))
            vf_prev_for_milestone = float(vf_now_milestone)

        theta_update_deg = np.degrees(theta_update)
        theta_update_sym_deg = np.degrees(theta_update_sym) if theta_update_sym is not None else None
        theta_opt_deg = np.degrees(theta_opt)
        psi_stats_now = None
        psi_stats_stride = max(1, int(cfg.get("psi_stats_stride", 1)))
        if it % psi_stats_stride == 0:
            psi_stats_now = psi_global_stats(lsf, dx)
        # Pre-filter theta gate on current (unfiltered) state uses the mechanical optimality indicator.
        step14_ok_raw = (theta_opt < eps_theta_rad)
        step14_ok = bool(step14_ok_raw)
        postprocess_active = bool(cfg.get("_postprocess_active", False))
        stage2_final_convergence_active = (
            bool(cfg.get("_stage2_final_convergence_active", False))
            and (not postprocess_active)
        )
        stage2_final_settle_confirmed = (
            stage2_final_convergence_active
            and bool(cfg.get("_stage2_final_settle_confirmed", False))
        )
        final_workflow_action = "start-postprocess" if stage2_final_convergence_active else "terminate"
        final_target = (
            float(stage2_convergence_vf)
            if stage2_final_convergence_active
            else float(current_postprocess_vf_final_target(cfg))
        )
        final_target_tol = stage_success_tolerance(cfg, dv=float(cfg.get("vf_stage_dv_min", 0.002)))
        uniform_exit_override_active = bool(cfg.get("_uniform_hard_shift_exit_override", False)) and postprocess_active
        final_target_satisfied = (
            (float(res["vf"]) <= final_target + final_target_tol)
            or uniform_exit_override_active
        )
        # A hard shift is a discrete plateau-escape move; use it for stage progression,
        # but let the next iterations re-evaluate the final theta gate on the updated lsf.
        conv_ok = (not solver_fail) and (bool(step13_ok) or (bool(line_search_stalled) and (not hard_shift_done)))
        # J_merit-gate for termination:
        # allow either normal step-13 J_merit-convergence, or reaching the kappa_min floor.
        kappa_min_reached_for_j_gate = bool(line_search_stalled) or (float(kappa) <= float(kappa_min) + 1e-15)
        j_conv_ok = bool(step13_ok) or bool(kappa_min_reached_for_j_gate)
        if stage2_final_filter_just_done:
            conv_ok = False
            j_conv_ok = False
        if stage2_final_settle_confirmed:
            # Stage-2 final convergence has already passed the constrained
            # settle gate based on theta_update/J_merit history. Do not require
            # the old pure-mechanical theta_opt gate, and do not rely on
            # refine-cap bypass to leave stage 2.
            conv_ok = (not solver_fail)
            j_conv_ok = (not solver_fail)
            step14_ok = (not solver_fail)
        final_filter_trigger = conv_ok and final_target_satisfied and j_conv_ok
        postprocess_target_ready = (
            postprocess_active
            and bool(cfg.get("postprocess_terminate_on_target", True))
            and final_target_satisfied
            and (not hard_shift_done)
            and (not cooldown_active)
            and (not recovery_active)
            and (not lambda_plateau_cooldown_active)
        )
        if postprocess_target_ready:
            conv_ok = True
            j_conv_ok = True
            step14_ok = True
            final_filter_trigger = False
            final_filter_post_action = "terminate"
            final_filter_post_pending_terminate = True
            cfg["_postprocess_target_stop_reason"] = "target-ready"
            if MPI.rank(MPI.comm_world) == 0:
                _pp_hit, _pp_vf, _pp_target, _pp_tol = postprocess_target_status(cfg, res)
                print("[postprocess-target] it=%03d: target ready vf=%.6f <= %.6f + %.3e; terminate without extra final filter" %
                      (int(it), float(_pp_vf), float(_pp_target), float(_pp_tol)),
                      flush=True)
        # If mesh is already at refine cap and theta still fails, bypass theta gate:
        # do not keep trying impossible refine, go to final-filter + cooldown + terminate.
        at_refine_cap_now = (
            int(Nx) >= int(cfg.get("refine_n_max", Nx))
            and int(Ny) >= int(cfg.get("refine_n_max", Ny))
            and int(Nz) >= int(cfg.get("refine_n_max", Nz))
        )
        theta_bypass_due_to_refine_cap = bool(
            final_filter_trigger
            and (not step14_ok_raw)
            and at_refine_cap_now
            and (not stage2_final_convergence_active)
        )
        if theta_bypass_due_to_refine_cap:
            step14_ok = True
            if MPI.rank(MPI.comm_world) == 0:
                print("[14] theta_opt bypass: refine cap reached (%dx%dx%d, n_max=%d), skip theta gate and enter final-filter workflow" %
                      (int(Nx), int(Ny), int(Nz), int(cfg.get("refine_n_max", Nx))))
        if conv_ok and (not final_target_satisfied):
            if MPI.rank(MPI.comm_world) == 0:
                if stage2_final_convergence_active:
                    action_msg = "continue stage2 final convergence"
                elif postprocess_active:
                    action_msg = "continue post-processing hard shift"
                else:
                    action_msg = "continue stage controller"
                print("[14] skip termination: vf=%.4f > final_target=%.4f (tol=%.3e); %s" %
                      (float(res["vf"]), final_target, final_target_tol, action_msg))
        elif conv_ok and uniform_exit_override_active and (float(res["vf"]) > final_target + final_target_tol):
            if MPI.rank(MPI.comm_world) == 0:
                print("[14] vf target override active after repeated uniform hard-shift rebounds: vf=%.4f > final_target=%.4f (tol=%.3e), use J_merit/theta convergence gates" %
                      (float(res["vf"]), final_target, final_target_tol))
        elif conv_ok and final_target_satisfied and (not j_conv_ok):
            if MPI.rank(MPI.comm_world) == 0:
                _rel_msg = ("nan" if not np.isfinite(rel_tol) else ("%.3e" % float(rel_tol)))
                print("[14] skip termination: vf target satisfied but J_merit gate not met yet (rel_tol=%s, eps_J=%.3e, kappa=%.4e, kappa_min=%.4e)" %
                      (_rel_msg, float(eps_J), float(kappa), float(kappa_min)))

        final_filter_can_start = (
            final_filter_trigger
            and bool(step14_ok)
            and (int(final_filter_post_cooldown_left) <= 0)
            and (not bool(final_filter_post_pending_terminate))
        )
        if final_filter_can_start:
            final_filter_post_action = str(final_workflow_action)
            stage2_filter_already_done = (
                final_filter_post_action == "start-postprocess"
                and bool(cfg.get("_stage2_final_filter_done", False))
            )
            if stage2_filter_already_done:
                final_filter_post_cooldown_left = 0
                final_filter_post_pending_terminate = True
                step14_ok = True
                if MPI.rank(MPI.comm_world) == 0:
                    print("[stage2-final] convergence confirmed after prior stage2-final filter; start post-processing without a second filter")
            elif (
                final_filter_post_action == "start-postprocess"
                and (not bool(cfg.get("stage2_final_filter_enabled", True)))
            ):
                cfg["_stage2_final_filter_done"] = True
                write_stage2_raw_xdmf_once(
                    outdir, mesh, it, materials, lsf, cfg,
                    reason="stage2-final-filter-disabled",
                )
                final_filter_post_cooldown_left = 0
                final_filter_post_pending_terminate = True
                step14_ok = True
                if MPI.rank(MPI.comm_world) == 0:
                    print("[stage2-final] convergence confirmed; stage2_final_filter_enabled=False, start post-processing without final filter")
            else:
                # Apply final filter only after the relevant convergence gate is confirmed.
                # Stage-2 final filtering uses a more permissive radius search but is
                # capped in absolute vf so smoothing cannot add too much material.
                if final_filter_post_action == "start-postprocess":
                    vf_rel_limit = max(0.0, float(cfg.get("stage2_final_filter_vf_rel_change_max", 0.05)))
                    radius_factor_final = max(0.0, float(cfg.get("stage2_final_filter_radius_factor", 1.0)))
                    context_name = "stage2-final-helmholtz-search"
                    log_tag = "[stage2-final-helmholtz]"
                else:
                    vf_rel_limit = max(0.0, float(cfg.get("final_filter_vf_rel_change_max", 0.10)))
                    radius_factor_final = max(0.0, float(cfg.get("final_filter_radius_max_factor", 1.0)))
                    context_name = "final-helmholtz-search"
                    log_tag = "[final-helmholtz]"
                radius_final_cap = float(radius_factor_final) * float(mesh.hmin())
                lsf_before_final = Function(Vls)
                copy_function_values(lsf_before_final, lsf)
                res_before_final = res
                vf_before_final = float(res_before_final.get("vf", float("nan")))
                if final_filter_post_action == "start-postprocess":
                    write_stage2_raw_xdmf_once(
                        outdir, mesh, it, materials, lsf_before_final, cfg,
                        reason="stage2-final-helmholtz",
                    )
                best_lsf_final, best_res_final, final_info = adaptive_helmholtz_filter_with_vf_guard(
                    lsf_before_final, vf_before_final, radius_final_cap, vf_rel_limit,
                    filter_radius_search_max_iter,
                    context_prefix="[it %03d %s]" % (it, context_name),
                    alpha_eval=alpha_now,
                    apply_hard_vf=False,
                    base_res=res_before_final,
                    use_plain_cg_helmholtz=True,
                )
                if (best_lsf_final is not None) and (best_res_final is not None):
                    if final_filter_post_action == "start-postprocess":
                        vf_inc_allow = max(0.0, float(cfg.get("stage2_final_filter_vf_increase_allow_abs", 0.0)))
                        vf_abs_max = max(
                            float(cfg.get("stage2_final_filter_vf_max", 0.11)),
                            float(stage2_convergence_vf) + vf_inc_allow,
                        )
                        if bool(cfg.get("stage2_final_filter_delete_bias", True)):
                            vf_cap = min(vf_abs_max, max(vf_before_final, stage2_convergence_vf) + vf_inc_allow)
                        else:
                            vf_cap = vf_abs_max
                        if float(best_res_final.get("vf", 0.0)) > vf_cap + 1e-15:
                            lsf_capped, cap_info, vf_cap_exact = enforce_specific_volume_fraction_by_shift(
                                best_lsf_final, mesh, cfg, vf_cap
                            )
                            if bool(cap_info.get("active", False)):
                                if use_symmetry_parameterization and (symmetry_map is not None):
                                    lsf_capped = expand_symmetry_to_full(lsf_capped, symmetry_map)
                                _lsf_norm_cap = renormalize_lsf_inplace(lsf_capped, dx)
                                res_capped, solver_fail_cap = evaluate_safe(
                                    mesh, W, lsf_capped, materials, cfg, alpha=alpha_now,
                                    context="[it %03d stage2-final-helmholtz-vf-cap]" % int(it)
                                )
                                if (not solver_fail_cap) and (res_capped is not None):
                                    best_lsf_final = lsf_capped
                                    best_res_final = res_capped
                                    final_info["vf_cap_applied"] = True
                                    final_info["vf_cap"] = float(vf_cap)
                                    final_info["vf_cap_exact"] = float(vf_cap_exact)
                                    final_info["vf_cap_lsf_norm_before"] = float(_lsf_norm_cap)
                        if float(best_res_final.get("vf", 0.0)) > vf_abs_max + 1e-15:
                            best_lsf_final = lsf_before_final
                            best_res_final = res_before_final
                            final_info["vf_cap_rejected"] = True
                    lsf = Function(Vls)
                    copy_function_values(lsf, best_lsf_final)
                    mark_materials_from_lsf(mesh, lsf, materials, threshold=cfg["threshold"])
                    res = best_res_final
                else:
                    lsf = Function(Vls)
                    copy_function_values(lsf, lsf_before_final)
                    mark_materials_from_lsf(mesh, lsf, materials, threshold=cfg["threshold"])
                    res = res_before_final
                theta_update, pn_update, gn_update = angle_between(lsf, g_proj, M=l2_mass)
                theta_update_sym = None
                if use_symmetry_parameterization and (symmetry_map is not None):
                    theta_update_sym, _, _ = symmetry_reduced_angle_between(lsf, g_proj, symmetry_map)
                theta_opt, _, _ = angle_between(lsf, res.get("g_obj", res["g"]), M=l2_mass)
                theta_update_deg = np.degrees(theta_update)
                theta_update_sym_deg = np.degrees(theta_update_sym) if theta_update_sym is not None else None
                theta_opt_deg = np.degrees(theta_opt)
                if final_filter_post_action == "start-postprocess":
                    cfg["_stage2_final_filter_done"] = True
                    final_filter_post_steps = max(0, int(cfg.get("stage2_final_filter_post_cooldown_steps", 0)))
                else:
                    final_filter_post_steps = max(0, int(cfg.get("final_filter_post_cooldown_steps", 5)))
                final_filter_post_cooldown_left = int(final_filter_post_steps)
                final_filter_post_pending_terminate = bool(final_filter_post_steps <= 0)
                # Once final-filter phase starts, skip refinement-by-convergence and
                # complete the selected action only after post-filter cooldown.
                step14_ok = True
                if it % psi_stats_stride == 0:
                    psi_stats_now = psi_global_stats(lsf, dx)
                if MPI.rank(MPI.comm_world) == 0:
                    cap_msg = ""
                    if final_filter_post_action == "start-postprocess":
                        cap_msg = ", vf_max=%.3f, cap_applied=%s" % (
                            float(vf_abs_max),
                            str(bool(final_info.get("vf_cap_applied", False))),
                        )
                    print("%s adaptive radius: mode=%s, r*=%.3e (cap=%.3e), vf_rel_limit=%.3f, vf_rel_change=%.3e%s, vf %.6f -> %.6f, J_mech %.6e -> %.6e" %
                          (log_tag,
                           str(final_info.get("mode", "none")),
                           float(final_info.get("radius_star", 0.0)),
                           float(final_info.get("radius_cap", radius_final_cap)),
                           float(vf_rel_limit),
                           float(final_info.get("rel_change", float("nan"))),
                           cap_msg,
                           float(res_before_final.get("vf", float("nan"))),
                           float(res.get("vf", float("nan"))),
                           float(res_before_final.get("J", float("nan"))),
                           float(res.get("J", float("nan")))))
                    if final_filter_post_pending_terminate:
                        print("[14] final-filter cooldown steps=0 => %s now" %
                              ("start post-processing" if final_filter_post_action == "start-postprocess" else "terminate"))
                    else:
                        print("[14] final-filter cooldown started: steps=%d, action=%s, current post-filter theta_update=%.2f deg, theta_opt=%.2f deg" %
                              (int(final_filter_post_steps), str(final_filter_post_action), float(theta_update_deg), float(theta_opt_deg)))

        if final_filter_post_pending_terminate:
            if str(final_filter_post_action) == "start-postprocess":
                cfg["_stage2_optimization_converged"] = True
                cfg["_stage2_final_convergence_active"] = False
                cfg["_postprocess_active"] = True
                cfg["_postprocess_start_it"] = int(it)
                cfg["_postprocess_started_announced"] = True
                cfg["_hard_shift_switch_reached_once"] = True
                cfg["_hard_shift_force_uniform_early"] = False
                cfg["_uniform_hard_shift_aggressive_active"] = False
                cfg["_uniform_hard_shift_exit_hits"] = 0
                cfg["_uniform_hard_shift_exit_override"] = False
                cfg["_last_postprocess_shift_it"] = None
                cfg["_vf_stage_frozen_in_hard_shift_announced"] = False
                final_filter_post_pending_terminate = False
                final_filter_post_action = "terminate"
                final_filter_post_cooldown_left = 0
                kappa = float(cfg["kappa0"])
                J_old = res["J"]
                J_compare_old = res["J_compare"]
                J_history = [J_compare_old]
                cfg["_postprocess_j_mech_history"] = [float(res.get("J", J_compare_old))]
                vf_history = [float(res["vf"])]
                lambda_plateau_vf_history = [float(res["vf"])]
                plateau_kappa_min_hits = 0
                plateau_last_vf = None
                hard_shift_cooldown_left = 0
                hard_shift_recovery_left = 0
                refresh_volume_merit_inplace(cfg, res, hard_shift_only=True)
                write_xdmf(outdir, mesh, it, materials, lsf)
                if MPI.rank(MPI.comm_world) == 0:
                    print("[postprocess-start] it=%03d: stage-2 optimization converged at vf=%.6f, J_mech=%.6e. Enter LSF hard-shift post-processing toward vf_final=%.6f (base=%.6f, extra_drop=%.6f)" %
                          (int(it), float(res.get("vf", 0.0)), float(res.get("J", 0.0)),
                           float(current_postprocess_vf_final_target(cfg)),
                           float(cfg.get("vf_final_target", 0.0)),
                           float(cfg.get("postprocess_loss_envelope_extra_drop", 0.10))))
            else:
                converged = True
                write_xdmf(outdir, mesh, it, materials, lsf)
                termination_reason = str(cfg.get("_postprocess_target_stop_reason", "final-filter-cooldown"))
                termination_label = (
                    "POSTPROCESS TARGET"
                    if termination_reason.startswith("target")
                    else "FINAL-FILTER COOLDOWN"
                )
                sym_max_last = None
                sym_mean_last = None
                if use_symmetry_parameterization and (symmetry_map is not None):
                    # Collective: all ranks must participate.
                    sym_max_last, sym_mean_last = symmetry_residual_stats(lsf, symmetry_map)
                if MPI.rank(MPI.comm_world) == 0:
                    theta_msg = "theta_update=%.2f deg  theta_opt=%.2f deg" % (theta_update_deg, theta_opt_deg)
                    if theta_update_sym_deg is not None:
                        theta_msg += "  theta_update_sym=%.2f deg" % theta_update_sym_deg
                    print("\n[it %03d] vf=%.4f  vf_stage=%.4f  J_mech=%.6e = j_hb(%.6e) + j_ti(%.6e) ; J_merit=%.6e ; diag_j_vol(%.6e)  alpha=%.3e  %s  kappa=%.3e  time=%.2fs  [TERMINATED: %s]" %
                          (it, res["vf"], float(current_vf_target(cfg)), res["J"], res["J_hb"], res["J_ti"], res["J_compare"], res["J_vol"], res["alpha"], theta_msg, kappa, time.time()-t0, termination_label))
                    print("[it %03d-metrics] ha=%.3e  H=%.3e  hb=%.3e  R_TI=%.3e  cref=%.3e  phi_TI=%.3e  vf=%.4f  vf_stage=%.4f  vf-vf_stage=%.6f  dv=%.4f  dir=%s  lambda_v=%.3e  vol_strength=%.3f  ||g_obj_t||=%.3e  ||g_mech_perp||=%.3e  ||g_vol_t||=%.3e  lambda_ratio_eff=%.3e  <g,psi>/||psi||^2=%.3e  <gv,psi>/||psi||^2=%.3e  <gt,gv_t>/||gv_t||^2=%.3e  ||gv_t||^2=%.3e" %
                          (it, float(res["ha"]), float(res["H"]), float(res["hb"]), float(res["R_ti"]), float(res["cref"]), float(res["phi_TI"]), float(res["vf"]), float(current_vf_target(cfg)), float(res["vf_constraint_residual"]), float(current_stage_dv(cfg)), str(direction_info.get("mode", "n/a")), float(direction_info.get("lambda_v_active", 0.0)), float(direction_info.get("recovery_strength", 0.0)), float(direction_info.get("mech_tangent_norm", float("nan"))), float(direction_info.get("mech_perp_norm", float("nan"))), float(direction_info.get("vol_tangent_norm", float("nan"))), float(direction_info.get("lambda_ratio_eff", float("nan"))), float(direction_info.get("mech_tangent_coeff", 0.0)), float(direction_info.get("vol_tangent_coeff", 0.0)), float(direction_info.get("mech_vol_coeff", 0.0)), float(direction_info.get("vol_tangent_norm_sq", 0.0))))
                    if psi_stats_now is not None:
                        print("[psi-stats] it=%03d min=%.4e max=%.4e mean=%.4e std=%.4e l2=%.4e" %
                              (it, float(psi_stats_now["min"]), float(psi_stats_now["max"]), float(psi_stats_now["mean"]), float(psi_stats_now["std"]), float(psi_stats_now["l2"])))
                        print("[psi-stats] it=%03d q01=%.4e q10=%.4e q50=%.4e q90=%.4e q99=%.4e" %
                              (it, float(psi_stats_now["q01"]), float(psi_stats_now["q10"]), float(psi_stats_now["q50"]), float(psi_stats_now["q90"]), float(psi_stats_now["q99"])))
                    if symmetry_mode == "cubic":
                        cubic_last = cubic_residual_metrics(res["Chom"])
                        print("[it %03d-cubic] rN=%.3e rS=%.3e rF=%.3e" %
                              (it, cubic_last["rN"], cubic_last["rS"], cubic_last["rF"]))
                    if use_symmetry_parameterization and (symmetry_map is not None):
                        print("[it %03d-symmetry] mode=%s reps=%d residual_max=%.3e residual_mean=%.3e (tol=%.3e)" %
                              (it, symmetry_mode, int(symmetry_map["n_reps"]), sym_max_last, sym_mean_last, symmetry_assert_tol))
                        if sym_max_last > symmetry_assert_tol:
                            print("[symmetry-warning] it=%03d residual %.3e exceeds tol %.3e" %
                                  (it, sym_max_last, symmetry_assert_tol))
                    # Final iteration: always print homogenized tensor in Voigt form.
                    print("[it %03d-Chom] Voigt6x6=\n%s" %
                          (it, np.array2string(res["Chom"], precision=4, suppress_small=True)))
                break

        # Refinement policy:
        # - Always refine on persistent solver failure.
        # - Do NOT refine just because kappa hit kappa_min; let stage controller continue.
        # - Only refine for topology convergence once final vf target is satisfied and theta gate fails.
        refine_due_to_convergence = final_target_satisfied and conv_ok and (not step14_ok)
        if (solver_fail or refine_due_to_convergence) and (not refine_cap_blocked):
            # mesh refinement: linear growth N <- N + step, capped by refine_n_max
            vf_before_refine = float(res["vf"])
            step = int(cfg.get("refine_step", 10))
            n_max = int(cfg["refine_n_max"])
            new_Nx = min(n_max, Nx + step)
            new_Ny = min(n_max, Ny + step)
            new_Nz = min(n_max, Nz + step)

            if (new_Nx, new_Ny, new_Nz) == (Nx, Ny, Nz):
                refine_cap_blocked = True
                if MPI.rank(MPI.comm_world) == 0:
                    print("[refine] cap reached (%d); keep current mesh %dx%dx%d and disable further refinement attempts" %
                          (n_max, Nx, Ny, Nz))
            else:
                Nx, Ny, Nz = new_Nx, new_Ny, new_Nz
                if MPI.rank(MPI.comm_world) == 0:
                    print("\n===== [refine] new mesh %dx%dx%d =====" % (Nx, Ny, Nz))
                # Re-mesh route (2D-style): rebuild a new structured mesh at target resolution.
                mesh_new = UnitCubeMesh(Nx, Ny, Nz)
                W, Vls, VtDG, VsDG, VDG0 = build_spaces(mesh_new, deg=cfg["deg"], tol=1e-10)
                l2_mass = build_l2_mass_matrix(Vls)
                # Cross-mesh lsf transfer: periodic CG1 → plain CG1 (old mesh) → plain CG1 (new mesh).
                # Step A: copy lsf from periodic CG1 to PLAIN CG1 on the SAME (old) mesh.
                #   This removes periodic slave DOFs that confuse cross-mesh evaluation.
                #   Same-mesh interpolate is exact and MPI-safe (no cross-partition needed).
                V_plain_old = FunctionSpace(mesh, "CG", 1)    # plain CG1 on OLD mesh, no periodic BC
                lsf_plain_old = Function(V_plain_old)
                lsf.set_allow_extrapolation(True)
                LagrangeInterpolator.interpolate(lsf_plain_old, lsf)
                # Verify: plain copy should have same min/max/vf as original
                _lsf_old_min = lsf.vector().min()
                _lsf_old_max = lsf.vector().max()
                _lsf_plain_min = lsf_plain_old.vector().min()
                _lsf_plain_max = lsf_plain_old.vector().max()
                if MPI.rank(MPI.comm_world) == 0:
                    print("[dbg-copy] periodic lsf: min=%.4e max=%.4e  →  plain lsf: min=%.4e max=%.4e" %
                          (_lsf_old_min, _lsf_old_max, _lsf_plain_min, _lsf_plain_max))

                # Step B: LagrangeInterpolator from plain CG1 (old mesh) → plain CG1 (new mesh).
                #   Both are standard Lagrange elements, no periodic BC → clean cross-mesh transfer.
                V_plain_new = FunctionSpace(mesh_new, "CG", 1)   # plain CG1 on NEW mesh
                lsf_transferred = Function(V_plain_new)
                if MPI.rank(MPI.comm_world) == 0:
                    print("[refine] cross-mesh lsf via LagrangeInterpolator (plain CG1 → plain CG1)")
                LagrangeInterpolator.interpolate(lsf_transferred, lsf_plain_old)

                # ---- DIAGNOSIS: verify transfer preserved zero-level set ----
                _dx_new = Measure("dx", domain=mesh_new)
                _Vol_new = float(assemble(Constant(1.0) * _dx_new))
                _lsf_xfer_min = lsf_transferred.vector().min()
                _lsf_xfer_max = lsf_transferred.vector().max()
                _vf_xfer_cts = float(assemble(
                    conditional(lt(lsf_transferred, Constant(cfg["threshold"])),
                                Constant(1.0), Constant(0.0)) * _dx_new
                )) / max(_Vol_new, 1e-30)
                if MPI.rank(MPI.comm_world) == 0:
                    print("[dbg-xfer] lsf_transferred: min=%.4e  max=%.4e  vf_cts=%.4f  (target %.4f  delta=%.4f)" %
                          (_lsf_xfer_min, _lsf_xfer_max, _vf_xfer_cts, vf_before_refine,
                           _vf_xfer_cts - vf_before_refine))

                # Mark materials on new mesh from transferred lsf (same-mesh project to DG0 → threshold)
                materials_new = MeshFunction("size_t", mesh_new, mesh_new.topology().dim())
                materials_new.set_all(0)
                mark_materials_from_lsf(mesh_new, lsf_transferred, materials_new, threshold=cfg["threshold"])
                chi_xfer = materials_to_chi(mesh_new, materials_new)
                vf_from_xfer = volume_fraction_from_chi(chi_xfer, mesh_new)
                if MPI.rank(MPI.comm_world) == 0:
                    print("[dbg-mat] vf_materials=%.4f  (target %.4f  delta=%.4f)" %
                          (vf_from_xfer, vf_before_refine, vf_from_xfer - vf_before_refine))

                # Keep transferred level-set field on the refined mesh instead of
                # rebuilding from chi_xfer, so we preserve continuous lsf information.
                lsf = _project(lsf_transferred, Vls)
                # Rebuild symmetry map on the refined mesh, then expand once immediately.
                if use_symmetry_parameterization:
                    symmetry_map = build_symmetry_index_map(Vls, Nx, Ny, Nz, symmetry_mode)
                    nucleation_map = symmetry_map
                    lsf = expand_symmetry_to_full(lsf, symmetry_map)
                else:
                    nucleation_map = build_full_dof_index_map(Vls, Nx, Ny, Nz)
                # Re-normalise to unit L2 norm so slerp N=1.0 assumption holds on new mesh.
                _lsf_norm_refine = float(np.sqrt(max(assemble(lsf * lsf * _dx_new), 0.0)))
                if _lsf_norm_refine > 1e-14:
                    lsf.vector()[:] /= _lsf_norm_refine
                    lsf.vector().apply("insert")
                if MPI.rank(MPI.comm_world) == 0:
                    print("[refine] lsf renorm: ||lsf||_L2 was %.4f, rescaled to 1.0" % _lsf_norm_refine)

                if use_symmetry_parameterization and (symmetry_map is not None):
                    lsf = expand_symmetry_to_full(lsf, symmetry_map)
                res_new, solver_fail_refine = evaluate_safe(mesh_new, W, lsf, materials_new, cfg, alpha=alpha_now, context="[it %03d post-refine]" % it)
                if solver_fail_refine:
                    raise RuntimeError("Solver still failed after refinement at it=%d." % it)
                mesh = mesh_new
                materials = materials_new
                dx = Measure("dx", domain=mesh)
                res = res_new
                J_old = res["J"]
                J_compare_old = res["J_compare"]
                J_history = [J_compare_old]  # reset after refinement (comparison objective not comparable across meshes)
                vf_history = [float(res["vf"])]
                lambda_plateau_vf_history = [float(res["vf"])]
                plateau_kappa_min_hits = 0
                plateau_last_vf = None
                refine_post_cooldown_left = int(cfg.get("refine_cooldown_steps", 2))
                hard_shift_cooldown_left = 0
                hard_shift_recovery_left = 0

        # Periodic XDMF output: every xdmf_stride steps, and always on the final
        # iteration if not converged (so we keep last state even without [14] success).
        if (it % xdmf_stride == 0) or (it == cfg["it_max"]):
            write_xdmf(outdir, mesh, it, materials, lsf)
        sym_max = None
        sym_mean = None
        if use_symmetry_parameterization and (symmetry_map is not None) and (it % max(1, symmetry_diag_stride) == 0):
            sym_max, sym_mean = symmetry_residual_stats(lsf, symmetry_map)
            if MPI.rank(MPI.comm_world) == 0:
                print("[it %03d-symmetry] mode=%s reps=%d residual_max=%.3e residual_mean=%.3e (tol=%.3e)" %
                      (it, symmetry_mode, int(symmetry_map["n_reps"]), sym_max, sym_mean, symmetry_assert_tol))
                if sym_max > symmetry_assert_tol:
                    print("[symmetry-warning] it=%03d residual %.3e exceeds tol %.3e" %
                          (it, sym_max, symmetry_assert_tol))
        if MPI.rank(MPI.comm_world) == 0:
            theta_msg = "theta_update=%.2f deg  theta_opt=%.2f deg" % (theta_update_deg, theta_opt_deg)
            if theta_update_sym_deg is not None:
                theta_msg += "  theta_update_sym=%.2f deg" % theta_update_sym_deg
            print("\n[it %03d] vf=%.4f  vf_stage=%.4f  J_mech=%.6e = j_hb(%.6e) + j_ti(%.6e) ; J_merit=%.6e ; diag_j_vol(%.6e)  alpha=%.3e  %s  kappa=%.3e  time=%.2fs" %
                  (it, res["vf"], float(current_vf_target(cfg)), res["J"], res["J_hb"], res["J_ti"], res["J_compare"], res["J_vol"], res["alpha"], theta_msg, kappa, time.time()-t0))
            print("[it %03d-metrics] ha=%.3e  H=%.3e  hb=%.3e  R_TI=%.3e  cref=%.3e  phi_TI=%.3e  vf=%.4f  vf_stage=%.4f  vf-vf_stage=%.6f  dv=%.4f  dir=%s  lambda_v=%.3e  vol_strength=%.3f  ||g_obj_t||=%.3e  ||g_mech_perp||=%.3e  ||g_vol_t||=%.3e  lambda_ratio_eff=%.3e  <g,psi>/||psi||^2=%.3e  <gv,psi>/||psi||^2=%.3e  <gt,gv_t>/||gv_t||^2=%.3e  ||gv_t||^2=%.3e" %
                  (it, float(res["ha"]), float(res["H"]), float(res["hb"]), float(res["R_ti"]), float(res["cref"]), float(res["phi_TI"]), float(res["vf"]), float(current_vf_target(cfg)), float(res["vf_constraint_residual"]), float(current_stage_dv(cfg)), str(direction_info.get("mode", "n/a")), float(direction_info.get("lambda_v_active", 0.0)), float(direction_info.get("recovery_strength", 0.0)), float(direction_info.get("mech_tangent_norm", float("nan"))), float(direction_info.get("mech_perp_norm", float("nan"))), float(direction_info.get("vol_tangent_norm", float("nan"))), float(direction_info.get("lambda_ratio_eff", float("nan"))), float(direction_info.get("mech_tangent_coeff", 0.0)), float(direction_info.get("vol_tangent_coeff", 0.0)), float(direction_info.get("mech_vol_coeff", 0.0)), float(direction_info.get("vol_tangent_norm_sq", 0.0))))
            if psi_stats_now is not None:
                print("[psi-stats] it=%03d min=%.4e max=%.4e mean=%.4e std=%.4e l2=%.4e" %
                      (it, float(psi_stats_now["min"]), float(psi_stats_now["max"]), float(psi_stats_now["mean"]), float(psi_stats_now["std"]), float(psi_stats_now["l2"])))
                print("[psi-stats] it=%03d q01=%.4e q10=%.4e q50=%.4e q90=%.4e q99=%.4e" %
                      (it, float(psi_stats_now["q01"]), float(psi_stats_now["q10"]), float(psi_stats_now["q50"]), float(psi_stats_now["q90"]), float(psi_stats_now["q99"])))
            if symmetry_mode == "cubic":
                cubic_now = cubic_residual_metrics(res["Chom"])
                print("[it %03d-cubic] rN=%.3e rS=%.3e rF=%.3e" %
                      (it, cubic_now["rN"], cubic_now["rS"], cubic_now["rF"]))
            chom_event = bool(
                stage_target_just_changed
                or hard_shift_done
                or stage2_final_filter_just_done
                or (
                    postprocess_active
                    and int(cfg.get("_postprocess_start_it", -1)) == int(it)
                )
            )
            if should_print_chom_voigt(
                cfg,
                it,
                event=chom_event,
                final_iteration=(int(it) == int(cfg.get("it_max", it))),
            ):
                print("[it %03d-Chom] Voigt6x6=\n%s" %
                      (it, np.array2string(res["Chom"], precision=4, suppress_small=True)))

    if MPI.rank(MPI.comm_world) == 0:
        stage2_converged_report = bool(cfg.get("_stage2_optimization_converged", False))
        postprocess_active_report = bool(cfg.get("_postprocess_active", False))
        final_vf_report = float(res.get("vf", float("nan")))
        postprocess_target_report = float(current_postprocess_vf_final_target(cfg))
        postprocess_tol_report = stage_success_tolerance(
            cfg, dv=float(cfg.get("vf_stage_dv_min", 0.002))
        )
        postprocess_target_reached_report = bool(
            postprocess_active_report
            and np.isfinite(final_vf_report)
            and np.isfinite(postprocess_target_report)
            and final_vf_report <= postprocess_target_report + postprocess_tol_report
        )
        reported_converged = bool(converged or stage2_converged_report)
        print("[done] results in: %s  (converged=%s, stage2_converged=%s, postprocess_active=%s, postprocess_target_reached=%s, vf=%.6f, vf_final_target=%.6f, vf_final_tol=%.3e)" %
              (outdir, reported_converged, stage2_converged_report,
               postprocess_active_report, postprocess_target_reached_report,
               final_vf_report, postprocess_target_report, postprocess_tol_report))
    if MPI.rank(MPI.comm_world) == 0:
        sys.stdout = _stdout
        log_file.close()

if __name__ == "__main__":
    run()
