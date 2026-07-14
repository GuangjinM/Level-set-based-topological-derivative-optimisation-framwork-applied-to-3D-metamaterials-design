# -*- coding: utf-8 -*-
"""
Init3d.py
Initialization/configuration for 3D TD + Level-set microstructure design (FEniCS 2019)

This mirrors your 2D file structure: definition.py / Init.py / main.ipynb (here main_3d.py).
"""

from fenics import *
import numpy as np

def init_3d():
    # ============================================================
    # Key knobs for the stage-2 volume-continuation experiment
    # ============================================================
    # Keep the reported objective purely mechanical.  These settings only
    # control the optimizer's volume target, KKT/AL multiplier and acceptance
    # merit during the second stage.
    stage2_volume_continuation_enabled = True
    stage2_merit_linesearch_enabled = True
    stage2_rare_nucleation_enabled = True
    stage2_enter_on_plateau_nucleation = True
    stage2_plateau_lambda_warmstart_iters = 8
    stage2_plateau_lambda_warmstart_floor_factor = 1.0
    # Keep stage 1 purely mechanical.  Stage-2 volume continuation starts only
    # when the plateau/nucleation trigger calls force_stage2_volume_continuation_start().
    stage2_auto_start_enabled = False
    stage2_start_iter = 30
    stage2_start_vf = 0.40
    stage2_volume_end_vf = 0.10
    # Stage 2 is the actual constrained-optimization phase.  Once vf reaches
    # this target, stop lowering the stage target, smooth the topology under a
    # vf guard, and judge convergence before post-processing begins.
    stage2_stop_for_convergence_enabled = True
    stage2_convergence_vf = stage2_volume_end_vf
    stage2_final_filter_enabled = True
    stage2_final_filter_radius_factor = 1.0
    stage2_final_filter_vf_rel_change_max = 0.05
    stage2_final_filter_vf_max = 0.11
    stage2_final_filter_vf_increase_allow_abs = 0.005
    stage2_final_filter_delete_bias = True
    stage2_final_filter_post_cooldown_steps = 0

    # Volume-continuation target ladder.  Use the R03D closed-loop strategy:
    # moderate stage gaps give a meaningful AL residual without making the
    # volume direction dominate the mechanical search.
    stage2_vf_stage_dv0 = 0.040
    stage2_vf_stage_dv_min = 0.007
    stage2_vf_stage_dv_levels = (0.040, 0.025, 0.015, 0.007)
    # Low-vf stage target drops are capped because a fixed absolute dv becomes
    # mechanically severe once the topology is sparse.
    stage2_vf_stage_dv_caps = (
        (0.35, 0.020),
        (0.20, 0.015),
    )
    stage2_vf_stage_tol = 0.0005
    stage2_vf_stage_success_window = 8
    stage2_vf_stage_plateau_tol = 2e-4
    # Stage-2 settle gate: reaching vf_stage starts a local constrained
    # stationarity check. The stage target advances only after this gate passes.
    stage2_settle_enabled = True
    stage2_settle_min_iters = 6
    stage2_settle_min_accepts = 4
    stage2_settle_j_window = 6
    stage2_settle_j_rel_tol = 5e-3
    stage2_settle_theta_deg = 60.0
    stage2_settle_theta_low_vf_deg = 55.0
    stage2_settle_theta_final_deg = 50.0
    stage2_settle_low_vf_threshold = 0.20
    stage2_settle_use_symmetry_theta = True
    stage2_settle_theta_sym_margin_deg = 5.0
    # Safety fallback for settle mode: if theta_update stays high but the
    # current constrained subproblem stops improving and J does not worsen,
    # allow a guarded advance with a very small next dv.
    stage2_settle_exhaust_enabled = True
    stage2_settle_max_iters = 40
    stage2_settle_final_max_iters = 80
    stage2_settle_exhaust_window = 12
    stage2_settle_exhaust_theta_drop_tol_deg = 2.0
    stage2_settle_exhaust_j_improve_tol = 1e-3
    stage2_settle_exhaust_j_worsen_tol = 5e-3
    stage2_settle_exhaust_max_advance_dv = stage2_vf_stage_dv_min
    stage2_settle_record_all_evals = True
    stage2_settle_exhaust_allow_stall_release = True
    # Final-stage escape: if the 0.10 target has already been overshot and the
    # angle gates are satisfied, stop optimizing stage2 instead of letting a
    # worsening merit window keep deleting material indefinitely.
    stage2_final_settle_force_enabled = True
    stage2_final_settle_force_min_iters = 12
    stage2_final_settle_force_overshoot_abs = 0.002
    # If a non-final stage has already overshot below vf_stage, release the
    # settle gate once either mechanical plateau signal is present: local
    # merit flattening or a hard line-search stall. The force_max_iters cap
    # is counted from the first overshoot hit, so min age/accept counts are
    # included in the same window instead of added after it.
    stage2_settle_overshoot_release_enabled = True
    stage2_settle_overshoot_release_final_enabled = False
    stage2_settle_overshoot_min_iters = 6
    stage2_settle_overshoot_min_accepts = stage2_settle_min_accepts
    stage2_settle_overshoot_tol_factor = 1.0
    stage2_settle_overshoot_require_stall = True
    stage2_settle_overshoot_require_j_ok = True
    stage2_settle_overshoot_force_max_iters = 40
    # None means overshoot-safe advances use the normal stage-dv ladder above,
    # still subject to low-vf caps.
    stage2_settle_overshoot_max_advance_dv = None
    # If vf is within the target tolerance but not below the overshoot
    # threshold, a hard line-search stall or a flat merit window means the
    # local constrained subproblem has no useful accepted steps left.  Release
    # this non-overshoot hit gate after the normal minimum age, without waiting
    # for additional accepts that may never occur.
    stage2_settle_hit_release_enabled = True
    stage2_settle_hit_release_final_enabled = False
    stage2_settle_hit_require_stall = True
    stage2_settle_hit_require_j_ok = True

    # AL/KKT multiplier used in the Hilbertian direction:
    #   d = g_mech_perp + lambda_v * g_vol_tangent
    # with lambda_v derived from the active volume constraint residual.
    stage2_al_mu0 = 0.0
    # The raw AL response can be firm because the actual lambda_v authority is
    # capped by the ratio/mechanical-reference/vf-rate controls below.
    stage2_al_rho0 = 0.40
    stage2_al_rho_grow = 1.25
    stage2_al_rho_shrink = 0.90
    stage2_al_rho_min = 0.10
    stage2_al_rho_max = 1.20
    stage2_al_mu_max = 0.010
    stage2_al_mu_overshoot_release_enabled = True
    stage2_al_mu_overshoot_decay = 0.5
    stage2_al_mu_overshoot_tol_factor = 1.0
    stage2_lambda_abs_cap = 0.012
    stage2_lambda_ratio_cap = 0.18
    stage2_lambda_ratio_cap_low_vf = stage2_lambda_ratio_cap
    stage2_lambda_ratio_low_vf_threshold = 0.35
    stage2_lambda_ratio_low_vf_blend_width = 0.0
    # R03D-style controller: do not hard-bind AL lambda_v to the dv ladder.
    # The vf-rate gain below is the main closed-loop speed knob.
    stage2_lambda_respect_dv_ratio_cap = False
    stage2_lambda_smoothing = 0.50
    stage2_vf_rate_control_enabled = True
    stage2_vf_rate_window = 12
    stage2_vf_rate_target_iters = 80
    stage2_vf_rate_min_drop_abs = 8e-5
    stage2_vf_rate_max_drop_abs = 7e-4
    stage2_vf_rate_slow_factor = 0.55
    stage2_vf_rate_fast_factor = 1.60
    stage2_vf_rate_gain0 = 1.0
    stage2_vf_rate_gain_min = 0.45
    stage2_vf_rate_gain_max = 2.20
    stage2_vf_rate_gain_grow = 1.20
    stage2_vf_rate_gain_shrink = 0.75
    stage2_vf_rate_gain_relax = 0.08
    stage2_vf_rate_update_cooldown_iters = 5
    stage2_vf_rate_min_gap_abs = 1.5e-3
    stage2_lambda_mech_ref_floor_enabled = True
    stage2_lambda_mech_ref_floor_frac = 0.15
    stage2_lambda_mech_ref_decay = 0.98
    stage2_merit_relax = 5e-3
    stage2_merit_relax_below_target = 2e-4
    stage2_merit_relax_below_target_disable_auto = True
    # R03D globalization: allow a separate near-miss rescue path to
    # temporarily raise the effective AL-merit allowance, then decay it back.
    stage2_merit_relax_auto_enabled = True
    stage2_merit_relax_auto_max = 2e-2
    stage2_merit_relax_auto_safety = 1.20
    stage2_merit_relax_auto_decay = 0.70
    stage2_stall_rescue_accept_enabled = True
    stage2_stall_rescue_vf_reduction_frac = 0.01
    stage2_stall_rescue_vf_reduction_abs = 1e-5
    stage2_stall_rescue_vf_tol_factor = 0.05
    stage2_stall_rescue_kappa_increase_factor = 1.05
    # Dynamic AL merit: mu/rho are fixed during each line-search and may update
    # after accepted iterations.  The high-level loss envelope below governs
    # the low-vf mechanical/volume tradeoff.
    stage2_merit_freeze_enabled = False
    stage2_merit_freeze_runtime_penalty_cap_enabled = False
    stage2_merit_stage_budget_enabled = False
    stage2_merit_stage_budget_eta = 0.10
    stage2_merit_stage_budget_eta_min = 0.005
    stage2_merit_stage_budget_mu_share = 0.50
    stage2_merit_stage_budget_j_floor = 1e-8
    stage2_merit_stage_budget_eps = 1e-12
    stage2_merit_stage_mu_max = stage2_al_mu_max
    stage2_merit_stage_rho_max = stage2_al_rho_max
    stage2_loss_envelope_enabled = True
    stage2_loss_envelope_vf_anchor = 0.30
    stage2_loss_envelope_vf_goal = stage2_convergence_vf
    stage2_loss_envelope_total_rel = 1.0
    stage2_loss_envelope_power = 1.0
    stage2_loss_envelope_update_anchor_on_improve = True
    stage2_loss_envelope_improve_rel_tol = 1e-4
    stage2_loss_envelope_stop_tol_rel = 1e-3
    stage2_loss_envelope_min_budget_abs = 0.0
    stage2_merit_stage_budget_lambda_projection_enabled = False
    # Optional legacy mechanical cap.  Disabled by default so line-search
    # acceptance remains merit-based, matching the R03D production path.
    stage2_merit_step_jmech_cap_enabled = False
    stage2_merit_step_jmech_cap_stage2_only = False
    stage2_merit_step_jmech_cap_rel = 2e-3
    stage2_merit_step_jmech_cap_j_floor = 1e-12
    stage2_merit_below_target_mech_strict = False
    stage2_merit_below_target_mech_relax = 0.0
    # Legacy one-sided slow-progress watchdog.  Kept disabled; the vf-rate
    # controller and stall diagnosis own this behavior.
    stage2_slow_progress_watchdog_enabled = False
    stage2_slow_progress_window = 20
    stage2_slow_progress_consecutive_hits = 1
    stage2_slow_progress_min_avg_drop_abs = 1.2e-4
    stage2_slow_progress_min_avg_drop_rel = 0.0
    stage2_slow_progress_min_gap_abs = 1.5e-3
    stage2_slow_progress_min_gap_tol_factor = 3.0
    stage2_slow_progress_kappa_max = 3e-3
    stage2_slow_progress_require_controller_cap = True
    stage2_slow_progress_controller_cap_fraction = 0.95
    stage2_slow_progress_lambda_ratio_step = 0.02
    stage2_slow_progress_lambda_ratio_grow = 1.25
    stage2_slow_progress_lambda_ratio_cap_low_vf_max = stage2_lambda_ratio_cap
    stage2_slow_progress_lambda_ratio_decay = 0.75
    stage2_slow_progress_cooldown_iters = 10
    # Keep the AL merit penalty comparable to the mechanical scale at low vf.
    stage2_merit_penalty_ratio_cap_enabled = True
    stage2_merit_penalty_ratio_cap_vf_threshold = stage2_lambda_ratio_low_vf_threshold
    stage2_merit_penalty_ratio_cap = 1.5
    stage2_merit_penalty_ratio_cap_j_floor = 1e-8
    # Diagnose hard line-search stalls before changing the AL controller.
    stage2_stall_diagnosis_enabled = True
    stage2_stall_overdrive_vf_reduction_frac = 0.20
    stage2_stall_overdrive_vf_reduction_abs = 2e-5
    stage2_stall_overdrive_vf_tol_factor = 0.25
    stage2_stall_overdrive_mech_increase_rel = 5e-3
    stage2_stall_overdrive_mech_increase_abs = 0.0
    stage2_stall_overdrive_mech_penalty_dominance = 1.0
    stage2_stall_merit_excess_rel = 1e-5
    stage2_stall_weak_vf_reduction_frac = 0.08
    stage2_stall_weak_vf_reduction_abs = 1e-5
    stage2_stall_weak_vf_tol_factor = 0.10
    stage2_stall_overdrive_rho_shrink = 0.75
    stage2_stall_overdrive_mu_decay = 0.65
    stage2_stall_overdrive_lambda_decay = 0.70
    stage2_stall_conflict_lambda_decay = 0.85
    stage2_stall_watchdog_enabled = True
    stage2_stall_weak_grow_before_dv_shrink = 2
    stage2_stall_nonweak_soft_reset_after = 8
    stage2_stall_nonweak_allow_fallback_after = 12
    stage2_stall_escape_rho_shrink = 0.80
    stage2_stall_escape_mu_decay = 0.70
    stage2_stall_escape_lambda_decay = 0.60
    stage2_stall_escape_hold_when_fallback_allowed = True
    stage2_stall_nonweak_shrink_dv_after_fallback = True
    stage2_stall_wait_enabled = True
    stage2_stall_wait_max_skip_iters = 120
    stage2_stall_wait_vf_tol = 1e-10
    stage2_same_state_stall_wait_enabled = True
    stage2_same_state_stall_wait_after = 2
    stage2_same_state_stall_wait_max_skip_iters = 30
    stage2_same_state_stall_wait_j_rel_tol = 1e-8
    stage2_same_state_stall_wait_lambda_rel_tol = 1e-8
    stage2_post_nucleation_takeover_enabled = True
    stage2_post_nucleation_takeover_iters = 120
    stage2_post_nucleation_takeover_min_gap_abs = 1.5e-3
    stage2_post_nucleation_takeover_gain_grow = 1.20
    stage2_post_nucleation_takeover_gain_floor = 0.90
    stage2_post_nucleation_takeover_suppress_fallback = True
    # Keep stage2 acceptance merit-only.  Once vf is low, use the same tight
    # J_merit relaxation as the below-target gate instead of adding separate
    # J_mech/J_hb acceptance tests.
    stage2_merit_relax_low_vf_threshold = 0.25
    # Legacy/debug knobs kept in the config.  Runtime now follows R03D:
    # below-target tightening applies to the whole line search, not only to
    # trials whose vf drop exceeds a threshold.
    stage2_merit_relax_below_target_drop = 2e-4
    stage2_merit_relax_below_target_drop_tol = 1e-4
    # Kept for quick ablation/debugging, but disabled by default so the
    # R03D vf-rate controller is the single runtime lambda_v gain loop.
    stage2_adaptive_lambda_ratio_enabled = False
    stage2_adaptive_lambda_ratio_scale_init = 1.0
    stage2_adaptive_lambda_ratio_scale_min = 0.25
    stage2_adaptive_lambda_ratio_scale_max = 4.0
    stage2_adaptive_lambda_ratio_low_vf_scale_max = 2.0
    stage2_adaptive_lambda_ratio_grow = 1.20
    stage2_adaptive_lambda_ratio_shrink = 0.60
    stage2_adaptive_lambda_ratio_mild_shrink = 0.80
    stage2_adaptive_lambda_ratio_stall_shrink = 0.50
    stage2_adaptive_lambda_ratio_slow_hits = 3
    stage2_adaptive_lambda_ratio_stall_low_eff = 0.015
    stage2_adaptive_lambda_ratio_low_stall_hits = 2
    stage2_adaptive_lambda_ratio_vf_drop_min_abs = 2.0e-4
    stage2_adaptive_lambda_ratio_vf_drop_min_gap_fraction = 0.015
    stage2_adaptive_lambda_ratio_vf_drop_max_gap_fraction = 0.12
    stage2_adaptive_lambda_ratio_j_worsen_tol = 2.0e-3
    stage2_adaptive_lambda_ratio_j_ok_tol = 5.0e-3
    stage2_adaptive_hold_al_on_ratio_shrink = True

    # Nucleation is now a rare fallback, not the volume controller.
    stage2_min_nucleation_gap_iters = 60
    stage2_nucleation_reject_cooldown_iters = 30
    stage2_nucleation_max_abs_drop = 0.004
    stage2_nucleation_max_gap_fraction = 0.25
    stage2_nucleation_merit_relax = 5e-3
    stage2_nucleation_merit_relax_auto_enabled = True
    stage2_nucleation_merit_relax_auto_max = 3e-2
    stage2_nucleation_merit_relax_auto_small_drop = 1e-3
    stage2_nucleation_max_trials = 3
    stage2_nucleation_retry_drop_factor = 0.5
    stage2_nucleation_retry_next_batch = True
    stage2_nucleation_retry_min_abs_drop = 5e-4

    # ------------------------------
    # Mesh resolution (start small; refine later)
    # ------------------------------
    Nx = 120
    Ny = 120
    Nz = 120
    deg = 1  # CG degree for level-set
    vf_final_target = 0.05
    it_max = 600     # max iterations
    # Switch hard-shift channel by current vf:
    # - vf >= switch_to_shift_vf: use ranked nucleation channel
    # - vf <  switch_to_shift_vf: use uniform shift channel psi <- psi - c
    hard_shift_switch_to_shift_vf = stage2_convergence_vf
    hard_shift_shift_factor = 0.95  # UTI run-result-tuned LSF hard shift
    hard_shift_aggressive_shift_factor = 0.8
    # Post-processing begins only after stage-2 convergence at vf ~= 0.10.
    # It uses uniform LSF hard shifts toward vf_final_target, separated by
    # mechanical-relaxation iterations, instead of being counted as stage-2
    # optimization progress.
    postprocess_loss_envelope_extra_drop = 0.10
    postprocess_hard_shift_gap_iters = 20
    postprocess_hard_shift_min_gap_iters = 20
    postprocess_hard_shift_max_gap_iters = 30
    postprocess_hard_shift_plateau_window = 5
    postprocess_hard_shift_plateau_rel_tol = 2e-4
    postprocess_hard_shift_max_abs_drop = 0.008
    postprocess_hard_shift_gap_fraction = 0.45
    postprocess_hard_shift_min_abs_drop = 0.003
    postprocess_terminate_on_target = True
    postprocess_terminate_immediate_on_target = True
    hard_shift_rebound_early_shift_enabled = True
    hard_shift_rebound_trigger_vf = 0.3
    hard_shift_rebound_hits_needed = 2

    hard_shift_factor = 0.99 # rare stage-2 nucleation; capped again by stage2_nucleation_max_abs_drop

    lambda_v_seed_enabled = False
    lambda_v_seed_ratio = 0.002

    # Global absolute cap for lambda_v across seed / stage / nucleation / plateau updates.
    lambda_v_adapt_max_abs = stage2_lambda_abs_cap
    lambda_v_adapt_max_abs_after_iter = stage2_lambda_abs_cap
    lambda_v_adapt_max_abs_after_iter_start = 30

    # ------------------------------
    # Level-set update parameters (Amstutz-style)
    # ------------------------------
    kappa0 = 0.03 # step length in Eq. (4.11)-style slerp
    kappa_min = 1e-4  # minimum line-search step length
    delta = 0.7  # line search: first two kappa shrinks multiply by delta
    delta_ls_tail = 0.5  # line search: from 3rd shrink onward multiply kappa by this
    # After J_try<=J_allow accept: first N accepts use early factor, then default factor (cap at kappa0).
    kappa_increase_factor_early = 1.1
    kappa_increase_early_count = 2
    kappa_increase_factor = 1.2


    # Line search (re-evaluate full objective each trial => expensive in 3D)
    do_full_linesearch = True  # True = use J decrease to accept; False = accept first trial
    # Iteration-to-iteration acceptance objective.  Stage 1 uses pure J;
    # active stage 2 switches the same J_accept/J_compare field to AL merit.
    J_compare_vf_weight = 0.003

    # Termination ([13] comparison-objective history then [14] theta)
    eps_theta_deg = 30.0   # final-angle tolerance in degrees
    refine_step = 10      # mesh refinement: linear growth N <- N + refine_step
    refine_n_max = Nx + 10    # mesh refinement cap: stop at initial Nx + 10
    xdmf_stride = 10      # write XDMF every 10 iterations (reduce I/O)
    # Pentamode-like diagnostic from homogenized tensor eigen spectrum (Voigt 6x6):
    # significant eigen if |lambda| > max(abs_tol, rel_tol*lambda_max_abs)
    print_chom_voigt_each_iter = True  # True => print full Chom(6x6) every iteration
    print_chom_voigt_stride = 1  # when not printing each iteration, print every N iterations
    # [13] relative comparison-objective convergence:
    # max|J_cmp_{n-k}-J_cmp_n| / max(|J_cmp_n|, J_min) < eps_J (m=5 step window)
    J_min = 1e-12         # safeguard to avoid division by zero
    J_window = 5
    eps_J = 2e-4          # relative tolerance

    # ------------------------------
    # Material (two-phase isotropic). Formula: gamma* in matrix (solid), 1/gamma* in inclusion (void).
    # ------------------------------
    E0         = 1.0    # solid Young's modulus
    gamma_star = 1e-3   # E_void/E_solid => E_void = gamma_star*E0
    nu         = 0.30   # Poisson ratio (kept constant for both phases)

    # ------------------------------
    # Objective settings (uncoupled transverse isotropic / UTI):
    #   The raw Chom matrix is kept unchanged. The objective scalars use
    #   tetragonal representative components:
    #     C11_t = 0.5*(C1111 + C2222)
    #     C12_t = 0.5*(C1122 + C2211)
    #     C13_t = 0.25*(C1133 + C3311 + C2233 + C3322)
    #     C44_t = 0.5*(C1313 + C2323)
    #     C33_t = C3333
    #     C66_t = C1212
    #   hb = C11_t + C12_t - C13_t - C33_t
    #   ha = 5*C11_t - 2*C33_t - 7*C12_t + 4*C13_t - 6*C44_t
    #   H  = C11_t + C33_t - 2*C13_t - 4*C44_t
    #   R_TI = C11_t - C12_t - 2*C66_t
    #   cref = 0.5 * (C11_t + C33_t) + eps_denom  (diagnostics / J_ti normalization)
    #   J_hb = beta_a*hb^2 + beta_b/(ha^2+H^2+eps_denom)
    #   J_ti uses one of:
    #     - "ha_h": alpha * R_TI^2 / (ha^2 + H^2 + eps)
    #     - "cref": alpha * (R_TI / cref)^2
    #     - "none": alpha * R_TI^2
    #   Volume control is now handled outside Phi(C) by a Hilbertian projection
    #   direction; J_vol=lambda_v*vf is kept only as a diagnostic/adaptation scale.
    # ------------------------------
    # Keep the stable UTI denominator barrier, but give hb and the normalized
    # TI residual a modestly stronger late-stage pull toward the 1e-5 regime.
    beta_a = 3.0  # objective weight for hb^2
    beta_b = 1e-5  # weak barrier on 1/(ha^2+H^2+eps)
    beta = beta_a  # legacy single-beta alias
    alpha = 0.10  # maximum weight on the cref-normalized TI residual
    # TI-penalty continuation is stage-wise: alpha stays fixed inside each
    # vf_stage subproblem and can only change after settle/advance. Once stage2
    # is active, alpha grows by a fixed multiplier until alpha_max; no volume
    # schedule cap or J_ti/J_hb ratio cap is applied.
    alpha_ti_continuation_enabled = True
    alpha_ti_min_factor = 0.10
    alpha_ti_min = 5e-4
    alpha_ti_floor = 5e-4
    alpha_ti_stage2_only = True
    alpha_ti_schedule_cap_enabled = False
    alpha_ti_ramp_start_vf = 0.35
    alpha_ti_ramp_end_vf = 0.15
    alpha_ti_ramp_power = 1.5
    alpha_ti_growth_per_stage = 2.5
    alpha_ti_ratio_cap = None
    alpha_ti_ratio_cap_strict = False
    alpha_ti_ratio_ref_floor = 0.0
    alpha_ti_allow_decrease = True
    alpha_ti_update_rel_tol = 0.0
    alpha_ti_update_abs_tol = 1e-12
    # Re-evaluate the current state immediately after a stage-wise alpha update
    # so objective/gradient/line-search comparisons use the same alpha.
    alpha_ti_refresh_rel_tol = 0.0
    alpha_ti_refresh_abs_tol = 1e-12
    # Hilbertian volume-control coefficient:
    #   d = g_mech_perp + lambda_v * g_vol_tangent
    # It starts from zero and is seeded/adapted later from the current
    # objective magnitude and volume-gap information.
    lambda_v = 0
    # Adapt lambda_v from the current stage difficulty. After the initial
    # evaluate, lambda_v is seeded to a nonzero value. Later stage /
    # nucleation / plateau updates are free to move away from that initial
    # seed and do not keep it as a persistent lower bound.
    # The target is set from the desired ratio between the volume term
    # and the pure mechanical term inside the Hilbertian direction:
    #   d = g_mech_perp + lambda_v * g_vol_tangent
    # with
    #   ||lambda_v g_vol_tangent|| ~= ratio_target * ||g_mech_perp||.
    # The ratio_target is now driven mainly by the current stage dv:
    #   dv=dv0   => use the mild floor ratio
    #   dv=dvmin => use the strongest ratio
    # so repeated dv shrink operations automatically ask for a stronger
    # volume component before another nucleation is needed.
    # When the configured dv ladder is active, use the matching discrete
    # ratio ladder below instead of interpolating between the endpoints.
    # Current warmstart / non-AL target range for the volume component:
    # about 4% to 16% of ||g_mech_perp||, with a softer low-vf ladder.
    # If True, keep lambda_v fixed at its initial seeded value during the
    # early geometry-forming stage. Release this freeze on the first accepted
    # lambda_v plateau event or the first accepted hard nucleation event.
    lambda_v_early_freeze_enabled = False

    # Master switch: update lambda_v right after accepted hard nucleation.
    nucleation_update_lambda_v = True
    # Backward-compatible alias (kept for old configs/logics).
    lambda_v_adapt_on_nucleation = True
    lambda_v_direction_ratio_min = 0.04
    lambda_v_direction_ratio_gain = 0.03529411764705882
    lambda_v_direction_ratio_max = 0.20
    lambda_v_direction_ratio_levels = (0.04, 0.08, 0.12, 0.16)
    lambda_v_low_vf_ratio_threshold = 0.35
    lambda_v_low_vf_direction_ratio_levels = (0.03, 0.05, 0.07, 0.09)
    lambda_v_direction_ratio_eps = 1e-12
    lambda_v_adapt_min_abs = 0.0
  
    # Plateau boost for lambda_v:
    # master switch for the whole plateau-lambda mechanism.
    # False => disable both lambda_v boost and its accept-cooldown branch.
    lambda_v_plateau_boost_enabled = False
    lambda_v_plateau_window = 20
    lambda_v_plateau_abs_drop_thresh = 2e-4
    lambda_v_plateau_rel_drop_thresh = 3.5e-4
    lambda_v_plateau_target_gap_abs = 5e-4
    lambda_v_plateau_target_gap_ratio = 0
    lambda_v_plateau_consecutive_hits = 1
    # If lambda_v is still zero when plateau boost triggers, seed from the
    # current adaptive target and only apply a mild multiplicative lift.
    lambda_v_plateau_boost_factor = 1.1
    # If lambda_v is already active, plateau boost does not multiply it by a
    # large factor anymore; instead it increases the target volume-component
    # ratio by an absolute 2 percentage points.
    lambda_v_plateau_ratio_step = 0.02
    lambda_v_plateau_cooldown_iters = 3
    # If True, after one successful plateau boost, the next plateau trigger
    # skips lambda_v boosting and directly forces a nucleation move.
    lambda_v_plateau_second_hit_force_nucleation = False
    lambda_v_plateau_accept_cooldown_enabled = False
    lambda_v_plateau_accept_cooldown_steps = 0
    # Optional hard cap for plateau-triggered boosts only.
    lambda_v_plateau_boost_max_abs = None


    # TI penalty normalization mode. Use the same cref-normalized residual as
    # the R03D optimizer so the TI term measures relative anisotropy instead of
    # shrinking automatically when the whole stiffness matrix scales down.
    ti_penalty_normalization_mode = "cref"
    # Backward-compat fallback used only when ti_penalty_normalization_mode is unset.
    ti_penalty_normalize_by_cref = True
    eps_denom = 1e-12
   


    # Geometric symmetry parameterization.
    # Options:
    #   "none"              : full DOF, no geometric symmetry reduction
    #   "cubic"             : old strict minimal wedge, 0 <= z <= y <= x <= 0.5
    #   "tetragonal_z"      : x-y fourfold symmetry with x/y/z mid-plane mirrors
    #   "tetragonal_z_rot4" : pure C4 rotation about z, without z mid-plane mirror
    symmetry_mode = "tetragonal_z_rot4"
    symmetry_assert_tol = 1e-12
    symmetry_diag_stride = 1
    psi_stats_stride = 1

    # Backward-compat aliases for older cubic-only configuration keys.
    use_strict_wedge_parameterization = (symmetry_mode == "cubic")
    use_minimal_wedge_parameterization = (symmetry_mode == "cubic")
    minimal_wedge_decimals = 10
    minimal_wedge_soft_eta = 1.0
    wedge_assert_tol = symmetry_assert_tol
    wedge_diag_stride = symmetry_diag_stride
    # Helmholtz smoothing on level-set:
    # - regular light smoothing every N accepted steps
    # - stronger emergency smoothing when solver fails (before refine)
    use_helmholtz_filter = True
    # Fail-recover: try Helmholtz radii = factor_i * h_min in order (weak -> stronger).
    # e.g. (0.5, 1.0, 1.5) => try 0.5*h_min, then 1*h_min, then 1.5*h_min.
    helmholtz_fail_radius_factors = (0.5, 1.0, 1.5)
    helmholtz_fail_max_tries = 3
    helmholtz_refine_radius_factor = 1
    # Trial-state PDE rescue from the R03D workflow.
    pre_jacobi_filter_enabled = True
    pre_jacobi_filter_radius_factors = (0.25, 0.50)
    pre_jacobi_filter_vf_rel_change_max = 0.002
    # Debug hook: force entering fail-recover branch even without solver failure.
    # - Set debug_force_fail_recover=True to enable.
    # - If debug_force_fail_recover_iters is empty, force every iteration.
    # - Otherwise force only listed iteration indices, e.g. (420, 557).
    debug_force_fail_recover = False
    debug_force_fail_recover_iters = ()
    # Volume control:
    # Outer controller: Deng-style stage-wise volume target
    #   current target v_stage, current decrement dv
    #   success -> next stage, failure -> shrink dv
    # Continuous correction now uses the Hilbertian projected direction above.
    # (Legacy/compat) Augmented-Lagrangian volume penalty parameters.
    # They remain disabled by default.
    rho_v = stage2_al_rho0
    mu_v0 = stage2_al_mu0
    vf_al_adapt_eps = 1e-12
    vf_al_adapt_good_ratio = 0.70
    vf_al_adapt_bad_ratio = 0.95
    vf_al_rho_grow = stage2_al_rho_grow
    vf_al_rho_shrink = stage2_al_rho_shrink
    vf_al_rho_min = stage2_al_rho_min
    vf_al_rho_max = stage2_al_rho_max

    use_hard_vf = False
    vf_target = None
    vf_constraint_target = None
    vf_bisect_tol = 5e-4
    vf_bisect_max_iter = 40

    # Backward-compat: older configs used vf_stage_ratio/vf_stage_tol.
    vf_stage_ratio = 1.0
    vf_stage_tol = stage2_vf_stage_tol
    vf_stage_dv0 = stage2_vf_stage_dv0
    vf_stage_dv_min = stage2_vf_stage_dv_min
    vf_stage_dv_levels = stage2_vf_stage_dv_levels
    vf_stage_shrink = 0.5
    vf_stage_tol_min = 0.002
    # Stage-controller hysteresis cap:
    # effective stage tolerance = min(vf_stage_tol, vf_stage_hysteresis_factor * dv)
    # so a stage can never be declared "success" inside the overlap with the
    # next shrunken stage when dv is already near dv_min.
    vf_stage_hysteresis_factor = 0.49
    vf_stage_success_window = stage2_vf_stage_success_window
    vf_stage_plateau_tol = stage2_vf_stage_plateau_tol
    # Optional relaxed-acceptance window right after stage target changes.
    stage_relax_tau = 0.0
    stage_relax_steps_total = 0
    # Lightweight progress markers for HPC debugging.
    # Prints entry/exit around init XDMF write and evaluate/homogenisation blocks.
    debug_progress_markers = False
    print_direction_chain_diagnostics = False

    # Plateau-triggered hard volume nucleation:
    # if vf is unchanged for the last `hard_shift_plateau_window` iterations,
    # line-search touches kappa_min on that same vf plateau, and the current
    # stage step has already shrunk to dv_min,
    # apply a local delete-material operation inside an adaptive inner band
    # defined by quantiles of d = threshold - psi over the current solid region.
    # First build the seed band from [q_lo, q_hi], then cap its geometric depth
    # by d_cap, and if that capped band is empty or insufficient, expand both
    # d-bounds outward by d_expand_step until it becomes usable.
    # The target remains:
    #   vf_new ~= hard_shift_factor * vf_current.
    use_plateau_hard_shift = True
    hard_shift_plateau_window = 30
    hard_shift_plateau_vf_tol = 1e-4
    hard_shift_kappa_min_hits = 3
    hard_shift_nucleation_quantile_lo = 0.10
    hard_shift_nucleation_quantile_hi = 0.35
    hard_shift_nucleation_d_cap = 0.40
    hard_shift_nucleation_d_expand_step = 0.03
    hard_shift_nucleation_psi_value = None
    post_nucleation_freeze_steps = 60
    post_nucleation_freeze_early_release_enabled = False
    post_nucleation_lambda_v_boost_factor = 1.0
    post_nucleation_lambda_zero_steps = 0
    # Aggressive low-vf uniform hard-shift regime:
    # if the next pre-shift vf rises above the previous post-shift vf for this
    # many consecutive uniform-shift triggers, switch to a stronger shift factor.
    hard_shift_uniform_exit_enabled = True
    hard_shift_uniform_exit_consecutive_hits = 3
    hard_shift_aggressive_max_steps = 50

    # If True, when hard nucleation cannot fully hit vf_target even after fallback expansion,
    # accept the best achievable reduction instead of raising an error.
    hard_shift_allow_partial_target = True
    hard_shift_kappa_reset = 0.5 * kappa0
    hard_shift_relax_steps = stage_relax_steps_total
    # Hard-shift cooldown:
    # after a hard nucleation move, use a few unconditional small-kappa steps with the
    # plain objective direction (no ranked deletion) so the interface can
    # rebuild on the new volume level before normal ranked deletion resumes.
    hard_shift_cooldown_steps = 3
    hard_shift_cooldown_kappa_factor = 0.1
    # Refine cooldown:
    # after remeshing, take a few unconditional small-kappa steps so the
    # transferred level-set can relax on the new mesh before full J-based
    # line-search resumes.
    refine_cooldown_steps = 2
    refine_cooldown_kappa_factor = 0.1
    # Ranked-deletion recovery after cooldown:
    # restore the extra deletion bias gradually over a few accepted steps.
    hard_shift_recovery_steps = 2
    hard_shift_recovery_stall_counts = True
    # Milestone smoothing (adaptive radius under vf-change guard):
    # when vf first crosses each target value, search the largest radius in
    # [0, vf_milestone_filter_radius_factor*hmin] that keeps relative vf change
    # <= milestone_filter_vf_rel_change_max.
    vf_milestone_filter_targets = (hard_shift_switch_to_shift_vf, vf_final_target)
    vf_milestone_filter_radius_factor = 0.5
    milestone_filter_vf_rel_change_max = 0.005
    # Fail-recover adaptive vf guard:
    # start from 10%, then 15%, 20%, ... (step) until recovery succeeds,
    # and finally one unconstrained pass if still needed.
    fail_recover_filter_vf_rel_change_start = 0.005
    fail_recover_filter_vf_rel_change_step = 0.05
    fail_recover_filter_vf_rel_change_max = 1.00
    # Final smoothing at termination:
    # choose the largest radius r in [0, final_filter_radius_max_factor*hmin]
    # such that relative vf change after filtering stays within:
    #   |vf_after - vf_before| / max(vf_before, 1e-12) <= final_filter_vf_rel_change_max
    final_filter_vf_rel_change_max = 0.005
    final_filter_radius_max_factor = 1.0
    final_filter_radius_search_max_iter = 10
    # After final filter is applied (only after pre-filter convergence is confirmed),
    # run a few cooldown iterations and then terminate.
    final_filter_post_cooldown_steps = 5
    final_filter_post_cooldown_kappa_factor = 0.1

    # Legacy ranked-deletion controls kept only for backward compatibility.
    # They are no longer used once the Hilbertian continuous volume controller
    # is active, but the discrete nucleation / hard-shift logic is unchanged.
    use_ranked_volume_enhancement = True
    # Extreme test: select ALL candidates in the narrow band (n_sel = n_candidates).
    # This is only for debugging; it can easily make line-search fail.
    vf_rank_select_all_candidates = False
    # Narrow band definition:
    # - Optional stage-gap scaling (fixed |psi| band, not hmin):
    #     band = vf_rank_band_min + (vf_rank_band_max - vf_rank_band_min)
    #            * min(1, (vf - vf_stage) / vf_rank_band_gap_scale_d0)
    #   Large gap -> wide band (strong deletion spread); small gap -> narrow band (less J_try noise).
    # - If vf_rank_band_stage_scaled=False: constant |psi| <= vf_rank_band
    # - If vf_rank_use_hmin_band=True: band = vf_rank_band_hmin_factor * hmin
    vf_rank_band_stage_scaled = True
    vf_rank_band_min = 0.03
    vf_rank_band_max = 0.20
    vf_rank_band_gap_scale_d0 = vf_stage_dv0
    vf_rank_use_hmin_band = False
    vf_rank_band_hmin_factor = 2.4
    vf_rank_band = 0.20  # legacy constant when vf_rank_band_stage_scaled=False
    vf_rank_kp = 6            # selected fraction p = clip(kp * e_stage, pmin, pmax)
    vf_rank_pmin = 0.01
    vf_rank_pmax = 0.20
    vf_rank_beta0 = 0.0
    vf_rank_kbeta = 3
    # Dual-band ranked deletion:
    # - boundary and interior channels can be toggled independently
    # - interior band is an auxiliary channel, starts at zero and ramps up as vf decreases
    vf_rank_use_boundary_band = True
    vf_rank_use_interior_band = False
    vf_rank_interior_depth_abs = 0.25
    vf_rank_interior_depth_band_mult = 2.0
    vf_rank_interior_progress_start = 0.20
    vf_rank_interior_progress_full = 0.85
    vf_rank_interior_progress_power = 1.0
    vf_rank_interior_max_strength = 0.35
    vf_rank_interior_p_scale = 0.50
    vf_rank_interior_pmax = 0.08
    vf_rank_interior_beta_scale = 0.35
    # ------------------------------
    # Cell PDE solver (periodic semi-definite + explicit nullspace)
    # ------------------------------
    cell_solver = {
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-12,
        "ksp_max_it": 20000,
        # Choose one: "gamg" (default robust) or "hypre" (boomeramg)
        "pc_type": "gamg",
        "pc_hypre_type": "boomeramg",
        # Keep hypre smoother symmetric if you run CG.
        "pc_hypre_boomeramg_relax_type_all": "symmetric-SOR/Jacobi",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "set_near_nullspace": True,
        # Robustness at scale: if CG fails with DIVERGED_INDEFINITE_PC, try these in order.
        "enable_ksp_fallback": True,
        "fallback_ksp_types": ("minres", "cg"),
    }

    # ------------------------------
    # Level-set initialisation
    # ------------------------------
    # Available modes:
    #   "corner_plus_center_spheres" : one central ellipsoidal void plus corner voids
    #   "center_sphere_only"         : only one central spherical void
    lsf_init_mode = "corner_plus_center_spheres"

    # Sign convention is unchanged for all modes:
    # material if lsf < 0, void if lsf > 0.
    if lsf_init_mode == "corner_plus_center_spheres":
        # Inside the unit cell this appears as one full void at the center
        # plus eight 1/8-voids at the corners.
        # Corner voids remain spherical; center void is switched to an ellipsoid.
        # Base sphere radius follows the same nominal void fraction ~= 0.40.
        sphere_radius = (0.40 * 3.0 / (8.0 * np.pi)) ** (1.0 / 3.0)
        # Raw axis ratios for center ellipsoid. We normalize them so that
        # ax*ay*az = sphere_radius^3 (center void volume close to original sphere).
        center_ellipsoid_ratio_x = 0.90
        center_ellipsoid_ratio_y = 1.00
        center_ellipsoid_ratio_z = 1.30
        ratio_geom_mean = (center_ellipsoid_ratio_x * center_ellipsoid_ratio_y * center_ellipsoid_ratio_z) ** (1.0 / 3.0)
        ax = sphere_radius * center_ellipsoid_ratio_x / ratio_geom_mean
        ay = sphere_radius * center_ellipsoid_ratio_y / ratio_geom_mean
        az = sphere_radius * center_ellipsoid_ratio_z / ratio_geom_mean
        lsf_expr = Expression(
            "((1.0 - sqrt(((x[0]-cx)*(x[0]-cx))/(ax*ax) + ((x[1]-cy)*(x[1]-cy))/(ay*ay) + ((x[2]-cz)*(x[2]-cz))/(az*az))) > "
            " (r - sqrt(((x[0] < 0.5) ? x[0] : 1.0-x[0]) * ((x[0] < 0.5) ? x[0] : 1.0-x[0]) + "
            "            ((x[1] < 0.5) ? x[1] : 1.0-x[1]) * ((x[1] < 0.5) ? x[1] : 1.0-x[1]) + "
            "            ((x[2] < 0.5) ? x[2] : 1.0-x[2]) * ((x[2] < 0.5) ? x[2] : 1.0-x[2])))) ? "
            " (1.0 - sqrt(((x[0]-cx)*(x[0]-cx))/(ax*ax) + ((x[1]-cy)*(x[1]-cy))/(ay*ay) + ((x[2]-cz)*(x[2]-cz))/(az*az))) : "
            " (r - sqrt(((x[0] < 0.5) ? x[0] : 1.0-x[0]) * ((x[0] < 0.5) ? x[0] : 1.0-x[0]) + "
            "            ((x[1] < 0.5) ? x[1] : 1.0-x[1]) * ((x[1] < 0.5) ? x[1] : 1.0-x[1]) + "
            "            ((x[2] < 0.5) ? x[2] : 1.0-x[2]) * ((x[2] < 0.5) ? x[2] : 1.0-x[2])))",
            degree=deg,
            r=float(sphere_radius),
            ax=float(ax),
            ay=float(ay),
            az=float(az),
            cx=0.5,
            cy=0.5,
            cz=0.5
        )
    elif lsf_init_mode == "center_sphere_only":
        # Choose void fraction ~= 0.35 so the material fraction is vf ~= 0.65.
        sphere_radius = (0.35 * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        lsf_expr = Expression(
            "r - sqrt((x[0]-cx)*(x[0]-cx) + (x[1]-cy)*(x[1]-cy) + (x[2]-cz)*(x[2]-cz))",
            degree=deg,
            r=float(sphere_radius),
            cx=0.5,
            cy=0.5,
            cz=0.5
        )
    else:
        raise ValueError("Unknown lsf_init_mode: %s" % str(lsf_init_mode))

    # Threshold: material if lsf < 0, void if lsf > 0
    threshold = 0.0+1e-6

    return dict(
        Nx=Nx, Ny=Ny, Nz=Nz, deg=deg,
        kappa0=kappa0, kappa_min=kappa_min, delta=delta, delta_ls_tail=delta_ls_tail,
        kappa_increase_factor_early=kappa_increase_factor_early,
        kappa_increase_early_count=kappa_increase_early_count,
        kappa_increase_factor=kappa_increase_factor, it_max=it_max,
        do_full_linesearch=do_full_linesearch,
        J_compare_vf_weight=J_compare_vf_weight,
        eps_theta_deg=eps_theta_deg, refine_step=refine_step, refine_n_max=refine_n_max, xdmf_stride=xdmf_stride,
        print_chom_voigt_each_iter=print_chom_voigt_each_iter,
        print_chom_voigt_stride=print_chom_voigt_stride,
        J_min=J_min, J_window=J_window, eps_J=eps_J,
        E0=E0, gamma_star=gamma_star, nu=nu,
        beta_a=beta_a, beta_b=beta_b, beta=beta, alpha=alpha,
        alpha_ti_continuation_enabled=alpha_ti_continuation_enabled,
        alpha_ti_min_factor=alpha_ti_min_factor,
        alpha_ti_min=alpha_ti_min,
        alpha_ti_floor=alpha_ti_floor,
        alpha_ti_stage2_only=alpha_ti_stage2_only,
        alpha_ti_schedule_cap_enabled=alpha_ti_schedule_cap_enabled,
        alpha_ti_ramp_start_vf=alpha_ti_ramp_start_vf,
        alpha_ti_ramp_end_vf=alpha_ti_ramp_end_vf,
        alpha_ti_ramp_power=alpha_ti_ramp_power,
        alpha_ti_growth_per_stage=alpha_ti_growth_per_stage,
        alpha_ti_ratio_cap=alpha_ti_ratio_cap,
        alpha_ti_ratio_cap_strict=alpha_ti_ratio_cap_strict,
        alpha_ti_ratio_ref_floor=alpha_ti_ratio_ref_floor,
        alpha_ti_allow_decrease=alpha_ti_allow_decrease,
        alpha_ti_update_rel_tol=alpha_ti_update_rel_tol,
        alpha_ti_update_abs_tol=alpha_ti_update_abs_tol,
        alpha_ti_refresh_rel_tol=alpha_ti_refresh_rel_tol,
        alpha_ti_refresh_abs_tol=alpha_ti_refresh_abs_tol,
        eps_denom=eps_denom,
        ti_penalty_normalize_by_cref=ti_penalty_normalize_by_cref,
        ti_penalty_normalization_mode=ti_penalty_normalization_mode,
        lambda_v=lambda_v,
        lambda_v_seed_enabled=lambda_v_seed_enabled,
        lambda_v_seed_ratio=lambda_v_seed_ratio,
        lambda_v_early_freeze_enabled=lambda_v_early_freeze_enabled,
        nucleation_update_lambda_v=nucleation_update_lambda_v,
        lambda_v_adapt_on_nucleation=lambda_v_adapt_on_nucleation,
        lambda_v_direction_ratio_min=lambda_v_direction_ratio_min,
        lambda_v_direction_ratio_gain=lambda_v_direction_ratio_gain,
        lambda_v_direction_ratio_max=lambda_v_direction_ratio_max,
        lambda_v_direction_ratio_levels=lambda_v_direction_ratio_levels,
        lambda_v_low_vf_ratio_threshold=lambda_v_low_vf_ratio_threshold,
        lambda_v_low_vf_direction_ratio_levels=lambda_v_low_vf_direction_ratio_levels,
        lambda_v_direction_ratio_eps=lambda_v_direction_ratio_eps,
        lambda_v_adapt_min_abs=lambda_v_adapt_min_abs,
        lambda_v_adapt_max_abs=lambda_v_adapt_max_abs,
        lambda_v_adapt_max_abs_after_iter=lambda_v_adapt_max_abs_after_iter,
        lambda_v_adapt_max_abs_after_iter_start=lambda_v_adapt_max_abs_after_iter_start,
        lambda_v_plateau_boost_enabled=lambda_v_plateau_boost_enabled,
        lambda_v_plateau_window=lambda_v_plateau_window,
        lambda_v_plateau_abs_drop_thresh=lambda_v_plateau_abs_drop_thresh,
        lambda_v_plateau_rel_drop_thresh=lambda_v_plateau_rel_drop_thresh,
        lambda_v_plateau_target_gap_abs=lambda_v_plateau_target_gap_abs,
        lambda_v_plateau_target_gap_ratio=lambda_v_plateau_target_gap_ratio,
        lambda_v_plateau_consecutive_hits=lambda_v_plateau_consecutive_hits,
        lambda_v_plateau_boost_factor=lambda_v_plateau_boost_factor,
        lambda_v_plateau_ratio_step=lambda_v_plateau_ratio_step,
        lambda_v_plateau_cooldown_iters=lambda_v_plateau_cooldown_iters,
        lambda_v_plateau_second_hit_force_nucleation=lambda_v_plateau_second_hit_force_nucleation,
        lambda_v_plateau_accept_cooldown_enabled=lambda_v_plateau_accept_cooldown_enabled,
        lambda_v_plateau_accept_cooldown_steps=lambda_v_plateau_accept_cooldown_steps,
        lambda_v_plateau_boost_max_abs=lambda_v_plateau_boost_max_abs,
        stage2_volume_continuation_enabled=stage2_volume_continuation_enabled,
        stage2_merit_linesearch_enabled=stage2_merit_linesearch_enabled,
        stage2_rare_nucleation_enabled=stage2_rare_nucleation_enabled,
        stage2_enter_on_plateau_nucleation=stage2_enter_on_plateau_nucleation,
        stage2_plateau_lambda_warmstart_iters=stage2_plateau_lambda_warmstart_iters,
        stage2_plateau_lambda_warmstart_floor_factor=stage2_plateau_lambda_warmstart_floor_factor,
        stage2_auto_start_enabled=stage2_auto_start_enabled,
        stage2_start_iter=stage2_start_iter,
        stage2_start_vf=stage2_start_vf,
        stage2_volume_end_vf=stage2_volume_end_vf,
        stage2_stop_for_convergence_enabled=stage2_stop_for_convergence_enabled,
        stage2_convergence_vf=stage2_convergence_vf,
        stage2_final_filter_enabled=stage2_final_filter_enabled,
        stage2_final_filter_radius_factor=stage2_final_filter_radius_factor,
        stage2_final_filter_vf_rel_change_max=stage2_final_filter_vf_rel_change_max,
        stage2_final_filter_vf_max=stage2_final_filter_vf_max,
        stage2_final_filter_vf_increase_allow_abs=stage2_final_filter_vf_increase_allow_abs,
        stage2_final_filter_delete_bias=stage2_final_filter_delete_bias,
        stage2_final_filter_post_cooldown_steps=stage2_final_filter_post_cooldown_steps,
        stage2_vf_stage_dv_caps=stage2_vf_stage_dv_caps,
        stage2_settle_enabled=stage2_settle_enabled,
        stage2_settle_min_iters=stage2_settle_min_iters,
        stage2_settle_min_accepts=stage2_settle_min_accepts,
        stage2_settle_j_window=stage2_settle_j_window,
        stage2_settle_j_rel_tol=stage2_settle_j_rel_tol,
        stage2_settle_theta_deg=stage2_settle_theta_deg,
        stage2_settle_theta_low_vf_deg=stage2_settle_theta_low_vf_deg,
        stage2_settle_theta_final_deg=stage2_settle_theta_final_deg,
        stage2_settle_low_vf_threshold=stage2_settle_low_vf_threshold,
        stage2_settle_use_symmetry_theta=stage2_settle_use_symmetry_theta,
        stage2_settle_theta_sym_margin_deg=stage2_settle_theta_sym_margin_deg,
        stage2_settle_exhaust_enabled=stage2_settle_exhaust_enabled,
        stage2_settle_max_iters=stage2_settle_max_iters,
        stage2_settle_final_max_iters=stage2_settle_final_max_iters,
        stage2_settle_exhaust_window=stage2_settle_exhaust_window,
        stage2_settle_exhaust_theta_drop_tol_deg=stage2_settle_exhaust_theta_drop_tol_deg,
        stage2_settle_exhaust_j_improve_tol=stage2_settle_exhaust_j_improve_tol,
        stage2_settle_exhaust_j_worsen_tol=stage2_settle_exhaust_j_worsen_tol,
        stage2_settle_exhaust_max_advance_dv=stage2_settle_exhaust_max_advance_dv,
        stage2_settle_record_all_evals=stage2_settle_record_all_evals,
        stage2_settle_exhaust_allow_stall_release=stage2_settle_exhaust_allow_stall_release,
        stage2_final_settle_force_enabled=stage2_final_settle_force_enabled,
        stage2_final_settle_force_min_iters=stage2_final_settle_force_min_iters,
        stage2_final_settle_force_overshoot_abs=stage2_final_settle_force_overshoot_abs,
        stage2_settle_overshoot_release_enabled=stage2_settle_overshoot_release_enabled,
        stage2_settle_overshoot_release_final_enabled=stage2_settle_overshoot_release_final_enabled,
        stage2_settle_overshoot_min_iters=stage2_settle_overshoot_min_iters,
        stage2_settle_overshoot_min_accepts=stage2_settle_overshoot_min_accepts,
        stage2_settle_overshoot_tol_factor=stage2_settle_overshoot_tol_factor,
        stage2_settle_overshoot_require_stall=stage2_settle_overshoot_require_stall,
        stage2_settle_overshoot_require_j_ok=stage2_settle_overshoot_require_j_ok,
        stage2_settle_overshoot_force_max_iters=stage2_settle_overshoot_force_max_iters,
        stage2_settle_overshoot_max_advance_dv=stage2_settle_overshoot_max_advance_dv,
        stage2_settle_hit_release_enabled=stage2_settle_hit_release_enabled,
        stage2_settle_hit_release_final_enabled=stage2_settle_hit_release_final_enabled,
        stage2_settle_hit_require_stall=stage2_settle_hit_require_stall,
        stage2_settle_hit_require_j_ok=stage2_settle_hit_require_j_ok,
        stage2_al_mu0=stage2_al_mu0,
        stage2_al_rho0=stage2_al_rho0,
        stage2_al_rho_grow=stage2_al_rho_grow,
        stage2_al_rho_shrink=stage2_al_rho_shrink,
        stage2_al_rho_min=stage2_al_rho_min,
        stage2_al_rho_max=stage2_al_rho_max,
        stage2_al_mu_max=stage2_al_mu_max,
        stage2_al_mu_overshoot_release_enabled=stage2_al_mu_overshoot_release_enabled,
        stage2_al_mu_overshoot_decay=stage2_al_mu_overshoot_decay,
        stage2_al_mu_overshoot_tol_factor=stage2_al_mu_overshoot_tol_factor,
        stage2_lambda_abs_cap=stage2_lambda_abs_cap,
        stage2_lambda_ratio_cap=stage2_lambda_ratio_cap,
        stage2_lambda_ratio_cap_low_vf=stage2_lambda_ratio_cap_low_vf,
        stage2_lambda_ratio_low_vf_threshold=stage2_lambda_ratio_low_vf_threshold,
        stage2_lambda_ratio_low_vf_blend_width=stage2_lambda_ratio_low_vf_blend_width,
        stage2_lambda_respect_dv_ratio_cap=stage2_lambda_respect_dv_ratio_cap,
        stage2_lambda_smoothing=stage2_lambda_smoothing,
        stage2_vf_rate_control_enabled=stage2_vf_rate_control_enabled,
        stage2_vf_rate_window=stage2_vf_rate_window,
        stage2_vf_rate_target_iters=stage2_vf_rate_target_iters,
        stage2_vf_rate_min_drop_abs=stage2_vf_rate_min_drop_abs,
        stage2_vf_rate_max_drop_abs=stage2_vf_rate_max_drop_abs,
        stage2_vf_rate_slow_factor=stage2_vf_rate_slow_factor,
        stage2_vf_rate_fast_factor=stage2_vf_rate_fast_factor,
        stage2_vf_rate_gain0=stage2_vf_rate_gain0,
        stage2_vf_rate_gain_min=stage2_vf_rate_gain_min,
        stage2_vf_rate_gain_max=stage2_vf_rate_gain_max,
        stage2_vf_rate_gain_grow=stage2_vf_rate_gain_grow,
        stage2_vf_rate_gain_shrink=stage2_vf_rate_gain_shrink,
        stage2_vf_rate_gain_relax=stage2_vf_rate_gain_relax,
        stage2_vf_rate_update_cooldown_iters=stage2_vf_rate_update_cooldown_iters,
        stage2_vf_rate_min_gap_abs=stage2_vf_rate_min_gap_abs,
        stage2_lambda_mech_ref_floor_enabled=stage2_lambda_mech_ref_floor_enabled,
        stage2_lambda_mech_ref_floor_frac=stage2_lambda_mech_ref_floor_frac,
        stage2_lambda_mech_ref_decay=stage2_lambda_mech_ref_decay,
        stage2_merit_relax=stage2_merit_relax,
        stage2_merit_relax_below_target=stage2_merit_relax_below_target,
        stage2_merit_relax_below_target_disable_auto=stage2_merit_relax_below_target_disable_auto,
        stage2_merit_relax_auto_enabled=stage2_merit_relax_auto_enabled,
        stage2_merit_relax_auto_max=stage2_merit_relax_auto_max,
        stage2_merit_relax_auto_safety=stage2_merit_relax_auto_safety,
        stage2_merit_relax_auto_decay=stage2_merit_relax_auto_decay,
        stage2_stall_rescue_accept_enabled=stage2_stall_rescue_accept_enabled,
        stage2_stall_rescue_vf_reduction_frac=stage2_stall_rescue_vf_reduction_frac,
        stage2_stall_rescue_vf_reduction_abs=stage2_stall_rescue_vf_reduction_abs,
        stage2_stall_rescue_vf_tol_factor=stage2_stall_rescue_vf_tol_factor,
        stage2_stall_rescue_kappa_increase_factor=stage2_stall_rescue_kappa_increase_factor,
        stage2_merit_freeze_enabled=stage2_merit_freeze_enabled,
        stage2_merit_freeze_runtime_penalty_cap_enabled=stage2_merit_freeze_runtime_penalty_cap_enabled,
        stage2_merit_stage_budget_enabled=stage2_merit_stage_budget_enabled,
        stage2_merit_stage_budget_eta=stage2_merit_stage_budget_eta,
        stage2_merit_stage_budget_eta_min=stage2_merit_stage_budget_eta_min,
        stage2_merit_stage_budget_mu_share=stage2_merit_stage_budget_mu_share,
        stage2_merit_stage_budget_j_floor=stage2_merit_stage_budget_j_floor,
        stage2_merit_stage_budget_eps=stage2_merit_stage_budget_eps,
        stage2_merit_stage_mu_max=stage2_merit_stage_mu_max,
        stage2_merit_stage_rho_max=stage2_merit_stage_rho_max,
        stage2_loss_envelope_enabled=stage2_loss_envelope_enabled,
        stage2_loss_envelope_vf_anchor=stage2_loss_envelope_vf_anchor,
        stage2_loss_envelope_vf_goal=stage2_loss_envelope_vf_goal,
        stage2_loss_envelope_total_rel=stage2_loss_envelope_total_rel,
        stage2_loss_envelope_power=stage2_loss_envelope_power,
        stage2_loss_envelope_update_anchor_on_improve=stage2_loss_envelope_update_anchor_on_improve,
        stage2_loss_envelope_improve_rel_tol=stage2_loss_envelope_improve_rel_tol,
        stage2_loss_envelope_stop_tol_rel=stage2_loss_envelope_stop_tol_rel,
        stage2_loss_envelope_min_budget_abs=stage2_loss_envelope_min_budget_abs,
        stage2_merit_stage_budget_lambda_projection_enabled=stage2_merit_stage_budget_lambda_projection_enabled,
        stage2_merit_step_jmech_cap_enabled=stage2_merit_step_jmech_cap_enabled,
        stage2_merit_step_jmech_cap_stage2_only=stage2_merit_step_jmech_cap_stage2_only,
        stage2_merit_step_jmech_cap_rel=stage2_merit_step_jmech_cap_rel,
        stage2_merit_step_jmech_cap_j_floor=stage2_merit_step_jmech_cap_j_floor,
        stage2_merit_below_target_mech_strict=stage2_merit_below_target_mech_strict,
        stage2_merit_below_target_mech_relax=stage2_merit_below_target_mech_relax,
        stage2_slow_progress_watchdog_enabled=stage2_slow_progress_watchdog_enabled,
        stage2_slow_progress_window=stage2_slow_progress_window,
        stage2_slow_progress_consecutive_hits=stage2_slow_progress_consecutive_hits,
        stage2_slow_progress_min_avg_drop_abs=stage2_slow_progress_min_avg_drop_abs,
        stage2_slow_progress_min_avg_drop_rel=stage2_slow_progress_min_avg_drop_rel,
        stage2_slow_progress_min_gap_abs=stage2_slow_progress_min_gap_abs,
        stage2_slow_progress_min_gap_tol_factor=stage2_slow_progress_min_gap_tol_factor,
        stage2_slow_progress_kappa_max=stage2_slow_progress_kappa_max,
        stage2_slow_progress_require_controller_cap=stage2_slow_progress_require_controller_cap,
        stage2_slow_progress_controller_cap_fraction=stage2_slow_progress_controller_cap_fraction,
        stage2_slow_progress_lambda_ratio_step=stage2_slow_progress_lambda_ratio_step,
        stage2_slow_progress_lambda_ratio_grow=stage2_slow_progress_lambda_ratio_grow,
        stage2_slow_progress_lambda_ratio_cap_low_vf_max=stage2_slow_progress_lambda_ratio_cap_low_vf_max,
        stage2_slow_progress_lambda_ratio_decay=stage2_slow_progress_lambda_ratio_decay,
        stage2_slow_progress_cooldown_iters=stage2_slow_progress_cooldown_iters,
        stage2_merit_penalty_ratio_cap_enabled=stage2_merit_penalty_ratio_cap_enabled,
        stage2_merit_penalty_ratio_cap_vf_threshold=stage2_merit_penalty_ratio_cap_vf_threshold,
        stage2_merit_penalty_ratio_cap=stage2_merit_penalty_ratio_cap,
        stage2_merit_penalty_ratio_cap_j_floor=stage2_merit_penalty_ratio_cap_j_floor,
        stage2_stall_diagnosis_enabled=stage2_stall_diagnosis_enabled,
        stage2_stall_overdrive_vf_reduction_frac=stage2_stall_overdrive_vf_reduction_frac,
        stage2_stall_overdrive_vf_reduction_abs=stage2_stall_overdrive_vf_reduction_abs,
        stage2_stall_overdrive_vf_tol_factor=stage2_stall_overdrive_vf_tol_factor,
        stage2_stall_overdrive_mech_increase_rel=stage2_stall_overdrive_mech_increase_rel,
        stage2_stall_overdrive_mech_increase_abs=stage2_stall_overdrive_mech_increase_abs,
        stage2_stall_overdrive_mech_penalty_dominance=stage2_stall_overdrive_mech_penalty_dominance,
        stage2_stall_merit_excess_rel=stage2_stall_merit_excess_rel,
        stage2_stall_weak_vf_reduction_frac=stage2_stall_weak_vf_reduction_frac,
        stage2_stall_weak_vf_reduction_abs=stage2_stall_weak_vf_reduction_abs,
        stage2_stall_weak_vf_tol_factor=stage2_stall_weak_vf_tol_factor,
        stage2_stall_overdrive_rho_shrink=stage2_stall_overdrive_rho_shrink,
        stage2_stall_overdrive_mu_decay=stage2_stall_overdrive_mu_decay,
        stage2_stall_overdrive_lambda_decay=stage2_stall_overdrive_lambda_decay,
        stage2_stall_conflict_lambda_decay=stage2_stall_conflict_lambda_decay,
        stage2_stall_watchdog_enabled=stage2_stall_watchdog_enabled,
        stage2_stall_weak_grow_before_dv_shrink=stage2_stall_weak_grow_before_dv_shrink,
        stage2_stall_nonweak_soft_reset_after=stage2_stall_nonweak_soft_reset_after,
        stage2_stall_nonweak_allow_fallback_after=stage2_stall_nonweak_allow_fallback_after,
        stage2_stall_escape_rho_shrink=stage2_stall_escape_rho_shrink,
        stage2_stall_escape_mu_decay=stage2_stall_escape_mu_decay,
        stage2_stall_escape_lambda_decay=stage2_stall_escape_lambda_decay,
        stage2_stall_escape_hold_when_fallback_allowed=stage2_stall_escape_hold_when_fallback_allowed,
        stage2_stall_nonweak_shrink_dv_after_fallback=stage2_stall_nonweak_shrink_dv_after_fallback,
        stage2_stall_wait_enabled=stage2_stall_wait_enabled,
        stage2_stall_wait_max_skip_iters=stage2_stall_wait_max_skip_iters,
        stage2_stall_wait_vf_tol=stage2_stall_wait_vf_tol,
        stage2_same_state_stall_wait_enabled=stage2_same_state_stall_wait_enabled,
        stage2_same_state_stall_wait_after=stage2_same_state_stall_wait_after,
        stage2_same_state_stall_wait_max_skip_iters=stage2_same_state_stall_wait_max_skip_iters,
        stage2_same_state_stall_wait_j_rel_tol=stage2_same_state_stall_wait_j_rel_tol,
        stage2_same_state_stall_wait_lambda_rel_tol=stage2_same_state_stall_wait_lambda_rel_tol,
        stage2_post_nucleation_takeover_enabled=stage2_post_nucleation_takeover_enabled,
        stage2_post_nucleation_takeover_iters=stage2_post_nucleation_takeover_iters,
        stage2_post_nucleation_takeover_min_gap_abs=stage2_post_nucleation_takeover_min_gap_abs,
        stage2_post_nucleation_takeover_gain_grow=stage2_post_nucleation_takeover_gain_grow,
        stage2_post_nucleation_takeover_gain_floor=stage2_post_nucleation_takeover_gain_floor,
        stage2_post_nucleation_takeover_suppress_fallback=stage2_post_nucleation_takeover_suppress_fallback,
        stage2_merit_relax_low_vf_threshold=stage2_merit_relax_low_vf_threshold,
        stage2_merit_relax_below_target_drop=stage2_merit_relax_below_target_drop,
        stage2_merit_relax_below_target_drop_tol=stage2_merit_relax_below_target_drop_tol,
        stage2_adaptive_lambda_ratio_enabled=stage2_adaptive_lambda_ratio_enabled,
        stage2_adaptive_lambda_ratio_scale_init=stage2_adaptive_lambda_ratio_scale_init,
        stage2_adaptive_lambda_ratio_scale_min=stage2_adaptive_lambda_ratio_scale_min,
        stage2_adaptive_lambda_ratio_scale_max=stage2_adaptive_lambda_ratio_scale_max,
        stage2_adaptive_lambda_ratio_low_vf_scale_max=stage2_adaptive_lambda_ratio_low_vf_scale_max,
        stage2_adaptive_lambda_ratio_grow=stage2_adaptive_lambda_ratio_grow,
        stage2_adaptive_lambda_ratio_shrink=stage2_adaptive_lambda_ratio_shrink,
        stage2_adaptive_lambda_ratio_mild_shrink=stage2_adaptive_lambda_ratio_mild_shrink,
        stage2_adaptive_lambda_ratio_stall_shrink=stage2_adaptive_lambda_ratio_stall_shrink,
        stage2_adaptive_lambda_ratio_slow_hits=stage2_adaptive_lambda_ratio_slow_hits,
        stage2_adaptive_lambda_ratio_stall_low_eff=stage2_adaptive_lambda_ratio_stall_low_eff,
        stage2_adaptive_lambda_ratio_low_stall_hits=stage2_adaptive_lambda_ratio_low_stall_hits,
        stage2_adaptive_lambda_ratio_vf_drop_min_abs=stage2_adaptive_lambda_ratio_vf_drop_min_abs,
        stage2_adaptive_lambda_ratio_vf_drop_min_gap_fraction=stage2_adaptive_lambda_ratio_vf_drop_min_gap_fraction,
        stage2_adaptive_lambda_ratio_vf_drop_max_gap_fraction=stage2_adaptive_lambda_ratio_vf_drop_max_gap_fraction,
        stage2_adaptive_lambda_ratio_j_worsen_tol=stage2_adaptive_lambda_ratio_j_worsen_tol,
        stage2_adaptive_lambda_ratio_j_ok_tol=stage2_adaptive_lambda_ratio_j_ok_tol,
        stage2_adaptive_hold_al_on_ratio_shrink=stage2_adaptive_hold_al_on_ratio_shrink,
        stage2_min_nucleation_gap_iters=stage2_min_nucleation_gap_iters,
        stage2_nucleation_reject_cooldown_iters=stage2_nucleation_reject_cooldown_iters,
        stage2_nucleation_max_abs_drop=stage2_nucleation_max_abs_drop,
        stage2_nucleation_max_gap_fraction=stage2_nucleation_max_gap_fraction,
        stage2_nucleation_merit_relax=stage2_nucleation_merit_relax,
        stage2_nucleation_merit_relax_auto_enabled=stage2_nucleation_merit_relax_auto_enabled,
        stage2_nucleation_merit_relax_auto_max=stage2_nucleation_merit_relax_auto_max,
        stage2_nucleation_merit_relax_auto_small_drop=stage2_nucleation_merit_relax_auto_small_drop,
        stage2_nucleation_max_trials=stage2_nucleation_max_trials,
        stage2_nucleation_retry_drop_factor=stage2_nucleation_retry_drop_factor,
        stage2_nucleation_retry_next_batch=stage2_nucleation_retry_next_batch,
        stage2_nucleation_retry_min_abs_drop=stage2_nucleation_retry_min_abs_drop,
        symmetry_mode=symmetry_mode,
        symmetry_assert_tol=symmetry_assert_tol,
        symmetry_diag_stride=symmetry_diag_stride,
        psi_stats_stride=psi_stats_stride,
        use_strict_wedge_parameterization=use_strict_wedge_parameterization,
        use_minimal_wedge_parameterization=use_minimal_wedge_parameterization,
        minimal_wedge_decimals=minimal_wedge_decimals,
        minimal_wedge_soft_eta=minimal_wedge_soft_eta,
        wedge_assert_tol=wedge_assert_tol,
        wedge_diag_stride=wedge_diag_stride,
        use_helmholtz_filter=use_helmholtz_filter,
        helmholtz_fail_radius_factors=helmholtz_fail_radius_factors,
        helmholtz_fail_max_tries=helmholtz_fail_max_tries,
        helmholtz_refine_radius_factor=helmholtz_refine_radius_factor,
        pre_jacobi_filter_enabled=pre_jacobi_filter_enabled,
        pre_jacobi_filter_radius_factors=pre_jacobi_filter_radius_factors,
        pre_jacobi_filter_vf_rel_change_max=pre_jacobi_filter_vf_rel_change_max,
        debug_force_fail_recover=debug_force_fail_recover,
        debug_force_fail_recover_iters=debug_force_fail_recover_iters,
        use_octant_parameterization=use_minimal_wedge_parameterization,
        octant_param_decimals=minimal_wedge_decimals,
        octant_param_soft_eta=minimal_wedge_soft_eta,
        # Backward-compat aliases for previous symmetry config keys:
        use_centerdiag_symmetry=use_minimal_wedge_parameterization,
        centerdiag_symmetry_decimals=minimal_wedge_decimals,
        centerdiag_symmetry_soft_eta=minimal_wedge_soft_eta,
        rho_v=rho_v, mu_v0=mu_v0,
        vf_al_adapt_good_ratio=vf_al_adapt_good_ratio,
        vf_al_adapt_bad_ratio=vf_al_adapt_bad_ratio,
        vf_al_rho_grow=vf_al_rho_grow,
        vf_al_rho_shrink=vf_al_rho_shrink,
        vf_al_rho_min=vf_al_rho_min,
        vf_al_rho_max=vf_al_rho_max,
        vf_al_adapt_eps=vf_al_adapt_eps,
        use_hard_vf=use_hard_vf, vf_target=vf_target, vf_bisect_tol=vf_bisect_tol, vf_bisect_max_iter=vf_bisect_max_iter,
        vf_constraint_target=vf_constraint_target,
        vf_final_target=vf_final_target,
        vf_stage_ratio=vf_stage_ratio, vf_stage_tol=vf_stage_tol,
        # Stage-wise volume controller parameters
        vf_stage_dv0=vf_stage_dv0, vf_stage_dv_min=vf_stage_dv_min, vf_stage_dv_levels=vf_stage_dv_levels, vf_stage_shrink=vf_stage_shrink,
        vf_stage_tol_min=vf_stage_tol_min, vf_stage_hysteresis_factor=vf_stage_hysteresis_factor,
        vf_stage_success_window=vf_stage_success_window,
        vf_stage_plateau_tol=vf_stage_plateau_tol,
        stage_relax_tau=stage_relax_tau, stage_relax_steps_total=stage_relax_steps_total,
        debug_progress_markers=debug_progress_markers,
        print_direction_chain_diagnostics=print_direction_chain_diagnostics,
        use_plateau_hard_shift=use_plateau_hard_shift,
        hard_shift_plateau_window=hard_shift_plateau_window,
        hard_shift_plateau_vf_tol=hard_shift_plateau_vf_tol,
        hard_shift_kappa_min_hits=hard_shift_kappa_min_hits,
        hard_shift_factor=hard_shift_factor,
        hard_shift_nucleation_quantile_lo=hard_shift_nucleation_quantile_lo,
        hard_shift_nucleation_quantile_hi=hard_shift_nucleation_quantile_hi,
        hard_shift_nucleation_d_cap=hard_shift_nucleation_d_cap,
        hard_shift_nucleation_d_expand_step=hard_shift_nucleation_d_expand_step,
        hard_shift_nucleation_psi_value=hard_shift_nucleation_psi_value,
        post_nucleation_freeze_steps=post_nucleation_freeze_steps,
        post_nucleation_freeze_early_release_enabled=post_nucleation_freeze_early_release_enabled,
        post_nucleation_lambda_v_boost_factor=post_nucleation_lambda_v_boost_factor,
        post_nucleation_lambda_zero_steps=post_nucleation_lambda_zero_steps,
        hard_shift_uniform_exit_enabled=hard_shift_uniform_exit_enabled,
        hard_shift_uniform_exit_consecutive_hits=hard_shift_uniform_exit_consecutive_hits,
        hard_shift_aggressive_shift_factor=hard_shift_aggressive_shift_factor,
        hard_shift_aggressive_max_steps=hard_shift_aggressive_max_steps,
        hard_shift_switch_to_shift_vf=hard_shift_switch_to_shift_vf,
        hard_shift_shift_factor=hard_shift_shift_factor,
        postprocess_loss_envelope_extra_drop=postprocess_loss_envelope_extra_drop,
        postprocess_hard_shift_gap_iters=postprocess_hard_shift_gap_iters,
        postprocess_hard_shift_min_gap_iters=postprocess_hard_shift_min_gap_iters,
        postprocess_hard_shift_max_gap_iters=postprocess_hard_shift_max_gap_iters,
        postprocess_hard_shift_plateau_window=postprocess_hard_shift_plateau_window,
        postprocess_hard_shift_plateau_rel_tol=postprocess_hard_shift_plateau_rel_tol,
        postprocess_hard_shift_max_abs_drop=postprocess_hard_shift_max_abs_drop,
        postprocess_hard_shift_gap_fraction=postprocess_hard_shift_gap_fraction,
        postprocess_hard_shift_min_abs_drop=postprocess_hard_shift_min_abs_drop,
        postprocess_terminate_on_target=postprocess_terminate_on_target,
        postprocess_terminate_immediate_on_target=postprocess_terminate_immediate_on_target,
        hard_shift_rebound_early_shift_enabled=hard_shift_rebound_early_shift_enabled,
        hard_shift_rebound_trigger_vf=hard_shift_rebound_trigger_vf,
        hard_shift_rebound_hits_needed=hard_shift_rebound_hits_needed,
        hard_shift_allow_partial_target=hard_shift_allow_partial_target,
        hard_shift_kappa_reset=hard_shift_kappa_reset,
        hard_shift_relax_steps=hard_shift_relax_steps,
        hard_shift_cooldown_steps=hard_shift_cooldown_steps,
        hard_shift_cooldown_kappa_factor=hard_shift_cooldown_kappa_factor,
        refine_cooldown_steps=refine_cooldown_steps,
        refine_cooldown_kappa_factor=refine_cooldown_kappa_factor,
        hard_shift_recovery_steps=hard_shift_recovery_steps,
        hard_shift_recovery_stall_counts=hard_shift_recovery_stall_counts,
        vf_milestone_filter_targets=vf_milestone_filter_targets,
        vf_milestone_filter_radius_factor=vf_milestone_filter_radius_factor,
        milestone_filter_vf_rel_change_max=milestone_filter_vf_rel_change_max,
        fail_recover_filter_vf_rel_change_start=fail_recover_filter_vf_rel_change_start,
        fail_recover_filter_vf_rel_change_step=fail_recover_filter_vf_rel_change_step,
        fail_recover_filter_vf_rel_change_max=fail_recover_filter_vf_rel_change_max,
        final_filter_vf_rel_change_max=final_filter_vf_rel_change_max,
        final_filter_radius_max_factor=final_filter_radius_max_factor,
        final_filter_radius_search_max_iter=final_filter_radius_search_max_iter,
        final_filter_post_cooldown_steps=final_filter_post_cooldown_steps,
        final_filter_post_cooldown_kappa_factor=final_filter_post_cooldown_kappa_factor,
        use_ranked_volume_enhancement=use_ranked_volume_enhancement,
        vf_rank_use_boundary_band=vf_rank_use_boundary_band,
        vf_rank_select_all_candidates=vf_rank_select_all_candidates,
        vf_rank_band_stage_scaled=vf_rank_band_stage_scaled,
        vf_rank_band_min=vf_rank_band_min,
        vf_rank_band_max=vf_rank_band_max,
        vf_rank_band_gap_scale_d0=vf_rank_band_gap_scale_d0,
        vf_rank_use_hmin_band=vf_rank_use_hmin_band,
        vf_rank_band_hmin_factor=vf_rank_band_hmin_factor,
        vf_rank_band=vf_rank_band,
        vf_rank_kp=vf_rank_kp,
        vf_rank_pmin=vf_rank_pmin,
        vf_rank_pmax=vf_rank_pmax,
        vf_rank_beta0=vf_rank_beta0,
        vf_rank_kbeta=vf_rank_kbeta,
        vf_rank_use_interior_band=vf_rank_use_interior_band,
        vf_rank_interior_depth_abs=vf_rank_interior_depth_abs,
        vf_rank_interior_depth_band_mult=vf_rank_interior_depth_band_mult,
        vf_rank_interior_progress_start=vf_rank_interior_progress_start,
        vf_rank_interior_progress_full=vf_rank_interior_progress_full,
        vf_rank_interior_progress_power=vf_rank_interior_progress_power,
        vf_rank_interior_max_strength=vf_rank_interior_max_strength,
        vf_rank_interior_p_scale=vf_rank_interior_p_scale,
        vf_rank_interior_pmax=vf_rank_interior_pmax,
        vf_rank_interior_beta_scale=vf_rank_interior_beta_scale,
        threshold=threshold, lsf_expr=lsf_expr,
        cell_solver=cell_solver
    )
