from __future__ import print_function
from fenics import *
import numpy as np

# ------------------------------------------------------------
# Project with PETSc iterative solver (avoids direct/LU/UMFPACK OOM)
# ------------------------------------------------------------
_original_project = project  # from fenics
_PROJECT_SOLVER = {"solver_type": "cg", "preconditioner_type": "jacobi"}
_PROJECT_FALLBACK_SOLVERS = (
    {"solver_type": "gmres", "preconditioner_type": "jacobi"},
    {"solver_type": "minres", "preconditioner_type": "jacobi"},
)
_project_solver_printed = False
_pde_solver_printed = False
_nullspace_cache = {}

def _project(u, V, **kwargs):
    """Project u onto V using PETSc iterative solver with fallback sequence."""
    global _project_solver_printed
    primary_solver = dict(_PROJECT_SOLVER)
    primary_solver.update(kwargs)
    solver_sequence = [primary_solver]
    for solver_cfg in _PROJECT_FALLBACK_SOLVERS:
        merged = dict(kwargs)
        merged.update(solver_cfg)
        if not any(
            (cfg.get("solver_type") == merged.get("solver_type"))
            and (cfg.get("preconditioner_type") == merged.get("preconditioner_type"))
            for cfg in solver_sequence
        ):
            solver_sequence.append(merged)
    if not _project_solver_printed and MPI.rank(MPI.comm_world) == 0:
        fallback_desc = [
            "%s+%s" % (cfg.get("solver_type"), cfg.get("preconditioner_type"))
            for cfg in solver_sequence[1:]
        ]
        print("[project] using PETSc iterative: solver=%s, preconditioner=%s, fallback=%s" %
              (primary_solver.get("solver_type"), primary_solver.get("preconditioner_type"), str(fallback_desc)))
        _project_solver_printed = True
    last_error = None
    for i_try, solver_cfg in enumerate(solver_sequence):
        try:
            result = _original_project(u, V, **solver_cfg)
            if (i_try > 0) and (MPI.rank(MPI.comm_world) == 0):
                print("[project] fallback success with solver=%s, preconditioner=%s" %
                      (solver_cfg.get("solver_type"), solver_cfg.get("preconditioner_type")))
            return result
        except RuntimeError as err:
            last_error = err
            if MPI.rank(MPI.comm_world) == 0:
                print("[project] solver=%s, preconditioner=%s failed; trying next fallback if available..." %
                      (solver_cfg.get("solver_type"), solver_cfg.get("preconditioner_type")))
    raise last_error

# Override so "from definition3d import *" gives iterative solver
project = _project

# ============================================================
# definition3d.py
# 3D Topological-derivative + Level-set framework (FEniCS 2019)
#
# Key features:
#   - Unit-cube periodic RVE
#   - Two-phase isotropic elasticity (E varies via level-set, nu fixed)
#   - 3D homogenisation: compute C_hom (Voigt 6x6, tensorial shear)
#   - Objective Phi(C_hom) as provided
#   - DTJ(y) computed via chain rule:
#         DTJ = sum_{I,J} (dPhi/dC_IJ) * DT_C_IJ(y)
#     where DT_C_IJ(y) is computed from the 6 stress fields via the isotropic spherical
#     polarization tensor P (given in the 3D flowchart):
#         (DT C)_{IJ}(y) = sigma^I(y) : P(y) : sigma^J(y).
#
# Notes:
#     Amstutz-style TD/Level-set workflows.
#   - The DT_C model here is a practical isotropic-spherical approximation.
#     If you already have an exact 3D polarization-tensor formula, you can
#     replace compute_DTC_fields() without touching the rest of the code.
# ============================================================

# ------------------------------------------------------------
# 1) Periodic boundary condition on unit cube
# ------------------------------------------------------------
class PeriodicBoundary3D(SubDomain):
    def __init__(self, tol=1e-10):
        super(PeriodicBoundary3D, self).__init__()
        self.tol = tol

    def inside(self, x, on_boundary):
        # Left/bottom/back boundaries are "inside" for periodicity
        return bool(on_boundary and
                    ((near(x[0], 0.0, self.tol) and (not near(x[0], 1.0, self.tol))) or
                     (near(x[1], 0.0, self.tol) and (not near(x[1], 1.0, self.tol))) or
                     (near(x[2], 0.0, self.tol) and (not near(x[2], 1.0, self.tol)))))

    def map(self, x, y):
        y[0] = x[0] - 1.0 if near(x[0], 1.0, self.tol) else x[0]
        y[1] = x[1] - 1.0 if near(x[1], 1.0, self.tol) else x[1]
        y[2] = x[2] - 1.0 if near(x[2], 1.0, self.tol) else x[2]


# ------------------------------------------------------------
# 2) Voigt helpers (tensorial shear convention)
#    Order: [11,22,33,23,13,12]
# ------------------------------------------------------------
def macro_strain_voigt(k):
    E = np.zeros((3,3), dtype=float)
    if k == 0: E[0,0] = 1.0
    if k == 1: E[1,1] = 1.0
    if k == 2: E[2,2] = 1.0
    if k == 3: E[1,2] = 0.5; E[2,1] = 0.5
    if k == 4: E[0,2] = 0.5; E[2,0] = 0.5
    if k == 5: E[0,1] = 0.5; E[1,0] = 0.5
    return E

def stress_to_voigt(sig):
    return as_vector([sig[0,0], sig[1,1], sig[2,2], sig[1,2], sig[0,2], sig[0,1]])

def strain_to_voigt(eps):
    return as_vector([eps[0,0], eps[1,1], eps[2,2], eps[1,2], eps[0,2], eps[0,1]])

def tr_eps(eps):
    return eps[0,0] + eps[1,1] + eps[2,2]

def dev_eps(eps):
    return eps - (tr_eps(eps)/3.0)*Identity(3)

# ------------------------------------------------------------
# 3) Two-phase isotropic elasticity (Lamé form)
# ------------------------------------------------------------
def lame_from_E_nu(E, nu):
    mu  = E/(2.0*(1.0+nu))
    lam = E*nu/((1.0+nu)*(1.0-2.0*nu))
    return lam, mu

def sigma(u_fluc, Eps_macro, E_expr, nu):
    """
    Total strain = Eps_macro + sym(grad(u_fluc))
    Stress = C(E_expr):eps
    """
    eps = Eps_macro + sym(grad(u_fluc))
    lam_expr = E_expr*nu/((1.0+nu)*(1.0-2.0*nu))
    mu_expr  = E_expr/(2.0*(1.0+nu))
    return lam_expr*tr(eps)*Identity(3) + 2.0*mu_expr*eps

# ------------------------------------------------------------
# 4) Mark materials from level-set
# ------------------------------------------------------------
def mark_materials_from_lsf(mesh, lsf, materials, threshold=0.0):
    """
    Mark cells: materials[c] = 1 if lsf < threshold, else 0.
    Uses project-to-DG0 + dofmap indexing — MPI-safe and avoids slow per-cell point evaluation.
    """
    V0 = FunctionSpace(mesh, "DG", 0)
    lsf_dg0 = project(lsf, V0)
    lsf_dg0.vector().update_ghost_values()
    dofmap = V0.dofmap()
    lsf_local = lsf_dg0.vector().get_local()
    mat = materials.array()
    n = mesh.num_cells()
    cell_to_dof = np.array([dofmap.cell_dofs(ci)[0] for ci in range(n)], dtype=np.intp)
    mat[:] = (lsf_local[cell_to_dof] < threshold).astype(np.uintp)

def mark_materials_from_reference_lsf(mesh, lsf_ref, materials, threshold=0.0):
    """
    Inherit topology by pointwise classification on a reference level-set (possibly from old mesh):
      materials[cell] = 1 if lsf_ref(cell_midpoint) < threshold else 0
    This mimics the 2D "inside(x)=lsf_old(x)<threshold" inheritance behavior after re-meshing.
    """
    lsf_ref.set_allow_extrapolation(True)
    for c in cells(mesh):
        p = c.midpoint()
        materials[c] = 1 if float(lsf_ref(p)) < threshold else 0

def mark_materials_from_reference_chi(mesh, chi_ref, materials, cutoff=0.5):
    """
    Inherit topology by pointwise classification on a reference indicator chi (0/1-like):
      materials[cell] = 1 if chi_ref(cell_midpoint) > cutoff else 0
    Using chi is often more robust than lsf transfer across re-meshing.
    """
    chi_ref.set_allow_extrapolation(True)
    for c in cells(mesh):
        p = c.midpoint()
        materials[c] = 1 if float(chi_ref(p)) > cutoff else 0

def materials_to_chi(mesh, materials):
    """
    1 for solid, 0 for void
    """
    V0 = FunctionSpace(mesh, "DG", 0)
    chi = Function(V0)
    vals = chi.vector().get_local()
    m = materials.array()
    vals[:] = m.astype(np.float64)
    chi.vector().set_local(vals)
    chi.vector().apply("insert")
    return chi

def E_from_materials(mesh, materials, E0, gamma_star):
    """
    Return E(x) as DG0 Function: E0 in material==1 else gamma_star*E0 (void). gamma_star = E_void/E_solid.
    """
    V0 = FunctionSpace(mesh, "DG", 0)
    Efun = Function(V0)
    vals = Efun.vector().get_local()
    m = materials.array()
    vals[:] = np.where(m == 1, E0, gamma_star * E0).astype(np.float64)
    Efun.vector().set_local(vals)
    Efun.vector().apply("insert")
    return Efun

# ------------------------------------------------------------
# 5) Cell problem solve (periodic + mean-zero constraint)
# ------------------------------------------------------------
def build_spaces(mesh, deg=1, tol=1e-10):
    V = VectorFunctionSpace(mesh, "CG", deg,
                            constrained_domain=PeriodicBoundary3D(tol=tol))
    Vls = FunctionSpace(mesh, "CG", deg, constrained_domain=PeriodicBoundary3D(tol=tol))
    VtDG = TensorFunctionSpace(mesh, "DG", 0)
    VsDG = VectorFunctionSpace(mesh, "DG", 0, dim=6)
    VDG0 = FunctionSpace(mesh, "DG", 0)
    return V, Vls, VtDG, VsDG, VDG0


def _space_cache_key(V):
    """Cache key per function space instance (changes after refinement)."""
    return int(id(V))


def _build_translation_nullspace(V):
    """Build 3 rigid-translation modes for periodic vector space in 3D.

    NOTE:
    Avoid interpolate(Constant(...), V) on periodic constrained vector spaces.
    Some HPC FEniCS builds fail on that code path at high MPI counts.
    """
    basis = []
    for comp in range(3):
        vec_fn = Function(V)
        vals = vec_fn.vector().get_local()
        vals[:] = 0.0
        # Fill parent-vector dofs belonging to one component with 1.0.
        # IMPORTANT (MPI): dofmap().dofs() are global indices, while get_local()
        # uses local ownership indexing [0, local_size). Convert safely.
        comp_dofs_global = np.asarray(V.sub(comp).dofmap().dofs(), dtype=np.int64)
        lo, hi = vec_fn.vector().local_range()
        owned_mask = (comp_dofs_global >= lo) & (comp_dofs_global < hi)
        comp_dofs_local = comp_dofs_global[owned_mask] - int(lo)
        vals[comp_dofs_local] = 1.0
        vec_fn.vector().set_local(vals)
        vec_fn.vector().apply("insert")
        basis.append(vec_fn.vector())
    ns = VectorSpaceBasis(basis)
    ns.orthonormalize()
    return ns


def _get_translation_nullspace(V):
    """Get cached translation nullspace for this vector function space."""
    key = _space_cache_key(V)
    ns = _nullspace_cache.get(key, None)
    if ns is None:
        ns = _build_translation_nullspace(V)
        _nullspace_cache[key] = ns
    return ns

def solve_cell_problem(V, dx, Eps_macro, E_expr, nu, solver_cfg=None):
    """Solve periodic fluctuation cell problem in semi-definite form.

    We keep the original periodic weak form (no pointwise anchor, no penalty, no multiplier)
    and explicitly attach the 3 translational nullspace vectors to PETSc.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    u_sol = Function(V)

    # F = sigma:grad(v); lhs/rhs split so a=bilinear(u,v), L=linear(v)
    F = inner(sigma(u, Eps_macro, E_expr, nu), sym(grad(v))) * dx
    a, L = lhs(F), rhs(F)

    # Assemble as a semi-definite periodic system (no Dirichlet anchor).
    A = PETScMatrix()
    b = PETScVector()
    assemble(a, tensor=A)
    assemble(L, tensor=b)

    if solver_cfg is None:
        raise ValueError("cell_solver config is required; define it in Init3d.py and pass via cfg['cell_solver'].")
    cfg = dict(solver_cfg)
    required_keys = (
        "ksp_type", "ksp_rtol", "ksp_atol", "ksp_max_it",
        "pc_type", "pc_hypre_type", "pc_hypre_boomeramg_relax_type_all",
        "mg_levels_ksp_type", "mg_levels_pc_type",
        "set_near_nullspace", "enable_ksp_fallback", "fallback_ksp_types",
    )
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError("cell_solver config missing keys: %s" % ", ".join(missing))

    # Attach exact translation nullspace and make RHS compatible.
    nullspace = _get_translation_nullspace(V)
    A.set_nullspace(nullspace)
    b *= 1.0  # ensure PETScVector ownership is finalized before orthogonalize
    nullspace.orthogonalize(b)

    # Also provide near-nullspace to AMG so coarse-space construction is robust.
    if bool(cfg["set_near_nullspace"]):
        try:
            A.set_near_nullspace(nullspace)
        except (RuntimeError, AttributeError):
            # Some PETSc/FEniCS builds expose only set_nullspace; continue safely.
            pass

    # PETSc options with a dedicated prefix to avoid side effects.
    prefix = "cell_"
    PETScOptions.set(prefix + "ksp_rtol", float(cfg["ksp_rtol"]))
    PETScOptions.set(prefix + "ksp_atol", float(cfg["ksp_atol"]))
    PETScOptions.set(prefix + "ksp_max_it", int(cfg["ksp_max_it"]))
    PETScOptions.set(prefix + "pc_type", str(cfg["pc_type"]))
    PETScOptions.set(prefix + "mg_levels_ksp_type", str(cfg["mg_levels_ksp_type"]))
    PETScOptions.set(prefix + "mg_levels_pc_type", str(cfg["mg_levels_pc_type"]))
    if str(cfg["pc_type"]).lower() == "hypre":
        PETScOptions.set(prefix + "pc_hypre_type", str(cfg["pc_hypre_type"]))
        PETScOptions.set(
            prefix + "pc_hypre_boomeramg_relax_type_all",
            str(cfg["pc_hypre_boomeramg_relax_type_all"])
        )

    ksp_seq = [str(cfg["ksp_type"]).lower()]
    if bool(cfg.get("enable_ksp_fallback", True)):
        for ksp_fb in cfg.get("fallback_ksp_types", ("minres", "gmres")):
            ksp_fb = str(ksp_fb).lower()
            if ksp_fb not in ksp_seq:
                ksp_seq.append(ksp_fb)

    global _pde_solver_printed
    if not _pde_solver_printed and MPI.rank(MPI.comm_world) == 0:
        print("[pde] periodic semi-definite solve: ksp=%s pc=%s near_nullspace=%s explicit_translation_nullspace=3 fallback=%s" %
              (str(cfg["ksp_type"]), str(cfg["pc_type"]), str(bool(cfg["set_near_nullspace"])), str(ksp_seq[1:])))
        _pde_solver_printed = True

    last_error = None
    for i_try, ksp_name in enumerate(ksp_seq):
        PETScOptions.set(prefix + "ksp_type", ksp_name)
        solver = PETScKrylovSolver()
        solver.set_options_prefix(prefix)
        solver.set_from_options()
        solver.set_operator(A)
        u_sol.vector().zero()
        u_sol.vector().apply("insert")
        try:
            solver.solve(u_sol.vector(), b)
            if i_try > 0 and MPI.rank(MPI.comm_world) == 0:
                print("[pde] fallback success with ksp=%s (primary=%s)" % (ksp_name, ksp_seq[0]))
            return u_sol
        except RuntimeError as err:
            last_error = err
            if MPI.rank(MPI.comm_world) == 0:
                print("[pde] ksp=%s failed, trying next fallback if available..." % ksp_name)

    raise last_error


# ------------------------------------------------------------
# 6) Homogenized stiffness C_hom (6x6) + sig_cache (for DTJ)
# ------------------------------------------------------------
def compute_homogenized_C(mesh, W, materials, E0, gamma_star, nu, threshold=0.0, solver_cfg=None):
    dx = Measure("dx", domain=mesh, subdomain_data=materials)
    Vol = assemble(Constant(1.0)*dx)

    # material field
    E_expr = E_from_materials(mesh, materials, E0, gamma_star)

    # sig_cache only (u/eps not stored to save memory; evaluate uses only sig_cache)
    sig_cache = []

    VtDG = TensorFunctionSpace(mesh, "DG", 0)
    VsDG = VectorFunctionSpace(mesh, "DG", 0, dim=6)

    Chom = np.zeros((6,6), dtype=float)

    for k in range(6):
        Eps = Constant(macro_strain_voigt(k))
        u_fluc = solve_cell_problem(W, dx, Eps, E_expr, nu, solver_cfg=solver_cfg)

        sig = _project(sigma(u_fluc, Eps, E_expr, nu), VtDG)
        sig_cache.append(sig)

        sig_v = _project(stress_to_voigt(sig), VsDG)
        for I in range(6):
            Chom[I, k] = assemble(sig_v[I]*dx)/Vol

    return Chom, sig_cache, float(Vol), E_expr

# ------------------------------------------------------------
# 7) Generic constitutive functionals h(C) and Fréchet derivatives
#    Voigt convention (tensorial shear): [11,22,33,23,13,12]
#
#    Type 3:  Phi(C) = C :: A
#       => dPhi/dC = A   (the direction tensor itself)
#
#    Type 4:  h(C) = (C :: A) (C :: B)
#       => dPhi/dC = (C::B) A + (C::A) B
#
#    For squares: (C::A)^2  => dPhi/dC = 2 (C::A) A
# ------------------------------------------------------------

# --- Voigt index map for tensorial shear (NOT doubled)
# 0:11, 1:22, 2:33, 3:23, 4:13, 5:12
_VOIGT_MAP = {
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
    (1, 2): 3, (2, 1): 3,
    (0, 2): 4, (2, 0): 4,
    (0, 1): 5, (1, 0): 5,
}

def voigt_index(i, j):
    """0-based indices i,j in {0,1,2} -> Voigt index in {0..5}."""
    return _VOIGT_MAP[(int(i), int(j))]


def Prod_varphi_voigt(i, j, k, l):
    """Return the 6x6 Voigt 'direction tensor' A selecting C_{ijkl}.

    This matches the paper notation A = varphi^{ij} ⊗ varphi^{kl} where
    varphi^{ij} is the symmetric unit strain basis used in the cell problems.

    With our Voigt convention, C :: A = C_IJ where I=voigt(i,j), J=voigt(k,l).

    Note: We intentionally return a (possibly) non-symmetric 6x6 matrix A.
    The Fréchet derivative of a linear measurement is the direction itself.
    """
    I = voigt_index(i, j)
    J = voigt_index(k, l)
    A = np.zeros((6, 6), dtype=float)
    A[I, J] = 1.0
    return A


def Type3_Phi(Chom, i, j, k, l):
    """Type-3 functional h(C)=C::(varphi^{ij}⊗varphi^{kl})."""
    A = Prod_varphi_voigt(i, j, k, l)
    return float(np.sum(Chom * A))


def Type4_Phi(Chom, i, j, k, l, m, n, p, q):
    """Type-4 functional h(C)=(C::A)(C::B)."""
    A = Prod_varphi_voigt(i, j, k, l)
    B = Prod_varphi_voigt(m, n, p, q)
    CA = float(np.sum(Chom * A))
    CB = float(np.sum(Chom * B))
    return CA * CB



def Type3_dPhi_dC(Chom, i, j, k, l):
    """Fréchet derivative dPhi_dC for Type-3: returns the direction tensor A."""
    return Prod_varphi_voigt(i, j, k, l)


def Type4_dPhi_dC(Chom, i, j, k, l, m, n, p, q):
    """Fréchet derivative for Type-4: (C::B)A + (C::A)B."""
    A = Prod_varphi_voigt(i, j, k, l)
    B = Prod_varphi_voigt(m, n, p, q)
    CA = float(np.sum(Chom * A))
    CB = float(np.sum(Chom * B))
    return CB * A + CA * B


# ------------------------------------------------------------
# 7b) UTI material objective (3D tetragonal design space) + TI soft penalty
#
#   The raw Chom matrix is not modified.  Because this optimizer runs in a
#   tetragonal z-rot4 design space, the scalar objective terms below use
#   tetragonal representative components:
#
#       C11t = 0.5*(C1111 + C2222)
#       C12t = 0.5*(C1122 + C2211)
#       C13t = 0.25*(C1133 + C3311 + C2233 + C3322)
#       C44t = 0.5*(C1313 + C2323)
#       C33t = C3333
#       C66t = C1212
#
#   Note: C44t averages C1313 with C2323, not C1212.  C1212 is C66 and enters
#   only the extra transverse-isotropy residual.
#
#   hb = C11t + C12t - C13t - C33t
#   ha = 5*C11t - 2*C33t - 7*C12t + 4*C13t - 6*C44t
#   H  = C11t + C33t - 2*C13t - 4*C44t
#   R_TI = C11t - C12t - 2*C66t
#
#   J_hb = beta_a * hb^2 + beta_b / (ha^2 + H^2 + eps)
#   (eps = eps_denom for stable division when ha,H -> 0)
#
#   Full objective in evaluate(): J = J_hb + J_ti
# ------------------------------------------------------------


def _component_direction(i, j, k, l):
    return Type3_dPhi_dC(None, i, j, k, l)


def c1111_raw_value(Chom):
    return float(Type3_Phi(Chom, 0, 0, 0, 0))


def c2222_raw_value(Chom):
    return float(Type3_Phi(Chom, 1, 1, 1, 1))


def c1122_raw_value(Chom):
    return float(Type3_Phi(Chom, 0, 0, 1, 1))


def c2211_raw_value(Chom):
    return float(Type3_Phi(Chom, 1, 1, 0, 0))


def c1133_raw_value(Chom):
    return float(Type3_Phi(Chom, 0, 0, 2, 2))


def c3311_raw_value(Chom):
    return float(Type3_Phi(Chom, 2, 2, 0, 0))


def c2233_raw_value(Chom):
    return float(Type3_Phi(Chom, 1, 1, 2, 2))


def c3322_raw_value(Chom):
    return float(Type3_Phi(Chom, 2, 2, 1, 1))


def c1212_raw_value(Chom):
    return float(Type3_Phi(Chom, 0, 1, 0, 1))


def c1313_raw_value(Chom):
    return float(Type3_Phi(Chom, 0, 2, 0, 2))


def c2323_raw_value(Chom):
    return float(Type3_Phi(Chom, 1, 2, 1, 2))


def c3333_raw_value(Chom):
    return float(Type3_Phi(Chom, 2, 2, 2, 2))


def c1111_value(Chom):
    return 0.5 * (c1111_raw_value(Chom) + c2222_raw_value(Chom))


def c1122_value(Chom):
    return 0.5 * (c1122_raw_value(Chom) + c2211_raw_value(Chom))


def c1133_value(Chom):
    return 0.25 * (
        c1133_raw_value(Chom)
        + c3311_raw_value(Chom)
        + c2233_raw_value(Chom)
        + c3322_raw_value(Chom)
    )


def c1212_value(Chom):
    return c1212_raw_value(Chom)


def c1313_value(Chom):
    return 0.5 * (c1313_raw_value(Chom) + c2323_raw_value(Chom))


def c3333_value(Chom):
    return c3333_raw_value(Chom)


def hb_value(Chom):
    return (
        c1111_value(Chom)
        + c1122_value(Chom)
        - c1133_value(Chom)
        - c3333_value(Chom)
    )


def ha_value(Chom):
    return (
        5.0 * c1111_value(Chom)
        - 2.0 * c3333_value(Chom)
        - 7.0 * c1122_value(Chom)
        + 4.0 * c1133_value(Chom)
        - 6.0 * c1313_value(Chom)
    )


def H_value(Chom):
    return (
        c1111_value(Chom)
        + c3333_value(Chom)
        - 2.0 * c1133_value(Chom)
        - 4.0 * c1313_value(Chom)
    )


def cref_value(Chom, eps_denom=1e-12):
    return 0.5 * (c1111_value(Chom) + c3333_value(Chom)) + float(eps_denom)


def ti_residual_value(Chom):
    # TI extra relation inside a tetragonal design space: C66 = (C11-C12)/2.
    return c1111_value(Chom) - c1122_value(Chom) - 2.0 * c1212_value(Chom)


def c1111_direction():
    return 0.5 * (
        _component_direction(0, 0, 0, 0)
        + _component_direction(1, 1, 1, 1)
    )


def c1122_direction():
    return 0.5 * (
        _component_direction(0, 0, 1, 1)
        + _component_direction(1, 1, 0, 0)
    )


def c1133_direction():
    return 0.25 * (
        _component_direction(0, 0, 2, 2)
        + _component_direction(2, 2, 0, 0)
        + _component_direction(1, 1, 2, 2)
        + _component_direction(2, 2, 1, 1)
    )


def c1212_direction():
    return _component_direction(0, 1, 0, 1)


def c1313_direction():
    return 0.5 * (
        _component_direction(0, 2, 0, 2)
        + _component_direction(1, 2, 1, 2)
    )


def c3333_direction():
    return _component_direction(2, 2, 2, 2)


def hb_direction():
    return (
        c1111_direction()
        + c1122_direction()
        - c1133_direction()
        - c3333_direction()
    )


def ha_direction():
    return (
        5.0 * c1111_direction()
        - 2.0 * c3333_direction()
        - 7.0 * c1122_direction()
        + 4.0 * c1133_direction()
        - 6.0 * c1313_direction()
    )


def H_direction():
    return (
        c1111_direction()
        + c3333_direction()
        - 2.0 * c1133_direction()
        - 4.0 * c1313_direction()
    )


def cref_direction():
    return 0.5 * (c1111_direction() + c3333_direction())


def ti_residual_direction():
    return (
        c1111_direction()
        - c1122_direction()
        - 2.0 * c1212_direction()
    )


def denominator_value(Chom, eps_denom=1e-12):
    ha = ha_value(Chom)
    HH = H_value(Chom)
    return ha * ha + HH * HH + float(eps_denom)


def uti_ratio_term(Chom, beta_a=1.0, beta_b=0.0, eps_denom=1e-12):
    """
    UTI material objective term:
      J_hb = beta_a * hb^2 + beta_b / (ha^2 + H^2 + eps)

    Parameters:
      - `beta_a`: weight on hb^2
      - `beta_b`: weight on 1/(ha^2 + H^2 + eps); eps = eps_denom
    """
    a = float(beta_a)
    b = float(beta_b)
    hb = hb_value(Chom)
    ha = ha_value(Chom)
    HH = H_value(Chom)
    den = ha * ha + HH * HH + float(eps_denom)
    return a * (hb * hb) + (b / den)


def grad_uti_ratio_term(Chom, beta_a=1.0, beta_b=0.0, eps_denom=1e-12):
    a = float(beta_a)
    b = float(beta_b)
    hb = hb_value(Chom)
    ha = ha_value(Chom)
    HH = H_value(Chom)

    d_hb = hb_direction()
    d_ha = ha_direction()
    d_H = H_direction()

    den = ha * ha + HH * HH + float(eps_denom)
    d_den = (2.0 * ha) * d_ha + (2.0 * HH) * d_H

    # d/dC [ a hb^2 + b/den ] = 2 a hb d_hb - b/den^2 * d_den
    return (2.0 * a * hb) * d_hb - (b / (den * den)) * d_den


def r0_material_term(Chom, beta_a=1.0, beta_b=0.0, eps_denom=1e-12):
    """Backward-compatible alias: the active objective is the UTI `J_hb` term."""
    return uti_ratio_term(Chom, beta_a=beta_a, beta_b=beta_b, eps_denom=eps_denom)


def grad_r0_material_term(Chom, beta_a=1.0, beta_b=0.0, eps_denom=1e-12):
    """Backward-compatible alias: gradient of the active UTI `J_hb` term."""
    return grad_uti_ratio_term(Chom, beta_a=beta_a, beta_b=beta_b, eps_denom=eps_denom)


def _resolve_ti_penalty_mode(normalization_mode=None, normalize_by_cref=True):
    if normalization_mode is not None:
        mode = str(normalization_mode).strip().lower()
        if mode in ("none", "cref", "ha_h"):
            return mode
        raise ValueError("ti penalty normalization mode must be one of: none, cref, ha_h")
    return "cref" if bool(normalize_by_cref) else "none"


def ti_penalty_term(Chom, alpha, eps_denom=1e-12, normalize_by_cref=True, normalization_mode=None):
    r_ti = ti_residual_value(Chom)
    a = float(alpha)
    mode = _resolve_ti_penalty_mode(normalization_mode=normalization_mode, normalize_by_cref=normalize_by_cref)
    if mode == "none":
        return a * (r_ti * r_ti)
    if mode == "cref":
        denom = float(cref_value(Chom, eps_denom=eps_denom))
        z = r_ti / denom
        return a * (z * z)
    else:
        denom = float(denominator_value(Chom, eps_denom=eps_denom))
        return a * (r_ti * r_ti) / denom


def grad_ti_penalty_term(Chom, alpha, eps_denom=1e-12, normalize_by_cref=True, normalization_mode=None):
    alpha = float(alpha)
    if alpha == 0.0:
        return np.zeros((6, 6), dtype=float)
    r_ti = ti_residual_value(Chom)
    d_r = ti_residual_direction()
    mode = _resolve_ti_penalty_mode(normalization_mode=normalization_mode, normalize_by_cref=normalize_by_cref)
    if mode == "none":
        return alpha * 2.0 * r_ti * d_r
    if mode == "cref":
        denom = float(cref_value(Chom, eps_denom=eps_denom))
        d_denom = cref_direction()
        inv_d2 = 1.0 / (denom * denom)
        inv_d3 = inv_d2 / denom
        return alpha * (2.0 * r_ti * inv_d2) * d_r - alpha * (2.0 * r_ti * r_ti * inv_d3) * d_denom
    else:
        ha = ha_value(Chom)
        HH = H_value(Chom)
        denom = ha * ha + HH * HH + float(eps_denom)
        d_denom = (2.0 * ha) * ha_direction() + (2.0 * HH) * H_direction()
        inv_d1 = 1.0 / denom
        inv_d2 = inv_d1 / denom
        return alpha * (2.0 * r_ti * inv_d1) * d_r - alpha * ((r_ti * r_ti) * inv_d2) * d_denom


def iso_term(Chom, alpha, eps_denom=1e-12, normalize_by_cref=True, normalization_mode=None):
    # Backward-compatible alias for the TI penalty term.
    return ti_penalty_term(
        Chom, alpha=alpha, eps_denom=eps_denom,
        normalize_by_cref=normalize_by_cref, normalization_mode=normalization_mode
    )


def grad_iso_term(Chom, alpha, eps_denom=1e-12, normalize_by_cref=True, normalization_mode=None):
    return grad_ti_penalty_term(
        Chom, alpha=alpha, eps_denom=eps_denom,
        normalize_by_cref=normalize_by_cref, normalization_mode=normalization_mode
    )


def ratio_hb_term(Chom, eps_denom=1e-12):
    # Backward-compatible alias: UTI J_hb with beta_a=1, beta_b=0 => hb^2
    return uti_ratio_term(Chom, beta_a=1.0, beta_b=0.0, eps_denom=eps_denom)


def grad_ratio_hb_term(Chom, eps_denom=1e-12):
    return grad_uti_ratio_term(Chom, beta_a=1.0, beta_b=0.0, eps_denom=eps_denom)


def Phi_uti(Chom, alpha, beta_a=1.0, beta_b=0.0, eps_denom=1e-12,
            ti_penalty_normalize_by_cref=True, ti_penalty_normalization_mode=None):
    return (
        uti_ratio_term(Chom, beta_a=beta_a, beta_b=beta_b, eps_denom=eps_denom)
        + ti_penalty_term(
            Chom, alpha=alpha, eps_denom=eps_denom,
            normalize_by_cref=ti_penalty_normalize_by_cref,
            normalization_mode=ti_penalty_normalization_mode
        )
    )


def grad_Phi_uti(Chom, alpha, beta_a=1.0, beta_b=0.0, eps_denom=1e-12,
                 ti_penalty_normalize_by_cref=True, ti_penalty_normalization_mode=None):
    return (
        grad_uti_ratio_term(Chom, beta_a=beta_a, beta_b=beta_b, eps_denom=eps_denom)
        + grad_ti_penalty_term(
            Chom, alpha=alpha, eps_denom=eps_denom,
            normalize_by_cref=ti_penalty_normalize_by_cref,
            normalization_mode=ti_penalty_normalization_mode
        )
    )


# ------------------------------------------------------------
# Backward-compatible generic aliases expected by the TD framework.
# ------------------------------------------------------------

Phi_from_Chom = Phi_uti

def grad_Phi_voigt(Chom, alpha, beta=1.0, beta_a=None, beta_b=0.0, eps_denom=1e-12):
    if beta_a is None:
        beta_a = beta
    return grad_Phi_uti(Chom, alpha=alpha, beta_a=beta_a, beta_b=beta_b, eps_denom=eps_denom)


def Phi_terms_debug(Chom):
    """Return objective components and raw Chom entries for debugging/paper matching."""
    return {
        "C11_t": c1111_value(Chom),
        "C12_t": c1122_value(Chom),
        "C13_t": c1133_value(Chom),
        "C44_t": c1313_value(Chom),
        "C33_t": c3333_value(Chom),
        "C66_t": c1212_value(Chom),
        "C1111_raw": c1111_raw_value(Chom),
        "C2222_raw": c2222_raw_value(Chom),
        "C1122_raw": c1122_raw_value(Chom),
        "C2211_raw": c2211_raw_value(Chom),
        "C1133_raw": c1133_raw_value(Chom),
        "C3311_raw": c3311_raw_value(Chom),
        "C2233_raw": c2233_raw_value(Chom),
        "C3322_raw": c3322_raw_value(Chom),
        "C1212_raw": c1212_raw_value(Chom),
        "C1313_raw": c1313_raw_value(Chom),
        "C2323_raw": c2323_raw_value(Chom),
        "C3333_raw": c3333_raw_value(Chom),
    }

# ------------------------------------------------------------
# 8) Topological derivative of C_hom using polarization tensor P (bidirectional)
#      (DT C)_{ijkl}(y) = sigma^(ij)(y) : P(y) : sigma^(kl)(y)
#
#    P(y) = (gamma(y)-1)/E(y) * [ c1(gamma) * I + c2(gamma) * (1⊗1) ]
#    c1 = 15(1-nu^2) / ((7-5*nu) + 2*gamma*(4-5*nu))
#    c2 = (1-nu)(1-2*nu)/((1+nu)*gamma+2(1-2*nu)) - 5(1-nu^2)/((7-5*nu)+2*gamma*(4-5*nu))
#
#    gamma(y): matrix (solid) -> gamma*;  inclusion (void) -> 1/gamma*  (gamma* = Init gamma_star).
# ------------------------------------------------------------

def _P_action_bidirectional(S, gamma_val, Ehost, nu):
    """P(S) = (gamma_val-1)/Ehost * [ c1*S + c2*tr(S)*I ] with c1,c2 from the image formula."""
    one = Constant(1.0)
    denom1 = Constant(7.0 - 5.0*nu) + Constant(2.0*(4.0 - 5.0*nu)) * gamma_val
    c1 = Constant(15.0*(1.0 - nu*nu)) / denom1
    denom2 = Constant((1.0 + nu)) * gamma_val + Constant(2.0*(1.0 - 2.0*nu))
    c2 = Constant((1.0 - nu)*(1.0 - 2.0*nu)) / denom2 - Constant(5.0*(1.0 - nu*nu)) / denom1
    factor = (gamma_val - one) / Ehost
    return factor * (c1 * S + c2 * tr(S) * Identity(3))


def compute_DTJ(mesh, sig_cache, chi, Vol, dPhi_dC, E0, gamma_star, nu):
    """Compute DTJ(y) from sig_cache via the chain rule.

    DTJ(y) = sum_{I,J} (dPhi/dC_IJ) * (DT C)_{IJ}(y)
    where (DT C)_{IJ}(y) = (1/Vol) * sigma^I(y) : P(y) : sigma^J(y)

    Merged with DTC computation - no intermediate DTC storage to save memory.

    Inputs:
      sig_cache: list of 6 DG0 tensor stress fields (Voigt order [11,22,33,23,13,12])
      chi: DG0 indicator (1=solid, 0=void)
      Vol: cell volume (float)
      dPhi_dC: 6x6 numpy array (Fréchet derivative direction tensor)
      E0, gamma_star, nu: material parameters (gamma* = E_void/E_solid; void E = gamma_star*E0)

    Returns:
      DTJ_CG1, DTJ_DG0
    """
    V0 = FunctionSpace(mesh, "DG", 0)
    Vls = FunctionSpace(mesh, "CG", 1)

    # E(y): host Young modulus (solid E0, void gamma_star*E0)
    # gamma(y): matrix (solid) -> gamma*; inclusion (void) -> 1/gamma*
    # Fast path: if chi is already DG0 on this mesh, fill vectors directly.
    # Fallback keeps the original projection-based construction unchanged.
    use_fast_path = False
    try:
        chi_V = chi.function_space()
        chi_el = chi_V.ufl_element()
        if (
            chi_V.mesh().id() == mesh.id()
            and chi_el.family() == "Discontinuous Lagrange"
            and int(chi_el.degree()) == 0
        ):
            chi_local = chi.vector().get_local()
            Ehost = Function(V0)
            gamma_val = Function(V0)

            E_void = float(gamma_star * E0)
            E_solid = float(E0)
            E_vals = E_void + (E_solid - E_void) * chi_local
            Ehost.vector().set_local(E_vals)
            Ehost.vector().apply("insert")
            Ehost.vector().update_ghost_values()

            gamma_void = float(1.0 / gamma_star)
            gamma_solid = float(gamma_star)
            gamma_vals = gamma_void + (gamma_solid - gamma_void) * chi_local
            gamma_val.vector().set_local(gamma_vals)
            gamma_val.vector().apply("insert")
            gamma_val.vector().update_ghost_values()
            use_fast_path = True
    except Exception:
        use_fast_path = False

    if not use_fast_path:
        Ehost = _project(chi * Constant(E0) + (1.0 - chi) * Constant(gamma_star * E0), V0)
        gamma_val = _project(chi * Constant(gamma_star) + (1.0 - chi) * Constant(1.0 / gamma_star), V0)

    # Accumulate DTJ with bidirectional P(gamma_val, Ehost, nu)
    # Performance: P(sig^J) depends only on J, so cache it once.
    P_sig_cache = [_P_action_bidirectional(sig_cache[J], gamma_val, Ehost, nu) for J in range(6)]
    DTJ_expr = Constant(0.0)
    for I in range(6):
        for J in range(6):
            coeff = float(dPhi_dC[I, J])
            if abs(coeff) < 1e-14:
                continue
            P_sigJ = P_sig_cache[J]
            DTC_IJ = (1.0 / Vol) * inner(sig_cache[I], P_sigJ)
            DTJ_expr = DTJ_expr + coeff * DTC_IJ

    DTJ_DG0 = _project(DTJ_expr, V0)
    DTJ_CG1 = _project(DTJ_DG0, Vls)

    return DTJ_CG1, DTJ_DG0

# ------------------------------------------------------------
# 9) Level-set update (Amstutz "slerp" on unit sphere in L2)
# ------------------------------------------------------------
def build_l2_mass_matrix(V):
    """
    Assemble L2 mass matrix M on FunctionSpace V:
      <u, v>_L2 = u^T M v
    Assembled once per mesh/space, reused in line-search to avoid repeated assemble/JIT.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    return assemble(u * v * dx(domain=V.mesh()))

def l2_inner_mass(f, g, M):
    """
    Compute L2 inner product using preassembled mass matrix M:
      <f, g>_L2 = f^T M g
    """
    Mg = g.vector().copy()
    M.mult(g.vector(), Mg)
    return f.vector().inner(Mg)


def build_generalized_volume_gradient(V, scale=1.0):
    """Build the generalized volume-gradient field g_v on the current level-set space."""
    return _project(Constant(float(scale)), V)


def remove_l2_component(field, basis, dx=None, M=None):
    """
    Remove the L2 component of `field` along `basis`.

    Returns `(field_perp, coeff, basis_norm_sq)` with
      field_perp = field - coeff * basis
      coeff      = <field, basis> / ||basis||^2.
    """
    if M is not None:
        basis_norm_sq = float(l2_inner_mass(basis, basis, M))
    else:
        basis_norm_sq = float(assemble(basis * basis * dx))
    out = Function(field.function_space())
    vals = field.vector().get_local().copy()
    if basis_norm_sq <= 1e-14:
        out.vector().set_local(vals)
        out.vector().apply("insert")
        return out, 0.0, float(basis_norm_sq)
    if M is not None:
        coeff = float(l2_inner_mass(field, basis, M)) / basis_norm_sq
    else:
        coeff = float(assemble(field * basis * dx)) / basis_norm_sq
    vals -= coeff * basis.vector().get_local()
    out.vector().set_local(vals)
    out.vector().apply("insert")
    return out, float(coeff), float(basis_norm_sq)


def tangent_project_l2(field, psi, dx=None, M=None):
    """Project `field` onto the tangent space of the L2 unit sphere at `psi`."""
    return remove_l2_component(field, psi, dx=dx, M=M)


def combine_l2_search_direction(primary, correction, correction_scale):
    """Return `primary + correction_scale * correction` as a Function."""
    out = Function(primary.function_space())
    vals = primary.vector().get_local().copy()
    vals += float(correction_scale) * correction.vector().get_local()
    out.vector().set_local(vals)
    out.vector().apply("insert")
    return out

def l2_norm(f, dx=None, M=None):
    """
    L2 norm. If M is provided, use matrix form sqrt(f^T M f); else fallback to assemble.
    """
    if M is not None:
        val = l2_inner_mass(f, f, M)
        return float(np.sqrt(max(val, 0.0)))
    return float(np.sqrt(assemble(f * f * dx)))

def angle_between(f, g, dx=None, M=None):
    fn = l2_norm(f, dx=dx, M=M)
    gn = l2_norm(g, dx=dx, M=M)
    if fn < 1e-14 or gn < 1e-14:
        return 0.0, fn, gn
    if M is not None:
        fg = l2_inner_mass(f, g, M)
    else:
        fg = assemble(f * g * dx)
    c = float(fg) / (fn * gn)
    c = max(-1.0, min(1.0, c))
    th = float(np.arccos(c))
    return th, fn, gn

def slerp_update(psi, g, V, dx, kappa, deg=1, it=1, g_in_V=None, psi_in_V=None, theta_pn_gn=None, l2_mass=None):
    """
    Renewed level-set (flowchart):
      N = sqrt(int_Omega psi^2) if It==0 else 1
      psi_{n+1} = (1/(N*sin(theta))) * [ sin((1-kappa)theta)*psi + sin(kappa*theta)*g/||g|| ]
    g_in_V / psi_in_V: if provided, use as g/psi already in V (avoids repeated project() in MPI line-search).
    theta_pn_gn: if provided, (theta, pn, gn) from angle_between; skips repeated angle evaluation.
    l2_mass: optional preassembled L2 mass matrix for angle_between when theta_pn_gn is not given.
    """
    g_proj = g_in_V if g_in_V is not None else _project(g, V)
    p_for_expr = psi_in_V if psi_in_V is not None else _project(psi, V)
    if theta_pn_gn is not None:
        theta, pn, gn = theta_pn_gn
    else:
        theta, pn, gn = angle_between(psi, g_proj, dx=dx, M=l2_mass)
    if theta < 1e-14 or gn < 1e-14:
        return psi, theta, gn

    # N = ||psi||_L2 at first iteration (It=0), else 1
    N = float(pn) if it == 1 else 1.0
    denom = N * np.sin(theta)  # 1/(N*sin(theta)) as in flowchart
    if abs(denom) < 1e-14:
        return psi, theta, gn

    psi_expr = Expression(
        "inv_Nsin * ( sin((1.0-k)*th)*p + sin(k*th)*g/gn )",
        degree=deg, inv_Nsin=1.0/denom, th=theta, k=float(kappa),
        p=p_for_expr, g=g_proj, gn=float(gn)
    )
    return psi_expr, theta, gn
