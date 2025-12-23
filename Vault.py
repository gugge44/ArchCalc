"""
Streamlit Web Application for Arch Thrust Line Analysis
Minimax formulation: find minimum thickness for thrust line to fit within arch.
Usage: streamlit run arch_thrust_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize_scalar
from io import BytesIO
import tempfile
from pathlib import Path


class LoadGrid:
    """
    Uniform x-grid for discretising vertical loads for thrust-line / funicular analysis.

    The grid stores:
      - q(x): distributed line load intensity sampled at nodes [kN/m]
      - pls : point loads as (P, x0) with P downward-positive [kN]

    Loads can be defined as:
      - udl: uniform distributed load over an interval
      - vdl: linearly varying distributed load over an interval
      - pl : concentrated point load
    """

    def __init__(self, L, dx):
        """
        Parameters
        ----------
        L : float
            Span / horizontal length of arch (and equivalent beam) [m]
        dx : float
            Grid spacing [m]
        """
        self.L, self.dx = float(L), float(dx)
        self.x = np.linspace(0.0, self.L, int(round(self.L / self.dx)) + 1)
        self.q = np.zeros_like(self.x)     # distributed intensity at nodes [kN/m]
        self.pls = []                      # list[(P [kN], x0 [m])]

    def udl(self, w, xa=0.0, xb=None):
        """
        Add a (possibly partial) uniform distributed load.

        Parameters
        ----------
        w : float
            Load intensity (downward-positive) [kN/m]
        xa : float, optional
            Start x-coordinate [m]
        xb : float or None, optional
            End x-coordinate [m]. If None, uses L.
        """
        if xb is None:
            xb = self.L
        xa, xb = float(xa), float(xb)
        if xb <= xa:
            return
        self.q[(self.x >= xa) & (self.x <= xb)] += float(w)

    def vdl(self, w0, w1, xa, xb):
        """
        Add a linearly varying distributed load from w0 at xa to w1 at xb.

        Parameters
        ----------
        w0 : float
            Intensity at xa (downward-positive) [kN/m]
        w1 : float
            Intensity at xb (downward-positive) [kN/m]
        xa : float
            Start x-coordinate [m]
        xb : float
            End x-coordinate [m]
        """
        xa, xb = float(xa), float(xb)
        if xb <= xa:
            return
        m = (self.x >= xa) & (self.x <= xb)
        t = (self.x[m] - xa) / (xb - xa)
        self.q[m] += (1.0 - t) * float(w0) + t * float(w1)

    def pl(self, P, x0):
        """
        Add a point load.

        Parameters
        ----------
        P : float
            Point load magnitude (downward-positive) [kN]
        x0 : float
            x-coordinate of load application [m]
        """
        x0 = float(x0)
        if 0.0 <= x0 <= self.L:
            self.pls.append((float(P), x0))


def beam_MV(g, lam=1.0):
    """
    Compute shear V(x) and bending moment M(x) for an equivalent simply-supported BEAM
    under the same vertical loads as the arch (used to generate the funicular shape).

    This is a *load equilibrium tool* for thrust-line analysis:
      y_funicular(x) = M_beam(x) / H

    Parameters
    ----------
    g : LoadGrid
        Discretised loads on [0, L]
    lam : float, optional
        Load factor / multiplier applied to all loads (distributed and point).
        - lam = 1.0 gives the base load case.
        - To find collapse factor, you vary lam and check if a contained thrust line exists.

    Returns
    -------
    x : ndarray
        Grid stations [m]
    V : ndarray
        Shear force just to the right of each station [kN]
    M : ndarray
        Bending moment diagram with M[0] = 0 [kNm]
    RA : float
        Left support reaction [kN]
    RB : float
        Right support reaction [kN]

    Notes
    -----
    Sign convention:
      - Downward loads are positive.
      - Reactions (upwards) are positive.
      - M is sagging-positive (standard beam convention).
    """
    x, dx = g.x, g.dx

    # Convert distributed intensity q [kN/m] at nodes to equivalent nodal forces F [kN]
    # using trapezoidal weights (end nodes are half weight).
    wts = np.ones_like(x)
    wts[0] = wts[-1] = 0.5
    F = float(lam) * g.q * dx * wts  # nodal forces from distributed loads [kN]

    # Deposit point loads to adjacent nodes using linear shape functions.
    for P, xp in g.pls:
        xp = float(np.clip(xp, 0.0, g.L))
        # Index of left node
        i0 = min(int(xp / dx), len(x) - 2)
        # Local coordinate between nodes
        t = xp / dx - i0
        F[i0]   += float(lam) * P * (1.0 - t)
        F[i0+1] += float(lam) * P * t

    # Support reactions from global equilibrium
    W  = float(np.sum(F))               # total vertical load [kN]
    RB = float(np.sum(F * x) / g.L) if g.L > 0 else 0.0
    RA = W - RB

    # Shear: start from RA and step through nodal loads.
    # V[i] here is "just right" of node i (after applying load at node i).
    V = np.empty_like(x)
    running = RA
    for i in range(len(x)):
        running -= F[i]
        V[i] = running

    # Moment: integrate shear using trapezoidal rule, enforcing M[0]=0.
    M = np.zeros_like(x)
    for i in range(1, len(x)):
        M[i] = M[i-1] + 0.5 * (V[i-1] + V[i]) * dx

    return x, V, M, RA, RB

def y_arch(x, L, f):
    """
    y-component of arch centerline at station x.

    Parameters
    ----------
    x : float
        Horizontal coordinate along the arch [m]
    L : float
        Span of the arch [m]
    f : float
        Rise of the arch centreline [m]

    Returns
    -------
    yc : float
        y-component of arch centerline at station x
    """
    return 4*f/L**2 * x * (L-x)

def unit_normal(x, L, f):
    """
    Unit normal vector to the arch centreline at station x.

    Parameters
    ----------
    x : float
        Horizontal coordinate along the arch [m]
    L : float
        Span of the arch [m]
    f : float
        Rise of the arch centreline [m]

    Returns
    -------
    nx : float
        x-component of unit normal (positive left)
    ny : float
        y-component of unit normal (positive up)
    """
    s = 4 * f / L**2 * (L - 2 * x)   # dy/dx of centreline
    d = np.sqrt(1.0 + s * s)        # normalisation factor
    return -s / d, 1.0 / d

def solve_minimax(x, ny, yc, M, H):
    """
    Solve the minimax (Chebyshev) problem for thrust-line placement
    for a FIXED horizontal thrust H.

    We assume the thrust line family:
        y_t(x) = M(x)/H + a + b*x
        (M = H*y with constant H, and while keeping ∂²M=q we can add ax and b)

    and define the normal offset from the arch centreline:
        e(x) = (y_t(x) - y_c(x)) / n_y(x)

    This function finds (a, b) that minimise:
        delta = max_x |e(x)|

    Parameters
    ----------
    x  : array
        Horizontal station along the arch [m]
    ny : array
        Vertical component of unit normal to the arch centreline at x [-]
    yc : array
        Arch centreline vertical coordinate y_c(x) [m]
    M  : array
        Simply-supported beam moment diagram M(x) for vertical loads [kNm]
    H  : float
        Horizontal thrust [kN]

    Returns
    -------
    dict with keys:
        a     : vertical shift of thrust line
        b     : rotation (tilt) of thrust line
        delta : maximum normal offset |e(x)| [m]
    or None if infeasible
    """

    if H <= 0 or np.any(np.abs(ny) < 1e-10):
        return None

    g  = (M / H - yc) / ny      # known term from loads + geometry
    ca = 1.0 / ny              # coefficient multiplying 'a'
    cb = x / ny                # coefficient multiplying 'b'

    # ------------------------------------------------------------------
    # Minimax constraints:
    #   -delta <= e(x) <= delta
    # which becomes two linear inequalities:
    #   ca*a + cb*b - delta <= -g
    #  -ca*a - cb*b - delta <=  g
    # Stack them for all x locations
    # ------------------------------------------------------------------
    A = np.vstack([
        np.column_stack([ ca,  cb, -np.ones_like(x)]),
        np.column_stack([-ca, -cb, -np.ones_like(x)])
    ])
    b = np.concatenate([-g, g])

    res = linprog(
        c=[0, 0, 1],                     # minimise delta
        A_ub=A,
        b_ub=b,
        bounds=[(None, None), (None, None), (0, None)],
        method="highs"
    )

    if res.success and res.x[2] >= 0:
        return {
            "a":     res.x[0],   # vertical shift of thrust line
            "b":     res.x[1],   # rotation/tilt of thrust line
            "delta": res.x[2],   # max |normal offset| (half required thickness)
        }

    return None
    
def compute_for_H(M_of, L, f, H, N=2500):
    """
    Construct the optimal (minimax) thrust-line placement for a *fixed* horizontal thrust H.

    We use the thrust-line family (vertical loads only):
        y_t(x) = M(x)/H + a + b*x

    For a given H, solve_minimax finds a and b that minimise:
        delta = max_x |e(x)|

    where e(x) is the normal offset from the arch centreline:
        e(x) = (y_t(x) - y_c(x)) / n_y(x)

    Parameters
    ----------
    M_of : callable
        Function returning beam moment M(x) [kNm] at x [m], e.g. from interpolation.
    L : float
        Span [m]
    f : float
        Rise of arch centreline [m]
    H : float
        Horizontal thrust [kN]. Controls curvature of funicular term M/H.
    N : int, optional
        Number of x sample stations used for containment checking / plotting.

    Returns
    -------
    sol : dict or None
        If feasible, returns a dictionary containing:
          - H : horizontal thrust [kN]
          - a, b : rigid-body placement parameters of thrust line
          - delta : maximum absolute normal offset |e| [m] (half required thickness)
          - t_req : required thickness to contain thrust line = 2*delta [m]
          - x : stations [m]
          - yc : centreline y at stations [m]
          - nx, ny : components of unit normal at stations [-]
          - M : beam moment at stations [kNm]
          - yt : thrust-line y at stations [m]
          - e : normal offset at stations [m]
        Returns None if LP is infeasible or numerically invalid for this H.

    Notes
    -----
    - This is a *mechanism* check under Heyman-style assumptions:
      existence of a contained thrust line implies no collapse mechanism for that load case.
    - It does not check crushing or sliding.
    """
    xx = np.linspace(0.0, L, int(N)) # Sample stations along the span
    yc = y_arch(xx, L, f) # Arch centreline geometry
    nx, ny = unit_normal(xx, L, f) # Unit normal components at each station
    M = M_of(xx) # Equivalent simply-supported beam moment under the same vertical loads
    
    ab = solve_minimax(xx, ny, yc, M, H) # For this fixed H, find the best (a,b) that minimises max |e|
    if ab is None:
        return None

    yt = M / H + ab["a"] + ab["b"] * xx # Assemble the thrust line (geometry, not a material bending line)
    e = (yt - yc) / ny # Convert vertical offset to normal offset (distance to centreline along local normal)
    
    return {
        "H": float(H), **ab,                       # a, b, delta
        "t_req": 2.0 * float(ab["delta"]),  # thickness required to contain thrust line [m]
        "x": xx, "yc": yc, "nx": nx, "ny": ny, "M": M, "yt": yt, "e": e,
    }


def find_best_H(M_of, L, f, H_lo=None, H_hi=None):
    """
    Find the horizontal thrust H that minimises the required thickness (delta)
    for a contained thrust line.

    This wraps a 1D scalar optimisation around compute_for_H:
        H* = argmin_H delta(H)

    Parameters
    ----------
    M_of : callable
        Moment function M(x) [kNm]
    L : float
        Span [m]
    f : float
        Rise [m]
    H_lo : float, optional
        Lower bound for horizontal thrust search [kN]
    H_hi : float, optional
        Upper bound for horizontal thrust search [kN]

    Returns
    -------
    sol : dict or None
        The compute_for_H solution at the best H, or None if optimisation fails.

    Notes
    -----
    - If you already have an estimate for H (e.g. for UDL: H ~ wL^2/(8f)),
      narrowing [H_lo, H_hi] improves robustness and speed.
    """
    if H_lo is None: H_lo = 0
    if H_hi is None: H_hi = 1e9
     
    def obj(H): # For each candidate H, compute best placement and take the max offset delta
        sol = compute_for_H(M_of, L, f, H)
        return sol["delta"] if sol else 1e30

    res = minimize_scalar(obj, bounds=(float(H_lo), float(H_hi)), method="bounded")
    if not res.success:
        return None

    return compute_for_H(M_of, L, f, float(res.x))

def contact_clusters(xx, e, delta):
    """
    Identify contiguous clusters where the thrust line is effectively on the boundary,
    i.e. where |e(x)| ≈ delta.

    These "active constraint" locations are where the limiting thrust line scrapes
    intrados/extrados (potential hinge locations in the limit mechanism sense).

    Parameters
    ----------
    xx : ndarray
        Stations [m]
    e : ndarray
        Normal offset e(x) [m]
    delta : float
        Envelope half-thickness, i.e. max|e| [m]

    Returns
    -------
    clusters : list[tuple[int,int,int]]
        Each cluster is (i0, i1, sgn) where:
          - i0..i1 are index bounds in xx/e inclusive
          - sgn = +1 means extrados contact (e>0)
          - sgn = -1 means intrados contact (e<0)

    Notes
    -----
    - Because everything is discretised, we use an absolute tolerance to decide "touching".
    - If you see long clusters, that usually means near-plateau behaviour from discretisation.
    """
    # Consider a point "touching" if |e| is close to delta within tolerance.
    # Tolerance scales weakly with delta to avoid missing contacts at larger delta.
    touch = np.isclose(np.abs(e), float(delta), atol=max(1e-5, 1e-6 * float(delta)), rtol=0.0)

    clusters = []
    i = 0
    n = len(xx)

    while i < n:
        if touch[i]: # Start of a cluster of consecutive touching points
            j = i + 1
            while j < n and touch[j]:
                j += 1

            mid = (i + j - 1) // 2 # Cluster is i..j-1 inclusive
            sgn = 1 if e[mid] >= 0 else -1

            clusters.append((i, j - 1, sgn))
            i = j
        else:
            i += 1

    return clusters

def plot_results(sol, L, f, clusters, a_sol=None, V_sol=None, N_sol=None):
    """
    Plot:
      1) Arch geometry band (±t/2) and the admissible thrust line.
         If a_sol is supplied, also plot dashed envelope for total thickness:
             ±(t/2 + a(x)/2)
      2) Equivalent simply-supported beam moment M(x)
      3) Normal eccentricity e(x) and the minimax bounds ±delta
      4) Resultant compression N(x) and (optionally) strut thickness a(x)

    Parameters
    ----------
    sol : dict
        Solution from compute_for_H / find_best_H. Must contain:
        x, yc, nx, ny, e, M, H, t_req, delta
    L, f : float
        Span and rise used to compute normals for hinge markers.
    clusters : list[(i0, i1, sgn)]
        Contact clusters from contact_clusters. sgn=+1 extrados, -1 intrados.
    a_sol : ndarray or None
        Compression strut thickness a(x) [m] (capacity-driven extra thickness).
    V_sol : ndarray or None
        Equivalent-beam shear V(x) [kN] at sol["x"] stations.
    N_sol : ndarray or None
        Resultant compression N(x) = sqrt(H^2 + V^2) [kN] at sol["x"] stations.
    """
    # --- unpack solution arrays ---
    xx = sol["x"]      # stations along span [m]
    yc = sol["yc"]     # centreline y(x) [m]
    nx = sol["nx"]     # unit normal x-component [-]
    ny = sol["ny"]     # unit normal y-component [-]
    e  = sol["e"]      # normal offset of thrust line from centreline [m]
    t  = sol["t_req"]  # geometric thickness required for containment = 2*delta [m]
    H  = sol["H"]      # horizontal thrust [kN]

    # If N(x) not supplied but V(x) is, infer N(x)
    if N_sol is None and V_sol is not None:
        N_sol = np.sqrt(H**2 + V_sol**2)

    # 4 stacked plots: geometry, M, e, N
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 12),
        gridspec_kw={"height_ratios": [2.2, 1.1, 1.1, 1.1]}
    )

    # ============================================================
    # (1) Geometry + thrust line
    # ============================================================
    ax = axes[0]

    # If a(x) is provided, plot dashed total envelope:
    #   ±(t/2 + a(x)/2) along the local normal
    if a_sol is not None:
        a = a_sol
        ax.plot(
            xx + (t/2 + a/2) * nx,
            yc + (t/2 + a/2) * ny,
            "k--", lw=1.2, label="Envelope + strut"
        )
        ax.plot(
            xx - (t/2 + a/2) * nx,
            yc - (t/2 + a/2) * ny,
            "k--", lw=1.2
        )

    # Plot the geometric band (±t/2) as a filled polygon
    x_up = xx + (t/2) * nx
    y_up = yc + (t/2) * ny
    x_dn = xx - (t/2) * nx
    y_dn = yc - (t/2) * ny
    ax.fill(np.r_[x_up, x_dn[::-1]], np.r_[y_up, y_dn[::-1]], alpha=0.2)

    # Intrados/extrados edges for the geometric thickness only
    ax.plot(x_up, y_up, "k-", lw=1.5)
    ax.plot(x_dn, y_dn, "k-", lw=1.5)

    # Centreline and thrust line (thrust line plotted by offsetting centreline by e along normal)
    ax.plot(xx, yc, "b-", lw=2, alpha=0.7, label="Centreline")
    ax.plot(xx + e * nx, yc + e * ny, "r--", lw=2, label="Thrust line")

    # Mark hinge/contact clusters at the boundary: extrados (+) or intrados (-)
    for i0, i1, sgn in clusters:
        xh = 0.5 * (xx[i0] + xx[i1])               # representative x within cluster
        nxh, nyh = unit_normal(np.array([xh]), L, f)
        ych = y_arch(xh, L, f)
        ax.plot(
            xh + sgn * t/2 * nxh[0],
            ych + sgn * t/2 * nyh[0],
            "go", ms=12, mec="k", mew=2
        )

    ax.set_title(f"H = {H:.1f} kN, t_req = {t*1000:.1f} mm")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")

    # ============================================================
    # (2) Equivalent beam moment M(x)
    # ============================================================
    axes[1].fill_between(xx, sol["M"], alpha=0.3)
    axes[1].plot(xx, sol["M"], lw=2)
    axes[1].set_title("Beam Moment")
    axes[1].grid(alpha=0.25)

    # ============================================================
    # (3) Normal offset e(x) with minimax bounds ±delta
    # ============================================================
    axes[2].fill_between(xx, e, alpha=0.3)
    axes[2].plot(xx, e, lw=2)
    axes[2].axhline(sol["delta"], ls="--")
    axes[2].axhline(-sol["delta"], ls="--")
    axes[2].set_title(f"Normal Offset δ = {sol['delta']*1000:.1f} mm")
    axes[2].grid(alpha=0.25)

    # ============================================================
    # (4) Compressive resultant N(x) and strut thickness a(x)
    # ============================================================
    axN = axes[3]

    if N_sol is not None:
        lineN, = axN.plot(xx, N_sol, lw=2, label="N(x) compressive resultant (kN)")
        axN.fill_between(xx, N_sol, alpha=0.2)
        axN.set_ylabel("N (kN)")
        axN.grid(alpha=0.25)

        lines = [lineN]
        labels = [lineN.get_label()]

        if a_sol is not None:
            axA = axN.twinx()
            lineA, = axA.plot(xx, a_sol * 1000.0, ls="--", lw=1.8, label="a(x) strut thickness (mm)")
            axA.set_ylabel("a (mm)")
            lines.append(lineA)
            labels.append(lineA.get_label())

        axN.set_title("Compressive resultant and required strut thickness")
        axN.legend(lines, labels, loc="upper center", ncol=2)
    else:
        axN.set_title("Compressive resultant N(x) (not available)")
        axN.grid(alpha=0.25)

    axN.set_xlabel("x (m)")

    plt.tight_layout()
    return fig
    
def plot_loading(g, L):
    """
    Plot distributed and point loads along the span.

    Parameters
    ----------
    g : LoadGrid
        Load discretisation
    L : float
        Span [m]
    """
    fig, ax = plt.subplots(figsize=(10, 3))

    # Distributed load intensity q(x)
    ax.fill_between(g.x, g.q, alpha=0.4, color="blue")
    ax.plot(g.x, g.q, "b-", lw=1.5)

    # Point loads (arrows)
    for P, xp in g.pls:
        q_ref = max(g.q) if max(g.q) > 0 else 10
        ax.annotate(
            "",
            xy=(xp, 0),
            xytext=(xp, q_ref * 0.3),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )
        ax.text(xp, q_ref * 0.35, f"{P:.0f} kN", ha="center", fontsize=9, color="red")

    ax.set_xlim(0, L)
    ax.set_ylabel("Load intensity q(x) [kN/m]")
    ax.set_xlabel("x (m)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def remesh_loadgrid(g: LoadGrid, L_new: float) -> LoadGrid:
    """
    Create a new LoadGrid on span L_new by RIGIDLY SHIFTING loads
    so that the original intrados-based load pattern remains centred.

    Assumption:
    - Loads are defined on the intrados geometry.
    - When span increases, the intrados load field shifts by ΔL/2.
    """
    L_new = float(L_new)
    g2 = LoadGrid(L_new, g.dx)

    dL = L_new - g.L
    shift = 0.5 * dL

    # --- distributed loads (shift, not scale) ---
    # evaluate old q at shifted coordinates
    x_old = g2.x - shift
    g2.q = np.interp(
        x_old,
        g.x,
        g.q,
        left=0.0,
        right=0.0
    )

    # --- point loads (shift, not scale) ---
    for P, xp in g.pls:
        xp_new = xp + shift
        if 0.0 <= xp_new <= L_new:
            g2.pls.append((float(P), float(xp_new)))

    return g2

def solve_from_intrados(
    inside_L,
    inside_f,
    g,
    b_eff,
    f_cd,
    *,
    tol=1e-4,
    max_iter=40,
    H_lo=None,
    H_hi=None,
    relax=0.5,      # 0<relax<=1 : under-relaxation for stability
    verbose=True,
):
    inside_L = float(inside_L)
    inside_f = float(inside_f)
    b_eff = float(b_eff)
    f_cd = float(f_cd)

    # Start with centreline ~ intrados
    L = inside_L
    f = inside_f

    t_prev = None
    history = []

    for it in range(1, max_iter + 1):

        # --- IMPORTANT: loads must live on the same span as the current geometry
        g_it = remesh_loadgrid(g, L)

        # Beam equilibrium on current span
        x_b, V_b, M_b, RA, RB = beam_MV(g_it)
        M_of = lambda x: np.interp(x, x_b, M_b)

        # Thrust-line minimax thickness on current geometry
        sol = find_best_H(M_of, L, f, H_lo=H_lo, H_hi=H_hi)
        if sol is None:
            raise RuntimeError(f"No admissible thrust line at iteration {it}")

        t_geo = float(sol["t_req"])  # geometric thickness = 2*delta

        # Compressive resultant and strut thickness
        V_sol = np.interp(sol["x"], x_b, V_b)
        N_sol = np.sqrt(sol["H"]**2 + V_sol**2)

        fcd_kNpm2 = 1000.0 * f_cd  # MPa -> kN/m²
        a_sol = N_sol / (b_eff * fcd_kNpm2)
        a_max = float(np.max(a_sol))

        # Total thickness demand
        t_total_raw = t_geo + a_max

        # Under-relaxation to avoid oscillation / runaway
        if t_prev is None:
            t_total = t_total_raw
        else:
            t_total = (1.0 - relax) * t_prev + relax * t_total_raw

        history.append({
            "iter": it,
            "L": L,
            "f": f,
            "t_geo": t_geo,
            "a_max": a_max,
            "t_total_raw": t_total_raw,
            "t_total": t_total,
            "H": sol["H"],
        })

        if verbose:
            print(
                f"[iter {it:02d}] "
                f"L={L:.3f} m, f={f:.3f} m | "
                f"t_geo={t_geo*1000:.1f} mm, a_max={a_max*1000:.1f} mm → "
                f"t_total={t_total*1000:.1f} mm"
            )

        # Convergence on total thickness
        if t_prev is not None and abs(t_total - t_prev) < tol:
            if verbose:
                print("Converged.")
            break

        # Update centreline geometry from intrados (your chosen rule)
        L = inside_L + t_total
        f = inside_f + 0.5 * t_total
        t_prev = t_total

    else:
        raise RuntimeError("Intrados-to-centreline iteration did not converge")

    return {
        "sol": sol,
        "L": L,
        "f": f,
        "t_geo": t_geo,
        "a_max": a_max,
        "t_total": t_total,
        "history": history,
        "V_sol": V_sol,
        "N_sol": N_sol,
        "a_sol": a_sol,
        "RA": RA,
        "RB": RB,
    }

if __name__ == "__main__":
    L_inside = 25.0 # [m] length of arch (cl)
    f_inside = 5.0 # [m] height of arch (cl)
    b_eff = 1.0 # [m] width into page
    f_cd = 0.5 # [MPa] compressive capacity
    dx = 0.05
    udls = [
        {"w": 3, "xa": 0.0, "xb": L_inside}
    ]
    vdls = [
        {"w0": 1.0, "w1": 3.0, "xa": 0.0, "xb": 3}
    ]
    pls = [
        {"P": 3, "x0": 4}
    ]
    
    g = LoadGrid(L_inside, dx)
    for udl in udls:
        g.udl(**udl)
    for vdl in vdls:
        g.vdl(**vdl)
    for pl in pls:
        g.pl(**pl)
    
    result = solve_from_intrados(L_inside, f_inside, g, b_eff, f_cd)
    
    sol = result["sol"]
    L = result["L"]
    f = result["f"]
    t_geo = result["t_geo"]
    a_max = result["a_max"]
    t_total = result["t_total"]
    V_sol = result["V_sol"]
    N_sol = result["N_sol"]
    a_sol = result["a_sol"]
    RA = result["RA"]
    RB = result["RB"]
    i = int(np.argmax(a_sol))
    x_a = float(sol["x"][i])
    
    
    clusters = contact_clusters(sol["x"], sol["e"], sol["delta"])
    
    fig = plot_results(sol, L, f, clusters, a_sol=a_sol, V_sol=V_sol, N_sol=N_sol)
    
    print("\n" + "=" * 70)
    print("ARCH THRUST-LINE CHECK (HEYMAN / LIMIT ANALYSIS)")
    print("=" * 70)
    
    print(f"Geometry:")
    print(f"  Span L           = {L:.2f} m")
    print(f"  Rise f           = {f:.2f} m")
    print(f"  Effective width  = {b_eff:.2f} m")
    
    print("\nEquivalent beam reactions:")
    print(f"  RA = {RA:.2f} kN")
    print(f"  RB = {RB:.2f} kN")
    
    print("\nThrust-line result:")
    print(f"  Horizontal thrust H* = {sol['H']:.2f} kN")
    print(f"  Required geometric thickness t = {t_geo*1000:.1f} mm")
    
    print("\nContact (hinge) locations:")
    for i0, i1, sgn in clusters:
        xh = 0.5 * (sol["x"][i0] + sol["x"][i1])
        side = "extrados" if sgn > 0 else "intrados"
        print(f"  x = {xh:.2f} m  ({side})")
    
    print("\nCompressive force check:")
    print(f"  Max compressive stress f_cd = {f_cd:.2f} MPa")
    
    if a_sol is not None:
        print(f"  Required strut thickness a_max = {a_max*1000:.1f} mm at x = {x_a:.2f} m")
        print(f"  Total required thickness (geom + strut) = {t_total*1000:.1f} mm")
    
    print("=" * 70 + "\n")
    
    plt.show()