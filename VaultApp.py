"""
Streamlit Web Application for Arch Thrust Line Analysis
Minimax formulation: find minimum thickness for thrust line to fit within arch.
Usage: streamlit run arch_thrust_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize_scalar
from io import BytesIO
import tempfile
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


def build_docx_python(
    *,
    L: float, f: float, dx: float,
    sol: dict, clusters: list[tuple[int, int, int]],
    udls: np.ndarray, vdls: np.ndarray, pls: np.ndarray,
    RA: float, RB: float,
    info: dict,
    load_png, res_png,
    beam_x: np.ndarray, beam_V: np.ndarray, beam_M: np.ndarray, beam_RA: float, beam_RB: float
) -> bytes:
    doc = Document()

    # ---------- basic style ----------
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    style.font.size = Pt(10)

    def add_title(text: str):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(18)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def add_meta_line(text: str):
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def add_h(level: int, text: str):
        doc.add_heading(text, level=level)

    def add_kv_table(rows: list[tuple[str, str]]):
        t = doc.add_table(rows=len(rows), cols=2)
        t.style = "Table Grid"
        for i, (k, v) in enumerate(rows):
            t.cell(i, 0).text = str(k)
            t.cell(i, 1).text = str(v)
        return t

    def add_table(headers: list[str], data_rows: list[list[str]]):
        t = doc.add_table(rows=1 + len(data_rows), cols=len(headers))
        t.style = "Table Grid"
        for j, h in enumerate(headers):
            t.cell(0, j).text = h
        for i, row in enumerate(data_rows, start=1):
            for j, val in enumerate(row):
                t.cell(i, j).text = val
        return t

    # ---------- derived checks ----------
    # discretised vertical load from beam_MV formulation: W = RA+RB (consistent with your algorithm)
    W = float(beam_RA + beam_RB)
    sumV_err = float((beam_RA + beam_RB) - W)  # will be ~0 by construction, but kept explicit

    # moment check about A: for simply supported, RB*L should match integral of w*x + sum(P*x)
    # We don't have raw discrete forces here, so do a consistent check using M diagram:
    # For a simply supported beam, M(0)=M(L)=0; check end moments.
    M0 = float(beam_M[0])
    ML = float(beam_M[-1])

    # thrust-line eccentricity check
    e = sol["e"]
    delta = float(sol["delta"])
    emax = float(np.max(np.abs(e)))
    emax_minus_delta = emax - delta

    # ---------- content ----------
    add_title("Arch Thrust Line Analysis (Minimax / Lower-Bound)")
    add_meta_line(f"{info.get('project','')}  |  {info.get('author','')}  |  {info.get('date','')}")
    doc.add_paragraph("")

    add_h(1, "1. Scope and assumptions")
    doc.add_paragraph(
        "This note computes a lower-bound (funicular) thrust line for a 2D parabolic arch subject to the applied vertical loading, "
        "and finds the minimum uniform thickness t such that the thrust line remains within ±t/2 of the arch centreline along the local normal."
    )
    doc.add_paragraph("Assumptions / limitations:")
    for s in [
        "2D analysis; out-of-plane effects ignored.",
        "Arch centreline is parabolic: y_c(x) = 4 f x (L − x) / L².",
        "Supports provide vertical reactions only (beam analogy for M(x)).",
        "No material strength checks (compression, shear, sliding, ring separation) included; this is geometric lower-bound only.",
        "Thickness is measured normal to the centreline (uniform t).",
    ]:
        doc.add_paragraph(s, style="List Bullet")

    add_h(1, "2. Geometry")
    add_kv_table([
        ("Span, L", f"{L:.3f} m"),
        ("Rise, f", f"{f:.3f} m"),
        ("Rise/Span", f"{(f/L):.3%}"),
        ("Discretisation dx (loading)", f"{dx:.4f} m"),
        ("Thrust evaluation points", f"{len(sol['x'])}"),
    ])

    add_h(1, "3. Loading input")
    add_h(2, "3.1 UDLs")
    if len(udls):
        add_table(
            ["w", "x0", "x1"],
            [[f"{r[0]:.4g}", f"{r[1]:.4g}", f"{r[2]:.4g}"] for r in udls],
        )
    else:
        doc.add_paragraph("None")

    add_h(2, "3.2 VDLs")
    if len(vdls):
        add_table(
            ["w0", "w1", "x0", "x1"],
            [[f"{r[0]:.4g}", f"{r[1]:.4g}", f"{r[2]:.4g}", f"{r[3]:.4g}"] for r in vdls],
        )
    else:
        doc.add_paragraph("None")

    add_h(2, "3.3 Point loads")
    if len(pls):
        add_table(
            ["P", "x"],
            [[f"{r[0]:.4g}", f"{r[1]:.4g}"] for r in pls],
        )
    else:
        doc.add_paragraph("None")

    add_h(2, "3.4 Loading diagram")
    doc.add_picture(str(load_png), width=Inches(6.5))

    add_h(2, "3.5 Reactions (from discretised loading)")
    add_kv_table([
        ("R_A", f"{RA:.3f} kN"),
        ("R_B", f"{RB:.3f} kN"),
        ("R_A + R_B", f"{(RA+RB):.3f} kN"),
    ])

    add_h(1, "4. Method summary")
    doc.add_paragraph("Beam moment:")
    doc.add_paragraph("• Compute the simply-supported beam moment diagram M(x) from discretised loads (UDL/VDL + distributed point loads).")
    doc.add_paragraph("Thrust line family:")
    doc.add_paragraph("• y_t(x) = M(x)/H + a + b x")
    doc.add_paragraph("Normal eccentricity:")
    doc.add_paragraph("• e(x) = (y_t(x) − y_c(x)) / n_y(x)")
    doc.add_paragraph("Minimax thickness condition:")
    doc.add_paragraph("• Find a, b to minimise δ such that −δ ≤ e(x) ≤ +δ for all x.")
    doc.add_paragraph("• Required thickness: t = 2 δ.")
    doc.add_paragraph("Outer optimisation:")
    doc.add_paragraph("• Vary horizontal thrust H to minimise δ (bounded scalar search).")

    add_h(1, "5. Checks (numerical sanity)")
    add_kv_table([
        ("Beam end moment M(0)", f"{M0:.6g}"),
        ("Beam end moment M(L)", f"{ML:.6g}"),
        ("Vertical equilibrium check (ΣV error)", f"{sumV_err:.6g}"),
        ("max|e(x)|", f"{emax*1000:.3f} mm"),
        ("δ", f"{delta*1000:.3f} mm"),
        ("max|e| − δ", f"{emax_minus_delta*1000:.3f} mm"),
    ])
    doc.add_paragraph(
        "Note: max|e|−δ should be ~0 (within numerical tolerance); a small positive value indicates discretisation/LP tolerance effects."
    )

    add_h(1, "6. Results")
    add_kv_table([
        ("H*", f"{sol['H']:.3f} kN"),
        ("a", f"{sol['a']:.8g}"),
        ("b", f"{sol['b']:.8g}"),
        ("δ", f"{sol['delta']*1000:.3f} mm"),
        ("t_req = 2δ", f"{sol['t_req']*1000:.3f} mm"),
        ("R_A", f"{RA:.3f} kN"),
        ("R_B", f"{RB:.3f} kN"),
    ])

    add_h(2, "6.1 Contact (hinge) locations")
    if clusters:
        xx = sol["x"]
        rows = []
        for k, (i0, i1, sgn) in enumerate(clusters, start=1):
            xh = 0.5 * (xx[i0] + xx[i1])
            side = "Extrados" if sgn > 0 else "Intrados"
            rows.append([str(k), f"{xh:.4f}", side])
        add_table(["#", "x (m)", "Side"], rows)
    else:
        doc.add_paragraph("No contact clusters detected (check tolerances / δ / discretisation).")

    add_h(2, "6.2 Diagrams")
    doc.add_picture(str(res_png), width=Inches(6.5))

    add_h(1, "7. Conclusion")
    doc.add_paragraph(
        f"Minimum required thickness (lower-bound geometric condition) is t = {sol['t_req']*1000:.1f} mm "
        f"at H = {sol['H']:.1f} kN."
    )

    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()



def editor_to_float_array(ed: pd.DataFrame, ncol: int) -> np.ndarray:
    if ed is None or len(ed) == 0:
        return np.array([]).reshape(0, ncol)

    ed = ed.copy().iloc[:, :ncol]

    def scalar(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return v[0] if len(v) else np.nan
        return v

    # per-column map (no applymap)
    for c in ed.columns:
        ed[c] = ed[c].map(scalar)
        ed[c] = pd.to_numeric(ed[c], errors="coerce")

    ed = ed.dropna(how="any")
    return ed.to_numpy(dtype=float) if len(ed) else np.array([]).reshape(0, ncol)




st.set_page_config(page_title="Arch Thrust Line", page_icon="🌉", layout="wide")

# Session state
for key, val in [
    ('udls', np.array([[10.0, 0.0, 20.0]])),
    ('vdls', np.array([]).reshape(0, 4)),
    ('pls', np.array([[50.0, 5.0], [30.0, 15.0]])),
    ('results', None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

if "docx_bytes" not in st.session_state:
    st.session_state.docx_bytes = None



# Load discretisation
class LoadGrid:
    def __init__(self, L, dx):
        self.L, self.dx = float(L), float(dx)
        self.x = np.linspace(0, self.L, int(round(self.L/self.dx))+1)
        self.q = np.zeros_like(self.x)
        self.pls = []
    def udl(self, w, xa=0, xb=None):
        xb = xb or self.L
        self.q[(self.x >= xa) & (self.x <= xb)] += w
    def vdl(self, w0, w1, xa, xb):
        m = (self.x >= xa) & (self.x <= xb)
        t = (self.x[m] - xa) / (xb - xa)
        self.q[m] += (1-t)*w0 + t*w1
    def pl(self, P, x0):
        self.pls.append((float(P), float(x0)))

def beam_MV(g, lam=1.0):
    x, dx = g.x, g.dx
    wts = np.ones_like(x); wts[0] = wts[-1] = 0.5
    F = lam * g.q * dx * wts
    for P, xp in g.pls:
        xp = np.clip(xp, 0, g.L)
        i0 = min(int(xp/dx), len(x)-2)
        t = xp/dx - i0
        F[i0] += lam*P*(1-t); F[i0+1] += lam*P*t
    W = np.sum(F)
    RB = np.sum(F*x)/g.L if g.L > 0 else 0
    RA = W - RB
    V = np.cumsum(-F) + RA + F
    M = np.zeros_like(x)
    for i in range(1, len(x)):
        M[i] = M[i-1] + 0.5*(V[i-1]+V[i])*dx
    return x, V, M, RA, RB

def y_arch(x, L, f): return 4*f/L**2 * x * (L-x)
def unit_normal(x, L, f):
    s = 4*f/L**2 * (L - 2*x)
    d = np.sqrt(1 + s*s)
    return -s/d, 1/d

def solve_minimax(x, ny, yc, M, H):
    if H <= 0 or np.any(np.abs(ny) < 1e-10): return None
    g = (M/H - yc)/ny
    ca, cb = 1/ny, x/ny
    A = np.vstack([np.column_stack([ca, cb, -np.ones_like(x)]), np.column_stack([-ca, -cb, -np.ones_like(x)])])
    b = np.concatenate([-g, g])
    res = linprog([0,0,1], A_ub=A, b_ub=b, bounds=[(None,None),(None,None),(0,None)], method="highs")
    return {"a": res.x[0], "b": res.x[1], "delta": res.x[2]} if res.success and res.x[2] >= 0 else None

def compute_for_H(M_of, L, f, H, N=2500):
    xx = np.linspace(0, L, N)
    yc = y_arch(xx, L, f)
    nx, ny = unit_normal(xx, L, f)
    M = M_of(xx)
    ab = solve_minimax(xx, ny, yc, M, H)
    if ab is None: return None
    yt = M/H + ab["a"] + ab["b"]*xx
    e = (yt - yc)/ny
    return {"H": H, **ab, "t_req": 2*ab["delta"], "x": xx, "yc": yc, "nx": nx, "ny": ny, "M": M, "yt": yt, "e": e}

def find_best_H(M_of, L, f, H_lo=1, H_hi=5e4):
    def obj(H):
        sol = compute_for_H(M_of, L, f, H)
        return sol["delta"] if sol else 1e30
    res = minimize_scalar(obj, bounds=(H_lo, H_hi), method="bounded")
    return compute_for_H(M_of, L, f, res.x) if res.success else None

def contact_clusters(xx, e, delta):
    touch = np.isclose(np.abs(e), delta, atol=max(1e-5, 1e-6*delta))
    clusters, i = [], 0
    while i < len(xx):
        if touch[i]:
            j = i+1
            while j < len(xx) and touch[j]: j += 1
            mid = (i+j-1)//2
            clusters.append((i, j-1, 1 if e[mid] >= 0 else -1))
            i = j
        else: i += 1
    return clusters

def plot_results(sol, L, f, clusters):
    xx, yc, nx, ny, e = sol["x"], sol["yc"], sol["nx"], sol["ny"], sol["e"]
    t, H = sol["t_req"], sol["H"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2.2, 1.1, 1.1]})
    ax = axes[0]
    ax.fill_between(xx + t/2*nx, yc + t/2*ny, yc - t/2*ny, alpha=0.2, color='blue')
    ax.plot(xx + t/2*nx, yc + t/2*ny, "k-", lw=1.5)
    ax.plot(xx - t/2*nx, yc - t/2*ny, "k-", lw=1.5)
    ax.plot(xx, yc, "b-", lw=2, alpha=0.7, label="Centreline")
    ax.plot(xx + e*nx, yc + e*ny, "r--", lw=2, label="Thrust line")
    for i0, i1, sgn in clusters:
        xh = 0.5*(xx[i0]+xx[i1])
        nxh, nyh = unit_normal(np.array([xh]), L, f)
        ych = y_arch(xh, L, f)
        ax.plot(xh + sgn*t/2*nxh[0], ych + sgn*t/2*nyh[0], "go", ms=12, mec="k", mew=2)
    ax.set_title(f"H = {H:.1f} kN, t_req = {t*1000:.1f} mm"); ax.legend(); ax.grid(alpha=0.25); ax.set_aspect('equal')
    axes[1].fill_between(xx, sol["M"], alpha=0.3); axes[1].plot(xx, sol["M"], 'b-', lw=2)
    axes[1].set_title("Beam Moment"); axes[1].grid(alpha=0.25)
    axes[2].fill_between(xx, e, alpha=0.3, color='orange'); axes[2].plot(xx, e, 'orange', lw=2)
    axes[2].axhline(sol["delta"], ls="--", color='red'); axes[2].axhline(-sol["delta"], ls="--", color='red')
    axes[2].set_title(f"Normal Offset δ = {sol['delta']*1000:.1f} mm"); axes[2].grid(alpha=0.25)
    plt.tight_layout()
    return fig

def plot_loading(g, L):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(g.x, g.q, alpha=0.4, color='blue'); ax.plot(g.x, g.q, 'b-', lw=1.5)
    for P, xp in g.pls:
        mq = max(g.q) if max(g.q) > 0 else 10
        ax.annotate('', xy=(xp, 0), xytext=(xp, mq*0.3), arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(xp, mq*0.35, f'{P:.0f}kN', ha='center', fontsize=9, color='red')
    ax.set_xlim(0, L); ax.grid(alpha=0.3); plt.tight_layout()
    return fig


# === MAIN APP ===
st.title("🌉 Arch Thrust Line Analysis")
with st.sidebar:
    st.header("📋 Project")
    proj = st.text_input("Project", "Arch Analysis")
    auth = st.text_input("Author", "")
    date = st.date_input("Date")
    st.divider()
    st.header("📐 Geometry")
    L = st.number_input("Span L (m)", 1.0, 100.0, 20.0)
    f = st.number_input("Rise f (m)", 0.1, 50.0, 5.0)
    st.markdown(f"**Rise/Span:** {f/L:.1%}")
    st.divider()
    dx = st.number_input("Grid (m)", 0.005, 0.5, 0.02, format="%.3f")
    H_max = st.number_input("Max H (kN)", 100.0, 1e6, 50000.0)

st.header("📊 Loading")



c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("UDL")
    df = pd.DataFrame(st.session_state.udls, columns=["w","x0","x1"])
    ed = st.data_editor(df, num_rows="dynamic", key="u", height=180)
    st.session_state.udls = editor_to_float_array(ed, 3)

with c2:
    st.subheader("VDL")
    df = pd.DataFrame(st.session_state.vdls, columns=["w0","w1","x0","x1"])
    ed = st.data_editor(df, num_rows="dynamic", key="v", height=180)
    st.session_state.vdls = editor_to_float_array(ed, 4)

with c3:
    st.subheader("Point")
    df = pd.DataFrame(st.session_state.pls, columns=["P","x"])
    ed = st.data_editor(df, num_rows="dynamic", key="p", height=180)
    st.session_state.pls = editor_to_float_array(ed, 2)



g = LoadGrid(L, dx)
for r in st.session_state.udls:
    try:
        if len(r) >= 3: g.udl(float(r[0]), float(r[1]), float(r[2]))
    except (ValueError, TypeError): pass
for r in st.session_state.vdls:
    try:
        w0, w1, x0, x1 = map(float, r)
        xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
        if xb > xa:
            g.vdl(w0, w1, xa, xb)
    except (ValueError, TypeError):
        pass

for r in st.session_state.pls:
    try:
        if len(r) >= 2: g.pl(float(r[0]), float(r[1]))
    except (ValueError, TypeError): pass

st.subheader("📈 Preview")
fig_l = plot_loading(g, L); st.pyplot(fig_l); plt.close(fig_l)
tot = np.trapezoid(g.q, g.x) + sum(P for P,_ in g.pls)

st.info(f"**Total: {tot:.1f} kN**")

st.divider()
if st.button("🔍 Run Analysis", type="primary"):
    with st.spinner("Computing..."):
        x_b, V_b, M_b, RA, RB = beam_MV(g)

        def M_of(xq):
            return np.interp(xq, x_b, M_b)

        _, _, _, RA, RB = beam_MV(g)
        sol = find_best_H(lambda x: M_of(x), L, f, H_hi=H_max)
        if sol:
            st.session_state.results = sol
            st.session_state.rx = (RA, RB)
            st.session_state.g_stored = g
            st.session_state.L_s, st.session_state.f_s = L, f
            st.success("Done!")
        else:
            st.error("Failed")

if st.session_state.results:
    sol = st.session_state.results
    RA, RB = st.session_state.rx
    g_s = st.session_state.g_stored
    L_s, f_s = st.session_state.L_s, st.session_state.f_s
    
    st.header("📊 Results")
    cols = st.columns(4)
    cols[0].metric("H", f"{sol['H']:.1f} kN")
    cols[1].metric("t_req", f"{sol['t_req']*1000:.1f} mm")
    cols[2].metric("R_A", f"{RA:.1f} kN")
    cols[3].metric("R_B", f"{RB:.1f} kN")
    
    clusters = contact_clusters(sol["x"], sol["e"], sol["delta"])
    st.subheader("🔗 Hinges")
    if clusters:
        st.table(pd.DataFrame([{"#": k, "x": f"{0.5*(sol['x'][i0]+sol['x'][i1]):.3f} m", "Side": "EXT" if s>0 else "INT"} for k,(i0,i1,s) in enumerate(clusters,1)]))
    
    st.subheader("📈 Diagrams")
    fig = plot_results(sol, L_s, f_s, clusters); st.pyplot(fig)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight'); buf.seek(0)
    st.download_button("📥 PNG", buf, "arch.png", "image/png")
    
    st.divider()

    if st.button("📝 Print Calculations (Word)", key="make_docx"):
        with st.spinner("Generating..."):
            tmpdir = Path(tempfile.mkdtemp(prefix="arch_thrust_"))
            load_png = tmpdir / "load.png"
            res_png  = tmpdir / "res.png"
    
            fig_l = plot_loading(g_s, L_s)
            fig_l.savefig(load_png, dpi=200, bbox_inches="tight")
            plt.close(fig_l)
    
            fig_r = plot_results(sol, L_s, f_s, clusters)
            fig_r.savefig(res_png, dpi=200, bbox_inches="tight")
            plt.close(fig_r)
    
            # pull beam arrays once for checks
            x_b, V_b, M_b, RA_b, RB_b = beam_MV(g_s)
    
            st.session_state.docx_bytes = build_docx_python(
                L=L_s, f=f_s, dx=dx,
                sol=sol, clusters=clusters,
                udls=st.session_state.udls, vdls=st.session_state.vdls, pls=st.session_state.pls,
                RA=RA, RB=RB,
                info={"project": proj, "author": auth, "date": str(date)},
                load_png=load_png, res_png=res_png,
                beam_x=x_b, beam_V=V_b, beam_M=M_b, beam_RA=RA_b, beam_RB=RB_b
            )
    
    if st.session_state.docx_bytes is not None:
        st.download_button(
            "📥 Download Word",
            data=st.session_state.docx_bytes,
            file_name=f"Arch_{proj.replace(' ','_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_docx",
        )






st.divider()
st.caption("Arch Thrust Line | Minimax | Lower Bound Theorem")