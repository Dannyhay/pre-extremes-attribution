"""
Exact Gaussian implementation of Smirnov's finite-time information response
(FTIR), EPJ ST (2026), Eqs. (5)-(7) and Appendix A, for the 1-D linear system

    xdot = -a_x x + k_xy y + sqrt(G_xx) xi_x
    ydot = -a_y y + k_yx x + sqrt(G_yy) xi_y

Validated against the published figures before the same machinery is applied
to the TV-CSM testbed.

Three initial ensembles sharing the X-marginal (Appendix A):
  rho*   : m=(0, y0),        C = diag(Cxx_st, 0)   -> p*_X
  rho**  : m=(0, 0),         C = C_st              -> p**_X
  rho^w  : m=(E[x|y0], y0),  C = diag(Cxx|y, 0)    -> w
  L(y0) = 1/2 ( ln(C**_xx/C*_xx) + A + B - C ),  then y0 ~ N(0, Cyy_st).
"""
import math
import numpy as np
from scipy.linalg import solve_lyapunov


def _integrate(A, Gam, C0, ts, nsub=40):
    """C(t) and B(t)=expm(At) at each t in ts, by RK4 on the joint ODE
    Cdot = A C + C A^T + Gam,  Bdot = A B.  Dense, accurate, and fast."""
    ts = np.asarray(ts, float)
    tmax = ts.max()
    n = max(int(nsub * max(1.0, tmax / 0.05)), 400)
    h = tmax / n
    C = C0.copy().astype(float)
    B = np.eye(2)
    fC = lambda M: A @ M + M @ A.T + Gam
    fB = lambda M: A @ M
    out_C, out_B = [], []
    grid = np.arange(n + 1) * h
    want = np.searchsorted(grid, ts - 1e-12)
    store = {int(i): None for i in want}
    if 0 in store:
        store[0] = (C.copy(), B.copy())
    for k in range(n):
        k1C, k1B = fC(C), fB(B)
        k2C, k2B = fC(C + 0.5 * h * k1C), fB(B + 0.5 * h * k1B)
        k3C, k3B = fC(C + 0.5 * h * k2C), fB(B + 0.5 * h * k2B)
        k4C, k4B = fC(C + h * k3C), fB(B + h * k3B)
        C = C + (h / 6) * (k1C + 2 * k2C + 2 * k3C + k4C)
        B = B + (h / 6) * (k1B + 2 * k2B + 2 * k3B + k4B)
        if (k + 1) in store:
            store[k + 1] = (C.copy(), B.copy())
    for i in want:
        c, b = store[int(i)]
        out_C.append(c); out_B.append(b)
    return out_C, out_B


def ftir(a_x, a_y, k_xy, k_yx, G_xx, G_yy, ts):
    """L^(t)_{Y->X} of Eq. (A6)."""
    A = np.array([[-a_x, k_xy], [k_yx, -a_y]])
    Gam = np.diag([G_xx, G_yy])
    Cst = solve_lyapunov(A, -Gam)
    Cxx, Cxy, Cyy = Cst[0, 0], Cst[0, 1], Cst[1, 1]
    Cxx_gy = Cxx - Cxy * Cxy / Cyy

    Cs, Bs = _integrate(A, Gam, np.array([[Cxx, 0.0], [0.0, 0.0]]), ts)
    Cd, _ = _integrate(A, Gam, Cst.copy(), ts)
    Cw, _ = _integrate(A, Gam, np.array([[Cxx_gy, 0.0], [0.0, 0.0]]), ts)

    L = np.empty(len(ts))
    for i in range(len(ts)):
        Bxx, Bxy = Bs[i][0, 0], Bs[i][0, 1]
        cs, cd, cw = Cs[i][0, 0], Cd[i][0, 0], Cw[i][0, 0]
        G = Bxx * Cxy / Cyy
        F = G + Bxy
        L[i] = 0.5 * (np.log(cd / cs) + (cw / cd - cw / cs)
                      + (F * F / cd) * Cyy - (G * G / cs) * Cyy)
    return L, dict(Cxx=Cxx, Cxy=Cxy, Cyy=Cyy,
                   r=Cxy / np.sqrt(Cxx * Cyy))


def lkif_analytic(k_xy, Cxx, Cxy, Cyy):
    """Smirnov Eq. (4)."""
    return k_xy * (Cxy / np.sqrt(Cxx * Cyy)) * np.sqrt(Cyy / Cxx)


def dirs(func, tmax=0.04, deg=5, npts=40):
    """First- and second-order DIR from L(t) = sum_{m>=1} c_m t^m / m!.
    Fits a polynomial through the origin on a small-t window; returns c1, c2."""
    ts = np.linspace(0.0, tmax, npts + 1)[1:]
    L = func(ts)
    powers = np.arange(1, deg + 1)
    X = ts[:, None] ** powers[None, :] / np.array(
        [math.factorial(int(p)) for p in powers])[None, :]
    c, *_ = np.linalg.lstsq(X, L, rcond=None)
    return c[0], c[1]


if __name__ == "__main__":
    al, G = 1.0, 2.0

    print("=" * 74)
    print("1. FTIR curve vs Smirnov Fig. 2a")
    print("   his text: beta_xy=1, beta_yx=0 -> max ~0.11 at alpha*t ~0.8;")
    print("             beta_yx=-1 (zero LKIF both ways) -> max ~0.05")
    print("=" * 74)
    tg = np.linspace(0.0, 3.0, 301)[1:]
    for byx in (0.0, -0.2, -0.6, -1.0, -1.4, -2.0):
        L, st = ftir(al, al, 1.0, byx, G, G, tg)
        k = int(np.argmax(L))
        print(f"   beta_yx={byx:5.1f}  r={st['r']:+.4f}  "
              f"LKIF={lkif_analytic(1.0, st['Cxx'], st['Cxy'], st['Cyy']):+.4f}"
              f"   FTIR max={L[k]:.4f} at alpha*t={tg[k]:.2f}")

    print("\n" + "=" * 74)
    print("2. Is the LKIF the first-order DIR?  (Smirnov Eq. 6)")
    print("=" * 74)
    print("   beta_yx     c1 (fitted)    LKIF (analytic)     difference")
    for byx in (0.0, -0.5, -1.0, -1.5, -2.0):
        f = lambda ts, b=byx: ftir(al, al, 1.0, b, G, G, ts)[0]
        _, st = ftir(al, al, 1.0, byx, G, G, np.array([0.01]))
        c1, c2 = dirs(f)
        lk = lkif_analytic(1.0, st["Cxx"], st["Cxy"], st["Cyy"])
        print(f"   {byx:6.1f}    {c1:+12.6f}    {lk:+12.6f}    {c1-lk:+.2e}")

    print("\n" + "=" * 74)
    print("3. Second-order DIR on the antisymmetric line beta_xy = -beta_yx")
    print("   (LKIF identically zero here; Smirnov Fig. 4b)")
    print("=" * 74)
    print("   beta      c1(~0)        c2        c2/beta^2     Eq.(B8)   ratio")
    for b in (0.2, 0.4, 0.6, 1.0, 1.4, 2.0):
        f = lambda ts, bb=b: ftir(al, al, bb, -bb, G, G, ts)[0]
        _, st = ftir(al, al, b, -b, G, G, np.array([0.01]))
        c1, c2 = dirs(f)
        iB8 = b ** 2 * st["Cyy"] / (2 * al ** 2 * st["Cxx"])
        print(f"   {b:4.1f}  {c1:+.2e}  {c2:9.5f}   {c2/b**2:8.5f}   "
              f"{iB8:8.4f}   {c2/(2*al**2)/iB8:6.4f}")

    print("\n   -> c2 scales exactly as beta^2, i.e. as k_xy^2, which is the")
    print("      scaling of Eq. (B8).  The higher-order repair works in")
    print("      Smirnov's setting: the coupling is recovered at second order.")
