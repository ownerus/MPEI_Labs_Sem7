"""Reproducible calculations for the TAU2 typical assignment, variant 103.

Dependencies: numpy, scipy, sympy.
Formula sources: Tipovoy_po_ISAU, printed pp. 11-15, 19-20, 31-33;
TAU lectures, pp. 47-54, 58-62, 76-81, 87-88.

The script prints results. Optional --export-dir writes CSV/JSON for checking.
Use --regime to choose the reference for --simintech-output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import sympy as sp
from scipy.optimize import brentq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


T = sp.Rational(1, 100)
T_FLOAT = float(T)
K_NOM = 10.0
p, q, z, v, K = sp.symbols("p q z v K", real=True)
t = sp.symbols("t", nonnegative=True)


def as_float(value: sp.Expr) -> float:
    return float(sp.N(value, 18))


def coefficients_desc(expr: sp.Expr, variable: sp.Symbol) -> np.ndarray:
    return np.array([as_float(c) for c in sp.Poly(sp.expand(expr), variable).all_coeffs()], dtype=float)


def coefficients_asc(expr: sp.Expr, variable: sp.Symbol) -> np.ndarray:
    poly = sp.Poly(sp.expand(expr), variable)
    return np.array([as_float(poly.nth(i)) for i in range(poly.degree() + 1)], dtype=float)


def simulate(input_values: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    output = np.zeros_like(input_values, dtype=float)
    for n in range(len(input_values)):
        current = 0.0
        for j, bj in enumerate(b):
            if n >= j:
                current += bj * input_values[n - j]
        for j in range(1, len(a)):
            if n >= j:
                current -= a[j] * output[n - j]
        output[n] = current / a[0]
    return output


def simintech_output(path: Path) -> list[tuple[float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.replace(",", ".").split()
        if len(fields) >= 2:
            try:
                y_column = 1 if len(fields) == 2 else 2
                rows.append((float(fields[0]), float(fields[y_column])))
            except ValueError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simintech-output", type=Path,
                        help="time/y or time/e/y[/xp[/x]] text file exported by SimInTech")
    parser.add_argument("--regime", choices=("nominal", "critical", "ramp"), default="nominal")
    parser.add_argument("--export-dir", type=Path, help="optional directory for reference CSV and results JSON")
    args = parser.parse_args()

    # Variant 103, Fig. 5б: W1=K/p is in the direct branch; W2 and W3 form
    # the feedback branch. The open-loop contour product is W1*W2*W3.
    W1 = K / p
    W2 = 2 / (1 + p)
    W3 = (1 + 5 * p) / (1 + 2 * p)
    Wn = sp.factor(W1 * W2 * W3)
    Wf = (1 - sp.exp(-p * T)) / p  # Methodical guide, p. 13; tau=0, IE 6a.
    Wpn = sp.factor(Wf * Wn)

    # D-transform of the sampled transition function, methodical p. 14 and
    # lecture PDF pp. 16, 28. K is factored out while forming the unit-gain model.
    H = sp.factor(Wpn.subs(K, 1) / (1 - sp.exp(-p * T)))
    h = 4 + 2 * t + 8 * sp.exp(-t) - 12 * sp.exp(-t / 2)
    assert sp.simplify(sp.laplace_transform(h, t, p, noconds=True) - H) == 0
    Hq = 4 / (1 - q) + 2 * T * q / (1 - q) ** 2 + 8 / (1 - sp.exp(-T) * q) - 12 / (1 - sp.exp(-T / 2) * q)
    Wunit_q = sp.cancel((1 - q) * Hq)
    Wunit_z = sp.cancel(Wunit_q.subs(q, 1 / z))
    Bz_raw, Az_raw = sp.fraction(Wunit_z)
    norm = sp.Poly(Az_raw, z).LC()
    Bz = sp.factor(Bz_raw / norm)
    Az = sp.factor(Az_raw / norm)
    assert sp.simplify(Bz / Az - Wunit_z) == 0

    az = coefficients_desc(Az, z)
    bz = coefficients_desc(Bz, z)

    def w_unit(theta: float | np.ndarray) -> complex | np.ndarray:
        delay = np.exp(-1j * np.asarray(theta))
        one_minus_delay = -np.expm1(-1j * np.asarray(theta))
        decay_a = float(sp.exp(-T))
        decay_b = float(sp.exp(-T / 2))
        return 4 + 2 * T_FLOAT * delay / one_minus_delay + 8 * one_minus_delay / (1 - decay_a * delay) - 12 * one_minus_delay / (1 - decay_b * delay)

    # Nyquist limit: locate the negative-real-axis crossing for unit gain.
    theta = np.linspace(1e-7, np.pi - 1e-8, 30001)
    imag = np.imag(w_unit(theta))
    crossings = np.where(imag[:-1] * imag[1:] < 0)[0]
    crossing_values: list[tuple[float, complex]] = []
    for i in crossings:
        th = brentq(lambda x: float(np.imag(w_unit(x))), theta[i], theta[i + 1], xtol=1e-14)
        val = complex(w_unit(th))
        if val.real < 0:
            crossing_values.append((th, val))
    if not crossing_values:
        raise RuntimeError("No negative-real-axis Nyquist crossing was found")
    theta_cr, w_cr = crossing_values[0]
    omega_cr = theta_cr / T_FLOAT
    K_nyquist = -1.0 / w_cr.real

    # Hurwitz test after z=(1+v)/(1-v), lecture pp. 58-62.
    characteristic_z = sp.expand(Az + K * Bz)
    characteristic_v = sp.Poly(
        sp.cancel((1 - v) ** 3 * characteristic_z.subs(z, (1 + v) / (1 - v))), v
    )
    c3, c2, c1, c0 = characteristic_v.all_coeffs()
    delta2 = sp.factor(c2 * c1 - c3 * c0)
    hurwitz_conditions = {"c3": c3, "c2": c2, "c1": c1, "c0": c0, "Delta2": delta2}
    hurwitz_boundaries = {}
    for name, expression in hurwitz_conditions.items():
        condition_roots = sp.nroots(sp.Poly(expression, K), n=30, maxsteps=200)
        hurwitz_boundaries[name] = [
            float(sp.re(root)) for root in condition_roots
            if abs(float(sp.im(root))) < 1e-10 and float(sp.re(root)) > 0
        ]
    hurwitz_candidates = [gain for gains in hurwitz_boundaries.values() for gain in gains]
    K_hurwitz = min(hurwitz_candidates)
    assert all(float(expression.subs(K, K_hurwitz / 2)) > 0 for expression in hurwitz_conditions.values())

    # Neutral open loop: shift its z=1 pole inside the unit circle, then let
    # beta tend to zero (lecture pp. 79-81). The finite crossing above is
    # supplemented by the arc at infinity.
    beta = 1e-6
    positive_angles = np.unique(np.r_[0, np.geomspace(1e-10, 0.01, 2000), np.linspace(0.01, np.pi, 30000)])
    full_angles = np.r_[-positive_angles[:0:-1], positive_angles]
    circle = np.exp(1j * full_angles)
    w_shifted_unit = np.polyval(bz, circle) / (
        (circle - 1 + beta) * (circle - float(sp.exp(-T))) * (circle - float(sp.exp(-T / 2)))
    )
    winding_numbers = {}
    for gain in [K_NOM, 1.01 * K_nyquist]:
        phases = np.unwrap(np.angle(1 + gain * w_shifted_unit))
        winding_numbers[gain] = int(round((phases[-1] - phases[0]) / (2 * np.pi)))
    assert winding_numbers[K_NOM] == 0

    # Root criterion: the boundary occurs when max_i |z_i(K)| reaches one.
    root_coefficients = [sp.lambdify(K, c, "numpy") for c in sp.Poly(characteristic_z, z).all_coeffs()]

    def characteristic_roots(gain: float) -> np.ndarray:
        return np.roots([float(fn(gain)) for fn in root_coefficients])

    def root_margin(gain: float) -> float:
        return float(np.max(np.abs(characteristic_roots(gain))) - 1.0)

    low, high = 1e-4, 1.0
    while root_margin(high) <= 0:
        high *= 2
    K_roots = brentq(root_margin, low, high, xtol=1e-12, rtol=1e-13)
    k_values = np.array([K_nyquist, K_hurwitz, K_roots])
    k_spread_percent = (k_values.max() - k_values.min()) / k_values.mean() * 100

    # Closed-loop recurrence Wz*(q)=K N(q)/(D(q)+K N(q)); lecture pp. 52-54.
    Nq, Dq = sp.fraction(Wunit_q)
    closed_num = sp.expand(K_NOM * Nq)
    closed_den = sp.expand(Dq + K_NOM * Nq)
    normalizer = sp.Poly(closed_den, q).nth(0)
    a = coefficients_asc(closed_den / normalizer, q)
    b = coefficients_asc(closed_num / normalizer, q)
    roots_nominal = characteristic_roots(K_NOM)

    n = np.arange(6001)
    time = n * T_FLOAT
    step = simulate(np.ones_like(time), a, b)
    ramp = simulate(time.copy(), a, b)
    ramp_error_series = time - ramp
    critical_a = az + K_hurwitz * np.r_[0.0, bz]
    critical_b = K_hurwitz * np.r_[0.0, bz]
    critical_step = simulate(np.ones_like(time), critical_a, critical_b)

    # Final-value error formulae for step/ramp inputs, lecture pp. 87-88.
    error_transfer = sp.cancel(Dq / (Dq + K_NOM * Nq))
    static_error = sp.limit(error_transfer, q, 1)
    kinetic_error = sp.limit(T * q * Dq / ((1 - q) * (Dq + K_NOM * Nq)), q, 1)

    # Exact and truncated alias AФЧХ, methodical p. 15. Sidorova permits either
    # example method; the 2/4/7-term comparison records the chosen truncations.
    omega0 = 2 * np.pi / T_FLOAT
    freq = np.linspace(0.02, np.pi / T_FLOAT, 5000)
    exact = K_NOM * w_unit(freq * T_FLOAT)

    def Wpn_numeric(s: np.ndarray, gain: float) -> np.ndarray:
        return gain * (-np.expm1(-s * T_FLOAT)) * 2 * (1 + 5 * s) / (s**2 * (1 + s) * (1 + 2 * s))

    def alias_sum(aliases: range, omega: np.ndarray) -> np.ndarray:
        total = np.zeros_like(omega, dtype=np.complex128)
        for ell in aliases:
            total += Wpn_numeric(1j * (omega - omega0 * ell), K_NOM)
        return total / T_FLOAT

    truncations = {"2 terms, l=0,1": range(0, 2), "4 terms, l=-1..2": range(-1, 3), "7 terms, l=-3..3": range(-3, 4)}
    max_errors = {name: float(np.max(np.abs(alias_sum(indices, freq) - exact))) for name, indices in truncations.items()}
    peak_error_frequencies = {
        name: float(freq[np.argmax(np.abs(alias_sum(indices, freq) - exact))])
        for name, indices in truncations.items()
    }
    errors_at_crossing = {
        name: float(abs(alias_sum(indices, np.array([omega_cr]))[0] - K_NOM * w_unit(theta_cr)))
        for name, indices in truncations.items()
    }
    control_frequencies = np.array([0.02, 1.0, 5.0, omega_cr, 30.0, 100.0, np.pi / T_FLOAT])
    control_exact = K_NOM * w_unit(control_frequencies * T_FLOAT)
    control_approx = alias_sum(range(-1, 3), control_frequencies)
    frequency_table = [
        {"omega": float(w), "exact_re": float(e.real), "exact_im": float(e.imag),
         "approx_re": float(ap.real), "approx_im": float(ap.imag),
         "abs_difference": float(abs(e - ap))}
        for w, e, ap in zip(control_frequencies, control_exact, control_approx)
    ]

    # Print the calculation in the same notation as the report.
    print(f"Tи = {T_FLOAT:.5f} с; Kном = {K_NOM:g}")
    print(f"Wн(p) = {sp.sstr(Wn.subs(K, K_NOM))}")
    print(f"Wф(p) = {sp.sstr(Wf)}")
    print(f"Wпн(p) = {sp.sstr(Wpn.subs(K, K_NOM))}")
    print(f"H(p) = {sp.apart(H, p)}")
    print(f"h(t) = {h}")
    print(f"A(z), descending = {az}")
    print(f"B(z), descending = {bz}")
    print(f"ωкр = {omega_cr:.10f} рад/с; Wр,1* = {w_cr.real:.12g} {w_cr.imag:+.3g}j")
    print(f"Kпр: Nyquist={K_nyquist:.12f}; Hurwitz={K_hurwitz:.12f}; roots={K_roots:.12f}")
    print(f"Positive Hurwitz boundaries: {hurwitz_boundaries}")
    print(f"Nyquist regularization beta={beta:g}; windings of 1+W_beta = {winding_numbers}")
    print(f"Spread of Kпр estimates = {k_spread_percent:.4g}%")
    print(f"Closed-loop roots at K=10 = {roots_nominal}")
    print(f"Recurrence a(q), ascending = {a}")
    print(f"Recurrence b(q), ascending = {b}")
    print(f"Step samples: y(0)={step[0]:.10g}; y(0.1)={step[10]:.10g}; y(1)={step[100]:.10g}; y(10)={step[1000]:.10g}; y(60)={step[-1]:.10g}")
    print(f"Static error = {sp.N(static_error, 10)}; unit-ramp kinetic error = {sp.N(kinetic_error, 10)}")
    print(f"Last 100 unit-ramp error mean = {np.mean(ramp_error_series[-100:]):.12g}")
    print(f"AФЧХ max errors: {max_errors}")
    print(f"AФЧХ frequencies of maximum errors: {peak_error_frequencies}")
    print(f"AФЧХ errors at ωкр: {errors_at_crossing}")
    print(f"Critical gain = {K_hurwitz:.12g}; roots = {characteristic_roots(K_hurwitz)}")
    print(f"Critical output on 40..60 s: min={critical_step[4000:].min():.10g}; max={critical_step[4000:].max():.10g}")

    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.export_dir / "reference_regimes.csv",
                   np.column_stack((time, step, critical_step, ramp, ramp_error_series)),
                   delimiter=",", header="t,step_nominal,step_critical,ramp_y,ramp_error", comments="", fmt="%.15g")
        results = {
            "K_nyquist": K_nyquist, "K_hurwitz": K_hurwitz, "K_roots": K_roots,
            "omega_critical": omega_cr, "static_error": float(static_error),
            "kinetic_error": float(kinetic_error), "frequency_table": frequency_table,
            "critical_a": critical_a.tolist(), "critical_b": critical_b.tolist(),
            "critical_tail_min": float(critical_step[4000:].min()),
            "critical_tail_max": float(critical_step[4000:].max()),
            "ramp_error_tail_mean": float(np.mean(ramp_error_series[-100:])),
        }
        (args.export_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.simintech_output:
        measured = simintech_output(args.simintech_output)
        reference = {"nominal": step, "critical": critical_step, "ramp": ramp}[args.regime]
        matched = []
        for time_s, output in measured:
            sample = round(time_s / T_FLOAT)
            if 0 <= sample < len(step) and abs(time_s - sample * T_FLOAT) <= 2e-5:
                matched.append((sample, time_s, output, reference[sample]))
        if not matched:
            raise RuntimeError("No SimInTech samples matched the calculation grid")
        maximum = max(abs(row[2] - row[3]) for row in matched)
        print(f"SimInTech rows matched = {len(matched)}; maximum |Δy| = {maximum:.12g}")

    assert abs(K_nyquist - 52.1204390183) < 1e-7
    assert abs(K_hurwitz - 52.1204390183) < 1e-7
    assert abs(K_roots - 52.1204390110) < 1e-6
    assert np.max(np.abs(roots_nominal)) < 1.0
    assert abs(float(static_error)) < 1e-12
    assert abs(float(kinetic_error) - 0.05) < 1e-12


if __name__ == "__main__":
    main()
