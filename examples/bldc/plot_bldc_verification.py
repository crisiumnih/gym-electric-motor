"""
Generate the BLDC model verification figures for the thesis.

Runs a set of physics checks against the BrushlessDCMotor implementation and renders
publication-ready figures into ``figures/bldc/`` (tracked by git; ``docs/plots`` is ignored upstream):

    1.  fig_bemf_shape.png            trapezoidal back-EMF shape functions (sharp + smoothed)
    2.  fig_bemf_dq_alignment.png     dq alignment sweep + instantaneous (e_d, e_q) over a cycle
    3.  fig_openloop_spinup.png       open-loop spin-up under constant q-voltage
    4.  fig_closedloop_speed.png      closed-loop PI speed control (tracking, currents, torque)
    5.  fig_torque_ripple.png         steady-state torque ripple (6th electrical harmonic)
    6.  fig_power_balance.png         instantaneous power balance T*omega = sum(e_j i_j)
    7.  fig_interface_equivalence.png BLDC vs PMSM under the same normalized controller interface

Also prints a verification summary with the key numbers used in
``docs/bldc_verification.md``.

Usage:
    python examples/bldc/plot_bldc_verification.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gym_electric_motor as gem
from gym_electric_motor.physical_systems import BrushlessDCMotor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "bldc")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.3,
    }
)
K_E = 0.0955  # default back-EMF constant of the Antigravity KV100


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  saved {name}")


def decimate(y, n_target):
    """Downsample to ``n_target`` samples keeping the shape (for clean ripple curves)."""
    n = len(y)
    if n <= n_target:
        return y
    idx = np.linspace(0, n - 1, n_target).astype(int)
    return y[idx]


def envelope(ax, t, y, window, color, mean_label=None, alpha=0.18, lw=1.3):
    """
    Rolling min/max envelope (translucent band) + rolling mean (solid line).
    For oscillating traces that would otherwise render as solid color bands.
    """
    n = len(y)
    window = max(1, min(int(window), n))
    if window % 2 == 0:
        window -= 1
    half = window // 2
    padded = np.pad(y, (half, half), mode="edge")
    rolling = np.lib.stride_tricks.sliding_window_view(padded, window)
    mean = rolling.mean(axis=-1)
    ymin = rolling.min(axis=-1)
    ymax = rolling.max(axis=-1)
    ax.fill_between(t, ymin, ymax, color=color, alpha=alpha, lw=0)
    (line,) = ax.plot(t, mean, color=color, lw=lw)
    ax.set_xlim(t[0], t[-1])
    if mean_label:
        return line
    return line


# ---------------------------------------------------------------------------
# 1. Back-EMF shape functions
# ---------------------------------------------------------------------------
def fig_bemf_shape():
    motor = BrushlessDCMotor()
    motor_s = BrushlessDCMotor({"bemf_smoothing": 1.0})
    theta = np.linspace(0, 2 * np.pi, 2001, endpoint=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, constrained_layout=True)
    for ax, m, title in (
        (axes[0], motor, r"sharp trapezoid (default, $b_{\mathrm{smooth}}=0$)"),
        (axes[1], motor_s, r"smoothed ($b_{\mathrm{smooth}}=1$)"),
    ):
        for j, c in zip(range(3), ["tab:blue", "tab:red", "tab:green"]):
            ax.plot(np.rad2deg(theta), m.bemf_shape(theta)[j], c, lw=1.6)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.set_xlabel(r"electrical angle $\epsilon$ [deg]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"shape function $f_j(\epsilon)$")
    axes[0].annotate("flat\n$2\\pi/3$", xy=(60, 0.55), ha="center", fontsize=9, color="0.2")
    axes[0].annotate("", xy=(150, 0.9), xytext=(120, 0.9), arrowprops=dict(arrowstyle="<->", lw=1.2, color="0.2"))
    axes[0].annotate("linear $\\pi/3$", xy=(150, 0.7), ha="center", fontsize=9, color="0.2")
    axes[0].annotate("", xy=(30, -0.5), xytext=(0, -0.5), arrowprops=dict(arrowstyle="<->", lw=1.2, color="0.2"))
    axes[0].legend(
        ["$f_a$", "$f_b$", "$f_c$"],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Trapezoidal back-EMF shape functions (Pillay & Krishnan 1989)", fontsize=12)
    save(fig, "fig_bemf_shape.png")


# ---------------------------------------------------------------------------
# 2. dq alignment
# ---------------------------------------------------------------------------
def fig_bemf_dq_alignment():
    motor = BrushlessDCMotor()
    theta = np.linspace(0, 2 * np.pi, 20001, endpoint=False)

    def park_dq(f_abc):
        f_ab = motor.t_23(f_abc)
        return (
            np.mean(f_ab[0] * np.cos(theta) + f_ab[1] * np.sin(theta)),
            np.mean(-f_ab[0] * np.sin(theta) + f_ab[1] * np.cos(theta)),
        )

    phi = np.linspace(0, 2 * np.pi, 49, endpoint=False)
    d_vals, q_vals = np.array([park_dq(motor.bemf_shape(theta - p)) for p in phi]).T
    phi_opt = np.deg2rad(210.0)
    f = motor.bemf_shape(theta - phi_opt)
    f_ab = motor.t_23(f)
    e_d = f_ab[0] * np.cos(theta) + f_ab[1] * np.sin(theta)
    e_q = -f_ab[0] * np.sin(theta) + f_ab[1] * np.cos(theta)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(np.rad2deg(phi), d_vals, label=r"$\bar e_d$", color="tab:blue", lw=1.6)
    ax.plot(np.rad2deg(phi), q_vals, label=r"$\bar e_q$", color="tab:red", lw=1.6)
    ax.axvline(np.rad2deg(phi_opt), color="k", ls="--", lw=1.1, label=r"chosen $\phi=210°$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xlabel(r"alignment shift $\phi$ [deg]")
    ax.set_ylabel("cycle mean of BEMF shape")
    ax.set_title("Fundamental placement vs. angle origin")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.08), fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    th_deg = np.rad2deg(theta) % 360
    ax.plot(th_deg, e_d, label=r"$e_d(\epsilon)$", color="tab:blue", lw=1.3)
    ax.plot(th_deg, e_q, label=r"$e_q(\epsilon)$", color="tab:red", lw=1.3)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(q_vals[np.argmin(np.abs(phi - phi_opt))], color="tab:red", ls="--", lw=1.1, label=r"$\bar e_q$")
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xlabel(r"electrical angle $\epsilon$ [deg]")
    ax.set_ylabel("BEMF shape in dq")
    ax.set_title(r"Instantaneous $e_d$, $e_q$ at $\phi=210°$")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, -0.08), fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "fig_bemf_dq_alignment.png")
    return q_vals[np.argmin(np.abs(phi - phi_opt))]


# ---------------------------------------------------------------------------
# 3. Open-loop spin-up
# ---------------------------------------------------------------------------
def run_openloop(u_q, n_steps, j_rotor=1e-4):
    env = gem.make("Cont-SC-BLDC-v0", motor=dict(motor_parameter=dict(j_rotor=j_rotor)))
    env.reset()
    ps = env.unwrapped.physical_system
    idx = {n: i for i, n in enumerate(ps.state_names)}
    lim = ps.limits
    omega, i_q = np.zeros(n_steps), np.zeros(n_steps)
    for k in range(n_steps):
        obs, _, terminated, truncated, _ = env.step([0.0, u_q])
        s = np.asarray(obs[0])
        omega[k] = s[idx["omega"]] * lim[idx["omega"]]
        i_q[k] = s[idx["i_sq"]] * lim[idx["i_sq"]]
        if terminated or truncated:
            break
    return env, omega, i_q


def fig_openloop_spinup():
    n_steps = 6000
    u_q = 0.05
    _, omega, i_q = run_openloop(u_q, n_steps)
    t = np.arange(n_steps) * 1e-4
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True, constrained_layout=True)
    envelope(axes[0], t, omega, 60, "tab:blue")
    axes[0].axhline(u_q * 22.2 / K_E, color="k", ls="--", lw=1.2, label=r"$u_q/K_e$ (no-load ideal)")
    axes[0].set_ylabel(r"$\omega_{\mathrm{me}}$ [rad/s]")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Speed")
    envelope(axes[1], t, i_q, 60, "tab:red")
    axes[1].set_ylabel(r"$i_{sq}$ [A]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(alpha=0.3)
    axes[1].set_title(r"q-axis current")
    fig.suptitle(
        r"Open-loop spin-up under constant q-axis voltage "
        r"($u_q = 0.05,\ 1.1\,\mathrm{V}$; band = ripple envelope)",
        fontsize=12,
    )
    save(fig, "fig_openloop_spinup.png")
    return omega, i_q


# ---------------------------------------------------------------------------
# 4. Closed-loop PI speed control
# ---------------------------------------------------------------------------
def run_closedloop(omega_ref, u_q_max, kp, ki, n_steps, j_rotor=3e-3):
    # j_rotor = 3e-3 kg m^2: rotor + attached propeller-class load inertia (placeholder
    # rotor inertia 1e-4 is unrealistically fast for a controller demonstration)
    env = gem.make("Cont-SC-BLDC-v0", motor=dict(motor_parameter=dict(j_rotor=j_rotor)))
    env.reset()
    ps = env.unwrapped.physical_system
    idx = {n: i for i, n in enumerate(ps.state_names)}
    lim = ps.limits
    log = {n: np.zeros(n_steps) for n in ["omega", "i_a", "i_b", "i_c", "i_sd", "i_sq", "torque", "u_q"]}
    integral, terminated = 0.0, True
    for k in range(n_steps):
        if terminated:
            obs, _ = env.reset()
            terminated = False
        omega = float(np.asarray(obs[0])[idx["omega"]] * lim[idx["omega"]])
        error = omega_ref - omega
        integral = np.clip(integral + ki * error, -u_q_max, u_q_max)
        u_q = np.clip(kp * error + integral, -u_q_max, u_q_max)
        obs, _, terminated, truncated, _ = env.step([0.0, u_q])
        s = np.asarray(obs[0])
        log["omega"][k] = omega
        log["u_q"][k] = u_q
        for n in ["i_a", "i_b", "i_c", "i_sd", "i_sq", "torque"]:
            log[n][k] = s[idx[n]] * lim[idx[n]]
        if truncated:
            terminated = True
    return env, log


def fig_closedloop_speed():
    omega_ref, n_steps = 10.0, 30000
    _, log = run_closedloop(omega_ref, u_q_max=0.12, kp=1e-3, ki=5e-6, n_steps=n_steps)
    t = np.arange(n_steps) * 1e-4
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    envelope(axes[0, 0], t, log["omega"], 200, "tab:blue")
    axes[0, 0].axhline(omega_ref, color="k", ls="--", lw=1.2, label=r"$\omega^*$")
    axes[0, 0].set_ylabel(r"$\omega_{\mathrm{me}}$ [rad/s]")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_title("Speed tracking (PI on $u_q$)")
    axes[1, 0].plot(t, log["u_q"], color="0.3", lw=1.1)
    axes[1, 0].set_ylabel(r"$u_q$ [norm.]")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title("q-axis voltage command")
    # currents and torque: short steady-state window with decimation -> clean waveforms
    window = slice(-4000, -2000)
    t_w = t[window]
    idx = np.linspace(0, len(t_w) - 1, 800).astype(int)
    t_wd = t_w[idx]
    for j, c in zip(range(3), ["tab:blue", "tab:red", "tab:green"]):
        axes[0, 1].plot(t_wd, log[["i_a", "i_b", "i_c"][j]][window][idx], c, lw=1.1)
    axes[0, 1].legend(
        [r"$i_a$", r"$i_b$", r"$i_c$"],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.08),
        frameon=False,
        ncol=3,
        fontsize=9,
    )
    axes[0, 1].set_ylabel("phase currents [A]")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_title(r"$i_a,\ i_b,\ i_c$ (steady state, 0.2 s)")
    axes[1, 1].plot(t_wd, log["torque"][window][idx], color="tab:purple", lw=1.1)
    axes[1, 1].axhline(log["torque"][window].mean(), color="k", ls="--", lw=1.1, label="mean")
    axes[1, 1].set_ylabel(r"$T_e$ [N·m]")
    axes[1, 1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, -0.08), fontsize=9)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_title("Torque (steady state)")
    for ax in axes[:, 0]:
        ax.set_xlim(0, t[-1])
    for ax in axes[:, 1]:
        ax.set_xlim(t_w[0], t_w[-1])
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    fig.suptitle(
        rf"Closed-loop speed control of the BLDC (clamped PI, $\omega^* = {omega_ref:.0f}$ rad/s, "
        r"$J = 3\times10^{-3}$ kg·m$^2$)",
        fontsize=12,
    )
    save(fig, "fig_closedloop_speed.png")
    return log, t


# ---------------------------------------------------------------------------
# 5. Torque ripple: intrinsic (shape functions) and simulated
# ---------------------------------------------------------------------------
def fig_torque_ripple(log, t):
    motor = BrushlessDCMotor()
    # (a) intrinsic ripple of the machine: unit q-current vector, torque from Eq. (12)
    theta = np.linspace(0, 2 * np.pi, 2001, endpoint=False)
    i_ab = np.stack([-np.sin(theta), np.cos(theta)])  # unit q-current in alpha-beta
    i_abc = motor.t_32(i_ab)  # rotating unit q-current in abc
    T_norm = np.sum(motor.bemf_shape(theta) * i_abc, axis=0)  # T / (k_e * i_q)
    ripple_pct = np.ptp(T_norm) / T_norm.mean() * 100.0

    # (b) simulated steady-state torque (unsaturated operating point)
    window = slice(-4000, None)
    T = log["torque"][window]
    t_w = t[window]
    idx = np.linspace(0, len(t_w) - 1, 900).astype(int)
    f_e = 21.0 * log["omega"][window].mean() / (2 * np.pi)
    ripple_sim_pct = np.ptp(T) / max(abs(T.mean()), 1e-12) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].plot(np.rad2deg(theta) % 360, T_norm, color="tab:purple", lw=1.5)
    axes[0].axhline(T_norm.mean(), color="k", ls="--", lw=1.2, label="mean")
    axes[0].set_xticks([0, 60, 120, 180, 240, 300, 360])
    axes[0].set_xlabel(r"electrical angle $\epsilon$ [deg]")
    axes[0].set_ylabel(r"$T_e\,/\,(k_e\, i_q)$")
    axes[0].set_title(f"Intrinsic ripple (dq current drive): {ripple_pct:.0f}% of mean")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[1].plot(t_w[idx], decimate(T, 900), color="tab:purple", lw=1.1)
    axes[1].axhline(T.mean(), color="k", ls="--", lw=1.2, label=f"mean {T.mean():.3f} N·m")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel(r"$T_e$ [N·m]")
    axes[1].set_title(rf"Simulated steady state ($6 f_e$ = {6 * f_e:.0f} Hz): p-p {np.ptp(T):.2f} N·m")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    axes[1].grid(alpha=0.3)
    save(fig, "fig_torque_ripple.png")
    return ripple_pct, ripple_sim_pct


# ---------------------------------------------------------------------------
# 6. Instantaneous power balance
# ---------------------------------------------------------------------------
def fig_power_balance():
    motor = BrushlessDCMotor()
    rng = np.random.default_rng(20260811)
    states = rng.uniform(-30, 30, (20000, 4))
    omegas = rng.uniform(-300, 300, 20000)
    p_mech = np.array([motor.torque(s) * w for s, w in zip(states, omegas)])
    p_el = np.array([np.sum(motor.back_emf(s, w) * s[:3]) for s, w in zip(states, omegas)])
    max_rel_err = np.max(np.abs(p_mech - p_el) / (np.abs(p_el) + 1e-12))
    fig, ax = plt.subplots(figsize=(6.5, 6.2), constrained_layout=True)
    hb = ax.hexbin(p_el, p_mech, gridsize=60, cmap="viridis", mincnt=1)
    lim = np.max(np.abs(np.concatenate([p_el, p_mech]))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\sum_j e_j\, i_j$ [W]")
    ax.set_ylabel(r"$T_e\, \omega_{\mathrm{me}}$ [W]")
    ax.set_title(f"Power balance over 20,000 random operating points\nmax rel. error {max_rel_err:.1e}")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.colorbar(hb, ax=ax, label="samples per bin")
    save(fig, "fig_power_balance.png")
    return max_rel_err


# ---------------------------------------------------------------------------
# 7. Interface equivalence: same controller on BLDC and PMSM
# ---------------------------------------------------------------------------
def fig_interface_equivalence():
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True, constrained_layout=True)
    for env_id, omega_ref, color in (
        ("Cont-SC-BLDC-v0", 10.0, "tab:blue"),
        ("Cont-SC-PMSM-v0", 100.0, "tab:red"),
    ):
        env_kwargs = dict(control_space="dq")
        if "BLDC" in env_id:
            u_max, kp, ki = 0.12, 1e-3, 5e-6
            env_kwargs["motor"] = dict(motor_parameter=dict(j_rotor=3e-3))
        else:
            u_max, kp, ki = 0.3, 2e-3, 2e-3
        env = gem.make(env_id, **env_kwargs)
        env.reset()
        ps = env.unwrapped.physical_system
        idx = {n: i for i, n in enumerate(ps.state_names)}
        lim = ps.limits
        n = 30000
        omega_norm, integral, terminated = np.zeros(n), 0.0, True
        for k in range(n):
            if terminated:
                obs, _ = env.reset()
                terminated = False
            omega = float(np.asarray(obs[0])[idx["omega"]])
            error = omega_ref / lim[idx["omega"]] - omega
            integral = np.clip(integral + ki * error, -u_max, u_max)
            u_q = np.clip(kp * error + integral, -u_max, u_max)
            obs, _, terminated, truncated, _ = env.step([0.0, u_q])
            omega_norm[k] = omega
            if truncated:
                terminated = True
        t = np.arange(n) * 1e-4
        ax = axes[0 if "BLDC" in env_id else 1]
        envelope(ax, t, omega_norm * lim[idx["omega"]], 200, color, mean_label=True)
        ax.axhline(omega_ref, color=color, ls="--", lw=1.1, label=fr"$\omega^*$ = {omega_ref:.0f} rad/s")
        ax.set_ylabel(r"$\omega_{\mathrm{me}}$ [rad/s]")
        ax.set_title("BLDC" if "BLDC" in env_id else "PMSM")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("time [s]")
    fig.suptitle("Same clamped-PI controller structure on both environments")
    save(fig, "fig_interface_equivalence.png")


# ---------------------------------------------------------------------------
def main():
    print(f"Saving figures to {OUT_DIR}")
    fig_bemf_shape()
    fig_bemf_dq_alignment()
    fig_openloop_spinup()
    log, t = fig_closedloop_speed()
    ripple_pct, ripple_sim_pct = fig_torque_ripple(log, t)
    max_rel_err = fig_power_balance()
    fig_interface_equivalence()

    # verification summary (numbers quoted in docs/bldc_verification.md)
    omega_ss = log["omega"][-5000:]
    T_ss = log["torque"][-5000:]
    print("\n=== verification summary ===")
    print(f"steady-state speed        : {omega_ss.mean():.2f} rad/s (ref 10)")
    print(f"steady-state error        : {10.0 - omega_ss.mean():.2f} rad/s")
    print(f"speed ripple (p-p)        : {np.ptp(omega_ss):.2f} rad/s")
    print(f"torque ripple (p-p)       : {np.ptp(T_ss):.4f} N·m on mean {T_ss.mean():.4f} N·m")
    print(f"intrinsic torque ripple   : {ripple_pct:.1f}% of mean (shape functions)")
    print(f"simulated torque ripple   : {ripple_sim_pct:.1f}% of mean")
    print(f"power balance max rel err : {max_rel_err:.2e}")
    print(
        "run the formal checks with: python -m pytest "
        "tests/test_physical_systems/test_electric_motors.py -k Brushless "
        "tests/test_environments/test_environments.py -q"
    )


if __name__ == "__main__":
    main()
