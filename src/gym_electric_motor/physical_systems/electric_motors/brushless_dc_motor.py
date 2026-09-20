import numpy as np

from .three_phase_motor import ThreePhaseMotor


class BrushlessDCMotor(ThreePhaseMotor):
    """
    =====================  ==========  ============= ===========================================
    Motor Parameter        Unit        Default Value Description
    =====================  ==========  ============= ===========================================
    p                      1           21            Pole pair number
    r_s                    Ohm         85e-3         Stator resistance
    l_s                    H           50e-6         Equivalent phase inductance (L-M)
    k_e                    V*s/rad     0.0955        Back-EMF constant (mechanical)
    j_rotor                kg/m^2      1e-4          Moment of inertia of the rotor
    bemf_smoothing         -           0.0           Numerical smoothing of the BEMF corners (0: sharp, 1: smooth)
    =====================  ==========  ============= ===========================================

    =============== ====== =============================================
    Motor Currents  Unit   Description
    =============== ====== =============================================
    i_a             A      Current through phase a
    i_b             A      Current through phase b
    i_c             A      Current through phase c
    i_sd            A      Direct axis current (observed, derived from i_abc)
    i_sq            A      Quadrature axis current (observed, derived from i_abc)
    =============== ====== =============================================
    =============== ====== =============================================
    Motor Voltages  Unit   Description
    =============== ====== =============================================
    u_a             V      Phase voltage for phase a
    u_b             V      Phase voltage for phase b
    u_c             V      Phase voltage for phase c
    u_sd            V      Direct axis voltage (observed, derived from u_abc)
    u_sq            V      Quadrature axis voltage (observed, derived from u_abc)
    =============== ====== =============================================

    ======== ===========================================================
    Limits / Nominal Value Dictionary Entries:
    -------- -----------------------------------------------------------
    Entry    Description
    ======== ===========================================================
    i        General current limit / nominal value
    i_a      Current in phase a
    i_b      Current in phase b
    i_c      Current in phase c
    i_sd     Current in direct axis
    i_sq     Current in quadrature axis
    omega    Mechanical angular Velocity
    torque   Motor generated torque
    epsilon  Electrical rotational angle
    u_a      Phase voltage in phase a
    u_b      Phase voltage in phase b
    u_c      Phase voltage in phase c
    u_sd     Phase voltage in direct axis
    u_sq     Phase voltage in quadrature axis
    ======== ===========================================================

    Note:
        The brushless DC motor (BLDC) is modeled in the phase-variable (abc) frame with a
        piecewise-linear trapezoidal back-EMF, following the model of Pillay & Krishnan (1989),
        "Modeling, Simulation, and Analysis of Permanent-Magnet Motor Drives, Part II: The
        Brushless DC Motor Drive", IEEE Trans. Ind. Appl. 25(2). Because the back-EMF is
        trapezoidal (not sinusoidal), the classical dq transformation is not used for the
        simulation; the dq quantities reported in the state vector are derived from the abc
        quantities via the Clarke/Park transform and are intended for observers and the
        RL state space only.

        The back-EMF of phase j is :math:`e_j = k_e \\cdot \\omega_{me} \\cdot f_j(\\epsilon)`
        with the shape function :math:`f_j` a unit trapezoid (flat :math:`\\pm 1` over 120 deg
        electrical, linear transitions of 60 deg electrical). The torque follows from the
        instantaneous power balance :math:`T = (e_a i_a + e_b i_b + e_c i_c)/\\omega_{me}`,
        i.e. :math:`T = k_e \\sum_j f_j(\\epsilon) i_j`, which is well defined at standstill.

        Angular alignment: the phase-variable trapezoid of Pillay & Krishnan (1989) is
        shifted by 150 deg electrical such that its fundamental component lies purely on
        the q-axis of the dq-transformation, i.e. :math:`(e_d, e_q) = (0, E)` with
        :math:`E > 0`, matching the PMSM convention used in this package
        (:math:`e_q = \\omega \\cdot p \\cdot \\psi_p`). The electrical angle origin is a
        free modeling convention; this choice makes dq-space observers and RL agents
        behave identically on the BLDC and PMSM environments. The trapezoidal (non-sinusoidal)
        waveform inherently produces a 6th-harmonic torque ripple that a PMSM does not have.

        The stator windings are assumed to be in star connection with symmetric rotor
        reluctance, such that :math:`i_c = -i_a - i_b` and all phase inductances are equal.
        The equivalent phase inductance :math:`l_s = L - M` is what a locked-rotor
        measurement between two phase terminals yields divided by two. The star connection
        is enforced in the ODE through the floating neutral potential (see
        ``electrical_ode``), which keeps the common-mode current identically zero; this
        matters here because the trapezoidal back-EMF has a nonzero common-mode component
        (unlike sinusoidal back-EMF, :math:`f_a + f_b + f_c \\neq 0` in general).

        Default parameters correspond to the T-Motor Antigravity KV100 (36N42P) used in the
        project; :math:`l_s` and :math:`j_rotor` are placeholders to be replaced by measured
        values and :math:`k_e` is derived from the motor's KV rating
        (:math:`k_e = 1/KV = 0.0955~\\text{V·s/rad}`) and to be refit from bench data.

        The parameter ``bemf_smoothing`` is a numerical, not physical, parameter. For
        ``0.0`` the back-EMF is the sharp piecewise-linear trapezoid of the original model;
        values in ``(0, 1]`` blend the transitions towards a smoothstep curve (at ``1.0`` the
        corners have zero slope), which may help stiff ODE solvers at high speed. This is a
        documented deviation from the reference model.
    """

    #### Parameters of the T-Motor Antigravity KV100 (36N42P), see docs/bldc_model.md
    _default_motor_parameter = {
        "p": 21,
        "r_s": 85e-3,
        "l_s": 50e-6,
        "k_e": 0.0955,
        "j_rotor": 1e-4,
        "bemf_smoothing": 0.0,
    }
    HAS_JACOBIAN = False
    _default_limits = dict(omega=465, torque=0.0, i=40, epsilon=np.pi, u=44.4)
    _default_nominal_values = dict(omega=300, torque=0.0, i=24, epsilon=np.pi, u=44.4)
    _default_initializer = {
        "states": {"i_a": 0.0, "i_b": 0.0, "i_c": 0.0, "epsilon": 0.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }

    IO_VOLTAGES = ["u_a", "u_b", "u_c", "u_sd", "u_sq"]
    IO_CURRENTS = ["i_a", "i_b", "i_c", "i_sd", "i_sq"]

    I_A_IDX = 0
    I_B_IDX = 1
    I_C_IDX = 2
    EPSILON_IDX = 3
    CURRENTS_IDX = [0, 1, 2]
    CURRENTS = ["i_a", "i_b", "i_c"]
    VOLTAGES = ["u_a", "u_b", "u_c"]

    def __init__(
        self,
        motor_parameter=None,
        nominal_values=None,
        limit_values=None,
        motor_initializer=None,
    ):
        # Docstring of superclass
        super().__init__(motor_parameter, nominal_values, limit_values, motor_initializer)
        smoothing = self._motor_parameter.get("bemf_smoothing", 0.0)
        if not np.isscalar(smoothing) or not 0.0 <= smoothing <= 1.0:
            raise ValueError("bemf_smoothing must be a scalar in the interval [0, 1]")
        self._update_limits()

    def reset(self, state_space, state_positions, **__):
        # Docstring of superclass
        if self._initializer and self._initializer["states"]:
            self.initialize(state_space, state_positions)
            return np.asarray(list(self._initial_states.values()))
        else:
            return np.zeros(len(self.CURRENTS) + 1)

    def bemf_shape(self, epsilon):
        """
        Unit trapezoidal back-EMF shape functions of the three phases.

        Args:
            epsilon(ndarray(float)/float): Electrical rotor angle(s) in rad

        Returns:
            ndarray(float): Shape function values ``[f_a, f_b, f_c]`` of the phases,
            each in the range ``[-1, 1]``. Flat ``+1``/``-1`` over 120 deg electrical with
            linear 60 deg transitions, per Pillay & Krishnan (1989) Fig. 1.
        """
        theta = np.asarray(epsilon, dtype=float) % (2 * np.pi)
        return np.stack(
            (
                self._bemf_phase_a(theta),
                self._bemf_phase_a((theta - 2 * np.pi / 3) % (2 * np.pi)),
                self._bemf_phase_a((theta - 4 * np.pi / 3) % (2 * np.pi)),
            )
        )

    def _bemf_phase_a(self, theta):
        mp = self._motor_parameter
        smoothing = mp.get("bemf_smoothing", 0.0)
        # Alignment shift of 150 deg electrical: the phase-variable trapezoid of
        # Pillay & Krishnan (1989) (flat +1 over [0, 2pi/3]) is shifted so that its
        # fundamental component lies on the q-axis of the dq-transformation, i.e.
        # (e_d, e_q) = (0, E) with E > 0, matching the PMSM convention of GEM
        # (e_q = omega * p * psi_p). The angle origin is a free modeling convention.
        theta = np.mod(theta + 5 * np.pi / 6, 2 * np.pi)
        return np.where(
            theta < 2 * np.pi / 3,
            1.0,
            np.where(
                theta < np.pi,
                self._bemf_ramp(theta, 2 * np.pi / 3, np.pi, 1.0, -1.0, smoothing),
                np.where(
                    theta < 5 * np.pi / 3,
                    -1.0,
                    self._bemf_ramp(theta, 5 * np.pi / 3, 2 * np.pi, -1.0, 1.0, smoothing),
                ),
            ),
        )

    @staticmethod
    def _bemf_ramp(theta, theta_lo, theta_hi, f_lo, f_hi, smoothing):
        tau = (theta - theta_lo) / (theta_hi - theta_lo)
        if smoothing <= 0.0:
            return f_lo + (f_hi - f_lo) * tau
        smooth = 3.0 * tau**2 - 2.0 * tau**3
        tau_s = (1.0 - smoothing) * tau + smoothing * smooth
        return f_lo + (f_hi - f_lo) * tau_s

    def back_emf(self, state, omega):
        """
        Back-EMF of the three phases.

        Args:
            state(ndarray(float)): Motor state ``[i_a, i_b, i_c, epsilon]``
            omega(float): Mechanical angular velocity in rad/s

        Returns:
            ndarray(float): Back-EMF voltages ``[e_a, e_b, e_c]`` with
            :math:`e_j = k_e \\cdot \\omega \\cdot f_j(\\epsilon)`.
        """
        mp = self._motor_parameter
        return mp["k_e"] * omega * self.bemf_shape(state[self.EPSILON_IDX])

    def electrical_ode(self, state, u_abc, omega, *_):
        """
        The differential equation of the brushless DC motor in phase variables.

        The star connection is enforced through the floating neutral potential
        :math:`u_n = \\frac{1}{3}\\sum_j (u_j - r_s i_j - e_j)`, which guarantees
        :math:`\\frac{\\mathrm{d}}{\\mathrm{d}t}(i_a + i_b + i_c) = 0` identically, i.e.
        the common-mode current stays at its initial value (zero). This is required
        because the trapezoidal back-EMF has a nonzero common-mode component
        (:math:`f_a + f_b + f_c \\neq 0`), which would otherwise drive a spurious
        circulating common-mode current in a three-independent-ODE formulation.

        Args:
            state(ndarray(float)): Motor state ``[i_a, i_b, i_c, epsilon]``
            u_abc(list(float)): The phase input voltages ``[u_a, u_b, u_c]``
            omega(float): The mechanical angular velocity

        Returns:
            ndarray(float): Derivatives of the state vector
            :math:`\\mathrm{d}/\\mathrm{d}t([i_a, i_b, i_c, \\epsilon])`
            with :math:`\\mathrm{d}i_j/\\mathrm{d}t = (u_j - u_n - r_s i_j - e_j)/l_s`
            and :math:`\\mathrm{d}\\epsilon/\\mathrm{d}t = p \\cdot \\omega`.
        """
        mp = self._motor_parameter
        u = np.asarray(u_abc)
        i = state[self.CURRENTS_IDX]
        e = self.back_emf(state, omega)
        u_n = (np.sum(u) - mp["r_s"] * np.sum(i) - np.sum(e)) / 3.0
        di = (u - u_n - mp["r_s"] * i - e) / mp["l_s"]
        return np.concatenate((di, np.array([mp["p"] * omega])))

    def i_in(self, state):
        # Docstring of superclass
        return state[self.CURRENTS_IDX]

    def torque(self, state):
        """
        Electromagnetic torque from the instantaneous power balance
        :math:`T = k_e \\sum_j f_j(\\epsilon) i_j`.

        Args:
            state(ndarray(float)): Motor state ``[i_a, i_b, i_c, epsilon]``

        Returns:
            float: Electromagnetic torque in N*m
        """
        mp = self._motor_parameter
        i = state[self.CURRENTS_IDX]
        return mp["k_e"] * np.sum(self.bemf_shape(state[self.EPSILON_IDX]) * i)

    def _torque_limit(self):
        # Docstring of superclass
        # upper bound: two phases conducting at the current limit
        mp = self._motor_parameter
        return 2.0 * mp["k_e"] * self._limits["i"]

    def _update_limits(self):
        # Docstring of superclass
        voltage_limit = 0.5 * self._limits["u"]
        voltage_nominal = 0.5 * self._nominal_values["u"]

        limits_agenda = {}
        nominal_agenda = {}
        for u, i in zip(self.IO_VOLTAGES, self.IO_CURRENTS):
            limits_agenda[u] = voltage_limit
            nominal_agenda[u] = voltage_nominal
            limits_agenda[i] = self._limits.get("i", None) or self._limits[u] / self._motor_parameter["r_s"]
            nominal_agenda[i] = (
                self._nominal_values.get("i", None) or self._nominal_values[u] / self._motor_parameter["r_s"]
            )
        super()._update_limits(limits_agenda, nominal_agenda)
