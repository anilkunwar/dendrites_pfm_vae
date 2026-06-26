#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electrochemical Scaling Recommender App
========================================
Author: Anil Kunwar (26.06.2026)
"""

import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import BasicTickFormatter, HoverTool
from scipy.optimize import minimize

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Electrochemical Scaling Recommender",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f4e79; }
    .subtle { color: #555; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Scaling Engine — mirrors MOOSE ParsedMaterial blocks
# ============================================================
class ScalingEngine:
    def __init__(self, Ls: float, Es: float, Ts: float, Vm: float):
        self.Ls = float(Ls)
        self.Es = float(Es)
        self.Ts = float(Ts)
        self.Vm = float(Vm)

    def scaled_kappa(self, ko: float) -> float:
        """(energy_scale/length_scale) * ko"""
        return (self.Es / self.Ls) * ko

    def scaled_L1(self, L1o: float) -> float:
        """(length_scale^3 / (energy_scale*time_scale)) * L1o"""
        return (self.Ls ** 3 / (self.Es * self.Ts)) * L1o

    def scaled_L2(self, L2o: float) -> float:
        """L2o / time_scale"""
        return L2o / self.Ts

    def scaled_Deff(self, Meo: float, Mso: float = None, h: float = 0.5) -> float:
        """(length_scale^2 / time_scale) * (Meo*h + Mso*(1-h))"""
        if Mso is None:
            Mso = 5.0 * Meo
        return (self.Ls ** 2 / self.Ts) * (Meo * h + Mso * (1.0 - h))

    def scaled_ElecEff(self, S1o: float, S2o: float, h: float = 0.5) -> float:
        """(S1o*h + S2o*(1-h)) / length_scale"""
        return (S1o * h + S2o * (1.0 - h)) / self.Ls

    def scaled_ChargeEff(self, valency: float, Faraday: float, c_s: float) -> float:
        """valency * Faraday * c_s / length_scale^3"""
        return valency * Faraday * c_s / (self.Ls ** 3)

    def scaled_free_energy_density(self, phys_density: float) -> float:
        """(energy_scale / length_scale^3) * phys_density"""
        return (self.Es / self.Ls ** 3) * phys_density

    @staticmethod
    def physical_curvature(A: float, fo: float) -> float:
        """Second derivative of G = fo*[A*(c-ceq)^2 + ...]  =>  2*A*fo  [J/mol]"""
        return 2.0 * A * fo


# ============================================================
# Safe-envelope validation (VAE-MDN insights)
# ============================================================
def check_safe_envelope(params: dict):
    checks = {}

    # 1. Gradient energy coefficient
    ko = params.get("ko", 0.0)
    checks["kappa"] = {
        "val": ko,
        "ok": 4.0e-10 <= ko <= 1.0e-9,
        "msg": f"κ = {ko:.2e} J/m  →  safe range [4×10⁻¹⁰, 1×10⁻⁹]",
    }

    # 2. Applied voltage
    pot = params.get("POT_LEFT", 0.0)
    checks["voltage"] = {
        "val": pot,
        "ok": -0.33 <= pot <= -0.2,
        "msg": f"U = {pot:.3f} V  →  safe range [–0.33, –0.20]",
    }

    # 3. Noise strength
    noise = params.get("Noise", 0.0)
    checks["noise"] = {
        "val": noise,
        "ok": 5.0e-4 <= noise <= 1.5e-3,
        "msg": f"ψ = {noise:.2e} s⁻¹  →  safe range [5×10⁻⁴, 1.5×10⁻³]",
    }

    # 4. Electrolyte thermodynamic stiffness (Al = liquid phase in fl)
    Al = params.get("Al", 0.0)
    fo = params.get("fo", 0.0)
    curvature = 2.0 * Al * fo
    checks["stiffness"] = {
        "val": curvature,
        "ok": curvature >= 145.0,
        "msg": f"2·Al·fo = {curvature:.2f} J/mol  →  safe ≥ 145 J/mol",
    }

    return checks


# ============================================================
# O(1) numerical-status helper
# ============================================================
def o1_status(val: float):
    if val == 0 or not np.isfinite(val):
        return "⚪ Zero / Inf", "gray"
    lg = np.log10(abs(val))
    if -2.0 <= lg <= 2.0:
        return "✅ O(1)", "safe"
    elif -4.0 <= lg < -2.0 or 2.0 < lg <= 4.0:
        return "⚠️ Marginal", "warning"
    else:
        return "❌ Ill-conditioned", "danger"


# ============================================================
# Auto-recommender: find Ls & Ts that bring key params to O(1)
# ============================================================
def recommend_scales(phys: dict, Es: float = 6.24150943e18):
    ko, L1o, L2o, Meo = phys["ko"], phys["L1o"], phys["L2o"], phys["Meo"]

    def objective(x):
        Ls, Ts = 10.0 ** x[0], 10.0 ** x[1]
        eng = ScalingEngine(Ls, Es, Ts, 2.2e-5)
        vals = [
            eng.scaled_kappa(ko),
            eng.scaled_L1(L1o),
            eng.scaled_L2(L2o),
            eng.scaled_Deff(Meo),
        ]
        vals = [max(v, 1e-30) for v in vals]
        devs = [abs(np.log10(v)) for v in vals]   # target = 1.0
        penalty = max(devs)
        # penalize runaway values
        for v in vals:
            if v < 1e-6 or v > 1e6:
                penalty += 1e3
        return penalty

    res = minimize(
        objective,
        x0=[6.0, 3.0],
        bounds=[(3.0, 10.0), (-3.0, 6.0)],
        method="L-BFGS-B",
    )
    return 10.0 ** res.x[0], 10.0 ** res.x[1], res.fun


# ============================================================
# Free-energy helpers
# ============================================================
def G_liquid(c, cleq, Al, Bl, Cl, fo):
    return fo * (Al * (c - cleq) ** 2 + Bl * (c - cleq) + Cl)

def G_solid(c, cseq, As, Bs, Cs, fo):
    return fo * (As * (c - cseq) ** 2 + Bs * (c - cseq) + Cs)


# ============================================================
# MOOSE input template (based on Hao Tang Oct 2025 input)
# ============================================================
MOOSE_TEMPLATE = r"""# ============================================================
# Auto-generated MOOSE input for Li dendrite growth
# Generated by Electrochemical Scaling Recommender App
# ============================================================

[GlobalParams]
  seed = 12345
[]

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 50
  ny = 50
  xmin = 0
  xmax = 50
  ymin = 0
  ymax = 50
  elem_type = QUAD4
[]

[Variables]
  [./eta]
    order = FIRST
    family = LAGRANGE
  [../]
  [./c]
    order = FIRST
    family = LAGRANGE
  [../]
  [./pot]
    order = FIRST
    family = LAGRANGE
  [../]
[]

[AuxVariables]
  [./delta_eta]
  [../]
  [./delta_c]
  [../]
  [./delta_pot]
  [../]
[]

[ICs]
  [./eta]
    variable = eta
    type = FunctionIC
    function = 'if(x>=0&x<=5,1,0)'
  [../]
  [./c]
    variable = c
    type = FunctionIC
    function = 'if(x>=0&x<=5,0.2,0.8)'
  [../]
[]

[BCs]
  [./left_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'left'
    value = {POT_LEFT}
  [../]
  [./right_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'right'
    value = 0
  [../]
  [./right_comp]
    type = DirichletBC
    variable = 'c'
    boundary = 'right'
    value = 0.8
  [../]
[]

[Materials]
  [./scale]
    type = GenericConstantMaterial
    prop_names = 'length_scale energy_scale time_scale'
    prop_values = '{length_scale} {energy_scale} {time_scale}'
  [../]
  [./constants]
    type = GenericConstantMaterial
    prop_names  = 'c_s c_0 valency molar_vol Faraday RT alpha'
    prop_values = '7.69e4 1e3 1 {molar_vol} 96485 2494.2 0.5'
  [../]
  [./system_constants]
    type = GenericConstantMaterial
    prop_names  = 'Meo Mso L1o L2o ko S1o S2o'
    prop_values = '{Meo} {Mso} {L1o} {L2o} {ko} {S1o} {S2o}'
  [../]
  [./material_constants]
    type = GenericConstantMaterial
    prop_names  = 'fo Al As Bl Bs Cl Cs cleq cseq'
    prop_values = '{fo} {Al} {As} {Bl} {Bs} {Cl} {Cs} {cleq} {cseq}'
  [../]
  [./L1]
    type = ParsedMaterial
    expression = '((length_scale)^3/(energy_scale*time_scale))*L1o'
    property_name = L1
    material_property_names = 'L1o length_scale energy_scale time_scale'
    outputs = exodus
  [../]
  [./L2]
    type = ParsedMaterial
    expression = 'L2o/time_scale'
    property_name = L2
    material_property_names = 'L2o time_scale'
    outputs = exodus
  [../]
  [./kappa_isotropy]
    type = ParsedMaterial
    property_name = kappa
    expression = '(energy_scale/length_scale)*ko'
    material_property_names = 'ko length_scale energy_scale'
    outputs = exodus
  [../]
  [./h]
    type = SwitchingFunctionMaterial
    h_order = HIGH
    eta = eta
    outputs = exodus
  [../]
  [./g]
    type = BarrierFunctionMaterial
    g_order = SIMPLE
    eta = eta
  [../]
  [./fl]
    type = DerivativeParsedMaterial
    property_name = fl
    material_property_names = 'fo Al Bl Cl cleq length_scale energy_scale molar_vol'
    coupled_variables = 'c'
    expression = '(energy_scale/(length_scale)^3)*(Al*(c-cleq)^2 + Bl*(c-cleq) + Cl)*fo/molar_vol'
    outputs = exodus
    derivative_order = 2
  [../]
  [./fs]
    type = DerivativeParsedMaterial
    property_name = fs
    material_property_names = 'fo As Bs Cs cseq length_scale energy_scale molar_vol'
    coupled_variables = 'c'
    expression = '(energy_scale/(length_scale)^3)*(As*(c-cseq)^2 + Bs*(c-cseq) + Cs)*fo/molar_vol'
    outputs = exodus
    derivative_order = 2
  [../]
  [./free_energy]
    type = DerivativeTwoPhaseMaterial
    property_name = F
    fa_name = fl
    fb_name = fs
    eta = eta
    coupled_variables = 'c'
    W = {W}
    outputs = exodus
    derivative_order = 2
  [../]
  [./Butlervolmer]
    type = DerivativeParsedMaterial
    expression = 'L2*dh*(eta*exp(pot*(1-alpha)*Faraday*valency/RT)-c*exp(-pot*alpha*Faraday*valency/RT))'
    coupled_variables = 'pot c eta'
    property_name = f_bv
    material_property_names = 'L2 alpha Faraday valency RT dh:=D[h,eta]'
    outputs = exodus
    derivative_order = 1
  [../]
  [./Deff]
    type = ParsedMaterial
    expression = '(length_scale)^2/time_scale*(Meo*h+Mso*(1-h))'
    property_name = Deff
    material_property_names = 'Meo Mso h length_scale time_scale'
    outputs = exodus
  [../]
  [./Deffe]
    type = DerivativeParsedMaterial
    expression = '(length_scale)^2/time_scale*(Meo*h+Mso*(1-h))*c*valency*Faraday'
    coupled_variables = 'c'
    property_name = Deffe
    material_property_names = 'Meo Mso h length_scale time_scale valency Faraday'
    outputs = exodus
    derivative_order = 1
  [../]
  [./coupled_eta_function]
    type = ParsedMaterial
    expression = 'c_s/c_0'
    property_name = ft
    material_property_names = 'c_s c_0'
    outputs = exodus
  [../]
  [./ElecEff]
    type = ParsedMaterial
    expression = '(S1o*h+S2o*(1-h))/length_scale'
    property_name = ElecEff
    material_property_names = 'S1o S2o h length_scale'
    outputs = exodus
  [../]
  [./ChargeEff]
    type = ParsedMaterial
    expression = 'valency*Faraday*c_s/(length_scale^3)'
    property_name = ChargeEff
    material_property_names = 'valency Faraday length_scale c_s'
    outputs = exodus
  [../]
[]

[Kernels]
  [./dcdt]
    type = TimeDerivative
    variable = c
  [../]
  [./ch]
    type = CahnHilliard
    variable = c
    f_name = F
    mob_name = Deff
    coupled_variables = 'eta'
  [../]
  [./elec]
    type = MatDiffusion
    variable = c
    v = pot
    diffusivity = Deffe
    args = 'eta c'
  [../]
  [./cSource]
    type = CoupledSusceptibilityTimeDerivative
    variable = c
    v = eta
    f_name = ft
  [../]
  [./detadt]
    type = TimeDerivative
    variable = eta
  [../]
  [./ac]
    type = AllenCahn
    variable = eta
    coupled_variables = 'eta c'
    f_name = F
    mob_name = L1
  [../]
  [./ACInterface]
    type = ACInterface
    variable = eta
    kappa_name = kappa
    mob_name = L1
  [../]
  [./BV]
    type = MKinetics
    variable = eta
    f_name = f_bv
    coupled_variables = 'c pot eta'
  [../]
  [./noise_interface]
    type = LangevinNoise
    variable = eta
    multiplier = dh/deta
    amplitude = {Noise}
  [../]
  [./Cond]
    type = MatDiffusion
    variable = pot
    diffusivity = ElecEff
    args = 'eta'
  [../]
  [./coupledSource]
    type = CoupledSusceptibilityTimeDerivative
    variable = pot
    v = eta
    f_name = ChargeEff
  [../]
[]

[AuxKernels]
  [./deta]
    type = DeltaUAux
    variable = delta_eta
    coupled_variable = eta
    execute_on = timestep_end
  [../]
  [./dc]
    type = DeltaUAux
    variable = delta_c
    coupled_variable = c
    execute_on = timestep_end
  [../]
  [./dpot]
    type = DeltaUAux
    variable = delta_pot
    coupled_variable = pot
    execute_on = timestep_end
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_type'
  petsc_options_value = 'preonly   lu        mumps'
  dtmax = 100
  end_time = 5E3
  [./TimeStepper]
    type = IterationAdaptiveDT
    dt = 1E-3
    growth_factor = 1.1
  [../]
  [./Adaptivity]
    interval = 5
    initial_adaptivity = 4
    refine_fraction = 0.8
    coarsen_fraction = 0.1
    max_h_level = 2
  [../]
[]

[Preconditioning]
  [./full]
    type = SMP
    full = true
  [../]
[]

[Postprocessors]
  [./eta_min]
    type = NodalExtremeValue
    variable = eta
    value_type = min
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./eta_max]
    type = NodalExtremeValue
    variable = eta
    value_type = max
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./c_min]
    type = NodalExtremeValue
    variable = c
    value_type = min
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./c_max]
    type = NodalExtremeValue
    variable = c
    value_type = max
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./pot_min]
    type = NodalExtremeValue
    variable = pot
    value_type = min
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./pot_max]
    type = NodalExtremeValue
    variable = pot
    value_type = max
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
[]

[UserObjects]
  [./eta_min_m]
    type = Terminator
    expression = 'eta_min >= 1'
    fail_mode = HARD
    error_level = INFO
    message = 'eta_min_m says this should end'
    execute_on = TIMESTEP_END
  [../]
  [./eta_max_m]
    type = Terminator
    expression = 'eta_max <= 0'
    fail_mode = HARD
    error_level = INFO
    message = 'eta_max_m says this should end'
    execute_on = TIMESTEP_END
  [../]
  [./c_min_m]
    type = Terminator
    expression = 'abs(c_min) >= 1'
    fail_mode = HARD
    error_level = INFO
    message = 'c_min_m says this should end'
    execute_on = TIMESTEP_END
  [../]
  [./c_max_m]
    type = Terminator
    expression = 'c_max <= 0'
    fail_mode = HARD
    error_level = INFO
    message = 'c_max_m says this should end'
    execute_on = TIMESTEP_END
  [../]
[]

[Outputs]
  exodus = true
  time_step_interval = 10
  file_base = results/{CASE}/{CASE}
[]
"""


# ============================================================
# Python generator script template (safe ranges enforced)
# ============================================================
GENERATOR_TEMPLATE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated MOOSE input generator
Safe ranges enforced from VAE-MDN dendrite-suppression envelope
"""

import random
import json
from pathlib import Path

TEMPLATE_FILE = "template.i"
OUTPUT_DIR = Path("MooseProject/generated_inputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Safe parameter ranges (VAE-MDN envelope)
PARAM_RANGES = {{
    "POT_LEFT": ({pot_left_min}, {pot_left_max}),
    "fo": ({fo_min}, {fo_max}),
    "Al": ({Al_min}, {Al_max}),
    "Bl": ({Bl_min}, {Bl_max}),
    "Cl": ({Cl_min}, {Cl_max}),
    "As": ({As_min}, {As_max}),
    "Bs": ({Bs_min}, {Bs_max}),
    "Cs": ({Cs_min}, {Cs_max}),
    "cleq": ({cleq_min}, {cleq_max}),
    "cseq": ({cseq_min}, {cseq_max}),
    "L1o": ({L1o_min}, {L1o_max}),
    "L2o": ({L2o_min}, {L2o_max}),
    "ko": ({ko_min}, {ko_max}),
    "Noise": ({noise_min}, {noise_max})
}}

NUM_CASES = {num_cases}

def generate_case(template_text, param_ranges):
    values = {{}}
    new_text = template_text
    for name, (low, high) in param_ranges.items():
        val = random.uniform(low, high)
        values[name] = val
        new_text = new_text.replace(f"${{name}}", f"{{val:.6g}}")
    return new_text, values

def main():
    template = Path(TEMPLATE_FILE).read_text()
    for i in range(1, NUM_CASES + 1):
        case_name = f"case_{{i:03d}}"
        new_text, params = generate_case(template, PARAM_RANGES)
        new_text = new_text.replace("$CASE", case_name)
        (OUTPUT_DIR / f"{{case_name}}.i").write_text(new_text)
        with open(OUTPUT_DIR / f"{{case_name}}.json", "w") as f:
            json.dump(params, f, indent=2)
        print(f"[+] Generated {{case_name}}.i & .json")

if __name__ == "__main__":
    main()
'''


# ============================================================
# Main App
# ============================================================
def main():
    st.markdown('<p class="main-header">🔋 Electrochemical Scaling Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Bridge physical design rules (VAE-MDN safe envelope) with '
                'numerical stability (JFNK solver convergence via O(1) scaled variables).</p>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # Sidebar — Scale factors & physical constants
    # --------------------------------------------------------
    st.sidebar.header("⚖️ Scale Factors")
    length_scale = st.sidebar.number_input(
        r"Length scale $L_s$ (e.g. 1e6 for m→µm):",
        value=1.0e6, format="%.2e", step=1.0e6, key="Ls"
    )
    energy_scale = st.sidebar.number_input(
        r"Energy scale $E_s$ (e.g. 6.2415e18 for J→eV):",
        value=6.24150943e18, format="%.2e", step=1.0e18, key="Es"
    )
    time_scale = st.sidebar.number_input(
        r"Time scale $T_s$ (e.g. 1e3 for s→ms):",
        value=1.0e3, format="%.2e", step=1.0e3, key="Ts"
    )

    st.sidebar.header("🔧 Constants")
    molar_vol = st.sidebar.number_input(
        r"Molar volume $V_m$ (m³/mol):", value=2.2e-5, format="%.2e", step=1e-6
    )

    with st.sidebar.expander("Advanced constants"):
        Meo = st.number_input("Meo (m²/s):", value=1.0e-13, format="%.2e")
        Mso = st.number_input("Mso (m²/s):", value=5.0e-13, format="%.2e")
        S1o = st.number_input("S1o (S/m):", value=1.0e7, format="%.2e")
        S2o = st.number_input("S2o (S/m):", value=1.19, format="%.2e")
        W = st.number_input("Double-well barrier W:", value=2.0e3, format="%.2e")

    # --------------------------------------------------------
    # Sidebar — Physical parameters (safe envelope)
    # --------------------------------------------------------
    st.sidebar.header("🔬 Physical Parameters")

    ko = st.sidebar.number_input(
        "κ  (ko)  [J/m]:", min_value=1e-12, max_value=1e-7,
        value=5.0e-10, format="%.2e", step=1e-11
    )
    if not (4.0e-10 <= ko <= 1.0e-9):
        st.sidebar.warning("⚠️ Safe: [4×10⁻¹⁰, 1×10⁻⁹] J/m")

    POT_LEFT = st.sidebar.number_input(
        "Applied voltage POT_LEFT [V]:", min_value=-1.0, max_value=0.0,
        value=-0.25, step=0.01, format="%.3f"
    )
    if not (-0.33 <= POT_LEFT <= -0.2):
        st.sidebar.warning("⚠️ Safe: [–0.33, –0.20] V")

    Noise = st.sidebar.number_input(
        "Noise amplitude ψ [s⁻¹]:", min_value=1e-5, max_value=1e-2,
        value=1.0e-3, format="%.2e", step=1e-4
    )
    if not (5.0e-4 <= Noise <= 1.5e-3):
        st.sidebar.warning("⚠️ Safe: [5×10⁻⁴, 1.5×10⁻³] s⁻¹")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Free-energy coefficients")

    fo = st.sidebar.number_input(
        "Amplitude fo [J/mol]:", value=1.0e4, format="%.2e", step=1e2
    )
    Al = st.sidebar.slider("Al  (electrolyte curvature):", 0.0, 50.0, 10.0, 0.1)
    Bl = st.sidebar.slider("Bl  (electrolyte linear):", -10.0, 10.0, 0.0, 0.1)
    Cl = st.sidebar.slider("Cl  (electrolyte constant):", -10.0, 10.0, 0.0, 0.1)
    cleq = st.sidebar.slider("c_eq liquid:", 0.01, 0.99, 0.80, 0.01)

    As = st.sidebar.slider("As  (solid curvature):", 0.0, 50.0, 10.0, 0.1)
    Bs = st.sidebar.slider("Bs  (solid linear):", -10.0, 10.0, 0.0, 0.1)
    Cs = st.sidebar.slider("Cs  (solid constant):", -10.0, 10.0, 0.0, 0.1)
    cseq = st.sidebar.slider("c_eq solid:", 0.01, 0.99, 0.20, 0.01)

    st.sidebar.subheader("Mobilities")
    L1o = st.sidebar.number_input(
        "L1o  [m³/J·s]:", value=1.0e-13, format="%.2e", step=1e-14
    )
    L2o = st.sidebar.number_input(
        "L2o  [s⁻¹]:", value=1.0e-1, format="%.2e", step=1e-2
    )

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------
    tab_safe, tab_diag, tab_fe, tab_rec, tab_gen = st.tabs([
        "🛡️ Safe Envelope", "📊 Diagnostics", "📈 Free Energy",
        "🔧 Auto-Recommender", "💾 Generator"
    ])

    engine = ScalingEngine(length_scale, energy_scale, time_scale, molar_vol)

    # ========================================================
    # Tab 1 — Safe Envelope
    # ========================================================
    with tab_safe:
        st.header("VAE-MDN Safe Operating Envelope")
        params = {"ko": ko, "POT_LEFT": POT_LEFT, "Noise": Noise, "Al": Al, "fo": fo}
        checks = check_safe_envelope(params)

        c1, c2 = st.columns(2)
        cols = [c1, c2]
        for idx, (key, chk) in enumerate(checks.items()):
            with cols[idx % 2]:
                if chk["ok"]:
                    st.success(f"✅ {chk['msg']}")
                else:
                    st.error(f"❌ {chk['msg']}")

        st.divider()
        st.subheader("Target-stiffness calculator")
        target_A = st.number_input("Target electrolyte stiffness [J/mol]:", value=150.0, step=5.0)
        req_fo = target_A / (2.0 * Al) if Al > 0 else np.inf
        st.info(f"With **Al = {Al}**, you need **fo ≥ {req_fo:.2e} J/mol** to hit the target.")

        st.divider()
        st.subheader("Quick summary")
        summary = pd.DataFrame({
            "Parameter": ["κ", "U (POT_LEFT)", "ψ (Noise)", "2·Al·fo", "L1o", "L2o", "fo"],
            "Value": [
                f"{ko:.2e} J/m", f"{POT_LEFT:.3f} V", f"{Noise:.2e} s⁻¹",
                f"{2*Al*fo:.2f} J/mol", f"{L1o:.2e} m³/J·s",
                f"{L2o:.2e} s⁻¹", f"{fo:.2e} J/mol"
            ],
            "Safe range": ["[4e-10, 1e-9]", "[–0.33, –0.2]", "[5e-4, 1.5e-3]", "≥ 145", "—", "—", "—"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ========================================================
    # Tab 2 — Diagnostics
    # ========================================================
    with tab_diag:
        st.header("Scaling Diagnostic Dashboard")

        # Compute scaled quantities
        s_kappa = engine.scaled_kappa(ko)
        s_L1 = engine.scaled_L1(L1o)
        s_L2 = engine.scaled_L2(L2o)
        s_D = engine.scaled_Deff(Meo, Mso)
        s_E = engine.scaled_ElecEff(S1o, S2o)
        s_Q = engine.scaled_ChargeEff(1.0, 96485.0, 7.69e4)

        phys_fl = fo * Cl / molar_vol          # at c = cleq
        phys_fs = fo * Cs / molar_vol          # at c = cseq
        s_fl = engine.scaled_free_energy_density(phys_fl)
        s_fs = engine.scaled_free_energy_density(phys_fs)

        diag_data = {
            "Parameter": [
                "κ (kappa)", "L1 (interface mob.)", "L2 (reaction rate)",
                "Deff (diffusivity)", "ElecEff (conductivity)", "ChargeEff",
                "fl (liq. energy dens.)", "fs (sol. energy dens.)"
            ],
            "SI value": [
                f"{ko:.2e} J/m", f"{L1o:.2e} m³/J·s", f"{L2o:.2e} s⁻¹",
                f"{Meo:.2e} m²/s", f"{S1o:.2e} S/m", "9.65e9 C/m³",
                f"{phys_fl:.2e} J/m³", f"{phys_fs:.2e} J/m³"
            ],
            "Scaled MOOSE": [
                f"{s_kappa:.4e}", f"{s_L1:.4e}", f"{s_L2:.4e}",
                f"{s_D:.4e}", f"{s_E:.4e}", f"{s_Q:.4e}",
                f"{s_fl:.4e}", f"{s_fs:.4e}"
            ],
            "Status": [
                o1_status(s_kappa)[0], o1_status(s_L1)[0], o1_status(s_L2)[0],
                o1_status(s_D)[0], o1_status(s_E)[0], o1_status(s_Q)[0],
                o1_status(s_fl)[0], o1_status(s_fs)[0]
            ]
        }
        df_diag = pd.DataFrame(diag_data)
        st.dataframe(df_diag, use_container_width=True, hide_index=True)

        st.subheader("Numerical-stability feedback")
        for name, val in [
            ("kappa", s_kappa), ("L1", s_L1), ("L2", s_L2),
            ("Deff", s_D), ("ElecEff", s_E), ("ChargeEff", s_Q)
        ]:
            status, level = o1_status(val)
            if level == "danger":
                st.error(f"**{name}** = {val:.4e} → {status}. "
                         f"Risk of ill-conditioning or truncation error in JFNK.")
            elif level == "warning":
                st.warning(f"**{name}** = {val:.4e} → {status}. "
                           f"May cause slow convergence or loss of coupling.")
            else:
                st.success(f"**{name}** = {val:.4e} → {status}. Well-scaled.")

        st.divider()
        st.subheader("Raw scaled physical quantities")
        sc = {
            "κ_scaled": s_kappa, "L1_scaled": s_L1, "L2_scaled": s_L2,
            "Deff_scaled": s_D, "ElecEff_scaled": s_E, "ChargeEff_scaled": s_Q
        }
        st.json(sc)

    # ========================================================
    # Tab 3 — Free Energy
    # ========================================================
    with tab_fe:
        st.header("Free Energy Visualization")

        c = np.linspace(0, 1, 200)
        Gl = G_liquid(c, cleq, Al, Bl, Cl, fo)
        Gs = G_solid(c, cseq, As, Bs, Cs, fo)

        # --- J/mol ---
        df_gl = pd.DataFrame({"c": c, "G": Gl, "phase": "Liquid (electrolyte)"})
        df_gs = pd.DataFrame({"c": c, "G": Gs, "phase": "Solid (electrode)"})
        df_G = pd.concat([df_gl, df_gs], ignore_index=True)

        col1, col2 = st.columns(2)
        colors = ["#338fff", "#ff5733"]

        with col1:
            st.subheader("Gibbs free energy [J/mol]")
            p = figure(width=500, height=400, title="G vs c")
            for i, ph in enumerate(["Liquid (electrolyte)", "Solid (electrode)"]):
                src = df_G[df_G["phase"] == ph]
                p.line(src["c"], src["G"], color=colors[i], line_width=3, legend_label=ph)
            p.xaxis.axis_label = "Concentration c"
            p.yaxis.axis_label = "G [J/mol]"
            p.legend.click_policy = "hide"
            p.add_tools(HoverTool(tooltips=[("c", "$x"), ("G", "$y")]))
            st.bokeh_chart(p, use_container_width=True)

        # --- J/m³ ---
        fl = Gl / molar_vol
        fs = Gs / molar_vol
        df_fl = pd.DataFrame({"c": c, "f": fl, "phase": "Liquid (electrolyte)"})
        df_fs = pd.DataFrame({"c": c, "f": fs, "phase": "Solid (electrode)"})
        df_f = pd.concat([df_fl, df_fs], ignore_index=True)

        with col2:
            st.subheader("Free energy density [J/m³]")
            p2 = figure(width=500, height=400, title="f vs c")
            for i, ph in enumerate(["Liquid (electrolyte)", "Solid (electrode)"]):
                src = df_f[df_f["phase"] == ph]
                p2.line(src["c"], src["f"], color=colors[i], line_width=3, legend_label=ph)
            p2.xaxis.axis_label = "Concentration c"
            p2.yaxis.axis_label = "f [J/m³]"
            p2.yaxis.formatter = BasicTickFormatter(use_scientific=True, precision=1)
            p2.legend.click_policy = "hide"
            p2.add_tools(HoverTool(tooltips=[("c", "$x"), ("f", "$y")]))
            st.bokeh_chart(p2, use_container_width=True)

        # --- Scaled ---
        f_l_sc = fl * (energy_scale / length_scale ** 3)
        f_s_sc = fs * (energy_scale / length_scale ** 3)
        df_flsc = pd.DataFrame({"c": c, "fsc": f_l_sc, "phase": "Liquid (electrolyte)"})
        df_fssc = pd.DataFrame({"c": c, "fsc": f_s_sc, "phase": "Solid (electrode)"})
        df_fsc = pd.concat([df_flsc, df_fssc], ignore_index=True)

        st.subheader("Scaled free energy density [dimensionless]")
        p3 = figure(width=900, height=400, title="f_scaled vs c")
        for i, ph in enumerate(["Liquid (electrolyte)", "Solid (electrode)"]):
            src = df_fsc[df_fsc["phase"] == ph]
            p3.line(src["c"], src["fsc"], color=colors[i], line_width=3, legend_label=ph)
        p3.xaxis.axis_label = "Concentration c"
        p3.yaxis.axis_label = "f_scaled"
        p3.yaxis.formatter = BasicTickFormatter(use_scientific=True, precision=1)
        p3.legend.click_policy = "hide"
        p3.add_tools(HoverTool(tooltips=[("c", "$x"), ("fsc", "$y")]))
        st.bokeh_chart(p3, use_container_width=True)

        st.latex(r"G_l = f_0\bigl[A_l(c-c_{eq}^l)^2 + B_l(c-c_{eq}^l) + C_l\bigr] \;\;[\mathrm{J/mol}]")
        st.latex(r"G_s = f_0\bigl[A_s(c-c_{eq}^s)^2 + B_s(c-c_{eq}^s) + C_s\bigr] \;\;[\mathrm{J/mol}]")
        st.latex(r"f_{\text{scaled}} = \frac{E_s}{L_s^3}\,\frac{G}{V_m}")

    # ========================================================
    # Tab 4 — Auto-Recommender
    # ========================================================
    with tab_rec:
        st.header("O(1) Auto-Recommender")
        st.markdown("Find $L_s$ and $T_s$ that bring κ, L1, L2, and Deff as close to **O(1)** as possible.")

        phys = {"ko": ko, "L1o": L1o, "L2o": L2o, "Meo": Meo}

        if st.button("🔍 Find optimal scale factors", type="primary"):
            with st.spinner("Running L-BFGS-B optimization…"):
                opt_Ls, opt_Ts, penalty = recommend_scales(phys, Es=energy_scale)

            st.success("Optimization complete!")
            m1, m2, m3 = st.columns(3)
            m1.metric("Optimal Ls", f"{opt_Ls:.4e}")
            m2.metric("Optimal Ts", f"{opt_Ts:.4e}")
            m3.metric("Max |log₁₀ deviation|", f"{penalty:.4f}")

            # Preview
            rec_eng = ScalingEngine(opt_Ls, energy_scale, opt_Ts, molar_vol)
            preview = pd.DataFrame({
                "Parameter": ["κ", "L1", "L2", "Deff"],
                "Scaled value": [
                    f"{rec_eng.scaled_kappa(ko):.4e}",
                    f"{rec_eng.scaled_L1(L1o):.4e}",
                    f"{rec_eng.scaled_L2(L2o):.4e}",
                    f"{rec_eng.scaled_Deff(Meo):.4e}",
                ]
            })
            st.table(preview)

            if st.button("⬅️ Apply recommended scales to sidebar"):
                st.session_state["Ls"] = opt_Ls
                st.session_state["Ts"] = opt_Ts
                st.rerun()

    # ========================================================
    # Tab 5 — Generator
    # ========================================================
    with tab_gen:
        st.header("MOOSE Input & Script Generator")

        g1, g2 = st.columns(2)

        with g1:
            st.subheader("1. MOOSE input file")
            case_name = st.text_input("Case name:", value="dendrite_safe_001")
            moose_text = MOOSE_TEMPLATE.format(
                length_scale=length_scale,
                energy_scale=energy_scale,
                time_scale=time_scale,
                molar_vol=molar_vol,
                Meo=Meo, Mso=Mso, L1o=L1o, L2o=L2o, ko=ko,
                S1o=S1o, S2o=S2o,
                fo=fo, Al=Al, As=As, Bl=Bl, Bs=Bs, Cl=Cl, Cs=Cs,
                cleq=cleq, cseq=cseq,
                POT_LEFT=POT_LEFT, Noise=Noise, W=W,
                CASE=case_name
            )
            st.download_button(
                label="⬇️ Download .i file",
                data=moose_text,
                file_name=f"{case_name}.i",
                mime="text/plain"
            )
            with st.expander("Preview (first 60 lines)"):
                st.code("\n".join(moose_text.splitlines()[:60]), language="ini")

        with g2:
            st.subheader("2. Python case generator")
            n_cases = int(st.number_input("Number of cases:", value=100, step=10))

            # Build generator with safe ranges
            gen_py = GENERATOR_TEMPLATE.format(
                pot_left_min=-0.33, pot_left_max=-0.2,
                fo_min=1e3, fo_max=5e4,
                Al_min=5.0, Al_max=50.0,
                Bl_min=-10.0, Bl_max=10.0,
                Cl_min=-10.0, Cl_max=10.0,
                As_min=0.0, As_max=50.0,
                Bs_min=-10.0, Bs_max=10.0,
                Cs_min=-10.0, Cs_max=10.0,
                cleq_min=0.01, cleq_max=0.99,
                cseq_min=0.01, cseq_max=0.99,
                L1o_min=1e-15, L1o_max=1e-11,
                L2o_min=1e-3, L2o_max=1.0,
                ko_min=4e-10, ko_max=1e-9,
                noise_min=5e-4, noise_max=1.5e-3,
                num_cases=n_cases
            )
            st.download_button(
                label="⬇️ Download generate_inputs.py",
                data=gen_py,
                file_name="generate_inputs.py",
                mime="text/x-python"
            )
            with st.expander("Preview (first 40 lines)"):
                st.code("\n".join(gen_py.splitlines()[:40]), language="python")

        st.divider()
        st.info("""
        **Citation**  
        If you use this tool, please cite:  
        Kunwar, A., Yousefi, E., Zuo, X., Sun, Y., Seveno, D., Guo, M., & Moelans, N. (2022).  
        Multi-phase field simulation of Al₃Ni₂ intermetallic growth at liquid Al/solid Ni interface  
        using MD computed interfacial energies. *International Journal of Mechanical Sciences*, 215, 106930.  
        https://doi.org/10.1016/j.ijmecsci.2021.106930
        """)


if __name__ == "__main__":
    main()
