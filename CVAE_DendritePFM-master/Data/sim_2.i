#
# KKS simple example in the split form
#

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 60	# 0.5 μm
  ny = 20
  xmin = 0
  xmax = 30 	# μm
  ymin = 0
  ymax = 10
  elem_type = QUAD4
[]

[AuxVariables]
  [./Fglobal]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[Variables]
  # order parameter
  [./eta]
    order = FIRST
    family = LAGRANGE
  [../]

  # solute concentration
  [./c]
    order = FIRST
    family = LAGRANGE
  [../]
  
  # Liquid phase solute concentration
  [./cl]
    order = FIRST
    family = LAGRANGE
    initial_condition = 0.0128	# 1/(1+76.9)
  [../]
  
  # Solid phase solute concentration
  [./cs]
    order = FIRST
    family = LAGRANGE
    initial_condition = 0.9872	#
  [../]
  
  # chemical potential
  [./w]
    order = FIRST
    family = LAGRANGE
  [../]
  
  # electric overpotential
  [./pot]
    order = FIRST
    family = LAGRANGE
  [../]
  
[]

[ICs]
  [./eta]
    variable = eta
    type = FunctionIC
    function = 'if(x>=0&x<=1,1,0)'
  [../]
  [./c]
    variable = c
    type = FunctionIC
    function = 'if(x>=0&x<=1,0.9872,0.0128)'
  [../]
[]

[BCs]
  [./left_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'left'
    value = -0.45
  [../]
  [./right_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'right'
    value = 0
  [../]
[]

[Materials]

  [./scale]
    type = GenericConstantMaterial
    prop_names = 'length_scale energy_scale time_scale'
    prop_values = '1e6 6.24150943e18 1.0e3'
  [../]
  
  [./constants]
    type = GenericConstantMaterial
    prop_names  = 'S1 S2 c_s valency molar_vol RT alpha'
    prop_values = '1e7 1.19 1e3 2.2e-5 1  2494.2 0.5'
  [../]
  
  [./generalDiff]
  	type = GenericConstantMaterial
    prop_names  = 'M L eps_sq Ls De Ds'
    prop_values = '0.04 0.04 0.65 0.1 2.5e-16 2.5e-15'
  [../]
  
#  [./fl]
#    type = DerivativeParsedMaterial
#    property_name = fl
#    constant_names = 'factor_f1'
#    constant_expressions = '1.0e-4'
#    material_property_names = 'length_scale energy_scale molar_vol Faraday valency'
#    coupled_variables = 'c eta'
#    expression = '(energy_scale/(length_scale)^3) * 1 *((2.05*(eta-0.01)^2 -3.98*(c-0.01)^3 +2.01*(eta-0.01)^4 -0.2 )* factor_f1/molar_vol )'
#    outputs = exodus
#  [../]

#  [./fs]
#    type = DerivativeParsedMaterial
#    property_name = fs
#    constant_names = 'factor_f2'
#    constant_expressions = '1.0e-4'
#    material_property_names = 'length_scale energy_scale molar_vol Faraday valency'
#    coupled_variables = 'c eta'
#    expression = '(energy_scale/(length_scale)^3) *1 *((2.05*(eta-0.01)^2 -3.98*(c-0.01)^3 +2.01*(eta-0.01)^4 -0.2 )* factor_f2/molar_vol )'
#    outputs = exodus
#  [../]
  
  # Free energy of the liquid
  [./fl]
    type = DerivativeParsedMaterial
    property_name = fl
    coupled_variables = 'cl'
    expression = '(0.0128-cl)^2'
    outputs = exodus
  [../]

  # Free energy of the solid
  [./fs]
    type = DerivativeParsedMaterial
    property_name = fs
    coupled_variables = 'cs'
    expression = '(0.9872-cs)^2'
    outputs = exodus
  [../]
  
  # h(eta)
  [./h_eta]
    type = SwitchingFunctionMaterial
    h_order = HIGH
    eta = eta
    outputs = exodus
  [../]

  # g(eta)
  [./g_eta]
    type = BarrierFunctionMaterial
    g_order = SIMPLE
    eta = eta
  [../]

  [./Butlervolmer]
    type = DerivativeParsedMaterial
    expression = 'Ls*(exp(pot*(1-alpha)*Faraday*valency/RT)-(c/0.0128)*exp(-pot*alpha*Faraday*valency/RT))/time_scale'
    coupled_variables = 'pot c'
    property_name = f_bv
    derivative_order = 1
    material_property_names = 'Ls alpha Faraday valency RT time_scale'
    outputs = exodus
  [../]
  [./monitor_BV1]
  	type = DerivativeParsedMaterial
    expression = 'exp(pot*(1-alpha)*Faraday*valency/RT)'
    coupled_variables = 'pot'
    property_name = monitor_BV1
    material_property_names = 'alpha Faraday valency RT'
    derivative_order = 1
    outputs = exodus
  []  
  [./monitor_BV2]
  	type = DerivativeParsedMaterial
    expression = '-(c/0.0128)*exp(-pot*alpha*Faraday*valency/RT)'
    coupled_variables = 'pot c'
    property_name = monitor_BV2
    material_property_names = 'alpha Faraday valency RT'
    derivative_order = 1
    outputs = exodus
  []  
  
  #
  [./DeffNorm]	# 扩散系数
    type = DerivativeParsedMaterial
    expression = '(De*h+Ds*(1-h))*(length_scale)^2/time_scale'
    coupled_variables = 'eta'
    property_name = DeffNorm
    material_property_names = 'De Ds h length_scale time_scale'
    derivative_order = 1
    outputs = exodus
  [../]
  [./DeffeNorm]	# 扩散系数
    type = DerivativeParsedMaterial
    expression = 'DeffNorm*c*valency*Faraday/RT'
    coupled_variables = 'eta c'
    property_name = DeffeNorm
    material_property_names = 'DeffNorm valency Faraday RT'
    derivative_order = 1
    outputs = exodus
  [../]
  [./coupled_eta_function]
    type = DerivativeParsedMaterial
    expression = 'c_s/c'
    coupled_variables = 'c'
    property_name = ft
    material_property_names = 'c_s'
    derivative_order = 1
  [../]
  # 
  [./ElecEff]	
    type = DerivativeParsedMaterial
    expression = '(S1*h+S2*(1-h))/length_scale'
    coupled_variables = 'eta'
    property_name = ElecEff
    material_property_names = 'S1 S2 h length_scale'
    derivative_order = 1
    outputs = exodus
  [../]
  [./ChargeEff]
    type = DerivativeParsedMaterial
    expression = 'valency*Faraday*c_s/(length_scale^3)'
    property_name = ChargeEff
    material_property_names = 'valency Faraday c_s length_scale'
    derivative_order = 1
  [../]
  
[]

[Kernels]

  # enforce c = (1-h(eta))*cl + h(eta)*cs
  [./PhaseConc]
    type = KKSPhaseConcentration
    ca       = cl
    variable = cs
    c        = c
    eta      = eta
  [../]
  # enforce pointwise equality of chemical potentials
  [./ChemPotSolute]
    type = KKSPhaseChemicalPotential
    variable = cl
    cb       = cs
    fa_name  = fl
    fb_name  = fs
    args_a = 'eta c'
    args_b = 'eta c'
  [../]

  #
  # Cahn-Hilliard Equation
  #
  [./CHBulk]
    type = KKSSplitCHCRes
    variable = c
    ca       = cl
    fa_name  = fl
    w        = w
    args_a = 'eta c'
  [../]
  [./dcdt]
    type = CoupledTimeDerivative
    variable = w
    v = c
  [../]
  [./ckernel]
    type = SplitCHWRes
    mob_name = M
    variable = w
  [../]
#  [./elec]	# 电迁移
#  	type = MatDiffusion
#    variable = c
#    v = pot
#    diffusivity = DeffeNorm
#  [../]
#  [./cSource] # 电荷源
#  	type = CoupledSusceptibilityTimeDerivative
#    variable = c
#    v = eta
#    f_name = ft
#  [../]

  #
  # Allen-Cahn Equation
  #
  [./ACBulkF]
    type = KKSACBulkF
    variable = eta
    fa_name  = fl
    fb_name  = fs
    w        = 1.0
    coupled_variables = 'cl cs eta c'
  [../]
  [./ACBulkC]
    type = KKSACBulkC
    variable = eta
    ca       = cl
    cb       = cs
    fa_name  = fl
    coupled_variables = 'eta c'
  [../]
  [./ACInterface]
    type = ACInterface
    variable = eta
    kappa_name = eps_sq
  [../]
  [./BV]
    type = Kinetics
    variable = eta
    f_name = f_bv
    cp = pot
    cv = eta
  [../]
  [./Noiseeta2]
    type = LangevinNoise
    variable = eta
    amplitude = 0.002
  [../]
  [./detadt]
    type = TimeDerivative
    variable = eta
  [../]
  
  # evolution of pot ▽(σ▽φ)
  [./Cond]
    type = MatDiffusion
    variable = pot
    diffusivity = ElecEff
  [../]
  # -nFcs(deta/dt)
  [./coupledSource]
    type = CoupledSusceptibilityTimeDerivative
    variable = pot
    v = eta
    f_name = ChargeEff
  [../]
[]

[AuxKernels]
  [./GlobalFreeEnergy]
    variable = Fglobal
    type = KKSGlobalFreeEnergy
    fa_name = fl
    fb_name = fs
    w = 1.0
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'PJFNK'

  petsc_options_iname = '-pc_type -sub_pc_type -sub_pc_factor_shift_type'
  petsc_options_value = 'asm      ilu          nonzero'

  l_max_its = 100
  nl_max_its = 100

  l_tol = 1.0e-4
  nl_rel_tol = 1.0e-7
  nl_abs_tol = 1.0e-8
  
  num_steps = 200
  dt = 10	
[]

#
# Precondition using handcoded off-diagonal terms
#
[Preconditioning]
  [./full]
    type = SMP
    full = true
  [../]
[]


[VectorPostprocessors]
  [./c]
    type =  LineValueSampler
    start_point = '0 0 0'
    end_point = '30 0 0'
    variable = c
    num_points = 60
    sort_by =  id
    execute_on = timestep_end
  [../]
  [./eta]
    type =  LineValueSampler
    start_point = '0 0 0'
    end_point = '30 0 0'
    variable = eta
    num_points = 60
    sort_by =  id
    execute_on = timestep_end
  [../]
[]

[Postprocessors]
  [./area_metal-electrode_eta]
    type = ElementIntegralMaterialProperty
    mat_prop = h
    execute_on = 'Initial TIMESTEP_END'
  [../]
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

[Outputs]
  exodus = true
  [./csv]
    type = CSV
    execute_on = final
  [../]
[]
