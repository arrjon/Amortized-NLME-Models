module SimulatorFroehlich

using SciMLBase
using DifferentialEquations
using ModelingToolkit

# Create parameter and variable symbolics
@parameters 𝛿_1m_0, 𝛿_2, e_0m_0, k_2m_0scale, k_2, k_1m_0, r_0m_0, 𝛾
@variables t, m(t), e(t), r(t), pro(t), start(t)

# From `ModelingToolkit`: differential operator
D = Differential(t)

# ODEs
eqs = [
    D(m) ~ start * (-𝛿_1m_0 * m * e - k_1m_0 * m * r + k_2 * (r_0m_0 - r)),
    D(e) ~ start * (-𝛿_1m_0 * m * e + 𝛿_2 * (e_0m_0 - e)),  # wrong in Fröhlich Supplement, but correct in implementation
    D(r) ~ start * (k_2 * (r_0m_0 - r) - k_1m_0 * m * r),
    D(pro) ~ start * (k_2m_0scale * (r_0m_0 - r) - 𝛾 * pro),
    D(start) ~ 0
]

# Set of variables defined at `@variables`
vars = [m, e, r, pro, start]::Vector{Num}
# Set of parameters defined at `@parameters`
pars = [𝛿_1m_0, 𝛿_2, e_0m_0, k_2m_0scale, k_2, k_1m_0, r_0m_0, 𝛾]::Vector{Num}

@named sys = ODESystem(eqs, t, vars, pars)
sys = structural_simplify(sys)


function simulateLargeModel(
    t_0::Float64,
    delta1_m0::Float64,
    delta2::Float64,
    e0_m0::Float64,
    k2_m0_scale::Float64,
    k2::Float64,
    k1_m0::Float64,
    r0_m0::Float64,
    gamma::Float64,
    offset::Float64
)
    x0 = [
        m => 1.0,
        e => e0_m0,
        r => r0_m0,
        pro => 0.0
    ]::Vector{Pair{Num, Float64}}

    p_var = [
        𝛿_1m_0 => delta1_m0,
        𝛿_2 => delta2,
        e_0m_0 => e0_m0,
        k_2m_0scale => k2_m0_scale,
        k_2 => k2,
        k_1m_0 => k1_m0,
        r_0m_0 => r0_m0,
        𝛾 => gamma,
        start => 0.0
    ]::Vector{Pair{Num, Float64}}

    affectStart!(integrator) = integrator.u[5] = 1.0
    cb = PresetTimeCallback(t_0, affectStart!)

    # solve ODE from t0 with stiff solver
    tspan = (0.0, 30.0)
    t_eval = 1/6:1/6:30
    prob = ODEProblem(sys, x0, tspan, p_var)
    sol = solve(
        prob,
        alg=Rodas5P(),
        saveat=t_eval,
        callback=cb;
        verbose=false)

    # apply measurement function
    p = hcat(sol(t_eval).u...)[4, :] .+ offset
    y = log.(p)

    return y
end

export simulateLargeModel

end # module SimulatorFroehlich
