module SimulatorFroehlich

using SciMLBase
using DifferentialEquations
using ModelingToolkit

# Create parameter and variable symbolics
@parameters 𝛿_1m_0, 𝛿_2, e_0m_0, k_2m_0scale, k_2, k_1m_0, r_0m_0, 𝛾
@variables t, m(t), e(t), r(t), pro(t)

# From `ModelingToolkit`: differential operator
D = Differential(t)

# ODEs
eqs = [
    D(m) ~ -𝛿_1m_0 * m * e - k_1m_0 * m * r + k_2 * (r_0m_0 - r),
    D(e) ~ 𝛿_1m_0 * m * e - 𝛿_2 * (e_0m_0 - e),
    D(r) ~ k_2 * (r_0m_0 - r) - k_1m_0 * m * r,
    D(pro) ~ k_2m_0scale * (r_0m_0 - r) - 𝛾 * pro
]

# Set of variables defined at `@variables`
vars = [m, e, r, pro]::Vector{Num}
# Set of parameters defined at `@parameters`
pars = [𝛿_1m_0, 𝛿_2, e_0m_0, k_2m_0scale, k_2, k_1m_0, r_0m_0, 𝛾]::Vector{Num}

@named sys = ODESystem(eqs, t, vars, pars)
sys = structural_simplify(sys)

function find_first_above_t0(t0::Float64)
    idx = 1
    for t in 1/6:1/6:30
        if t > t0
            return t, idx
        end
        idx += 1
    end
    return 30, 180
end


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

    simulations = fill(log(offset), 180)

    if t_0 >= 30
        # no simulation needed
        return simulations
    end

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
        𝛾 => gamma
    ]::Vector{Pair{Num, Float64}}

    # find right time spots, evaluate at (1/6:1/6:30) but t0 might be > 1/6
    t_first, t_idx = find_first_above_t0(t_0)
    t_eval = range.(t_first, 30.0, step=1/6)

    # solve ODE from t0 with stiff solver
    tspan = (t_0, 30.0)
    prob = ODEProblem(sys, x0, tspan, p_var)
    sol = solve(
        prob,
        alg=Rodas5P(),
        saveat=t_eval,
        verbose=false)

    # check if any simulation happened
    if t_first > sol.t[end]
        return simulations
    end

    # use saved time points to evaluate, but solver might stopped earlier so only up to sol.t[end]
    t_eval_sol = range.(t_first, sol.t[end], step=1/6)

    # apply measurement function
    p = hcat(sol(t_eval_sol).u...)[4, :] .+ offset
    # some simulations will oscillate around 0, and if offset is sampled small, might cause problems
    p[p.<0] .= 1e-12
    y = log.(p)

    # save simulations after t0 until solver finished
    if sol.t[end] < 30
        # if solver aborted, fill in the last value until the end
        t_idx_end = t_idx+size(t_eval_sol)[1]-1
        simulations[t_idx:t_idx_end] = y  # simulations
        simulations[t_idx_end:end] .= y[end]  # fill in last value
    else
        # if solver finished, fill in all simulations
        simulations[t_idx:end] = y
    end

    return simulations
end

export simulateLargeModel

end # module SimulatorFroehlich
