module SimulatorFroehlich

using SciMLBase
using DifferentialEquations
using ModelingToolkit

@parameters t 𝛿_1m_0 𝛿_2 e_0m_0 k_2m_0scale k_2 k_1m_0 r_0m_0 𝛾
@variables m(t) e(t) r(t) pro(t)
D = Differential(t)

eqs = [D(m) ~ -𝛿_1m_0 * m * e - k_1m_0 * m * r + k_2 * (r_0m_0 - r),
    D(e) ~ 𝛿_1m_0 * m * e - 𝛿_2 * (e_0m_0 - e),
    D(r) ~ k_2 * (r_0m_0 - r) - k_1m_0 * m * r,
    D(pro) ~ k_2m_0scale * (r_0m_0 - r) - 𝛾 * pro]

@named sys = ODESystem(eqs)
sys = structural_simplify(sys)

function find_first_above_t0(t0)
    idx = 1
    for t in 1/6:1/6:30
        if t > t0
            return t, idx
        end
        idx += 1
    end
    return 30, 180
end


function simulateLargeModel(t_0, delta1_m0, delta2, e0_m0, k2_m0_scale, k2, k1_m0, r0_m0, gamma, offset, sigma)

    simulations = Array{Float64}(undef, 180)

    if t_0 >= 30
        # no simulation needed
        simulations .= log(offset)
        return simulations
    end

    x0 = [m => 1.0,
        e => e0_m0,
        r => r0_m0,
        pro => 0.0]

    p_var = [𝛿_1m_0 => delta1_m0,
        𝛿_2 => delta2,
        e_0m_0 => e0_m0,
        k_2m_0scale => k2_m0_scale,
        k_2 => k2,
        k_1m_0 => k1_m0,
        r_0m_0 => r0_m0,
        𝛾 => gamma]

    # solve ODE from t0 with stiff solver
    tspan = (t_0, 30.0)
    prob = ODEProblem(sys, x0, tspan, p_var)
    sol = solve(prob, Rodas5P(), verbose=false)

    # find right time spots, evaluate at (1/6:1/6:30) but t0 might be > 1/6
    t0_sol, t_idx = find_first_above_t0(t_0)

    # check if simulation was measurable
    if t0_sol > sol.t[end]
        simulations .= log(offset)
        return simulations
    end

    t_eval = range.(t0_sol, sol.t[end], step=1/6)

    # apply measurement function
    p = hcat(sol(t_eval).u...)[4, :] .+ offset
    # some simulations will oscillate around 0, and if offset is sampled small, might cause problems
    p[p.<0] .= 1e-12
    y = log.(p)

    # save simulations after t0 until solver finished
    if sol.t[end] < 30
        t_idx_end = t_idx+size(t_eval)[1]-1
        simulations[t_idx:t_idx_end] = y
        # if solver aborted, fill in the last value until the end
        simulations[t_idx_end:end] .= y[end]
    else
        simulations[t_idx:end] = y
    end

    # fill offset in before t0
    simulations[begin:t_idx] .= log(offset)

    return simulations
end

export simulateLargeModel

end # module SimulatorFroehlich
