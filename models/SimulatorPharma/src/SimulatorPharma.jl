module SimulatorPharma

using SciMLBase
using DifferentialEquations
using ModelingToolkit

# Create parameter and variable symbolics
@parameters θ_1, θ_2_η_1, θ_4_η_3, θ_5, θ_6_η_2, θ_7, θ_8, θ_10, η_4, wt
@variables t, A1(t), A2(t), A3(t), A4(t), A5(t)

# From `ModelingToolkit`: differential operator
D = Differential(t)

ASCL = (wt / 70) ^ 0.75
ASV = wt / 70
k_a = θ_1
V_2 = θ_2_η_1 * ASV
Q_H = 80 * ASCL
CLP = θ_4_η_3 * ASCL
CLM = θ_5 * ASCL
V_3 = θ_6_η_2 * ASV
Q_34 = θ_7 * ASCL
V_4 = θ_8 * ASV
f_m = 0.21 * η_4
Q_25 = θ_10 * ASCL
V_5 = 588 * ASV
CLIV = (k_a * A1 + Q_H / V_2 * A2) / (Q_H + CLP)

eqs = [
    D(A1) ~ -k_a * A1,
    D(A2) ~ Q_H * CLIV - Q_H / V_2 * A2 - Q_25 / V_2 * A2 + Q_25 / V_5 * A5,
    D(A3) ~ f_m * CLP * CLIV - CLM / V_3 * A3 - Q_34 / V_3 * A3 + Q_34 / V_4 * A4,
    D(A4) ~ Q_34 / V_3 * A3 - Q_34 / V_4 * A4,
    D(A5) ~ Q_25 / V_2 * A2 - Q_25 / V_5 * A5
]

# Set of variables defined at `@variables`
vars = [A1, A2, A3, A4, A5]::Vector{Num}
# Set of parameters defined at `@parameters`
pars = [θ_1, θ_2_η_1, θ_4_η_3, θ_5, θ_6_η_2, θ_7, θ_8, θ_10, η_4, wt]::Vector{Num}

@named sys = ODESystem(eqs, t, vars, pars)
sys = structural_simplify(sys)

function simulatePharma(
    parameters::Vector{Float64},
    wt_indv::Float64,
    DOS::Float64,
    dosetimes::Vector{Float64},
    t_measurement::Vector{Float64}
    )

    parmap = [
        θ_1 => parameters[1],
        θ_2_η_1 => parameters[2],
        θ_4_η_3 => parameters[3],
        θ_5 => parameters[4],
        θ_6_η_2 => parameters[5],
        θ_7 => parameters[6],
        θ_8 => parameters[7],
        θ_10 => parameters[8],
        η_4 => parameters[9],
        wt => wt_indv
    ]::Vector{Pair{Num, Float64}}

    x0 = [
        A1 => 0.0,
        A2 => 0.0,
        A3 => 0.0,
        A4 => 0.0,
        A5 => 0.0
    ]::Vector{Pair{Num, Float64}}

    # At the time of `dosetimes` we'll add `DOS` to the first state
    affect!(integrator) = integrator.u[1] += DOS
    cb = PresetTimeCallback(dosetimes, affect!)

    # Set timespan to solve the ODE
    tspan = (0, t_measurement[end])

    # Define ODE problem
    problem = ODEProblem(sys, x0, tspan, parmap)  # using jac is a bit slower
    sol = solve(
        problem,
        alg=Rodas5P(),  #Tsit5 is a bit slower than Rodas5P
        tstops=dosetimes,
        saveat=t_measurement,
        callback=cb;
        verbose=false
    )

    # Return observed
    return hcat(sol(t_measurement).u...)[2:3, :]
end

# Export function
export simulatePharma

end # module SimulatorPharma