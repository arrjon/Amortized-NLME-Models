module SimulatorPharma

using SciMLBase
using DifferentialEquations
using ModelingToolkit

@parameters t, θ_0, θ_1, θ_2, θ_3, θ_4, θ_5, θ_6, θ_7, θ_8, θ_9, η_0, η_1, η_2, η_3, wt
@variables A1(t), A2(t), A3(t), A4(t), A5(t)
D = Differential(t)

ASCL = (wt / 70) ^ 0.75
ASV = wt / 70
k_a = θ_0
V_2 = θ_1 * ASV * η_0
Q_H = 80 * ASCL
CLP = θ_2 * ASCL * η_2
CLM = θ_3 * ASCL
V_3 = θ_4 * ASV * η_1
Q_34 = θ_5 * ASCL
V_4 = θ_6 * ASV
f_m = 0.21 * η_3
Q_25 = θ_7 * ASCL
V_5 = 588 * ASV
CLIV = (k_a * A1 + Q_H / V_2 * A2) / (Q_H + CLP)

eqs = [
    D(A1) ~ -k_a * A1,
    D(A2) ~ Q_H * CLIV - Q_H / V_2 * A2 - Q_25 / V_2 * A2 + Q_25 / V_5 * A5,
    D(A3) ~ f_m * CLP * CLIV - CLM / V_3 * A3 - Q_34 / V_3 * A3 + Q_34 / V_4 * A4,
    D(A4) ~ Q_34 / V_3 * A3 - Q_34 / V_4 * A4,
    D(A5) ~ Q_25 / V_2 * A2 - Q_25 / V_5 * A5
]

@named sys = ODESystem(eqs, t,
    [A1, A2, A3, A4, A5],
    [θ_0, θ_1, θ_2, θ_3, θ_4, θ_5, θ_6, θ_7, θ_8, θ_9, η_0, η_1, η_2, η_3, wt])
sys = structural_simplify(sys)


function simulatePharma(
        theta_0, theta_1, theta_2, theta_3, theta_4,
        theta_5, theta_6, theta_7, theta_8, theta_9,
        eta_0, eta_1, eta_2, eta_3,
        wt_indv, DOS, dosetimes, t_measurement)

    p_var = [
        θ_0 => theta_0,
        θ_1 => theta_1,
        θ_2 => theta_2,
        θ_3 => theta_3,
        θ_4 => theta_4,
        θ_5 => theta_5,
        θ_6 => theta_6,
        θ_7 => theta_7,
        θ_8 => theta_8,
        θ_9 => theta_9,
        η_0 => eta_0,
        η_1 => eta_1,
        η_2 => eta_2,
        η_3 => eta_3,
        wt => wt_indv
    ]

    x0 = [
        A1 => 0.0,
        A2 => 0.0,
        A3 => 0.0,
        A4 => 0.0,
        A5 => 0.0
    ]

    affect!(integrator) = integrator.u[1] += DOS
    cb = PresetTimeCallback(dosetimes, affect!)
    tspan = (0, t_measurement[end])

    prob = ODEProblem(sys, x0, tspan, p_var)
    sol = solve(prob, Tsit5(), tstops=t_measurement, callback=cb; verbose=false)

    observed = hcat(sol(t_measurement).u...)[2:3, :]
    return observed
end

export simulatePharma

end # module SimulatorPharma
