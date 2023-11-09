module SimulatorPharma

using SciMLBase
using DifferentialEquations
using ModelingToolkit

@parameters t, θ_1, θ_2_η_1, θ_4_η_3, θ_5, θ_6_η_2, θ_7, θ_8, θ_10, η_4, wt
@variables A1(t), A2(t), A3(t), A4(t), A5(t)
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

@named sys = ODESystem(eqs, t,
    [A1, A2, A3, A4, A5],
    [θ_1, θ_2_η_1, θ_4_η_3, θ_5, θ_6_η_2, θ_7, θ_8, θ_10, η_4, wt])
sys = structural_simplify(sys)


function simulatePharma(
        theta_1, theta_2_eta_1, theta_4_eta_3, theta_5, theta_6_eta_2,
        theta_7, theta_8, theta_10, theta_12, theta_13,
        eta_4,
        wt_indv, DOS, dosetimes, t_measurement)

    p_var = [
        θ_1 => theta_1,
        θ_2_η_1 => theta_2_eta_1,
        θ_4_η_3 => theta_4_eta_3,
        θ_5 => theta_5,
        θ_6_η_2 => theta_6_eta_2,
        θ_7 => theta_7,
        θ_8 => theta_8,
        θ_10 => theta_10,
        η_4 => eta_4,
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
