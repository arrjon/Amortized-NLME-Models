module SimulatorSmallClairon

using SciMLBase
using DifferentialEquations
using ModelingToolkit

# Create parameter and variable symbolics
# (They are of type `Num`)
@parameters fM2, fM3, theta, deltaAb, deltaV, deltaS
@variables t, S(t), AB(t), vaccineCounter(t), tDose(t), fMk(t)

# From `ModelingToolkit`: differential operator
D = Differential(t)

# ODEs
eqs = [
    D(S) ~ fMk * exp(- deltaV * (t - tDose)) - deltaS * S,
    D(AB) ~ theta * S - deltaAb * AB,
    D(vaccineCounter) ~ 0,
    D(tDose) ~ 0,
    D(fMk) ~ 0
]
# Set of variables defined at `@variables`
vars = [S, AB, vaccineCounter, tDose, fMk]::Vector{Num}
# Set of parameters defined at `@parameters`
pars = [fM2, fM3, theta, deltaAb, deltaV, deltaS]::Vector{Num}

@named sys = ODESystem(eqs, t, vars, pars)
sys = structural_simplify(sys)

function simulateSmallClairon(
    parameters::Vector{Float64},
    x0s::Vector{Float64},
    dose_amount::Float64,
    dosetimes::Vector{Float64},
    t_measurements::Vector{Float64}
)
    # Mapping from model parameter names to parameters for simulation
    parmap = [
        fM2 => parameters[1],
        fM3 => parameters[2],
        theta => parameters[3],
        deltaAb => parameters[4],
        deltaV => parameters[5],
        deltaS => parameters[6],
    ]::Vector{Pair{Num, Float64}}

    # Mapping from state variable names to initial values
    x0 = [
        S => x0s[1],
        AB => x0s[2],
        vaccineCounter => x0s[3],
        tDose => x0s[4],
        fMk => x0s[5]
    ]::Vector{Pair{Num, Float64}}

    # At the time of `dosetimes` we'll add `dose_amount` to the first state
    # variable, which in this case should be `V`
    affectCounter!(integrator) = integrator.u[3] += dose_amount
    cbCounter = PresetTimeCallback(dosetimes, affectCounter!)
    
    # Set `tDose` to the dosetime
    affectTDose!(integrator) = integrator.u[4] = integrator.t
    cbTDose = PresetTimeCallback(dosetimes, affectTDose!)

    # Make specific callback
    cbFm2Condition(u, t, integrator) = (t ∈ dosetimes) && u[3] == 2
    affectFm2!(integrator) = integrator.u[5] = parameters[1]
    cbFm2 = DiscreteCallback(cbFm2Condition, affectFm2!)
    cbFm3Condition(u, t, integrator) = (t ∈ dosetimes) && u[3] == 3
    affectFm3!(integrator) = integrator.u[5] = parameters[2]
    cbFm3 = DiscreteCallback(cbFm3Condition, affectFm3!)

    # Create callbackset
    cbs = CallbackSet(cbCounter, cbTDose, cbFm2, cbFm3)

    # Set timespan to solve the ODE
    tspan = (0, t_measurements[end])

    # Define ODE problem
    problem = ODEProblem(sys, x0, tspan, parmap)
    # Solve using `alg` algorithm with addition of callbacks
    solver = solve(
        problem,
        alg = Tsit5(),
        tstops = vcat(t_measurements, dosetimes),
        callback = cbs;
        verbose = false
    )

    # Return observed
    return hcat(solver(t_measurements).u...)[2, :]
end

# Export function
export simulateSmallClairon

end