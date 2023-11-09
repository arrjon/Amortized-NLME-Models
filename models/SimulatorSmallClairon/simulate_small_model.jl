using Pkg
Pkg.activate("/home/clemens/Documents/phd/modeling/ORCHESTRA_WP5")

cd("/home/clemens/Documents/phd/modeling/ORCHESTRA_WP5/jl_models")

# Load libraries for data frames
using DataFrames
using CSV
# For sampling from distributions
using Random
using Distributions
# Plotting
using Plots

# Load model
include("SimulatorSmallClairon.jl")

# Load priors
priors = CSV.read("priors_Clairon_small.csv", DataFrame)

# Sample many parameters from priors and simulate model to see if we get any failures
nSamples = 1000
parameterIds = unique(priors.parameter)
parameterValues = zeros(nSamples, length(parameterIds))
for (i, p) in enumerate(parameterIds)
    df = priors[priors.parameter .== p, :]
    dist = df.distribution[1]
    mean = df.mean[1]
    var = df.variance[1]
    if dist == "log-normal"
        mean = log(mean)
    end
    parameterValues[:, i] = exp.(rand(Normal(mean, sqrt(var)), nSamples))
end

# Plot sampled values to see if what I'm doing is correct
histogram(log.(parameterValues[:, 6]))

# Simulate model for all different parameters
time_measurements = [0.0:1.0:400.0;]
simulations = zeros(size(parameterValues, 1), length(time_measurements))
for i in [1:size(parameterValues, 1);]
    p = parameterValues[i, :]
    simulations[i, :] = SimulatorSmallClairon.simulateSmallClairon(p, [0.0, 0.0, 0.0, 0.0, 1.0], 1.0, [2.0, 20.0, 250.0], time_measurements)
    # simulations[i, :][simulations[i, :] .< 0] .= 0
end

# simulations

indices = rand([1:size(parameterValues, 1);], 100)
p = plot(time_measurements, log10.(simulations[1,:] .+ 1.0))
for i in indices
    p = plot!(
        time_measurements,
        log10.(simulations[i,:] .+ 1.0),
        alpha = 0.5,
        color = "blue"
    )
end
# p = plot!()
p = plot!(legend = false)
display(p)