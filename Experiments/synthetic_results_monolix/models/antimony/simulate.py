import tellurium as te
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("TkAgg")

model = te.loada("froehlich_detailed.ant")

# Include following selection for simulation
selection = ["time"] + ["y", "[p]"]

rdata = model.simulate(0, 400, 1000, selections=selection)


fig, ax = plt.subplots()
x = rdata['time']
y = rdata[:,1:]
ax.plot(x,y)
ax.set_yscale(matplotlib.scale.SymmetricalLogScale(ax.yaxis))

model.plot()
