
import rebound
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.cm as cm

def GetInput(prompt, default=None, type_func=float):
    inp = input(f"{prompt} [default: {default}]: ")
    if inp == "":
        return default
    try:
        return type_func(inp)
    except ValueError:
        print("Invalid input, using default.")
        return default


sim = rebound.Simulation()
sim.integrator = "ias15"  


star_mass = GetInput("Enter stellar mass (M_sun)", default=1.0)
sim.add(m=star_mass)


num_planets = int(GetInput("Enter number of planets", default=3, type_func=int))
planet_indices = []  
for i in range(num_planets):
    print(f"\nPlanet {i+1}")
    m = GetInput("Planet mass (M_sun)", default=0.001)
    a = GetInput("Semi-major axis (AU)", default=1.0 + i*0.5)
    e = GetInput("Eccentricity (0-1)", default=0.05)
    f = GetInput("True anomaly (rad)", default=np.random.uniform(0, 2*np.pi))
    
    sim.add(m=m, a=a, e=e, f=f)
    planet_indices.append(len(sim.particles)-1) 

sim.move_to_com()


plt.ion()
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.axis('on')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

op = rebound.OrbitPlot(sim, fig=fig, ax=ax, color=True, periastron=True, particles=planet_indices)

# Colormap
cmap = cm.get_cmap("RdYlGn") 


def ComputeStability(sim):
    
    stabilities = []
    for idx in planet_indices:
        p = sim.particles[idx]
        stability = max(0, min(1, 1 - p.e * 5))  
        stabilities.append(stability)
    return stabilities

# This is for Animation 
n_steps = 3000
dt = 0.02
pause = 0.01


for i in range(n_steps):
    sim.integrate(sim.t + dt)
    
    stability = ComputeStability(sim)
    for idx, s in enumerate(stability):
        op.orbits[idx].set_color(cmap(s))
    
    op.update(updateLimits=False)  
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(pause)

plt.ioff()
plt.show()


