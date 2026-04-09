import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants

g0 = 9.81              # Sea-level gravity (m/s^2)
R_earth = 6371000      # Earth radius (m)
rho0 = 1.225           # Sea-level density (kg/m^3)
H = 8500               # Scale height (m)

Cd = 0.5               # Drag coefficient
A = 0.1                # Cross-sectional area (m^2)

# Rocket parameters
T = 15000              # Thrust (N)
Isp = 300              # Specific impulse (s)

m0 = 500               # Initial mass (kg)
mf = 200               # Final mass (kg)

mdot = T / (Isp * g0)  # Mass flow rate (kg/s)
burn_time = (m0 - mf) / mdot

# Pitch Program

def pitch_program(t):
    """
    Simple gravity turn:
    - Start vertical
    - Gradually tilt over time
    """
    if t < 10:
        return np.deg2rad(90)  # vertical
    elif t < 50:
        return np.deg2rad(90 - 0.8 * (t - 10))  # gradual tilt
    else:
        return np.deg2rad(50)  # near horizontal



# Atmosphere Model

def air_density(h):
    return rho0 * np.exp(-h / H)

# Gravity Model

def gravity(h):
    return g0 * (R_earth / (R_earth + h))**2

# ODE System

def rocket_ode(t, y):
    x, z, vx, vz, m = y

    v = np.sqrt(vx**2 + vz**2) + 1e-6  # avoid division by zero
    h = max(z, 0)

    # Forces
    rho = air_density(h)
    D = 0.5 * rho * Cd * A * v**2

    g = gravity(h)

    # Thrust only during burn
    if t <= burn_time and m > mf:
        thrust = T
        dm_dt = -mdot
    else:
        thrust = 0
        dm_dt = 0

    theta = pitch_program(t)

    Tx = thrust * np.cos(theta)
    Tz = thrust * np.sin(theta)

    Dx = D * (vx / v)
    Dz = D * (vz / v)

    # Equations of motion
    dvx_dt = (Tx - Dx) / m
    dvz_dt = (Tz - Dz) / m - g

    return [vx, vz, dvx_dt, dvz_dt, dm_dt]

# Initial Conditions

x0 = 0
z0 = 0
vx0 = 0
vz0 = 0

y0 = [x0, z0, vx0, vz0, m0]

# Time span
t_span = (0, 200)
t_eval = np.linspace(0, 200, 1000)

# Solve ODE

sol = solve_ivp(rocket_ode, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-9)

# Extract results
t = sol.t
x = sol.y[0]
z = sol.y[1]
vx = sol.y[2]
vz = sol.y[3]
m = sol.y[4]

v = np.sqrt(vx**2 + vz**2)

# Plotting

plt.figure()
plt.plot(x/1000, z/1000)
plt.xlabel("Downrange Distance (km)")
plt.ylabel("Altitude (km)")
plt.title("Rocket Trajectory")
plt.grid()

plt.figure()
plt.plot(t, z/1000)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (km)")
plt.title("Altitude vs Time")
plt.grid()

plt.figure()
plt.plot(t, v)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs Time")
plt.grid()

plt.figure()
plt.plot(t, m)
plt.xlabel("Time (s)")
plt.ylabel("Mass (kg)")
plt.title("Mass vs Time")
plt.grid()

plt.show()