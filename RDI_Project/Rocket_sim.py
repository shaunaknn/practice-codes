import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Gravity model
g0 = 9.81              # Sea-level gravity (m/s^2)
R_earth = 6371000      # Earth radius (m)

def gravity(h): # Newton's law of gravity
    return g0 * (R_earth / (R_earth + h))**2

# Basic atmosphere model
#def air_density(h):
#    return rho0 * np.exp(-h / H)

### ISA Atmosphere model

# General Constants
gamma = 1.4            # Adiabatic constant for air
R_air = 287.05         # Gas constant (J/kgK)

# Troposhphere
L_tr = -0.0065         # Temperature lapse rate in troposphere (K/m)
T0 = 288.15            # Sea-level Temperature
rho0 = 1.225           # Sea-level density (kg/m^3) 
p0 = 101325            # Sea-level pressure (Pa)

# Lower Stratorsphere - 11 - 20 km
T11 = 216.65           # Temperature at 11 km (constant)
p11 = 22632            # Pressure at 11 km
rho11 = 0.364          # Density at 11 km (kg/m^3)

# Upper Stratosphere - 20 - 32 km
T20 = 216.65           # Temperature at 20 km
L_ustr = 0.001         # Temperature lapse rate in upper stratosphere
p20 = 5474.88          # Pressure at 20 km
rho20 = 0.088          # Density at 20 km (kg/m^3)

def isa_atmosphere(h):
    '''
    returns T(h), rho(h) and p(h) at given height
    '''
    if h < 11000:  # Troposphere
        T = T0 + L_tr * h
        p = p0 * (T / T0)**(-g0 / (L_tr * R_air)) # polytropic relation
    elif h >= 11000 and h < 20000:  # Lower Stratosphere
        T = T11
        p = p11 * np.exp(-g0 * (h - 11000) / (R_air * T))
    elif h >= 20000 and h < 32000:
        T = T20 + L_ustr*(h - 20000)
        p = p20 * (T / T20)**(-g0/ (L_ustr* R_air))
    else: # exponential extension above 32km
        T = T20 + L_ustr*(32000 - 20000)
        p = p20 * (T / T20)**(-g0/ (L_ustr* R_air)) * np.exp(-g0 * (h - 32000) / (R_air * T))
    
    rho = p / (R_air * T)
    return T, p, rho

def drag_coefficient(M):
    """
    Simple compressible Cd model:
    captures subsonic, transonic drag rise, and supersonic behavior
    """
    if M < 0.8: # subsonic
        return 0.5
    elif M < 1.2: # transonic drag rise
        return 0.5 + 0.4 * (M - 0.8) / 0.4
    elif M < 5: # supersonic decay
        return 0.9 - 0.3 * (M - 1.2) / 3.8
    else:
        return 0.6

#H = 8500               # Scale height (m)

# Aerodynamic parameters
#Cd = 0.5               # Drag coefficient
A = 0.1                # Cross-sectional area (m^2)

# Rocket parameters
Th = 15000              # Thrust (N)
Isp = 300              # Specific impulse (s)

m0 = 500               # Initial mass (kg)
mf = 200               # Final mass (kg)

mdot = Th / (Isp * g0)  # Mass flow rate (kg/s)
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

# ODE System

def rocket_ode(t, y):
    x, z, vx, vz, m = y

    v = np.sqrt(vx**2 + vz**2) + 1e-6  # avoid division by zero
    h = max(z, 0)

    # Forces
    #rho = air_density(h)
    T_atm, p_atm, rho = isa_atmosphere(h) # set temp, pressure and density from ISA

    a = np.sqrt(gamma*R_air*T_atm)

    Ma = max(v / a, 1e-3)

    #q = 0.5 * rho * v**2
    #q_vals = []
    #q_vals.append(q)

    Cd = drag_coefficient(Ma)

    D = 0.5 * rho * Cd * A * v**2

    g = gravity(h)

    # Thrust only during burn
    if t <= burn_time and m > mf:
        thrust = Th
        dm_dt = -mdot
    else:
        thrust = 0
        dm_dt = 0

    theta = pitch_program(t)

    Tx = thrust * np.cos(theta)
    Tz = thrust * np.sin(theta)

    if v > 1e-3: # to ensure drag direction isn't messed up at low velocities
        Dx = D * (vx / v)
        Dz = D * (vz / v)
    else:
        Dx, Dz = 0.0, 0.0
    
    #Dx = D * (vx / v)
    #Dz = D * (vz / v)

    m = max(m,1e-3)

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

# Compute atmosphere properties along trajectory
rho_profile = np.array([isa_atmosphere(max(zi, 0))[2] for zi in z])

# Dynamic pressure
q = 0.5 * rho_profile * v**2

q_max = np.max(q)
idx_max_q = np.argmax(q)

t_max_q = t[idx_max_q]
z_max_q = z[idx_max_q]
v_max_q = v[idx_max_q]

print(f"Max-Q: {q_max:.2f} Pa at t = {t_max_q:.2f} s, altitude = {z_max_q/1000:.2f} km")

# Plotting

plt.figure()
plt.plot(t, q)
plt.axvline(t_max_q, linestyle='--', label='Max-Q')
plt.xlabel("Time (s)")
plt.ylabel("Dynamic Pressure (Pa)")
plt.title("Dynamic Pressure vs Time")
plt.legend()
plt.grid()

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

T_profile = np.array([isa_atmosphere(max(zi, 0))[0] for zi in z])
a_profile = np.sqrt(gamma * R_air * T_profile)
Mach = v / a_profile

plt.figure()
plt.plot(t, Mach)
plt.xlabel("Time (s)")
plt.ylabel("Mach Number")
plt.title("Mach vs Time")
plt.grid()

Cd_vals = np.array([drag_coefficient(Mi) for Mi in Mach])

plt.figure()
plt.plot(t, Cd_vals)
plt.xlabel("Time (s)")
plt.ylabel("Cd")
plt.title("Drag Coefficient vs Time")
plt.grid()

plt.show()