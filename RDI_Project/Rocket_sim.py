import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Gravity model
g0 = 9.81              # Sea-level gravity (m/s^2)
R_earth = 6371000      # Earth radius (m)

def gravity(h): # Newton's law of gravity
    return g0 * (R_earth / (R_earth + h))**2

# only to be used if using dynamic throttling
g_limit = 4.0          # maximum g force tolerated by rocket
a_max = g_limit*g0     # max acceleration tolerable

q_limit = 50000 # Maximum allowed dynamic pressure in Pa

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
#rho11 = 0.364          # Density at 11 km (kg/m^3)

# Upper Stratosphere - 20 - 32 km
T20 = 216.65           # Temperature at 20 km
L_ustr = 0.001         # Temperature lapse rate in upper stratosphere
p20 = 5474.88          # Pressure at 20 km
#rho20 = 0.088          # Density at 20 km (kg/m^3)

def isa_atmosphere(h):
    '''
    returns Temperature, Pressure and Density at a given height
    '''
    if h < 11000:  # Troposphere
        T = T0 + L_tr * h
        p = p0 * (T / T0)**(-g0 / (L_tr * R_air)) # polytropic relation
    elif h < 20000:  # Lower Stratosphere
        T = T11
        p = p11 * np.exp(-g0 * (h - 11000) / (R_air * T))
    elif h < 32000:
        T = T20 + L_ustr*(h - 20000)
        p = p20 * (T / T20)**(-g0/ (L_ustr* R_air))
    elif h < 80000:
        T = T20 + L_ustr*(32000 - 20000)
        p = p20 * (T / T20)**(-g0/ (L_ustr* R_air)) * np.exp(-g0 * (h - 32000) / (R_air * T))
    else:
        T = 200.0
        p = 1e-9

    rho = p / (R_air * T)
    return T, p, rho

# Drag coefficient
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

# Geometric parameters
A = 0.1                 # Cross-sectional area (m^2)

# Rocket parameters
Th = 15000              # Thrust (N)
Isp = 300               # Specific impulse (s)

m0 = 500                # Initial mass (kg)
mf = 200                # Final mass (kg)

mdot = Th / (Isp * g0)  # Mass flow rate (kg/s)
burn_time = (m0 - mf) / mdot # Burning time (s)

# Rocket ODE System
def rocket_ode(t, y):
    x, z, vx, vz, m = y
    m = max(m,1e-3)

    v = max(np.sqrt(vx**2 + vz**2),1e-6)  # avoid division by zero
    h = max(z, 0)

    T_atm, p_atm, rho = isa_atmosphere(h) # set temp, pressure and density from ISA

    a = np.sqrt(gamma*R_air*T_atm)

    Ma = v / (a + 1e-9)

    q = 0.5 * rho * v**2

    Cd = drag_coefficient(Ma)

    D = 0.5 * rho * Cd * A * v**2

    g = gravity(h)
    
    # Simple pitch program (time-based)
    if t < 10:
        theta = np.deg2rad(90)
    elif t < 100:
        frac = (t - 10) / 90
        theta = np.deg2rad(90 - 70*frac)   # down to 20 deg slowly
    else:
        theta = np.deg2rad(20)
    
    ux = np.cos(theta)
    uz = np.sin(theta)

    # nominal thrust
    if t <= burn_time and m > mf: # can only thrust if fuel exists
        thrust_nominal = Th
        mdot_nominal = mdot
    else:
        thrust_nominal = 0.0
        mdot_nominal = 0.0

    if v > 1e-6:
        Dx = D * vx / v
        Dz = D * vz / v
    else:
        Dx, Dz = 0.0, 0.0

    # simple scaled throttle
    throttle = min(1.0, (q_limit / (q + 1e-6))**0.5)
    
    thrust = thrust_nominal*throttle
    dm_dt = -mdot_nominal*throttle

    Tx = thrust * ux
    Tz = thrust * uz

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
t_span = (0, 250)
t_eval = np.linspace(0, 200, 1000)

# Solve ODE

def hit_ground(t, y):
    return y[1]

hit_ground.terminal = True
hit_ground.direction = -1

sol = solve_ivp(
    rocket_ode, t_span, y0,
    t_eval=t_eval,
    events = hit_ground,
    rtol=1e-6, atol=1e-9
    )

print(sol.message)

# Extract results
t = sol.t
x = sol.y[0]
z = sol.y[1]
vx = sol.y[2]
vz = sol.y[3]
m = sol.y[4]

v = np.sqrt(vx**2 + vz**2)

# Compute atmosphere properties along trajectory
atm = np.array([isa_atmosphere(max(zi, 0)) for zi in z])
T_profile = atm[:,0]
rho_profile = atm[:,2]

# Dynamic pressure
q = 0.5 * rho_profile * v**2

q_max = np.max(q)
idx_max_q = np.argmax(q)

t_max_q = t[idx_max_q]
z_max_q = z[idx_max_q]
v_max_q = v[idx_max_q]

print(f"Max-Q: {q_max:.2f} Pa at t = {t_max_q:.2f} s, altitude = {z_max_q/1000:.2f} km")

# Maximum altitude (apogee)
idx_max_alt = np.argmax(z)
z_max = z[idx_max_alt]
t_max_alt = t[idx_max_alt]

print(f"Max altitude: {z_max/1000:.2f} km at t = {t_max_alt:.2f} s")

# Burn time
print(f"Burnout time: {burn_time:.2f} s")

# Plotting

# Dynamic Pressure vs time plot
plt.figure()
plt.plot(t, q)
plt.axvline(burn_time, linestyle=':', label='burnout')
plt.xlabel("Time (s)")
plt.ylabel("Dynamic Pressure (Pa)")
plt.title("Dynamic Pressure vs Time")
plt.scatter(t_max_q, q_max)
plt.legend()
plt.grid()

# Rocket trajectory plot
plt.figure()
plt.plot(x/1000, z/1000)
plt.xlabel("Downrange Distance (km)")
plt.ylabel("Altitude (km)")
plt.title("Rocket Trajectory")
plt.grid()

# Rocket altitude plot
plt.figure()
plt.plot(t, z/1000)
plt.axvline(burn_time, linestyle=':', label='burnout')
plt.scatter(t_max_alt, z_max/1000)
plt.axvline(t_max_alt, linestyle='--', label='apogee')
plt.axhline(z_max/1000, linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Altitude (km)")
plt.title("Altitude vs Time")
plt.legend()
plt.grid()

# Rocket velocity vs time plot
plt.figure()
plt.plot(t, v)
plt.axvline(burn_time, linestyle=':', label='burnout')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs Time")
plt.legend()
plt.grid()

# Rocket mass vs time plot
plt.figure()
plt.plot(t, m)
plt.axvline(burn_time, linestyle=':', label='burnout')
plt.xlabel("Time (s)")
plt.ylabel("Mass (kg)")
plt.title("Mass vs Time")
plt.legend()
plt.grid()

# Ma and Cd plots
a_profile = np.sqrt(gamma * R_air * T_profile)
Mach = v / (a_profile + 1e-9)
Cd_vals = np.array([drag_coefficient(Mi) for Mi in Mach])

# Mach vs Time
plt.figure()
plt.plot(t, Mach)
plt.axvline(burn_time, linestyle=':', label='burnout')
plt.xlabel("Time (s)")
plt.ylabel("Mach")
plt.title("Mach vs Time")
plt.legend()
plt.grid()

# Cd vs Time
plt.figure()
plt.plot(t, Cd_vals)
plt.axvline(burn_time, linestyle=':', label='burnout')
plt.xlabel("Time (s)")
plt.ylabel("Cd")
plt.title("Drag Coefficient vs Time")
plt.legend()
plt.grid()

plt.show()