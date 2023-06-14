# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 03:00:53 2023

@author: roboteknologies

"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math


class PlasmaSimulation:
    def __init__(self, num_cells, box_size, num_particles, dt, num_steps):
        self.num_cells = num_cells
        self.box_size = box_size
        self.num_particles = num_particles
        self.dt = dt
        self.num_steps = num_steps

        self.particle_position = None
        self.particle_velocity = None
        self.electric_field = None
        self.particle_charge = None

        self.initialize_simulation()

    def initialize_simulation(self):
        # Initialize plasma particles
        np.random.seed(42)
        self.particle_position = np.random.uniform(0, self.box_size, size=(self.num_particles, 2))
        self.particle_velocity = np.random.normal(0, 1, size=(self.num_particles, 2))
        self.particle_charge = np.zeros(self.num_particles) - 1.0

        # Initialize electric field
        self.electric_field = np.zeros((self.num_cells, self.num_cells))

    def advance_particles(self):
        self.particle_position += self.particle_velocity * self.dt


    def deposit_charge(self):
        for i in range(self.num_particles):
            x, y = self.particle_position[i]
            cell_x = int(x * (self.num_cells - 1) / self.box_size)  # Subtract 1 from num_cells
            cell_y = int(y * (self.num_cells - 1) / self.box_size)  # Subtract 1 from num_cells
            if 0 <= cell_x < self.num_cells and 0 <= cell_y < self.num_cells:
                self.electric_field[cell_x, cell_y] += self.particle_charge[i]


    def run_simulation(self):
        for step in range(self.num_steps):
            self.advance_particles()
            self.deposit_charge()

    def plot_particle_distribution(self):
        plt.scatter(self.particle_position[:, 0], self.particle_position[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Plasma Particle Distribution')
        plt.show()


class BeamSimulation:
    def __init__(self, num_cells, box_size, num_particles, dt, num_steps):
        self.num_cells = num_cells
        self.box_size = box_size
        self.num_particles = num_particles
        self.dt = dt
        self.num_steps = num_steps

        self.particle_position = None
        self.particle_velocity = None
        self.electric_field = None
        self.particle_charge = None

        self.initialize_simulation()

    def initialize_simulation(self):
        # Initialize beam particles
        np.random.seed(42)
        self.particle_position = np.random.uniform(0, self.box_size, size=(self.num_particles, 2))
        self.particle_velocity = np.random.normal(0, 1, size=(self.num_particles, 2))
        self.particle_charge = np.zeros(self.num_particles) + 1.0

        # Initialize electric field
        self.electric_field = np.zeros((self.num_cells, self.num_cells))

    def advance_particles(self):
        self.particle_position += self.particle_velocity * self.dt

    def deposit_charge(self):
        for i in range(self.num_particles):
            x, y = self.particle_position[i]
            cell_x = int(x * (self.num_cells - 1) / self.box_size)  # Subtract 1 from num_cells
            cell_y = int(y * (self.num_cells - 1) / self.box_size)  # Subtract 1 from num_cells
            if 0 <= cell_x < self.num_cells and 0 <= cell_y < self.num_cells:
                self.electric_field[cell_x, cell_y] += self.particle_charge[i]

    def run_simulation(self):
        for step in range(self.num_steps):
            self.advance_particles()
            self.deposit_charge()

    def plot_particle_distribution(self):
        plt.scatter(self.particle_position[:, 0], self.particle_position[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Beam Particle Distribution')
        plt.show()
        
# Create a plasma simulation
plasma_sim = PlasmaSimulation(num_cells=100, box_size=10, num_particles=1000, dt=0.01, num_steps=100)

# Run the plasma simulation
plasma_sim.run_simulation()

# Plot the particle distribution
plasma_sim.plot_particle_distribution()

# Create a beam simulation
beam_sim = BeamSimulation(num_cells=100, box_size=10, num_particles=1000, dt=0.01, num_steps=100)

# Run the beam simulation
beam_sim.run_simulation()

# Plot the particle distribution
beam_sim.plot_particle_distribution()


# Define simulation parameters
dt = 0.01  # Time step
num_particles = 100  # Number of particles
num_cells = 10  # Number of cells
grid_length = 10.0  # Length of the simulation grid
beam_type = "gaussian_ellipse"  # Beam type: "square", "gaussian_ellipse", "bi-gaussian"

# Define beam parameters
beam_width = 1.0  # Width of the beam
beam_length = 2.0  # Length of the beam
beam_offset = 0.5 * grid_length  # Offset of the beam

# Define plasma parameters
plasma_density = 1.0  # Density of the homogeneous plasma

# Initialize particle positions and velocities
positions = np.zeros(num_particles)
velocities = np.zeros(num_particles)

if beam_type == "square":
    # Square beam profile initialization
    positions = np.random.uniform(beam_offset - 0.5 * beam_length, beam_offset + 0.5 * beam_length, num_particles)
    velocities = np.random.uniform(-1.0, 1.0, num_particles)

elif beam_type == "gaussian_ellipse":
    # Gaussian beam with ellipsoid charge distribution profile initialization
    positions = np.random.normal(beam_offset, beam_width, num_particles)
    velocities = np.random.normal(0.0, 1.0, num_particles)

elif beam_type == "bi-gaussian":
    # Bi-Gaussian beam profile initialization
    half_particles = num_particles // 2
    positions[:half_particles] = np.random.normal(beam_offset - beam_length, beam_width, half_particles)
    positions[half_particles:] = np.random.normal(beam_offset + beam_length, beam_width, half_particles)
    velocities = np.random.normal(0.0, 1.0, num_particles)

# Main simulation loop
for step in range(100):
    # Clear electric field on the grid
    electric_field = np.zeros(num_cells)

    # Calculate charge density from particle positions
    charge_density = np.zeros(num_cells)
    for i in range(num_particles):
        cell_index = int(positions[i] / (grid_length / num_cells))
        charge_density[cell_index] += 1.0

    # Calculate electric field on the grid using relativistic adiabatic blowout approximation
    for i in range(1, num_cells - 1):
        electric_field[i] = (charge_density[i+1] - charge_density[i-1]) / (2 * (grid_length / num_cells))

    # Update particle velocities using the electric field (relativistic)
    for i in range(num_particles):
        cell_index = int(positions[i] / (grid_length / num_cells))
        velocity_factor = np.sqrt(1 + electric_field[cell_index]**2)
        velocities[i] += electric_field[cell_index] * dt / velocity_factor

    # Update particle positions using the velocities
    positions += velocities * dt

    # Apply periodic boundary conditions
    positions = np.mod(positions, grid_length)

    # Perform other calculations or data analysis as needed

# End of the simulation loop

# Constants
c = 299792458.0  # Speed of light
q_e = -1.60217662e-19  # Electron charge
m_e = 9.10938356e-31  # Electron mass

# Calculate relativistic gamma factor for each particle
gamma = np.sqrt(1.0 + np.sum(velocities**2) / c**2)

# Calculate relativistic kinetic energy for each particle
kinetic_energy = (gamma - 1.0) * m_e * c**2

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions, velocities, kinetic_energy)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('Kinetic Energy')
plt.show()



# Simulation parameters
N = 100  # Number of particles
Lx = 1.0  # Simulation box size in x
Ly = 1.0  # Simulation box size in y
dt = 0.1 * c / Lx  # Time step
T = 1000  # Total number of time steps
gamma = 100.0  # Relativistic factor
vx0 = 0.9 * c  # Initial x-velocity of particles
vy0 = 0.0  # Initial y-velocity of particles
ax = np.zeros((N, 1))  # External x-acceleration
ay = np.zeros((N, 1))  # External y-acceleration

# Initialize particle arrays
x = np.random.uniform(0.0, Lx, N)
y = np.random.uniform(0.0, Ly, N)
vx = np.random.normal(vx0, 0.1 * vx0, N)
vy = np.random.normal(vy0, 0.1 * vx0, N)
w = np.ones(N) * q_e * N / (Lx * Ly * N)  # Particle weights

# Initialize field arrays
Ex = np.zeros((N, N))
Ey = np.zeros((N, N))

# Define plasma density profile
def n(x, y):
    return 1.0

# Define Poisson solver using FFT
def poisson_solver(rho, dx, dy):
    rho_hat = np.fft.fftn(rho)
    kx = 2 * np.pi * np.fft.fftfreq(N, dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, dy)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx ** 2 + ky ** 2
    phi_hat = -rho_hat / (k2 + 1e-16)
    phi_hat[0, 0] = 0.0
    phi = np.fft.ifftn(phi_hat)
    return phi.real

# Define main simulation loop
for t in range(T):
    # Calculate charge density
    rho = q_e * np.sum(w * (n(x, y) - 1.0))

    # Solve Poisson equation for electric potential
    phi = poisson_solver(rho, Lx / N, Ly / N)

    # Calculate electric fields
    Ex, Ey = np.gradient(phi)

    # Calculate Lorentz force
    ax = q_e * Ex / gamma
    ay = q_e * Ey / gamma

    # Update particle velocities using Lorentz force
    vx += ax[:, 0] * dt
    vy += ay[:, 0] * dt

    # Update particle positions using updated velocities
    x += vx * dt
    y += vy * dt

    # Apply periodic boundary conditions
    x = np.mod(x, Lx)
    y = np.mod(y, Ly)

# Plot particle positions
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Scatter plot of Particle Positions")
plt.show()



# Code for B𝜃 computation and plot

# Define the equation parameters
𝜎x = 1.0
𝜎y = 1.0
nb0 = 1.0
𝜎𝜉 = 1.0

# Define the function for Ez and B𝜃
def Ez(x, y, 𝜉):
    r = np.sqrt(x**2 + y**2)
    term1 = 1 / (r * 𝜎x**2 * 𝜎y**2 * 𝜎𝜉**2)
    term2 = np.exp(-x**2 / (2 * 𝜎x**2) - y**2 / (2 * 𝜎y**2) - 𝜉**2 / (2 * 𝜎𝜉**2))
    return term1 * term2

def B𝜃(x, y, 𝜉):
    r = np.sqrt(x**2 + y**2)
    term1 = 1 / (r * 𝜎x**2 * 𝜎y**2 * 𝜎𝜉**2)
    term2 = np.exp(-x**2 / (2 * 𝜎x**2) - y**2 / (2 * 𝜎y**2) - 𝜉**2 / (2 * 𝜎𝜉**2))
    term3 = np.exp(-r**2 / (2 * 𝜎x**2) - 𝜉**2 / (2 * 𝜎𝜉**2)) - 1
    term4 = 𝜉**2 / 𝜎𝜉**2 - 1
    return term1 * term2 * term3 * term4

# Generate data for plotting B𝜃
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
𝜉 = 0.0

X, Y = np.meshgrid(x, y)
B_theta = B𝜃(X, Y, 𝜉)

# Plot B𝜃
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, B_theta, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('B_theta')
ax.set_title('Magnetic Field B_theta')
plt.show()


# Code for Er computation and plot

# Define the function for Er
def Er(x, y, 𝜉):
    r = np.sqrt(x**2 + y**2)
    B𝜃_values = B𝜃(x, y, 𝜉)
    return B𝜃_values + r**2

# Generate data for plotting Er
Er_values = Er(X, Y, 𝜉)

# Plot the electric field Er
plt.contourf(X, Y, Er_values, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot of Er(x, y, 𝜉={})'.format(𝜉))
plt.colorbar(label='Er')
plt.show()


def calculate_Ez(rb, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta):
    Ez = (np.exp(-rb**2 / (2 * sigma_x**2)) * (sigma_x**2 * nb0 * theta * Lb
          * np.exp(-xi**2 / (2 * sigma_zeta**2)) * np.exp(-rb**2 / (2 * sigma_x**2))
          - 2 * sigma_x**4 * nb0 * np.exp(-xi**2 / (2 * sigma_zeta**2))
          * np.exp(-rb**2 / (2 * sigma_x**2)))) / (2 * sigma_zeta**2 - 1)
    return Ez

def calculate_Er(rb, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta):
    Er = ((xi**2 / (2 * sigma_zeta**2) - 1) * np.exp(-rb**2 / (2 * sigma_x**2))
          * np.exp(-xi**2 / (2 * sigma_zeta**2)) * (sigma_x**2 * nb0 * theta * Lb
          * np.exp(-rb**2 / (2 * sigma_x**2)) - 2 * sigma_x**4 * nb0
          * np.exp(-rb**2 / (2 * sigma_x**2)))) / (2 * sigma_zeta**2 - 1)
    return Er

def calculate_Bphi(rb, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta):
    Bphi = (1 / rb * np.exp(-rb**2 / (2 * sigma_x**2)) * (sigma_x**2 * nb0 * theta * Lb
            * np.exp(-xi**2 / (2 * sigma_zeta**2)) * np.exp(-rb**2 / (2 * sigma_x**2))
            - 2 * sigma_x**4 * nb0 * np.exp(-xi**2 / (2 * sigma_zeta**2))
            * np.exp(-rb**2 / (2 * sigma_x**2)))) / (2 * sigma_zeta**2 - 1)
    return Bphi

def calculate_dEzdxi(rb, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta):
    dEz_dxi = (xi * np.exp(-rb**2 / (2 * sigma_x**2))
               * np.exp(-xi**2 / (2 * sigma_zeta**2)) * (2 * sigma_x**4 * nb0
               * np.exp(-rb**2 / (2 * sigma_x**2)) - sigma_x**2 * nb0 * theta * Lb
               * np.exp(-rb**2 / (2 * sigma_x**2)))) / (sigma_zeta**2 * (2 * sigma_zeta**2 - 1))
    return dEz_dxi

def calculate_dEr_dxi(rb, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta):
    dEr_dxi = ((2 * sigma_x**4 * nb0 * np.exp(-rb**2) / (2 * sigma_x**2))
               * np.exp(-xi**2 / (2 * sigma_zeta**2)) * (xi**2 / (2 * sigma_zeta**2) - 1))- (sigma_x**2 * nb0 * theta * Lb * np.exp(-rb**2 / (2 * sigma_x**2))
               * np.exp(-xi**2 / (2 * sigma_zeta**2)) * (2 * sigma_x**4 * np.exp(-rb**2 / (2 * sigma_x**2))
               - sigma_x**2 * theta * nb0 * Lb * np.exp(-rb**2 / (2 * sigma_x**2)))) / (sigma_zeta**2 * (2 * sigma_zeta**2 - 1))
    return dEr_dxi

# Parameters
sigma_x = 1.0
sigma_y = 1.5
nb0 = 1.0
theta = 0.8
Lb = 2.0
sigma_zeta = 0.5
rbmax1 = 8.0
rbmax2 = 1.0

# Define the range of xi values
xi = np.linspace(-10, 10, 100)

# Calculate Ez for homogeneous plasma
rb_homogeneous = 2 * sigma_zeta * np.sqrt(nb0)
Ez_homogeneous = calculate_Ez(rb_homogeneous, sigma_x, sigma_y, nb0, theta, Lb, xi,sigma_zeta)

# Calculate Ez for nonhomogeneous plasma
rb_nonhomogeneous = 2 * sigma_zeta**2 * np.sqrt(nb0)
Ez_nonhomogeneous = calculate_Ez(rb_nonhomogeneous, sigma_x, sigma_y, nb0, theta, Lb, xi,sigma_zeta)

# Calculate Er for homogeneous plasma
Er_homogeneous = calculate_Er(rb_homogeneous, sigma_x, sigma_y, nb0, theta, Lb, xi,sigma_zeta)

# Calculate Er for nonhomogeneous plasma
Er_nonhomogeneous = calculate_Er(rb_nonhomogeneous, sigma_x, sigma_y, nb0, theta, Lb, xi,sigma_zeta)


def plot_fields(rb_values, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta):
    Ez_values = calculate_Ez(rb_values, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta)
    Er_values = calculate_Er(rb_values, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta)
    Bphi_values = calculate_Bphi(rb_values, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta)
    dEz_dxi_values = calculate_dEzdxi(rb_values, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta)
    dEr_dxi_values = calculate_dEr_dxi(rb_values, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta)

    plt.plot(rb_values, Ez_values, label='Ez')
    plt.plot(rb_values, Er_values, label='Er')
    plt.plot(rb_values, Bphi_values, label='Bphi')
    plt.plot(rb_values, dEz_dxi_values, label='dEz/dxi')
    plt.plot(rb_values, dEr_dxi_values, label='dEr/dxi')
    plt.xlabel('rb')
    plt.ylabel('Field Components')
    plt.legend()
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(rb_values, Ez_values, label="Ez")
    plt.xlabel("rb")
    plt.ylabel("Ez")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(rb_values, Er_values, label="Er")
    plt.xlabel("rb")
    plt.ylabel("Er")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(rb_values, Bphi_values, label="Bphi")
    plt.xlabel("rb")
    plt.ylabel("Bphi")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(rb_values, dEz_dxi_values, label="dEz/dxi")
    plt.plot(rb_values, dEr_dxi_values, label="dEr/dxi")
    plt.xlabel("rb")
    plt.ylabel("dE/dxi")
    plt.legend()
    
    plt.tight_layout()

    # Plotting Ez and Er
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(xi, Ez_homogeneous, label='Homogeneous Plasma')
    plt.plot(xi, Ez_nonhomogeneous, label='Nonhomogeneous Plasma')
    plt.xlabel('ξ')
    plt.ylabel('Ez')
    plt.title('Longitudinal Electric Field (Ez)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(xi, Er_homogeneous, label='Homogeneous Plasma')
    plt.plot(xi, Er_nonhomogeneous, label='Nonhomogeneous Plasma')
    plt.xlabel('ξ')
    plt.ylabel('Er')
    plt.title('Radial Electric Field (Er)')
    plt.legend()

    plt.tight_layout()
    plt.show()
#  usage
rb_values = np.linspace(0, 10, 100)
sigma_x = 1
sigma_y = 1
nb0 = 1
theta = 1
Lb = 1
xi = np.linspace(-10, 10, 100)
sigma_zeta = 1

plot_fields(rb_values, sigma_x, sigma_y, nb0, theta, Lb, xi, sigma_zeta)

# Constants
c = 299792458.0  # Speed of light
k_beta = 8.617 * 10**(-5)  # Boltzmann constant times electron mass squared
# Define the electron mass
m_e = 9.10938356e-31  # Mass of an electron in kilograms
# Plasma parameters
Te = m_e * c**2  # Electron temperature
n0 = 10**20  # Number of plasma electrons

# Define the function for ξ = z - ct
def xi(z, t):
    return z - c * t

# Plotting parameters
z = np.linspace(-10, 10, 100)
t = np.linspace(0, 1, 100)

# Compute ξ values
xi_values = xi(z, t)

# Plot ξ
plt.figure()
plt.plot(z, xi_values)
plt.xlabel('z')
plt.ylabel('ξ')
plt.title('ξ = z - ct')
plt.grid(True)
plt.show()


# Relativistic Adiabatic Blowout Regime
def calculate_electric_field_rel(plasma_density, lorentz_factor):
    q = 1.602176634e-19  # elementary charge in Coulombs
    epsilon_0 = 8.8541878128e-12  # vacuum permittivity in Farads per meter
    return math.sqrt(2 * math.pi * plasma_density * q / (lorentz_factor * epsilon_0))

# Ultra-Relativistic Regime
def calculate_electric_field_ultra_rel():
    m_e = 9.10938356e-31  # electron mass in kilograms
    c = 299792458  # speed of light in meters per second
    e = 1.602176634e-19  # elementary charge in Coulombs
    lambda_c = 2.42631023867e-12  # Compton wavelength in meters
    return m_e * c**2 / (e * lambda_c)

# Example usage
plasma_density = 1e18  # plasma density in particles per cubic meter
lorentz_factor = 1.5  # Lorentz factor
electric_field_rel = calculate_electric_field_rel(plasma_density, lorentz_factor)
electric_field_ultra_rel = calculate_electric_field_ultra_rel()

print("Electric Field in Relativistic Adiabatic Blowout Regime:", electric_field_rel, "V/m")
print("Electric Field in Ultra-Relativistic Regime:", electric_field_ultra_rel, "V/m")

# Plotting
x = ["Relativistic Adiabatic Blowout", "Ultra-Relativistic"]
y = [electric_field_rel, electric_field_ultra_rel]

plt.bar(x, y)
plt.xlabel("Regime")
plt.ylabel("Electric Field (V/m)")
plt.title("Electric Field in Different Regimes")
plt.show()
