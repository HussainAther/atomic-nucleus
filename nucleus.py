"""nucleus_model.py: Simulate atomic nucleus using Monte Carlo methods
This module provides a Nucleus class to represent an atomic nucleus (with protons and neutrons)
and simulate its configuration using Monte Carlo techniques."""

import math
import random

class Nucleus:
    '''A class representing an atomic nucleus with protons and neutrons.'''
    
    def __init__(self, Z, N, V0=5.0, R0=3.0, K=1.0, container_radius=None, temperature=0.5, step_size=0.5):
        '''
        Initialize the Nucleus model.
        
        Parameters:
            Z (int): Number of protons.
            N (int): Number of neutrons.
            V0 (float): Depth of nuclear potential well (attraction strength).
            R0 (float): Range of the nuclear force (within this distance, nucleons attract).
            K (float): Coulomb's constant for proton-proton repulsion (in consistent units).
            container_radius (float): Radius of the spherical container (if None, use 1.2 * A^(1/3)).
            temperature (float): Effective temperature for Monte Carlo acceptance (higher values allow more uphill moves).
            step_size (float): Maximum move step size for Monte Carlo moves (in same distance units as R0).
        '''
        self.Z = Z
        self.N = N
        self.A = Z + N  # total nucleons
        # define container radius for nucleus (like radius of nucleus or potential well)
        if container_radius is None:
            self.container_radius = 1.2 * (self.A ** (1/3.0))
        else:
            self.container_radius = container_radius
        # Monte Carlo parameters
        self.temperature = temperature
        self.step_size = step_size
        # Physical constants for the model
        self.V0 = V0  # nuclear potential strength (attractive)
        self.R0 = R0  # nuclear force range
        self.K = K   # Coulomb constant (we use 1 for simplicity)
        
        # Initialize nucleon types (True for proton, False for neutron)
        types = [True]*Z + [False]*N
        random.shuffle(types)  # shuffle to randomize the order of protons and neutrons
        self.is_proton = types
        
        # Initialize positions randomly inside spherical container
        self.positions = [None] * self.A
        for i in range(self.A):
            placed = False
            attempts = 0
            while not placed:
                R = self.container_radius
                x = (random.random()*2 - 1) * R
                y = (random.random()*2 - 1) * R
                z = (random.random()*2 - 1) * R
                if x*x + y*y + z*z <= R*R:
                    # check distance from existing nucleons to avoid overlap
                    too_close = False
                    for j in range(i):
                        dx = x - self.positions[j][0]
                        dy = y - self.positions[j][1]
                        dz = z - self.positions[j][2]
                        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                        if dist < 1.0:
                            too_close = True
                            break
                    if not too_close:
                        self.positions[i] = (x, y, z)
                        placed = True
                attempts += 1
                if attempts > 10000:
                    # If we try too many times (high density scenario), place anyway
                    self.positions[i] = (x, y, z)
                    placed = True
        
        # Compute initial energy
        self.energy = self.compute_total_energy()
    
    def nuclear_potential(self, r):
        '''Nuclear potential: returns attractive energy (negative) if distance r is within range, else 0.'''
        if r < self.R0:
            return -self.V0
        else:
            return 0.0
    
    def coulomb_potential(self, r):
        '''Coulomb potential (repulsive) between two protons at distance r.'''
        if r == 0:
            return float('inf')
        return self.K / r
    
    def compute_total_energy(self):
        '''Compute the total potential energy of the nucleus (sum of pairwise interactions).'''
        E = 0.0
        for i in range(self.A):
            x_i, y_i, z_i = self.positions[i]
            for j in range(i+1, self.A):
                x_j, y_j, z_j = self.positions[j]
                dx = x_i - x_j
                dy = y_i - y_j
                dz = z_i - z_j
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                # Nuclear attraction
                E += self.nuclear_potential(dist)
                # Coulomb repulsion if both are protons
                if self.is_proton[i] and self.is_proton[j]:
                    E += self.coulomb_potential(dist)
        return E
    
    def monte_carlo_step(self):
        '''Perform a single Monte Carlo move (Metropolis algorithm step).'''
        i = random.randrange(0, self.A)
        x_i, y_i, z_i = self.positions[i]
        # propose a random move
        dx = (random.random()*2 - 1) * self.step_size
        dy = (random.random()*2 - 1) * self.step_size
        dz = (random.random()*2 - 1) * self.step_size
        new_x = x_i + dx
        new_y = y_i + dy
        new_z = z_i + dz
        # If outside container boundary, reject move immediately
        if new_x**2 + new_y**2 + new_z**2 > self.container_radius**2:
            return False
        # compute energy contribution involving i (old position)
        E_old_contrib = 0.0
        for j in range(self.A):
            if j == i: 
                continue
            x_j, y_j, z_j = self.positions[j]
            dx0 = x_i - x_j
            dy0 = y_i - y_j
            dz0 = z_i - z_j
            dist0 = math.sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0)
            E_old_contrib += self.nuclear_potential(dist0)
            if self.is_proton[i] and self.is_proton[j]:
                E_old_contrib += self.coulomb_potential(dist0)
        # compute energy contribution with i at new position
        E_new_contrib = 0.0
        for j in range(self.A):
            if j == i:
                continue
            x_j, y_j, z_j = self.positions[j]
            dx1 = new_x - x_j
            dy1 = new_y - y_j
            dz1 = new_z - z_j
            dist1 = math.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            E_new_contrib += self.nuclear_potential(dist1)
            if self.is_proton[i] and self.is_proton[j]:
                E_new_contrib += self.coulomb_potential(dist1)
        dE = E_new_contrib - E_old_contrib
        # Accept or reject the move
        if dE < 0:
            accept = True
        else:
            # Boltzmann acceptance probability
            accept_prob = math.exp(-dE / self.temperature) if self.temperature > 1e-12 else 0.0
            accept = (random.random() < accept_prob)
        if accept:
            self.positions[i] = (new_x, new_y, new_z)
            self.energy += dE
            return True
        return False
    
    def run_monte_carlo(self, steps=1000):
        '''Run a Monte Carlo simulation for a given number of steps.'''
        accepted_moves = 0
        for step in range(steps):
            if self.monte_carlo_step():
                accepted_moves += 1
        return accepted_moves
    
    def save_positions(self, filename):
        '''Save nucleon positions to a CSV file with columns: x,y,z,type (type: 1=proton, 0=neutron).'''
        with open(filename, 'w') as f:
            f.write("x,y,z,type\\n")
            for (x, y, z), is_p in zip(self.positions, self.is_proton):
                t = 1 if is_p else 0
                f.write(f"{x:.5f},{y:.5f},{z:.5f},{t}\\n")
"""
simulate_nucleus.py: Run a Monte Carlo simulation of an atomic nucleus and save results.
"""
import argparse
from nucleus_model import Nucleus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate an atomic nucleus configuration using Monte Carlo.")
    parser.add_argument('-Z', '--protons', type=int, default=10, help='Number of protons (default: 10)')
    parser.add_argument('-N', '--neutrons', type=int, default=10, help='Number of neutrons (default: 10)')
    parser.add_argument('-m', '--steps', type=int, default=1000, help='Number of Monte Carlo steps (default: 1000)')
    parser.add_argument('-T', '--temperature', type=float, default=0.5, help='Monte Carlo temperature for acceptance (default: 0.5)')
    parser.add_argument('-o', '--output', type=str, default='nucleus_positions.csv', help='Output CSV file for final positions (default: nucleus_positions.csv)')
    args = parser.parse_args()
    
    # Create nucleus and run simulation
    nucleus = Nucleus(Z=args.protons, N=args.neutrons, temperature=args.temperature)
    print(f"Initializing nucleus with {args.protons} protons and {args.neutrons} neutrons...")
    print(f"Initial total energy: {nucleus.energy:.3f}")
    accepted = nucleus.run_monte_carlo(steps=args.steps)
    acceptance_rate = accepted / args.steps * 100.0
    print(f"Simulation finished. Accepted moves: {accepted}/{args.steps} ({acceptance_rate:.1f}% acceptance)")
    print(f"Final total energy: {nucleus.energy:.3f}")
    # Calculate average radii for protons and neutrons
    distances = []
    proton_dists = []
    neutron_dists = []
    for (x, y, z), is_p in zip(nucleus.positions, nucleus.is_proton):
        dist = (x**2 + y**2 + z**2) ** 0.5
        distances.append(dist)
        if is_p:
            proton_dists.append(dist)
        else:
            neutron_dists.append(dist)
    if proton_dists:
        avg_p = sum(proton_dists) / len(proton_dists)
        print(f"Average proton distance from center: {avg_p:.3f}")
    if neutron_dists:
        avg_n = sum(neutron_dists) / len(neutron_dists)
        print(f"Average neutron distance from center: {avg_n:.3f}")
    # Save positions to file
    nucleus.save_positions(args.output)
    print(f"Final nucleon positions saved to {args.output}")

