import numpy as np
import matplotlib.pyplot as plt
import random

class AtomicNucleusSimulation:
    def __init__(self, num_nucleons=20, temperature=0.1, iterations=1000):
        self.num_nucleons = num_nucleons
        self.temperature = temperature
        self.iterations = iterations
        self.nucleons = self.initialize_nucleons()

    def initialize_nucleons(self):
        """Initialize nucleons in random 3D positions within a spherical boundary."""
        nucleons = np.random.uniform(-1, 1, (self.num_nucleons, 3))
        nucleons /= np.linalg.norm(nucleons, axis=1).reshape(-1, 1)  # Normalize to unit sphere
        return nucleons

    def nuclear_potential(self, r):
        """Approximate nuclear force potential (Yukawa-like)."""
        V0 = -50  # Depth of potential well (arbitrary units)
        alpha = 0.8  # Range of nuclear force
        return V0 * np.exp(-alpha * r) / r if r > 0 else 0

    def total_energy(self):
        """Compute total energy of nucleon configuration."""
        energy = 0
        for i in range(self.num_nucleons):
            for j in range(i + 1, self.num_nucleons):
                r = np.linalg.norm(self.nucleons[i] - self.nucleons[j])
                energy += self.nuclear_potential(r)
        return energy

    def monte_carlo_optimization(self):
        """Use Metropolis Monte Carlo to find low-energy nucleon configurations."""
        for _ in range(self.iterations):
            i = random.randint(0, self.num_nucleons - 1)  # Pick random nucleon
            move = np.random.uniform(-0.05, 0.05, 3)  # Small random displacement
            new_position = self.nucleons[i] + move
            new_position /= np.linalg.norm(new_position)  # Keep within unit sphere

            old_energy = self.total_energy()
            self.nucleons[i] = new_position  # Temporarily move
            new_energy = self.total_energy()

            delta_E = new_energy - old_energy
            if delta_E > 0 and np.exp(-delta_E / self.temperature) < np.random.rand():
                self.nucleons[i] -= move  # Reject move, revert position

    def visualize_nucleus(self):
        """Plot the nucleons in 3D space."""
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.nucleons[:, 0], self.nucleons[:, 1], self.nucleons[:, 2], c='blue', s=50)
        ax.set_title("Atomic Nucleus Simulation")
        plt.show()

if __name__ == "__main__":
    simulation = AtomicNucleusSimulation(num_nucleons=20, temperature=0.1, iterations=5000)
    simulation.monte_carlo_optimization()
    simulation.visualize_nucleus()

