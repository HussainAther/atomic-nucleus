"""
visualize_nucleus.py: Visualize the results of a nucleus simulation (radial distribution of nucleons).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize nucleus simulation results (radial distribution).")
    parser.add_argument('-i', '--input', type=str, default='nucleus_positions.csv', help='Input CSV file of nucleon positions (default: nucleus_positions.csv)')
    parser.add_argument('-o', '--output', type=str, default='radial_distribution.png', help='Output image file for plot (default: radial_distribution.png)')
    args = parser.parse_args()
    
    # Load data from CSV file
    data = np.loadtxt(args.input, delimiter=',', skiprows=1)
    # data columns: x, y, z, type
    positions = data[:, 0:3]
    types = data[:, 3]
    # compute radial distances
    radii = np.sqrt((positions ** 2).sum(axis=1))
    proton_r = radii[types == 1]
    neutron_r = radii[types == 0]
    
    # Plot histogram of radial distances for protons and neutrons
    plt.hist(proton_r, bins=10, alpha=0.5, label='Protons')
    plt.hist(neutron_r, bins=10, alpha=0.5, label='Neutrons')
    plt.xlabel('Radius (arbitrary units)')
    plt.ylabel('Frequency')
    plt.title('Radial Distribution of Nucleons')
    plt.legend()
    plt.savefig(args.output)
    print(f"Saved radial distribution histogram as {args.output}")
    # Uncomment the line below to display the plot window (if running interactively)
    # plt.show()

