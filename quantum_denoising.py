from qiskit.circuit.library import QFT

# Create Quantum Fourier Transform (QFT) circuit
qft_circuit = QFT(num_qubits=4)
qft_circuit.decompose().draw("mpl")

