import numpy as np


def get_operator(gate_matrix, target_qubit_index, total_qubits):
    I = np.eye(2, dtype=complex)

    op_list = []

    for i in range(total_qubits):

        if i == target_qubit_index:

            op_list.append(gate_matrix)

        else:

            op_list.append(I)

    full_operator = op_list[0]

    for i in range(1, len(op_list)):
        full_operator = np.kron(full_operator, op_list[i])

    return full_operator


H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

psi = np.array([1, 0, 0, 0], dtype=complex).T

op_step_1 = get_operator(H, 0, 2)

psi = op_step_1 @ psi

CNOT = np.array([

    [1, 0, 0, 0],

    [0, 1, 0, 0],

    [0, 0, 0, 1],

    [0, 0, 1, 0]

], dtype=complex)

psi = CNOT @ psi

expected = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1])

match = np.allclose(psi, expected)

print(f"Final State: {psi.real}")

print(f"Verification Successful: {match}")
