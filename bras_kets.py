import numpy as np

"""
Convention: 
Bras <v| are Row Vectors (1, N)
Kets |v> are Column Vectors (N, 1)
"""


def complex_dot(v1, v2) -> complex:
    conjugated_v1 = np.conj(v1).T

    dot_product = 0
    for row in range(conjugated_v1.shape[0]):
        for column in range(conjugated_v1.shape[1]):
            dot_product += conjugated_v1[row, column] * v2[column, 0]

    return dot_product


def normalize(v):
    norm = np.sqrt(complex_dot(v, v).real)
    return v / norm


def tensor_product(v1, v2):
    rows_1, rows_2 = v1.shape[0], v2.shape[0]

    tensor_product_vec = np.zeros((rows_1 * rows_2, 1), dtype=complex)

    current_row = 0

    for element_v1 in v1.flat:
        for element_v2 in v2.flat:
            tensor_product_vec[current_row][0] = element_v1 * element_v2
            current_row += 1

    return tensor_product_vec


if __name__ == "__main__":
    v1_test = np.array([[1],
                        [0]])
    v2_test = np.array([[0],
                        [1]])

    # Test 1: Normalization
    plus_ket = normalize(v1_test + v2_test)
    print("Normalized Plus Ket:\n", plus_ket)

    inner_prod = complex_dot(plus_ket, plus_ket)
    assert np.isclose(inner_prod, 1), f"Normalization failed: {inner_prod}"
    print("Normalization Test Passed.")

    tp_result = tensor_product(v1_test, v2_test)
    print("\nTensor Product Result:\n", tp_result)

    expected_tp = np.array([[0], [1], [0], [0]])
    assert np.allclose(tp_result, expected_tp), "Tensor Product failed"
    print("Tensor Product Test Passed.")
