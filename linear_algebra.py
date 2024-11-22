import math

def dot_product(v1, v2):
    """Calculates the dot product of two vectors.

    Args:
        v1: The first vector.
        v2: The second vector.

    Returns:
        The dot product of the two vectors.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension for dot product")
    return sum(x * y for x, y in zip(v1, v2))

def vector_addition(v1, v2):
    """Adds two vectors together.

    Args:
        v1: The first vector.
        v2: The second vector.

    Returns:
        The sum of the two vectors.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension for addition")
    return [x + y for x, y in zip(v1, v2)]

def scalar_multiplication(scalar, vector):
    """Multiplies a vector by a scalar.

    Args:
        scalar: The scalar value.
        vector: The vector.

    Returns:
        The result of scalar multiplication.
    """
    return [scalar * x for x in vector]

def vector_magnitude(vector):
    """Calculates the magnitude (length) of a vector.

    Args:
        vector: The vector.

    Returns:
        The magnitude of the vector.
    """
    return math.sqrt(sum(x**2 for x in vector))

def cross_product(v1, v2):
    """Calculates the cross product of two 3D vectors.

    Args:
        v1: The first 3D vector.
        v2: The second 3D vector.

    Returns:
        The cross product of the two vectors.
    """
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Cross product is only defined for 3D vectors")
    x = v1[1] * v2[2] - v1[2] * v2[1]
    y = v1[2] * v2[0] - v1[0] * v2[2]
    z = v1[0] * v2[1] - v1[1] * v2[0]
    return [x, y, z]

def normalize_vector(vector):
    """Normalizes a vector (creates a unit vector with the same direction).

    Args:
        vector: The vector to normalize.

    Returns:
        The normalized vector.
    """
    magnitude = vector_magnitude(vector)
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")
    return [x / magnitude for x in vector]

def matrix_addition(m1, m2):
    """Adds two matrices together.

    Args:
        m1: The first matrix.
        m2: The second matrix.

    Returns:
        The sum of the two matrices.
    """
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise ValueError("Matrices must have the same dimensions for addition")
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def matrix_subtraction(m1, m2):
    """Subtracts one matrix from another.

    Args:
        m1: The first matrix.
        m2: The second matrix.

    Returns:
        The difference between the two matrices.
    """
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise ValueError("Matrices must have the same dimensions for subtraction")
    return [[m1[i][j] - m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def matrix_multiplication(m1, m2):
    """Multiplies two matrices.

    Args:
        m1: The first matrix.
        m2: The second matrix.

    Returns:
        The product of the two matrices.
    """
    if len(m1[0]) != len(m2):
        raise ValueError("Number of columns in m1 must equal number of rows in m2")
    result = [[0 for _ in range(len(m2[0]))] for _ in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                result[i][j] += m1[i][k] * m2[k][j]
    return result

def matrix_transpose(matrix):
    """Calculates the transpose of a matrix.

    Args:
        matrix: The matrix.

    Returns:
        The transpose of the matrix.
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def determinant_2x2(matrix):
    """Calculates the determinant of a 2x2 matrix.

    Args:
        matrix: The 2x2 matrix.

    Returns:
        The determinant of the matrix.
    """
    if len(matrix) != 2 or len(matrix[0]) != 2:
        raise ValueError("Determinant_2x2 is only defined for 2x2 matrices")
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def determinant(matrix):
  """Calculates the determinant of a square matrix (any size).

  Args:
    matrix: The square matrix.

  Returns:
    The determinant of the matrix.
  """
  if len(matrix) != len(matrix[0]):
    raise ValueError("Determinant is only defined for square matrices")
  n = len(matrix)
  if n == 1:
    return matrix[0][0]
  elif n == 2:
    return determinant_2x2(matrix)
  else:
    det = 0
    for j in range(n):
      submatrix = [[matrix[i][k] for k in range(n) if k != j] for i in range(1, n)]
      det += (-1) ** j * matrix[0][j] * determinant(submatrix)
    return det


def invert_matrix_2x2(matrix):
    """Calculates the inverse of a 2x2 matrix.

    Args:
        matrix: The 2x2 matrix.

    Returns:
        The inverse of the matrix.
    """
    if len(matrix) != 2 or len(matrix[0]) != 2:
        raise ValueError("Invert_matrix_2x2 is only defined for 2x2 matrices")
    det = determinant_2x2(matrix)
    if det == 0:
        raise ValueError("Matrix is not invertible (determinant is 0)")
    return [[matrix[1][1] / det, -matrix[0][1] / det],
            [-matrix[1][0] / det, matrix[0][0] / det]]

def invert_matrix(matrix):
    """Calculates the inverse of a square matrix (any size).

    Args:
      matrix: The square matrix.

    Returns:
      The inverse of the matrix, if it exists.
    """
    n = len(matrix)
    if n != len(matrix[0]):
      raise ValueError("Inverse is only defined for square matrices")

    # Calculate the determinant
    det = determinant(matrix)
    if det == 0:
      raise ValueError("Matrix is not invertible (determinant is 0)")

    # Calculate the matrix of minors
    minors = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
      for j in range(n):
        submatrix = [[matrix[row][col] for col in range(n) if col != j] for row in range(n) if row != i]
        minors[i][j] = determinant(submatrix)

    # Calculate the matrix of cofactors
    cofactors = [[(-1) ** (i + j) * minors[i][j] for j in range(n)] for i in range(n)]

    # Calculate the adjugate (transpose of the matrix of cofactors)
    adjugate = matrix_transpose(cofactors)

    # Calculate the inverse
    inverse = [[adjugate[i][j] / det for j in range(n)] for i in range(n)]
    return inverse