# Mathhelper
Will do most of your math homework
def gradient_descent is not working right. Working on it.
# mymath: A Python Package for Math Homework and More

This is a Python package designed to help with math homework, explore basic machine learning concepts, and provide a foundation for building more complex mathematical applications.

## Features

**Modules:**

* **`basic`:**
    * `add(x, y)`: Adds two numbers.
    *  (Add more basic math functions here as you implement them)

* **`geometry`:**
    * `area_of_rectangle(length, breadth)`: Calculates the area of a rectangle.
    * `area_of_circle(radius)`: Calculates the area of a circle.
    * (Add more geometry functions here as you implement them)

* **`linear_algebra`:**
    * `dot_product(v1, v2)`: Calculates the dot product of two vectors.
    * `vector_addition(v1, v2)`: Adds two vectors.
    * `scalar_multiplication(scalar, vector)`: Multiplies a vector by a scalar.
    * `vector_magnitude(vector)`: Calculates the magnitude of a vector.
    * `cross_product(v1, v2)`: Calculates the cross product of two 3D vectors.
    * `normalize_vector(vector)`: Normalizes a vector.
    * `matrix_addition(m1, m2)`: Adds two matrices.
    * `matrix_subtraction(m1, m2)`: Subtracts two matrices.
    * `matrix_multiplication(m1, m2)`: Multiplies two matrices.
    * `matrix_transpose(matrix)`: Calculates the transpose of a matrix.
    * `determinant(matrix)`: Calculates the determinant of a square matrix.
    * `invert_matrix(matrix)`: Calculates the inverse of a square matrix. 

* **`machine_learning`:**
    * `euclidean_distance(p1, p2)`: Calculates the Euclidean distance between two points.
    * `manhattan_distance(p1, p2)`: Calculates the Manhattan distance between two points.
    * `cosine_similarity(v1, v2)`: Calculates the cosine similarity between two vectors.
    * `knn_classify(k, data, labels, new_point)`: Classifies a new point using KNN.
    * `gradient_descent(X, y, learning_rate, num_iterations)`: Performs gradient descent for linear regression.
    * `normalize_data(data)`: Normalizes data to have zero mean and unit variance.
    * (Add more machine learning functions and algorithms here as you implement them)

* **`statistics`:**
    * `mean(data)`: Calculates the mean of a dataset.
    * `median(data)`: Calculates the median of a dataset.
    * `mode(data)`: Calculates the mode(s) of a dataset.
    * `variance(data)`: Calculates the variance of a dataset.
    * `standard_deviation(data)`: Calculates the standard deviation of a dataset.
    * (Add more statistics functions here as you implement them)


## Installation

1.  **Download:** Download the `mymath` package (as a `.tar.gz` or `.zip` file).
2.  **Extract:** Extract the contents of the archive.
3.  **Place in Python Path:** Place the `mymath` directory in a location where Python can find it (e.g., in the same directory as your script or in your Python path).

## Usage

**Example:**

```python
import mymath.linear_algebra

v1 = [1, 2, 3]
v2 = [4, 5, 6]

dot_product_result = mymath.linear_algebra.dot_product(v1, v2)
print(f"Dot product: {dot_product_result}")
