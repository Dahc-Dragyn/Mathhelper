import math
from collections import Counter

def euclidean_distance(p1, p2):
  """Calculates the Euclidean distance between two points.

  Args:
      p1: The first point (a list or tuple of coordinates).
      p2: The second point (a list or tuple of coordinates).

  Returns:
      The Euclidean distance between the two points.
  """
  if len(p1) != len(p2):
    raise ValueError("Points must have the same dimension")
  return math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(p1, p2)))

def manhattan_distance(p1, p2):
  """Calculates the Manhattan distance between two points.

  Args:
      p1: The first point (a list or tuple of coordinates).
      p2: The second point (a list or tuple of coordinates).

  Returns:
      The Manhattan distance between the two points.
  """
  if len(p1) != len(p2):
    raise ValueError("Points must have the same dimension")
  return sum(abs(x1 - x2) for x1, x2 in zip(p1, p2))

def cosine_similarity(v1, v2):
  """Calculates the cosine similarity between two vectors.

  Args:
      v1: The first vector.
      v2: The second vector.

  Returns:
      The cosine similarity between the two vectors.
  """
  dot_product = sum(x1 * x2 for x1, x2 in zip(v1, v2))
  magnitude_v1 = math.sqrt(sum(x**2 for x in v1))
  magnitude_v2 = math.sqrt(sum(x**2 for x in v2))
  if magnitude_v1 == 0 or magnitude_v2 == 0:
    raise ValueError("Cannot calculate cosine similarity with zero-magnitude vectors")
  return dot_product / (magnitude_v1 * magnitude_v2)

def knn_classify(k, data, labels, new_point):
  """Classifies a new point using the K-Nearest Neighbors algorithm.

  Args:
      k: The number of nearest neighbors to consider.
      data: A list of data points (each point is a list or tuple of coordinates).
      labels: A list of labels corresponding to the data points.
      new_point: The point to classify.

  Returns:
      The predicted label for the new point.
  """
  if len(data) != len(labels):
    raise ValueError("Data and labels must have the same length")

  distances = [euclidean_distance(new_point, point) for point in data] 
  k_nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
  k_nearest_labels = [labels[i] for i in k_nearest_indices]
  return Counter(k_nearest_labels).most_common(1)[0][0]  # Most frequent label

def gradient_descent(X, y, learning_rate, num_iterations):
  """Performs gradient descent to find the parameters of a linear regression model.

  Args:
      X: A list of data points (each point is a list or tuple of features).
      y: A list of target values corresponding to the data points.
      learning_rate: The learning rate for gradient descent.
      num_iterations: The number of iterations to run gradient descent.

  Returns:
      A tuple containing the intercept (w0) and slope (w1) of the linear regression model.
  """
  w0 = 0  # Initial intercept
  w1 = 0  # Initial slope
  n = len(X)

  for _ in range(num_iterations):
    y_predicted = [w0 + w1 * x[0] for x in X]  # Assuming one feature
    error = [y_pred - y_true for y_pred, y_true in zip(y_predicted, y)]

    # Calculate gradients
    w0_gradient = (1/n) * sum(error)
    w1_gradient = (1/n) * sum(x[0] * err for x, err in zip(X, error))

    # Update parameters
    w0 = w0 - learning_rate * w0_gradient
    w1 = w1 - learning_rate * w1_gradient

  return w0, w1

def normalize_data(data):
  """Normalizes data to have zero mean and unit variance.

  Args:
      data: A list of data points.

  Returns:
      The normalized data.
  """
  mean_value = sum(data) / len(data)
  std_dev = math.sqrt(sum((x - mean_value)**2 for x in data) / len(data))
  return [(x - mean_value) / std_dev for x in data]