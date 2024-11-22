import math

def mean(data):
  """Calculates the mean (average) of a dataset.

  Args:
      data: A list of numerical values.

  Returns:
      The mean of the dataset.
  """
  if not data:
    raise ValueError("Input list cannot be empty")
  return sum(data) / len(data)

def median(data):
  """Calculates the median of a dataset.

  Args:
      data: A list of numerical values.

  Returns:
      The median of the dataset.
  """
  if not data:
    raise ValueError("Input list cannot be empty")
  data.sort()
  n = len(data)
  if n % 2 == 0:  # Even number of elements
    mid1 = n // 2 - 1
    mid2 = mid1 + 1
    return (data[mid1] + data[mid2]) / 2
  else:  # Odd number of elements
    mid = n // 2
    return data[mid]

def mode(data):
  """Calculates the mode(s) of a dataset.

  Args:
      data: A list of values.

  Returns:
      A list containing the mode(s) of the dataset.
  """
  if not data:
    raise ValueError("Input list cannot be empty")
  frequency = {}
  for value in data:
    frequency[value] = frequency.get(value, 0) + 1
  max_count = max(frequency.values())
  modes = [key for key, count in frequency.items() if count == max_count]
  return modes

def variance(data):
  """Calculates the variance of a dataset.

  Args:
      data: A list of numerical values.

  Returns:
      The variance of the dataset.
  """
  if not data:
    raise ValueError("Input list cannot be empty")
  n = len(data)
  if n < 2:
    raise ValueError("Variance requires at least two data points")
  mean_value = mean(data)
  squared_diff = [(x - mean_value) ** 2 for x in data]
  return sum(squared_diff) / (n - 1)  # Sample variance

def standard_deviation(data):
  """Calculates the standard deviation of a dataset.

  Args:
      data: A list of numerical values.

  Returns:
      The standard deviation of the dataset.
  """
  return math.sqrt(variance(data))