import numpy as np

x_weights = [0.89116931, 0.80307853, 0.92203534]
biases = [0.9015398]

x_data = [[2, 2, 2], [2, 2, 3], [3, 2, 4]]
y_data = np.matmul(x_data, [[1], [1], [1]])
print(y_data)

y_hat = np.matmul(x_data, x_weights) + biases

print(y_hat)
