import numpy as np

output = np.zeros((16, 1))
output[output == 0] = -1

target = np.zeros((16, 1))
target[target == 0] = 1

print(np.abs(output - target))
