import numpy as np

np_array = np.array([[1, 2], [4, 1], [0, 4]])
np_map = map(lambda x: x*3, np_array)

print([i for i in np_map])