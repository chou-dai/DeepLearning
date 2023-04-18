import numpy as np

np_array = np.array([[0, 1, 2],
                     [3, 4, 5]])

# shape
print(np_array.shape)

# reshape
print(np_array.reshape(-1))

# sum
print(np_array.sum()) # 全合計
print(np.sum(np_array)) # 全合計
print(np.sum(np_array, axis=0)) # 列方向
print(np.sum(np_array, axis=1)) # 行方向

# average
# print(np_array.average()) error
print(np.average(np_array))

# max
print(np_array.max())
print(np.max(np_array))

# min
print(np_array.min())
print(np.min(np_array))