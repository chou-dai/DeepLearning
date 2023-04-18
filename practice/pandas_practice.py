import pandas as pd
import numpy as np

# list
pd_series = pd.Series([60, 80, 70, 50, 30],
                      index=["label1", "label2", "label3", "label4", "label5"])
# Numpy
np_pd_series = pd.Series(np.array([60, 80, 70, 50, 30]),
                         index=np.array(["label1", "label2", "label3", "label4", "label5"]))
# dict
dict_pd_series = pd.Series({"label1": 60, "label2": 80, "label3": 70, "label4": 50, "label5": 30})

print(dict_pd_series)

# DataFrame（行列）
pd_dataFrame = pd.DataFrame([[80, 60, 70, True],
                             [90, 80, 70, True],
                             [70, 60, 75, True],
                             [40, 60, 50, False],
                             [20, 30, 40, False],
                             [50, 20, 10, False]])
# label
pd_dataFrame.index = ["Taro", "Hanako", "Jiro", "Sachiko", "Saburo", "Yoko"]
pd_dataFrame.columns = ["Japanese", "English", "Math", "Result"]

# shape = (行数, 列数)
print(pd_dataFrame.shape)

# head = 最初の5行
print(pd_dataFrame.head())

# tail = 最後の5行
print(pd_dataFrame.tail())

# describe = 統計量
print(pd_dataFrame.describe())