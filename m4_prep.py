import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# %%
reproduciblity = True
if reproduciblity:
    random.seed(10)
# %%
m4s = ["Daily", "Hourly", "Weekly"]
M4_TYPE = m4s[1]

df_train = pd.read_csv(f"m4_dataset/{M4_TYPE}-train.csv")
df_test = pd.read_csv(f"m4_dataset/{M4_TYPE}-test.csv")
# %%
data_arr = df_train.iloc[:, 1:].values
data_arr = list(data_arr)
for i, ts in enumerate(data_arr):
    data_arr[i] = ts[~np.isnan(ts)][None, :]
# %%
len(data_arr)


# %%
def give_indexes(df_train):
    data_arr = df_train.iloc[:, 1:].values
    data_arr = list(data_arr)
    for i, ts in enumerate(data_arr):
        data_arr[i] = ts[~np.isnan(ts)][None, :]
    indexes = []
    for ind in range(len(data_arr)):
        if data_arr[ind].shape[1] > 0:
            indexes.append(ind)
    return indexes


# %%
indexes = give_indexes(df_train)

# %%
len(indexes)
# %%
random.seed(10)
randomlist = random.sample(indexes, 300)
randomlist
# %%
train_series = df_train.iloc[randomlist, :]
test_series = df_test.iloc[randomlist, :]
# %%
sum(train_series.index == test_series.index) == len(train_series)
# %%
train_series
# %%
M4_TYPE
# %%
test_series
# %%
for data_i in range(len(train_series)):
    Xdf = train_series.iloc[data_i]
    Xdf = Xdf.dropna()
    Xdf = Xdf.iloc[1:]
    ydf = test_series.iloc[data_i]
    ydf = ydf.iloc[1:]
    df = pd.concat([Xdf, ydf])
    data = pd.DataFrame({"y": df})
    data.index = np.arange(len(data))
    print(f"{data_i} shape of new data {data.shape}")
    data.to_csv(f"m4_selections/{data_i}_M4_{M4_TYPE}.csv")