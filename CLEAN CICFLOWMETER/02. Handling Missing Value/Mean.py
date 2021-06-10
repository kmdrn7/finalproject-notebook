# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np

pd.options.display.max_rows = 100


# %%
dataset = pd.read_csv("/media/kmdr7/Seagate/TA/DATASETS/Dataset.csv")
dataset.head(10)


# %%
# replace infinite val with nan
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)


# %%
# Cek missing values
dataset.isna().sum()

# %% [markdown]
# ## Replace Missing with Mean

# %%
dataset["Flow Bytes/s"].fillna(value=dataset["Flow Bytes/s"].mean(), inplace=True)
dataset["Flow Packets/s"].fillna(value=dataset["Flow Packets/s"].mean(), inplace=True)
dataset.to_csv('/media/kmdr7/Seagate/TA/DATASETS/Dataset-Mean.csv', index=False)


# %%
# Cek missing values
dataset.isna().sum().sum()


