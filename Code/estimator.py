import pandas as pd
import numpy as np
import os

print(os.getcwd())

df = pd.read_parquet(os.path.join("Datasets", "train.parquet"))

print(df.info())
X = df['counter_id','']