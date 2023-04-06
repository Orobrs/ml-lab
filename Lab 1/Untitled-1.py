import pandas as pd
import numpy as np
df=pd.read_csv("D:\\Academic\\sem 6\\ML\\Lab\\Lab 1\\bike-sharing-daily.csv")
df.columns
df.shape
df.describe()
df.isna().sum()
from sklearn.model_selection import train_test_split
training,testing=train_test_split(df,test_size=0.30,random_state=24)
training.shape
testing.shape
training.info()
testing.info()