
import pandas as pd
import numpy as np
import preprocessing


full_dataset = pd.read_csv('data/CEAS-08.csv')
full_dataset= preprocessing.preprocess(full_dataset)

#For debugging:
print(full_dataset.columns)
print(full_dataset['sender'].head())


# # ====== Dataset Splits ======

