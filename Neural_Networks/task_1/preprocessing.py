
import pandas as pd
import numpy as np

data = pd.read_csv('dry_bean_dataset.csv')
data['MinorAxisLength'].interpolate(method='linear', inplace=True)
df = pd.DataFrame(data)