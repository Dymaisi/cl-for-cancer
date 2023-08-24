# 导入必要的库
import torch
import pandas as pd
from sklearn.decomposition import PCA

#Read csv file and skip first row, then set second row to be headers
df = pd.read_csv('processed_data/CPCGEA-FPKM.csv') 
#print(df.head())   #uncomment this line to check if you have read the file correctly
#Create a numpy array for columns number 3 and below
data = df.iloc[:,2:].values.T
#Set up PCA model with desired components

model = PCA(n_components=2)
#Fit and transform your data
transformedData = model.fit_transform(data)

output = torch.tensor(transformedData.T)
print(output[0][135])