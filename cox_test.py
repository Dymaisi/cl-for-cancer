import torch
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
file_path = 'data/CPGEA_parsed_cox.csv'
survival_data = pd.read_csv(file_path).drop('Sample_ID', axis=1)
# feature_data = pd.read_csv(file_path).drop('time', axis=1).drop('dead', axis=1).drop('Sample_ID', axis=1)

continuous_columns = survival_data.select_dtypes(include=['float64']).columns
    # 归一化连续数据
scaler = MinMaxScaler()
survival_data[continuous_columns] = scaler.fit_transform(survival_data[continuous_columns])

cph = CoxPHFitter()

# 使用数据拟合Cox-PH模型
cph.fit(survival_data, duration_col='time', event_col='dead')
cph.print_summary() 
# 计算患者的Risk Score
# Risk Score = sum(feature_i * feature_i_coef)
'''risk_scores = np.zeros(survival_data.shape[0])
for i in range(feature_data.shape[0]):
    feature_coef = cph.params_[i]
    risk_scores += feature_data[i] * feature_coef

print("Risk Scores:")
print(risk_scores)

# 计算C-Index
c_index = concordance_index(survival_data['time'], -risk_scores, survival_data['dead'])
print("C-Index:", c_index)
'''