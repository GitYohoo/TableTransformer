#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

class Read_data(object):
    def __init__(self) -> None:
        pass       

    def Transformer_data_486(self):
        feature_name_data = pd.read_csv("data\\score_selected_feature_name.csv", header=0)
        feature_name = feature_name_data.iloc[0, :]
        rowdata = pd.read_csv("data\\com_patient_sample_mrna.csv", header=None, index_col=0, low_memory=False)
        data = rowdata.loc[feature_name, :]
        data = data.T
        data  = data.astype(float)
        targets = rowdata.loc["label", :]
        targets = targets.astype(int)

        scaler = MinMaxScaler() 
        normalized_data = scaler.fit_transform(data)
        data = pd.DataFrame(normalized_data, columns=data.columns)

        # 划分训练集和测试集,训练集占80%,测试集占20%
        train_data, x_test, train_targets, y_test = train_test_split(
            data, targets, test_size=0.2
        )

        # 标签值减1
        train_targets = train_targets - 1
        y_test = y_test - 1

        return train_data, x_test, train_targets, y_test

    def Transformer_data_486_Kmeans(self, category=5):
        feature_name_data = pd.read_csv("data\\score_selected_feature_name.csv", header=0)
        feature_name = feature_name_data.iloc[0, :]
        rowdata = pd.read_csv("data\\com_patient_sample_mrna.csv", header=None, index_col=0, low_memory=False)
        data = rowdata.loc[feature_name, :]
        data = data.T
        data  = data.astype(float)
        targets = rowdata.loc["label", :]
        targets = targets.astype(int)

        scaler = MinMaxScaler() 
        normalized_data = scaler.fit_transform(data)
        data = pd.DataFrame(normalized_data, columns=data.columns)

        if category == 1:
            zeros = np.zeros((data.shape[0], 3))
            zeros_df = pd.DataFrame(zeros, columns=["Zero1", "Zero2", "Zero3"])
            data = pd.concat([data, zeros_df], axis=1)      
        # 进行K-Means聚类
        kmeans = KMeans(n_init = 10, n_clusters=category, random_state=0).fit(data.T) 
        # 获取每个样本的聚类标签
        labels = kmeans.labels_
        # 根据标签将特征分组
        groups = {}
        for idx, label in enumerate(labels):
            feature = data.columns[idx]
            if label not in groups:
                groups[label] = [feature]
            else:
                groups[label].append(feature) 

        total_sample = data.shape[0]
        max_group_length = 0
        for i in range(category):
            group = groups[i]
            group_length = len(group)
            #找出最大的组的长度
            if group_length > max_group_length:
                max_group_length = group_length
        DATA = np.zeros((total_sample, category, max_group_length))
        # DATA = [[] for _ in range(category)]
        for i in range(category):
            group = groups[i]
            group_length = len(group)
            for j in range(group_length):
                feature = group[j]
                DATA[:, i, j] = data[feature].values
        #DATA转化为tensor
        DATA = torch.tensor(DATA, dtype=torch.float32)
        
        # tensor_list = [torch.tensor(l, dtype=torch.float32) for l in DATA]
        # DATA = torch.stack(tensor_list, dim=1)

        return DATA, np.array(targets)-1

    def Transformer_data_286(self, Normalization=True, data2tensor=False, zero=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 读取数据,去除第一行
        data = pd.read_csv("data\\new_data.csv", header=1)
        # 取最后一列作为标签
        targets = data.iloc[:, -1]  
        # 其他列为特征
        features = data.iloc[:, :-1]  
        
        if Normalization:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
            features = pd.DataFrame(features, columns=data.columns[:-1])
        if zero:
            # 添加4列全0特征
            zeros = np.zeros((features.shape[0], 4))
            zeros_df = pd.DataFrame(zeros, columns=["f1", "f2", "f3", "f4"])
            features = pd.concat([features, zeros_df], axis=1)
        
        # 打乱数据并划分训练集和测试集, 80%训练,20%测试
        train_features, x_test, train_targets, y_test = train_test_split(features, targets, test_size=0.2, shuffle=True)
        # 标签减1
        train_targets -= 1
        y_test -= 1
        if data2tensor:
            # 转为Tensor
            train_features, x_test, train_targets, y_test = map(lambda x: torch.tensor(x.values, dtype=torch.float32).to(device), 
                                                            [train_features, x_test, train_targets, y_test])
        return train_features, x_test, train_targets, y_test

    def Transformer_data_286_NLP(self, Normalization=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 读取数据,去除第一行
        data = pd.read_csv("data\\new_data.csv", header=1)
        # 取最后一列作为标签
        targets = data.iloc[:, -1]  
        # 其他列为特征
        features = data.iloc[:, :-1]  
        
        if Normalization:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
            features = pd.DataFrame(features, columns=data.columns[:-1])
        total_sample = features.shape[0]
        category = features.shape[1]
        DATA = np.zeros((total_sample, category, 1)) 
        for i in range(category):
            DATA[:, i, 0] = features.iloc[:, i].values

        #DATA转化为tensor
        DATA = torch.tensor(DATA, dtype=torch.float32)

        #划分训练集和测试集,训练集占80%,测试集占20%
        train_data, x_test, train_targets, y_test = train_test_split(
            DATA, targets, test_size=0.2
        )
        # 标签减1
        train_targets -= 1
        y_test -= 1
        train_targets = torch.tensor(train_targets.values, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)
         
        return train_data, x_test, train_targets, y_test
    

