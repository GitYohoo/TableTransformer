import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset

# 定义Transformer模型
class myTransformer(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )  # 初始化一个Transformer编码器层对象,该层由多头自注意力和前馈神经网络组成
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )  # 初始化一个由多个Transformer编码器层组成的Transformer编码器对象
        self.dropout = nn.Dropout(dropout)  # 添加一个Dropout层,用于防止过拟合
        self.fc = nn.Linear(
            input_dim, output_dim
        )  # 初始化一个全连接层,用于将Transformer编码器的输出映射到所需的输出维度
       
    def forward(self, x):
        x = self.transformer_encoder(x)  # 传入Transformer编码器
        x = self.dropout(x)  # 传入Dropout层
        # attention_weights = self.calculate_attention_weights(x)  # 计算注意力权重
        x = self.fc(x)  # 传入全连接层
        return x

    def calculate_attention_weights(self, x):
        attn_weights = []
        for layer in self.transformer_encoder.layers:
            weight = layer.self_attn.in_proj_weight
            dim = layer.self_attn.embed_dim
            # 获取q,k,v的权重
            q_weight = weight[:dim, :]
            k_weight = weight[dim:2*dim, :] 
            v_weight = weight[2*dim:, :]  
            Q = x @ q_weight / math.sqrt(dim)
            K = x @ k_weight / math.sqrt(dim)
            V = x @ v_weight
            attn_output_weights = F.softmax(Q @ K.T / math.sqrt(dim), dim=1) @ V
            attn_weights.append(attn_output_weights)

        return attn_weights

    def predict_proba(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)  # 如果x是numpy数组,则转换为tensor
        x = self.forward(x)
        x = F.softmax(abs(x), dim=1)  # 计算模型输出的概率分布
        x = x.detach().numpy()  # 将张量转换为numpy数组
        return x

    def plot_attention_weights(self, attention_weights):
        count = 0
        #将attention_weights复制20行
        # attention_weights = np.repeat(attention_weights, 20, axis=0)
        fig, axes = plt.subplots(ncols=len(attention_weights), figsize=(20, 10))
        for ax, attn_weights in zip(axes, attention_weights):
            im = ax.imshow(attn_weights, origin="upper")
            ax.set_title("Attention Weights Layer {}".format(count))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.invert_yaxis()  # 反向显示y轴
            count += 1
        plt.show()

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # 返回数据和其对应的目标值
        x = self.data[idx]
        y = self.targets[idx]
        # 将目标值转换为1维张量的类索引形式
        return x, y.flatten().long()

class ThreeD_CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# 定义Focal Loss损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
