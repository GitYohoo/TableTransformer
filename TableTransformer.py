#%%
# 导入所需的包
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from myTransformer import myTransformer, CustomDataset, FocalLoss, ThreeD_CustomDataset
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
from read_data import Read_data
import numpy as np

# 读取数据
Data = Read_data()
# train_data, x_test, train_targets, y_test = Data.Transformer_data_286_NLP(Normalization=True)
train_data, x_test, train_targets, y_test = Data.Transformer_data_286(Normalization=True, zero=True, data2tensor=True)

epochs = 150
# 初始化模型,优化器和损失函数
model = myTransformer(
    input_dim=290,
    output_dim=4,
    hidden_dim=256,
    num_layers=2,
    num_heads=10,
    dropout=0.5,
)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(reduction="none")  # 交叉熵损失函数
# criterion = FocalLoss(gamma=2, alpha=1)
train_dataset = ThreeD_CustomDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 创建数据集和数据加载器

lr_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    anneal_strategy="linear",
)  # 定义学习率策略

model.to(device)

train_losses = []
train_accs = []
attention_weights = []
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for data1, target in train_loader:
        optimizer.zero_grad()  # 梯度重置
        data1 = data1.to(torch.float32).to(device)  # 转换为浮点数tensor
        target = target.squeeze()  # 压缩目标值到1D
        output = model(data1)  # 计算输出和损失
        loss = criterion(output, target)
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        lr_scheduler.step()  # 根据学习率策略更新学习率
        running_loss += loss.mean().item()  # 记录损失和准确率
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        attention_weights.append(attention_weights)
        #如果loss10次下降低于0.0001，停止训练
        if len(train_losses) > 10 and sum(train_losses[-10:]) / 10 < 0.0001:
            break

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # 打印周期统计信息
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}, Current lr: {lr_scheduler.get_last_lr()[0]}")

# 绘制训练曲线
plt.plot(train_losses, label="Training Loss")
plt.plot(train_accs, label="Training Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()
#%%
model.eval()  # 设置模型为评估模式以禁用dropout
with torch.no_grad():
    output = model(x_test)
    pred = output.argmax(dim=1)  # 获取每个样本的预测类
    accuracy = (pred == y_test).sum().item() / len(y_test)  # 计算模型在测试集上的准确率
    print(f"Accuracy: {accuracy}")
# model.plot_attention_weights([w.cpu().numpy() for w in _])

#绘制混淆矩阵
y_pred = pred.cpu().numpy()
y_true = y_test.cpu().numpy()
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')

print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true, y_pred, target_names=['AWNP', 'AWP', 'DWNP', 'DWP']))

#计算F1值
f1 = f1_score(y_true, y_pred, average='weighted')
print('F1 score:', f1)
#计算召回率
recall = recall_score(y_true, y_pred, average='weighted')
print('Recall:', recall)
#计算精确率
precision = precision_score(y_true, y_pred, average='weighted')
print('Precision:', precision)

#计算MCC
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
print('MCC:', mcc)

#%%
i = 8
#保存y_pred, y_true到同一个csv文件
import pandas as pd
y_pred = pd.DataFrame(y_pred)
y_true = pd.DataFrame(y_true)
#合并两个dataframe
result = pd.concat([y_pred, y_true], axis=1)
result.to_csv(f'result\\result_{i}.csv', index=False, header=False)

#保存f1, recall, precision到csv文件
import csv
with open(f'result\\point_{i}.csv', 'a', newline='') as csvfile:
    #如果有内容，先清空
    csvfile.seek(0)
    csvfile.truncate()
    writer = csv.writer(csvfile)
    writer.writerow(['F1', f1])
    writer.writerow(['Recall', recall])
    writer.writerow(['Precision', precision])
    writer.writerow(['MCC', mcc])

#%%
f1_avg = []
recall_avg = []
precision_avg = []
mcc_avg = []
for i in range(10):
    #读取point_i.csv文件,并计算对应的平均值
    with open(f'result\\point_{i}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        f1 = float(rows[0][1])
        recall = float(rows[1][1])
        precision = float(rows[2][1])
        mcc = float(rows[3][1])
        f1_avg.append(f1)
        recall_avg.append(recall)
        precision_avg.append(precision)
        mcc_avg.append(mcc)
        print(f1, recall, precision, mcc)
#计算平均值
f1_avg = sum(f1_avg) / len(f1_avg)
recall_avg = sum(recall_avg) / len(recall_avg)
precision_avg = sum(precision_avg) / len(precision_avg)
mcc_avg = sum(mcc_avg) / len(mcc_avg)
print(f1_avg, recall_avg, precision_avg, mcc_avg)

# %%
#读取