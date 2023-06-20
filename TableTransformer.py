#%%
# 导入所需的包
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from myTransformer import Transformer, CustomDataset, FocalLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
from read_data import Read_data

# 读取数据
Data = Read_data()
train_data, x_test, train_targets, y_test = Data.Transformer_data_286(Normalization=True, data2tensor=True, zero=True)

epochs = 150
# 初始化模型,优化器和损失函数
model = Transformer(
    input_dim=train_data.shape[1],
    output_dim=4,
    hidden_dim=128,
    num_layers=2,
    num_heads=10,
    dropout=0.5,
)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# criterion = nn.CrossEntropyLoss(reduction="none")  # 交叉熵损失函数
criterion = FocalLoss(gamma=2, alpha=1)
train_dataset = CustomDataset(train_data, train_targets)
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
    for batch_idx, (data1, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度重置
        data1 = data1.to(torch.float32)  # 转换为浮点数tensor
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

#%%
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# #计算ROC曲线的值
# fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# #计算AUC的值
# roc_auc = auc(fpr, tpr)
# #绘制ROC曲线
# plt.figure()
# lw = 2
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='darkorange',
#             lw=lw, label='ROC curve (area = %0.2f)'% roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# #设置x轴和y轴的取值范围
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# #设置x轴和y轴的标签
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# #设置标题
# plt.title('Receiver operating characteristic example')
# #显示图例
# plt.legend(loc="lower right")
# plt.show()

