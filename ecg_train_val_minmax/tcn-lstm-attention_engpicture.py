import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from tqdm import tqdm

# 移除中文显示设置，使用默认英文字体

# 数据集类
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 数据加载与预处理
def load_and_preprocess_data(train_path, val_path):
    # 加载数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # 提取信号和标签
    X_train = train_df['signals'].apply(lambda x: np.array([float(i) for i in x.split(',')])).values
    y_train = train_df['type'].values
    X_val = val_df['signals'].apply(lambda x: np.array([float(i) for i in x.split(',')])).values
    y_val = val_df['type'].values
    
    # 确定最大序列长度并填充序列
    max_length = max([len(seq) for seq in X_train] + [len(seq) for seq in X_val])
    X_train = np.array([np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in X_train])
    X_val = np.array([np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in X_val])
    
    # 标签编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # 重塑数据 (样本数, 通道数, 时间步) - PyTorch使用通道优先
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    
    print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train_encoded.shape}")
    print(f"Validation set shape: {X_val.shape}, Validation labels shape: {y_val_encoded.shape}")
    print(f"Classes: {label_encoder.classes_}")
    
    return X_train, y_train_encoded, X_val, y_val_encoded, label_encoder, max_length

# 多头注意力版本的TCN-LSTM-Attention模型（带注意力权重保存）
class TCNLSTM多头AttentionModel(nn.Module):
    def __init__(self, input_length, num_classes, num_heads=8):  # 增加多头参数，默认8头
        super(TCNLSTM多头AttentionModel, self).__init__()
        
        # TCN部分
        self.tcn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding='same', dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding='same', dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding='same', dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2)
        )
        
        # LSTM部分
        lstm_input_length = input_length // (2*2*2)
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        
        # 注意力层（多头注意力）
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # 保存注意力权重
        self.attn_weights = None
        
    def forward(self, x):
        # TCN部分
        x = self.tcn(x)  # (batch, 256, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 256)
        
        # LSTM部分
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)  # (batch, seq_len, 64)
        
        # 注意力部分（保存权重）
        attn_output, attn_weights = self.attention(x, x, x)  # (batch, seq_len, 64), (batch, num_heads, seq_len, seq_len)
        self.attn_weights = attn_weights  # 保存注意力权重用于后续可视化
        x = torch.mean(attn_output, dim=1)  # 全局平均池化
        
        # 全连接层
        x = self.fc(x)
        return x

# 训练模型
def train_model(model, train_loader, val_loader, num_classes, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, min_lr=0.0001)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []  # 新增：记录学习率变化
    }
    
    best_val_acc = 0.0
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_loss /= len(val_loader)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        print(f'Current learning rate: {current_lr:.6f}\n')  # 打印当前学习率
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_ecg_model_multihead.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 10:
                print("Early stopping triggered, stopping training")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_ecg_model_multihead.pth'))
    return model, history

# 注意力权重可视化函数（适配多头注意力）
def plot_attention_weights(ecg_signal, attn_weights, true_label, pred_label, class_names, idx):
    """
    可视化ECG信号和对应的注意力权重
    ecg_signal: 原始ECG信号 (时间步,)
    attn_weights: 注意力权重 (时间步,)
    true_label: 真实标签
    pred_label: 预测标签
    class_names: 类别名称列表
    idx: 样本索引
    """
    fig, ax1 = plt.subplots(figsize=(15, 12))
    
    # 绘制ECG信号
    color = 'tab:blue'
    ax1.set_xlabel('Time Step', fontsize=14)
    ax1.set_ylabel('Signal Strength', color=color, fontsize=14)
    ax1.plot(ecg_signal, color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    
    # 创建第二个y轴用于绘制注意力权重
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Attention Weight', color=color, fontsize=14)
    ax2.plot(attn_weights, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.set_ylim(0, max(attn_weights) * 1.2)  # 调整权重轴范围
    
    fig.tight_layout()
    plt.title(f'\nSample {idx} - True Label: {class_names[true_label]} - Predicted Label: {class_names[pred_label]} (Multi-head Attention)', 
              fontsize=16)
    # 改动1：保存为SVG格式，添加bbox_inches='tight'防止文字截断
    plt.savefig(f'attention_visualization_multihead_sample_{idx}.svg', format='svg', bbox_inches='tight')
    plt.show()

# 新增：绘制学习率衰减图
def plot_learning_rate(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['learning_rates'], label='Learning Rate', color='green', linewidth=2)
    plt.title('TCN-LSTM-Multi-head Attention Model Learning Rate Decay', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')  # 使用对数刻度更清晰地展示学习率变化
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    # 改动2：保存为SVG格式
    plt.savefig('tcn_lstm_multihead_learning_rate.svg', format='svg', bbox_inches='tight')
    plt.show()

# 模型评估（适配多头注意力）
def evaluate_model(model, val_loader, label_encoder, X_val_raw):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    y_pred = []
    y_true = []
    all_attn_weights = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # 获取注意力权重并检查形状
            attn_weights = model.attn_weights.cpu().numpy()
            
            # 调整注意力权重处理逻辑
            if attn_weights.ndim == 4:
                # 多头注意力权重: (batch, num_heads, seq_len, seq_len)
                avg_attn_weights = np.mean(attn_weights, axis=1)  # 平均多头注意力
                avg_attn_weights = np.mean(avg_attn_weights, axis=1)  # 平均第一个序列维度
            elif attn_weights.ndim == 3:
                # 可能的其他形状: (batch, seq_len, seq_len)
                avg_attn_weights = np.mean(attn_weights, axis=1)
            else:
                # 处理意外形状
                raise ValueError(f"Unexpected attention weights shape: {attn_weights.shape}")
                
            all_attn_weights.extend(avg_attn_weights)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.numpy())
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # 分类报告
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    colors = [(1, 1, 1), (0, 0, 1)]
    cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=100)
    
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im)
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('TCN-LSTM-Multi-head Attention Confusion Matrix', fontsize=16)
    plt.tight_layout()
    # 改动3：保存为SVG格式
    plt.savefig('tcn_lstm_multihead_confusion_matrix.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    # 可视化每个类别的注意力权重（各选一个样本）
    for class_idx in range(len(class_names)):
        # 找到该类别的第一个样本
        sample_idx = next(i for i, label in enumerate(y_true) if label == class_idx)
        # 获取原始信号
        raw_signal = X_val_raw[sample_idx].flatten()
        # 获取注意力权重
        attn_weights = all_attn_weights[sample_idx]
        # 调整注意力权重长度以匹配原始信号
        attn_weights_upsampled = np.interp(
            np.linspace(0, len(attn_weights)-1, len(raw_signal)),
            np.arange(len(attn_weights)),
            attn_weights
        )
        # 绘制注意力可视化
        plot_attention_weights(
            raw_signal, 
            attn_weights_upsampled,
            y_true[sample_idx], 
            y_pred[sample_idx],
            class_names,
            sample_idx
        )
    
    return y_pred, all_attn_weights

# 绘制训练过程
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('TCN-LSTM-Multi-head Attention Model Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('TCN-LSTM-Multi-head Attention Model Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    # 改动4：保存为SVG格式
    plt.savefig('tcn_lstm_multihead_training_history.svg', format='svg', bbox_inches='tight')
    plt.show()

# 主函数
def main():
    # 数据路径 - 请根据实际情况修改
    train_path = r"/public/home/wuchengliang/ecg_train_val_minmax/train.csv"
    val_path = r"/public/home/wuchengliang/ecg_train_val_minmax/val.csv"
    
    # 加载和预处理数据
    X_train, y_train, X_val, y_val, label_encoder, max_length = load_and_preprocess_data(train_path, val_path)
    
    # 保存原始验证集信号用于可视化
    val_df = pd.read_csv(val_path)
    X_val_raw = val_df['signals'].apply(lambda x: np.array([float(i) for i in x.split(',')])).values
    
    # 创建数据集和数据加载器
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 构建多头注意力模型
    num_classes = len(label_encoder.classes_)
    model = TCNLSTM多头AttentionModel(max_length, num_classes, num_heads=8)  # 使用8头注意力
    
    # 训练模型
    model, history = train_model(model, train_loader, val_loader, num_classes, epochs=50)
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 新增：绘制学习率衰减图
    plot_learning_rate(history)
    
    # 评估模型（包含注意力可视化）
    evaluate_model(model, val_loader, label_encoder, X_val_raw)
    
    print("TCN-LSTM-Multi-head Attention model training and evaluation completed!")

if __name__ == "__main__":
    main()