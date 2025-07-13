import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

DAYS_FOR_TRAIN = 15
TRAIN_SIZE = None
TRAIN_EPOCHS = 100

class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x, hidden=None):
        x, hidden = self.lstm(_x, hidden)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x, hidden

def create_dataset(data, days_for_train=DAYS_FOR_TRAIN) -> (np.array, np.array):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))

model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2)  # 导入模型并设置模型的参数输入输出层、隐藏层等

# model_total = sum([param.nelement() for param in model.parameters()])  # 计算模型参数
# print("Number of model_total parameter: %.8fM" % (model_total / 1e6))

class Train_Single_Stock:

    train_data = pd.read_csv('train.csv')

    def __init__(self, symbol, model=model, epochs=TRAIN_EPOCHS):
        self.model = model
        self.symbol = symbol
        self.data = Train_Single_Stock.train_data[Train_Single_Stock.train_data['symbol']==symbol]#[:1000]
        self.data_close = self.data[['close']].astype('float32').values# 转换数据类型
        self.normalize_data()
        self.train_loss = []
        self.epochs = epochs

    def normalize_data(self):
        # 将价格标准化到0~1
        self.max_value = np.max(self.data_close)
        self.min_value = np.min(self.data_close)
        self.data_close = (self.data_close - self.min_value) / (self.max_value - self.min_value)

        self.dataset_x, self.dataset_y = create_dataset(self.data_close)
        # if TRAIN_SIZE < 1.0:
        self.train_size = int(len(self.dataset_x) * TRAIN_SIZE)
        self.train_x = self.dataset_x[:self.train_size]
        self.train_y = self.dataset_y[:self.train_size]

        self.train_x = self.train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
        self.train_y = self.train_y.reshape(-1, 1, 1)

        self.train_x = torch.from_numpy(self.train_x)
        self.train_y = torch.from_numpy(self.train_y)

    def train(self, hidden=None):
        loss_function = nn.HuberLoss()#@
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        for i in range(self.epochs):
            out, _ = self.model(self.train_x, hidden)
            loss = loss_function(out, self.train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.train_loss.append(loss.item())
            with open('log.txt', 'a+') as f:
                f.write('{} - {}\n'.format(i + 1, loss.item()))
            if __name__ == '__main__':
                if (i + 1) % 1 == 0:
                    print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    def eval(self, period=None, hidden=None):
        model = self.model.eval()  # 转换成评估模式
        if period is None:
            dataset_x = self.dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
            dataset_x = torch.from_numpy(dataset_x)
            pred_test, hidden = model(dataset_x, hidden)  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
            pred_test = pred_test.view(-1).data.numpy()
            pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))  # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
            assert len(pred_test) == len(self.data_close)
        else:
            dataset_x = self.dataset_x[-1:]
            dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)
            dataset_x = torch.from_numpy(dataset_x)
            y_ = [model(dataset_x, hidden)[0]]
            while len(y_) < period:
                y = [[i[0] for i in y_[-DAYS_FOR_TRAIN:]]]
                y_tensor = torch.tensor(y, dtype=dataset_x.dtype)
                if (-DAYS_FOR_TRAIN + len(y_)) < 0:
                    part1 = dataset_x[-1][:, -DAYS_FOR_TRAIN + len(y_):]
                else:
                    part1 = torch.empty((1, 0), dtype=dataset_x.dtype)
                new_input = torch.cat([part1, y_tensor], dim=1)  # 须手动指定拼接维度
                new_input = new_input.reshape(1, 1, DAYS_FOR_TRAIN)
                next_pred = model(new_input, hidden)[0]
                y_.append(next_pred)
            pred_test = np.array([i[0].view(-1).data.numpy() for i in y_])
        if __name__ == '__main__':
            plt.figure()
            plt.plot(self.train_loss, 'b', label='loss')
            plt.title("Train_Loss_Curve")
            plt.ylabel('train_loss')
            plt.xlabel('epoch_num')
            plt.savefig('loss.png', format='png', dpi=200)
            plt.close()
            # 绘制真实值和预测值
            plt.plot(self.data_close, 'b', label='real')
            plt.plot(pred_test, 'r', label='prediction')
            plt.plot((self.train_size, self.train_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
            plt.legend(loc='best')
            plt.savefig('result.png', format='png', dpi=200)
            plt.close()
        return pred_test, hidden

if __name__ == '__main__':
    instance = Train_Single_Stock('ILMN')
    instance.train()
    print('Epoch: {}, Loss:{:.5f}'.format(TRAIN_EPOCHS, instance.train_loss[-1]))
    pred_test = instance.eval()
    plt.figure()
    plt.plot(instance.data[['close']].reset_index(drop=True), 'b', label='real')
    plt.plot(pred_test * (instance.max_value-instance.min_value) + instance.min_value, 'r', label='prediction')
    plt.show()

# torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用
# model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数
