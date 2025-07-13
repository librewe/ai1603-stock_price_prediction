import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

DAYS_FOR_TRAIN = 15
DAYS_TO_PREDICT = 100
TRAIN_SIZE = 1.0
TRAIN_EPOCHS = 1000

class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=DAYS_TO_PREDICT, num_layers=2):
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

def normalize_data(data):
    max_value = np.max(data)
    min_value = np.min(data)
    return (data - min_value) / (max_value - min_value), max_value, min_value

def split_data_resize(data, train_size=TRAIN_SIZE):
    def create_dataset(data, days_for_train=DAYS_FOR_TRAIN) -> (np.array, np.array):
        dataset_x, dataset_y = [], []
        for i in range(len(data) - days_for_train - DAYS_TO_PREDICT+1):
            _x = data[i:(i + days_for_train)]
            dataset_x.append(_x)
            dataset_y.append(data[i + days_for_train:i + days_for_train + DAYS_TO_PREDICT])
        return (np.array(dataset_x), np.array(dataset_y))
    dataset_x, dataset_y = create_dataset(data)
    train_size = int(len(dataset_x) * train_size)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, DAYS_TO_PREDICT)
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    return train_x, train_y

model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=DAYS_TO_PREDICT, num_layers=2)  # 导入模型并设置模型的参数输入输出层、隐藏层等

# model_total = sum([param.nelement() for param in model.parameters()])  # 计算模型参数
# print("Number of model_total parameter: %.8fM" % (model_total / 1e6))

class Train_Single_Stock:
    train_data = pd.read_csv('train.csv')
    def __init__(self, symbol, model=model, epochs=TRAIN_EPOCHS):
        self.model = model
        self.symbol = symbol
        self.data = Train_Single_Stock.train_data[Train_Single_Stock.train_data['symbol']==symbol]#[:1000]
        self.data_close = self.data[['close']].astype('float32').values# 转换数据类型
        self.data_close, self.max_value, self.min_value = normalize_data(self.data_close)
        self.train_size = int(len(self.data_close) * TRAIN_SIZE)
        self.train_xy = split_data_resize(self.data_close)
        self.train_loss = []
        self.epochs = epochs

    def set_train_data(self,data=None):
        if data is None:
            self.train_x, self.train_y = self.train_xy
        else:
            data = data.view(-1).data.numpy()
            data = np.concatenate((np.zeros(DAYS_FOR_TRAIN), data),dtype=np.float32)
            assert len(data) == len(self.data_close)
            self.train_x_pred, _ = split_data_resize(data)

    def train(self, hidden=None):
        self.set_train_data()
        loss_function = nn.MSELoss()#@
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        hidden_old = hidden
        for i in range(self.epochs):
            out, hidden = self.model(self.train_x, hidden)
            out += (torch.randn_like(out)-0.5) * 0.2
            loss = loss_function(out, self.train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # self.set_train_data(out[:,:, :1])
            # for j in range(len(self.train_x)):
            #     if np.random.rand() < 2*i / self.epochs - 1.2:
            #         self.train_x[j] = self.train_x_pred[j]  # 随机替换一部分数据
            self.train_loss.append(loss.item())
            with open('log.txt', 'a+') as f:
                f.write('{} - {}\n'.format(i + 1, loss.item()))
            if __name__ == '__main__':
                if (i + 1) % 1 == 0:
                    print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    def eval(self, period=None, hidden=None):
        model = self.model.eval()  # 转换成评估模式
        test_x, _ = split_data_resize(self.data_close, 1.0)  # (seq_size, batch_size, feature_size)
        if period is None:
            pred_test, hidden = model(test_x, hidden)  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
            pred_test_1 = pred_test[:, :, :1]
            pred_test = pred_test[-1].view(-1).data.numpy()
            pred_test_1 = pred_test_1.view(-1).data.numpy()
            pred_test_1 = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test_1))  # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
            pred_test = np.concatenate((pred_test_1,pred_test[-DAYS_TO_PREDICT+1:]))  # 补齐预测长度
            assert len(pred_test) == len(self.data_close)
        else:
            dataset_x = test_x[-1:]
            dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)
            pred_test = model(dataset_x, hidden)[0].view(-1).data.numpy()
            # y_ = [model(dataset_x, hidden)[0][:, :1]]
            # while len(y_) < period:
            #     y = [[i[0][:, :1] for i in y_[-DAYS_FOR_TRAIN:]]]
            #     y_tensor = torch.tensor(y, dtype=dataset_x.dtype)
            #     if (-DAYS_FOR_TRAIN + len(y_)) < 0:
            #         part1 = dataset_x[-1][:, -DAYS_FOR_TRAIN + len(y_):]
            #     else:
            #         part1 = torch.empty((1, 0), dtype=dataset_x.dtype)
            #     new_input = torch.cat([part1, y_tensor], dim=1)  # 须手动指定拼接维度
            #     new_input = new_input.reshape(1, 1, DAYS_FOR_TRAIN)
            #     # new_input = new_input.mean()+ 2* (new_input - new_input.mean())#@
            #     next_pred = model(new_input, hidden)[0]
            #     y_.append(next_pred)
            # pred_test = np.array([i[0][:, :1].view(-1).data.numpy() for i in y_])
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
