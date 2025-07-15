import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

DAYS_FOR_TRAIN = 50
TRAIN_SIZE = 0.8
TRAIN_EPOCHS = 500

class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.1)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)
        self.relu = nn.ReLU()

    def forward(self, _x, hidden=None):
        x, hidden = self.lstm(_x, hidden)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc2(self.relu(self.fc1(x)))
        x = x.view(s, b, -1)  # 把形状改回来
        return x, hidden

# 数据预处理：缺失值处理，异常值处理以及数据归一化
def preprocess_data(data):
    print("检查缺失值...")
    if data.isnull().sum().sum() > 0:
        print("存在缺失值，进行填充...")
        data.fillna(method='ffill', inplace=True)# 前向填充
        data.fillna(method='bfill', inplace=True)# 后向填充
        if data.isnull().sum().sum() > 0:# 检查是否还有缺失值
            print("仍有缺失值，考虑删除或进一步处理...")
            data.dropna(inplace=True)# 删除包含缺失值的行
    else:
        print("数据中没有缺失值。")
    print("检查异常值...")
    if (data['close'] <= 0).any():
        print("存在异常值，删除包含负值或零值的行...")
        data = data[(data['close'] > 0)]
    else:
        print("没有异常值。")
    return data

def normalize_data(data):
    max_value = np.max(data)
    min_value = np.min(data)
    return (data - min_value) / (max_value - min_value), max_value, min_value

def split_data_resize(data, train_size=TRAIN_SIZE):
    def create_dataset(data, days_for_train=DAYS_FOR_TRAIN):
        dataset_x, dataset_y = [], []
        for i in range(len(data) - days_for_train):
            _x = data[i:(i + days_for_train)]
            dataset_x.append(_x)
            dataset_y.append(data[i + days_for_train])
        return (np.array(dataset_x), np.array(dataset_y))
    dataset_x, dataset_y = create_dataset(data)
    train_size = int(len(dataset_x) * train_size)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, 1)
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    return train_x, train_y

model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2)  # 导入模型并设置模型的参数输入输出层、隐藏层等

# model_total = sum([param.nelement() for param in model.parameters()])  # 计算模型参数
# print("Number of model_total parameter: %.8fM" % (model_total / 1e6))

class Train_Single_Stock:
    train_data = pd.read_csv('train.csv')
    def __init__(self, symbol, model=model, epochs=TRAIN_EPOCHS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.symbol = symbol
        self.data = Train_Single_Stock.train_data[Train_Single_Stock.train_data['symbol']==symbol]#[:1000]
        self.data_close = self.data[['close']].astype('float32').values# 转换数据类型
        self.data_close, self.max_value, self.min_value = normalize_data(self.data_close)
        self.train_size = int(len(self.data_close) * TRAIN_SIZE)
        self.test_size = len(self.data_close) - self.train_size
        self.train_xy = split_data_resize(self.data_close)
        self.train_loss = []
        self.epochs = epochs

    def set_train_data(self,data=None):
        self.train_x, self.train_y = self.train_xy
        self.train_x = self.train_x.to(self.device)
        self.train_y = self.train_y.to(self.device)
        if data is None:
            return
        else:
            data = data.view(-1).data.numpy()
            data = np.concatenate((np.zeros(DAYS_FOR_TRAIN), data),dtype=np.float32)
            assert len(data) == len(self.train_x)
            self.train_x_pred, self.train_y_pred = split_data_resize(data)

    def train(self, hidden=None):
        self.set_train_data()
        loss_function = nn.MSELoss()#@
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        for i in range(self.epochs):
            out, _ = self.model(self.train_x, hidden)
            # out += (torch.randn_like(out)-0.5) * 0.1
            loss = loss_function(out, self.train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # self.set_train_data(out)
            # for j in range(len(self.train_x)):
            #     if np.random.rand() < 2*i / self.epochs - 1.3:
            #         self.train_x[j] = self.train_x_pred[j]  # 随机替换一部分数据
            for j in range(len(self.train_x)):
                # for k in range(DAYS_FOR_TRAIN):
                #     # if np.random.rand() < 2 * i / self.epochs - 1.2:
                #     #     self.train_x[j][0][k] = self.train_x_pred[j][0][k] 
                #         # self.train_y[j][0][k] = self.train_y_pred[j][0][k]
                #     if np.random.randn() < 0.3:
                #         if k == DAYS_FOR_TRAIN-1:
                #             self.train_x[j][0][k] = self.train_x_pred[j][0][k-1] 
                # k = np.random.randint(4, DAYS_FOR_TRAIN-2)
                # self.train_x[j][0][k+1:] = self.train_x_pred[j][0][k]+np.random.randn() * 0.01
                pass
            self.train_loss.append(loss.item())
            with open('log.txt', 'a+') as f:
                f.write('{} - {}\n'.format(i + 1, loss.item()))
            if __name__ == '__main__':
                if (i + 1) % 1 == 0:
                    print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    def eval(self, period=None, hidden=None):
        model = self.model.eval()  # 转换成评估模式
        test_x, _ = split_data_resize(self.data_close, TRAIN_SIZE)  # (seq_size, batch_size, feature_size)
        test_x = test_x.to(self.device)
        if period is None:#False:#
            pred_test, hidden = model(test_x, hidden)  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
            pred_test = pred_test.cpu()
            pred_test = pred_test.view(-1).data.numpy()
            pred_test = np.concatenate((np.zeros(int(DAYS_FOR_TRAIN*TRAIN_SIZE)), pred_test),dtype=np.float32)  # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
            assert len(pred_test) == self.train_size
        else:
            dataset_x = test_x[-1:]
            dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)
            dataset_x = dataset_x.to(self.device)
            # dataset_x = torch.from_numpy(dataset_x)
            hidden_old = hidden
            y_ = [model(dataset_x, hidden)[0].cpu()]
            while len(y_) < period:
                y = [[i[0] for i in y_[-DAYS_FOR_TRAIN:]]]
                y_tensor = torch.tensor(y, dtype=dataset_x.dtype).to(self.device)
                if (-DAYS_FOR_TRAIN + len(y_)) < 0:
                    part1 = dataset_x[-1][:, -DAYS_FOR_TRAIN + len(y_):]
                else:
                    part1 = torch.empty((1, 0), dtype=dataset_x.dtype).to(self.device)
                new_input = torch.cat([part1, y_tensor], dim=1).to(self.device)  # 须手动指定拼接维度
                new_input = new_input.reshape(1, 1, DAYS_FOR_TRAIN)
                # new_input = new_input.mean()+ 2* (new_input - new_input.mean())#@
                # next_pred = model(new_input, hidden)[0]
                next_pred, hidden = model(new_input, hidden)
                if len(y_) % 40 == 30: #np.random.randint(0, 15):
                    hidden = hidden_old
                next_pred = next_pred.cpu()
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
    ins = Train_Single_Stock('ILMN')
    ins.train()
    # print('Epoch: {}, Loss:{:.5f}'.format(TRAIN_EPOCHS, instance.train_loss[-1]))
    pred_train, hidden = ins.eval()
    pred_test, _ = ins.eval(ins.test_size, hidden)
    plt.figure()
    plt.plot(ins.data[['close']].reset_index(drop=True), 'b', label='real')
        # plt.plot(ins.data[['close']].reset_index(drop=True), 'b', label='real')
    plt.plot(pred_train * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='real')
    x = len(pred_train)
    plt.plot(range(x, x + len(pred_test)), pred_test * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='prediction')
    # plt.plot(pred_test * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='prediction')
    plt.plot((ins.train_size, ins.train_size), (0, ins.max_value), 'g--')
    plt.show()

# torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用
# model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数
