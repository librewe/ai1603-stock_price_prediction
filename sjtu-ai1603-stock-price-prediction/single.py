import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

DAYS_FOR_TRAIN = 30
TRAIN_SIZE = 0.9
TRAIN_EPOCHS = 100

class LSTM_Regression(nn.Module):
    def __init__(self, input_size, output_size, 
                 hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=False, dropout=dropout, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)
        self.relu = nn.ReLU()

    def forward(self, _x, hidden=None):
        x, hidden = self.lstm(_x, hidden)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc2(self.relu(self.fc1(x)))
        x = x.view(s, b, -1)
        return x, hidden

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

def split_data_resize(data, train_size=TRAIN_SIZE, drop_last=True, device=None):
    def create_dataset(data, days_for_train=DAYS_FOR_TRAIN):
        dataset_x, dataset_y = [], []
        for i in range(len(data) - days_for_train):
            _x = data[i:(i + days_for_train)]
            dataset_x.append(_x)
            dataset_y.append(data[i + days_for_train])
        return (np.array(dataset_x), np.array(dataset_y))
    data = data[:int(len(data)*train_size)]
    train_x, train_y = create_dataset(data)
    # if drop_last:
    #     train_x, train_y = train_x[:-1], train_y[:-1] #最后一个data是None
    ## 然而并不是
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, 1)
    train_x = torch.from_numpy(train_x).to(device)
    train_y = torch.from_numpy(train_y).to(device)
    return train_x, train_y

class Train_Single_Stock:
    train_data = pd.read_csv('train.csv')
    def __init__(self, symbol, epochs=TRAIN_EPOCHS,
                 model=LSTM_Regression(DAYS_FOR_TRAIN, 1)):
        self.symbol = symbol
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prepare_data()
        self.model = model.to(self.device)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=2e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.train_loss = []
        self.test_loss = []

    def prepare_data(self):
        self.data = Train_Single_Stock.train_data[Train_Single_Stock.train_data['symbol']==self.symbol]#[:1000]
        self.data_close = self.data[['close']].astype('float32').values# 转换数据类型
        self.data_close, self.max_value, self.min_value = normalize_data(self.data_close)
        
        self.train_size = int(len(self.data_close) * TRAIN_SIZE)
        self.test_size = len(self.data_close) - self.train_size
        self.train_xy = split_data_resize(self.data_close)
        self.test_xy = split_data_resize(self.data_close[self.train_size:], train_size=1.0, drop_last=False, device=self.device)

    def set_train_data(self,data=None):
        self.train_x, self.train_y = self.train_xy
        self.test_x, self.test_y = self.test_xy
        self.train_x = self.train_x.to(self.device)
        self.train_y = self.train_y.to(self.device)
        self.test_x = self.test_x.to(self.device)
        self.test_y = self.test_y.to(self.device)
        if data is None:
            return
        else:
            data = data.view(-1).data.cpu().numpy()
            data = np.concatenate((self.train_x[0].view(-1).data.cpu().numpy(), data),dtype=np.float32)
            self.train_x_pred, self.train_y_pred = split_data_resize(data, train_size=1.0, drop_last=False, device=self.device)
            self.train_x_pred = self.train_x_pred.to(self.device)
            self.train_y_pred = self.train_y_pred.to(self.device)

    def train(self, hidden=None):
        self.model.train()
        self.set_train_data()
        loss_function = self.loss_function
        optimizer = self.optimizer
        for i in range(self.epochs):
            out, _ = self.model(self.train_x, hidden)
            # out += (torch.randn_like(out)-0.5) * 0.001 #@1
            loss = loss_function(out, self.train_y)
            if __name__ == '__main__':
                loss1 = loss_function(torch.tensor(self.eval()[0],device=self.device ), self.test_y)
                self.test_loss.append(loss1.item())
                self.model.train()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.set_train_data(out)
            # for j in range(len(self.train_x)):# 计划采样-随机替换一部分输入数据 #@2
            #     if np.random.rand() < 1/(1+np.exp(-((1-i/self.epochs)*10-5))):
            #         self.train_x[j] = self.train_x_pred[j]
            self.train_loss.append(loss.item())
            if __name__ == '__main__':
                print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    def eval(self, period=None, hidden=None):
        model = self.model.eval()
        train_x, _ = split_data_resize(self.data_close, TRAIN_SIZE,
                                       drop_last=False, device=self.device)
        if period is None:
            pred_train, hidden = model(train_x, hidden)  # (seq_size, batch_size, output_size)
            pred_train = pred_train.cpu()
            pred_train = pred_train.view(-1).data.numpy()
            pred_train = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_train),dtype=np.float32)  # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
            assert len(pred_train) == self.train_size
            return pred_train, hidden
        else:
            dataset_x = train_x[-1:].reshape(-1, 1, DAYS_FOR_TRAIN).to(self.device)
            hidden_old = hidden
            y_ = [model(dataset_x, hidden)[0].cpu()]
            while len(y_) < period:
                y = [[i[0] for i in y_[-DAYS_FOR_TRAIN:]]]
                y_tensor = torch.tensor(y, dtype=dataset_x.dtype).to(self.device)
                if (-DAYS_FOR_TRAIN + len(y_)) < 0:
                    part1 = dataset_x[-1][:, -DAYS_FOR_TRAIN + len(y_):]
                else:
                    part1 = torch.empty((1, 0), dtype=dataset_x.dtype).to(self.device)
                new_input = torch.cat([part1, y_tensor], dim=1).to(self.device)
                new_input = new_input.reshape(1, 1, DAYS_FOR_TRAIN)
                # new_input = new_input.mean()+ 1.02* (new_input - new_input.mean())  #@3
                next_pred, hidden = model(new_input, hidden)
                # if len(y_) % 30 == 29:#np.random.randint(0, 30): #@4
                #     hidden = hidden_old  # 随机地复位隐藏层
                next_pred = next_pred.cpu()
                y_.append(next_pred)
            pred_test = np.array([i[0].view(-1).data.numpy() for i in y_])
            return pred_test, hidden

if __name__ == '__main__':
    ins = Train_Single_Stock('NVDA')
    ins.train()
    pred_train, hidden = ins.eval()
    pred_test, _ = ins.eval(ins.test_size, hidden)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(ins.train_loss, label='train_loss')
    plt.plot(ins.test_loss, label='test_loss')
    plt.subplot(1, 2, 2)
    plt.plot(ins.data[['close']].reset_index(drop=True), 'b', label='real')
    plt.plot(pred_train * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='real')
    x = len(pred_train)
    plt.plot(range(x, x + len(pred_test)), pred_test * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='prediction')
    plt.plot((ins.train_size, ins.train_size), (0, ins.max_value), 'g--')
    plt.show()

# torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用
# model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数

# model_total = sum([param.nelement() for param in model.parameters()])  # 计算模型参数
# print("Number of model_total parameter: %.8fM" % (model_total / 1e6))

# for j in range(len(self.train_x)):
    # for k in range(DAYS_FOR_TRAIN):
    #     # if np.random.rand() < 2 * i / self.epochs - 1.2:
    #     #     self.train_x[j][0][k] = self.train_x_pred[j][0][k] 
    #         # self.train_y[j][0][k] = self.train_y_pred[j][0][k]
    #     if np.random.randn() < 0.3:
    #         if k == DAYS_FOR_TRAIN-1:
    #             self.train_x[j][0][k] = self.train_x_pred[j][0][k-1] 
    # k = np.random.randint(4, DAYS_FOR_TRAIN-2)
    # self.train_x[j][0][k+1:] = self.train_x_pred[j][0][k]+np.random.randn() * 0.01
    # pass