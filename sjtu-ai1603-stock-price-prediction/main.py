# import single_multistep as single
import single
import pandas as pd
import matplotlib.pyplot as plt
single.TRAIN_SIZE = 1.0

train_data=pd.read_csv('train.csv')
# print(single.preprocess_data(train_data))

test_data = pd.read_csv('test_template.csv')
data_symbol = test_data['symbol'].value_counts()
data_close =test_data[['close']].astype('float32').values  # 转换数据类型

def stock_prediction(stock_name):
    ins = single.Train_Single_Stock(stock_name)
    ins.train()
    pred_train, hidden = ins.eval(period=None)
    pred_test, _ = ins.eval(period=data_symbol[stock_name], hidden=hidden)
    test_data.loc[test_data['symbol']==stock_name, 'close'] = pred_test * (ins.max_value - ins.min_value) + ins.min_value
    # plt.figure()
    # plt.plot(ins.data[['close']].reset_index(drop=True), 'b', label='real')
    # plt.plot(pred_train * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='real')
    # x = len(pred_train)
    # plt.plot(range(x, x + len(pred_test)), pred_test * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='prediction')
    # plt.show()
cnt=0
for stock in test_data['symbol'].unique():
    stock_prediction(stock)
    print(f'Finished predicting for {stock}. ', cnt:= cnt+1, f'/{len(test_data["symbol"].unique())}',sep='')
    # break
test_data = test_data[['id', 'close']]
test_data.to_csv('submission.csv', index=False)