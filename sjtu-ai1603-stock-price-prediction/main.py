# import single_multistep as single
import single
import pandas as pd
import matplotlib.pyplot as plt
single.TRAIN_SIZE = 1.0
test_data = pd.read_csv('test_template.csv')

data_close =test_data[['close']].astype('float32').values  # 转换数据类型
ins = single.Train_Single_Stock('SYMC')
ins.train() 
pred_train, hidden = ins.eval(period=None)
pred_test, _ = ins.eval(period=len(test_data[test_data['symbol']=='SYMC']), hidden=hidden)
plt.figure()
plt.plot(ins.data[['close']].reset_index(drop=True), 'b', label='real')
plt.plot(pred_train * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='real')
x = len(pred_train)
plt.plot(range(x, x + len(pred_test)), pred_test * (ins.max_value - ins.min_value) + ins.min_value, 'r', label='prediction')
plt.show()