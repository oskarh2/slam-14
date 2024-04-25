import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

def scaling_window(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x),np.array(y)
def prepare_data(training_set):
    #print(training_set)
    sc = MinMaxScaler()
    #print('sc')
    #print(sc)
    training_data = sc.fit_transform(training_set)    
    x, y = scaling_window(training_data, seq_length)
    #x, y = scaling_window(training_set, seq_length)
    #print('scaling_window')
    #print('x:',x,'y:',y)
    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size
    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))
    return sc, train_size,test_size, dataX, dataY



class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

def lstm_exec(dataX, dataY,num_epochs,learning_rate):
    input_size = 1
    hidden_size = 2
    num_layers = 1
    num_classes = 1
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
# Train the model
    for epoch in range(num_epochs):
        outputs = lstm(dataX)
        optimizer.zero_grad()
    # obtain the loss function
        loss = criterion(outputs, dataY)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    lstm.eval()
    train_predict = lstm(dataX)
    data_predict = train_predict.data.numpy()
    dataY_plot = dataY.data.numpy()    
    data_predict = sc.inverse_transform(data_predict)
    print(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)
    plt.axvline(x=train_size, c='r', linestyle='--')
    plt.plot(dataY_plot)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    plt.show()



loteria = input('file loteria_test ?')

loteria = loteria + '_test.csv'
print(loteria)
col  = input('(n1,n2,n3,n4) ?')

num_epochs = input('num_epochs :(3000,4000,5000) ?')
num_epochs =int(num_epochs)
learning_rate = input(' learning_rate(0.1,0.01,0.001) ?')
print(learning_rate)
learning_rate = float(learning_rate)

df = pd.read_csv(loteria)

df.fecha = pd.to_datetime(df.fecha)
seq_length = 4
df = df.set_index('fecha').interpolate()
df1 = df.reset_index().rename(columns={"index":"fecha"})
df1 = df1.sort_values(by = 'fecha', ascending = True)
df2 = df1.tail(50)
print(df2)
for column in df2.columns[5:]: 
    print(column, '-col-',col)
    if column == col:    
        df3 = df2[['fecha',column]]
        column_names = list(df3.columns.values)
        print('column_names:',column_names)
        fechas = df3.fecha.tolist()
        fechas = fechas[-45:]
        df3 = df3.iloc[:,1:2].values
        sc, train_size,test_size, dataX, dataY = prepare_data(df3)
        dataX1 = dataX
        dataY1 = dataY
        lstm_exec(dataX, dataY,num_epochs,learning_rate)
