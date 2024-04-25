import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
from itertools import chain

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
    sc = MinMaxScaler()
    print(training_set)
    training_data = sc.fit_transform(training_set)  
    x, y = scaling_window(training_data, seq_length)
    #print(x)    
    #var = input('continue?')
    train_size = int(len(y) * 0.7)
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

def lstm_exec(dataX, dataY,fechas, df3):
    num_epochs = 3000
    learning_rate = 0.01
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
    data_predict = sc.inverse_transform(data_predict)
    predictions = data_predict
    dataY_plot = dataY.data.numpy() 
    dataY_plot = sc.inverse_transform(dataY_plot)
    expected = dataY_plot    
    #print(dataX)
    #var = input('continue ??')
    plt.axvline(x=train_size, c='r', linestyle='--')
    print('len (expected):', len(expected))
    for i in  chain (range(0, len(expected))):
        print (f'This is {fechas[i]} and {expected[i]} and {predictions[i]}')
    #forecast_errors = [ x - y for x, y in zip(expected, predictions)] 
    #forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
    #print('Forecast Errors: %s' % forecast_errors)
    #bias = sum(forecast_errors) * 1.0/len(expected)
    #print('Bias: %f' % bias)
    mae = mean_absolute_error(expected, predictions)
    print('MAE: %f' % mae)
    mse = mean_squared_error(expected, predictions)
    print('MSE: %f' % mse)
    rmse = math.sqrt(mse)
    print('RMSE: %f' % rmse)
    ###### calcular el siguiente numero
    df4 = df3[-6:]
    #print('df4',df4)
    #var = input('continue?')
    sc1, train_size1,test1_size1, dataX1, dataY1 =prepare_data(df4)
    train_predict1 = lstm(dataX1)
    data_predict1 = train_predict1.data.numpy()       
    data_predict1 = sc.inverse_transform(data_predict1)
    predictions1 = data_predict1
    print(predictions1)
    plt.plot(dataY_plot)
    plt.plot(data_predict)
    plt.plot(data_predict1)
    plt.suptitle('Time-Series Prediction')
    plt.show()
    
df = pd.read_csv('loterias_test.csv')
df.fecha = pd.to_datetime(df.fecha)
seq_length = 4
df = df.set_index('fecha').interpolate()
df1 = df.reset_index().rename(columns={"index":"fecha"})
df1 = df1.sort_values(by = 'fecha', ascending = True)
df2 = df1.tail(50)
df3 = df2[['fecha','n1']]
fechas = df3.fecha.tolist()
fechas = fechas[-45:]
df3 = df3.iloc[:,1:2].values
#plt.plot(training_set, label = 'Data')
#plt.show()
sc, train_size,test_size, dataX, dataY = prepare_data(df3)
#print(dataX)
#var = input('continue ?')
lstm_exec(dataX, dataY,fechas,df3)
