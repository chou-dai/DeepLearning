# LSTMの実装
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

# 訓練データの作成（ノイズを含むsin関数のデータ）
x_data = np.linspace(-2*np.pi, 2*np.pi)  # -2πから2πまで
sin_data = np.sin(x_data)  + 0.1*np.random.randn(len(x_data))  # sin関数に乱数でノイズを加える


# 入力データと正解データの設定
n_rnn = 10  # 時系列の数 = 再帰の数
n_sample = len(x_data)-n_rnn  # サンプル数
x = np.zeros((n_sample, n_rnn))  # 入力 (行,列)=(n_sample,n_rnn)
t = np.zeros((n_sample, n_rnn))  # 正解
for i in range(0, n_sample):
    x[i] = sin_data[i:i+n_rnn]
    t[i] = sin_data[i+1:i+n_rnn+1]  # 時系列を入力よりも一つ後にずらす
x = x.reshape(n_sample, n_rnn, 1)  # KerasにおけるRNNでは、入力を（サンプル数、時系列の数、入力層のニューロン数）にする
t = t.reshape(n_sample, n_rnn, 1)  # 今回は入力と同じ形状


# RNNのモデル構築
batch_size = 8  # バッチサイズ
n_in = 1  # 入力層のニューロン数
n_mid = 20  # 中間層のニューロン数
n_out = 1  # 出力層のニューロン数
# 比較のための通常のRNN
model_rnn = Sequential()
model_rnn.add(SimpleRNN(n_mid, input_shape=(n_rnn, n_in), return_sequences=True))
model_rnn.add(Dense(n_out, activation="linear"))
model_rnn.compile(loss="mean_squared_error", optimizer="sgd")
print(model_rnn.summary())
# LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(n_mid, input_shape=(n_rnn, n_in), return_sequences=True))
model_lstm.add(Dense(n_out, activation="linear"))
model_lstm.compile(loss="mean_squared_error", optimizer="sgd")
print(model_lstm.summary())


# 学習
history_lstm = model_lstm.fit(x, t, epochs=1000, batch_size=batch_size, verbose=0)


# sin関数の次の値の予測
predicted = x[0].reshape(-1)  # 最初の入力。reshape(-1)で一次元のベクトルにする。
for i in range(0, n_sample):
    y = model_lstm.predict(predicted[-n_rnn:].reshape(1, n_rnn, 1))  # 直近のデータを使って予測を行う（バッチサイズ,時系列数,入力層のニューロン数）
    predicted = np.append(predicted, y[0][n_rnn-1][0])  # 出力の最後の結果をpredictedに追加する