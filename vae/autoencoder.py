# オートエンコーダ（自己符号器）の実装
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt

# 訓練用データの生成
(x_train, t_train), (x_test, t_test) = mnist.load_data()  # MNISTの読み込み
print(x_train.shape, x_test.shape)  # 28x28の手書き文字画像が6万枚
# 各ピクセルの値を0-1の範囲に収める
x_train = x_train / 255
x_test = x_test / 255
# 一次元に変換する
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape, x_test.shape)


# 各種設定
epochs = 20  
batch_size = 128
n_in_out = 784  # 入出力層のニューロン数
n_mid = 64  # 中間層のニューロン数


# モデルの構築
# オートエンコーダのネットワーク構築
x = Input(shape=(n_in_out,))
h = Dense(n_mid, activation="relu")(x)  # Encoder
decoder = Dense(n_in_out, activation="sigmoid")  #Decoder 後ほど再利用
y = decoder(h)
# オートエンコーダのモデルを作成
model_autoencoder = Model(x, y)
model_autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
model_autoencoder.summary()
# Encoderのみのモデル
model_encoder = Model(x, h)
# Decoderのみのモデル
input_decoder = Input(shape=(n_mid,))
model_decoder = Model(input_decoder, decoder(input_decoder))


# 学習
model_autoencoder.fit(x_train, x_train,
                      shuffle=True,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(x_test, x_test))


# 画像生成
encoded = model_encoder.predict(x_test)
decoded = model_decoder.predict(encoded)
n = 8  # 表示する画像の数
plt.figure(figsize=(16, 4))
for i in range(n):
    # 入力画像
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="Greys_r")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 中間層の出力
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(encoded[i].reshape(8,8), cmap="Greys_r") #画像サイズは、中間層のニューロン数に合わせて変更する
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    # 出力画像
    ax = plt.subplot(3, n, i+1+2*n)
    plt.imshow(decoded[i].reshape(28, 28), cmap="Greys_r")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()