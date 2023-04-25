# VAEの実装
import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist
from keras.models import Model
from keras import metrics  # 評価関数
from keras.layers import Input, Dense, Lambda
from keras import backend as K  # 乱数の発生に使用


# 訓練データの用意
(x_train, t_train), (x_test, t_test) = mnist.load_data()  # MNISTの読み込み
# 各ピクセルの値を0-1の範囲に収める
x_train = x_train / 255
x_test = x_test / 255
# 一次元に変換する
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


# 各種設定
epochs = 10  
batch_size = 128
n_in_out = 784  # 入出力層のニューロン数
n_z = 2  # 潜在変数の数（次元数）
n_mid = 256  # 中間層のニューロン数


# モデルの構築
# 潜在変数をサンプリングするための関数
def z_sample(args):
    mu, log_var = args  # 潜在変数の平均値と、分散の対数
    epsilon = K.random_normal(shape=K.shape(log_var), mean=0, stddev=1)
    return mu + epsilon * K.exp(log_var / 2)  # Reparametrization Trickにより潜在変数を求める
# VAEのネットワーク構築
x = Input(shape=(n_in_out,))
h_encoder = Dense(n_mid, activation="relu")(x)
# 平均、分散
mu = Dense(n_z)(h_encoder)
log_var = Dense(n_z)(h_encoder)
z = Lambda(z_sample, output_shape=(n_z,))([mu, log_var])
# デコーダ
mid_decoder = Dense(n_mid, activation="relu")
h_decoder = mid_decoder(z)
out_decoder = Dense(n_in_out, activation="sigmoid")
y = out_decoder(h_decoder)
# VAEのモデルを生成
model_vae = Model(x, y)
# 損失関数
rec_loss = n_in_out * metrics.binary_crossentropy(x, y)
reg_loss = - 0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
vae_loss = K.mean(rec_loss + reg_loss)
model_vae.add_loss(vae_loss)
model_vae.compile(optimizer="rmsprop")


# 学習
model_vae.fit(x_train,
              shuffle=True,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_test, None))



# 潜在空間の可視化（潜在変数はベクトル）
encoder = Model(x, z)
# 訓練データから作った潜在変数を2次元プロット
z_train = encoder.predict(x_train, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(z_train[:, 0], z_train[:, 1], c=t_train)  # ラベルを色で表す
plt.title("Train")
plt.colorbar()
plt.show()
# テストデータを入力して潜在空間に2次元プロットする 正解ラベルを色で表示
z_test = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(z_test[:, 0], z_test[:, 1], c=t_test)
plt.title("Test")
plt.colorbar()
plt.show()


# 画像の生成
input_decoder = Input(shape=(n_z,))
h_decoder = mid_decoder(input_decoder)
y = out_decoder(h_decoder)
generator = Model(input_decoder, y)
# 画像を並べる設定
n = 16  # 手書き文字画像を16x16並べる
image_size = 28
matrix_image = np.zeros((image_size*n, image_size*n))  # 全体の画像
# 潜在変数
z_1 = np.linspace(5, -5, n)  # 各行
z_2 = np.linspace(-5, 5, n)  # 各列
#  潜在変数を変化させて画像を生成
for i, z1 in enumerate(z_1):
    for j, z2 in enumerate(z_2):
        decoded = generator.predict(np.array([[z2, z1]]))  # x軸、y軸の順に入れる
        image = decoded[0].reshape(image_size, image_size)
        matrix_image[i*image_size : (i+1)*image_size, j*image_size: (j+1)*image_size] = image
plt.figure(figsize=(10, 10))
plt.imshow(matrix_image, cmap="Greys_r")
plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  # 軸目盛りのラベルと線を消す
plt.show()