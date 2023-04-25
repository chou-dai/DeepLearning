# 敵対的生成ネットワークの実装
import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.models import Model
from keras.layers import Input


# 訓練用データの用意
(x_train, t_train), (x_test, t_test) = mnist.load_data()  # MNISTの読み込み
# 各ピクセルの値を-1から1の範囲に収める
x_train = x_train / 255 * 2 - 1
x_test = x_test / 255 * 2 - 1
# 一次元に変換する
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


# 各種設定
n_learn = 10001 # 学習回数
interval = 1000  # 画像を生成する間隔
batch_size = 32
n_noize = 128  # ノイズの数
img_size = 28  # 生成される画像の高さと幅
alpha = 0.2  # Leaky ReLUの負の領域での傾き
optimizer = Adam(0.0002, 0.5)


# Generatorの構築
generator = Sequential()
generator.add(Dense(256, input_shape=(n_noize,)))
generator.add(LeakyReLU(alpha=alpha)) 
generator.add(Dense(512))
generator.add(LeakyReLU(alpha=alpha)) 
generator.add(Dense(1024))
generator.add(LeakyReLU(alpha=alpha)) 
generator.add(Dense(img_size**2, activation="tanh"))


# Discriminatiorの構築
discriminator = Sequential()
discriminator.add(Dense(512, input_shape=(img_size**2,)))
discriminator.add(LeakyReLU(alpha=alpha)) 
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(alpha=alpha)) 
discriminator.add(Dense(1, activation="sigmoid"))
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])


# モデルの結合
# 結合時はGeneratorのみ訓練する
discriminator.trainable = False
# Generatorによってノイズから生成された画像を、Discriminatorが判定する
noise = Input(shape=(n_noize,))
img = generator(noise)
reality = discriminator(img)
# GeneratorとDiscriminatorの結合
combined = Model(noise, reality)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)
print(combined.summary())


# 画像の生成
def generate_images(i):
    n_rows = 5  # 行数
    n_cols = 5  # 列数
    noise = np.random.normal(0, 1, (n_rows*n_cols, n_noize))
    g_imgs = generator.predict(noise)
    g_imgs = g_imgs/2 + 0.5  # 0-1の範囲にする
    matrix_image = np.zeros((img_size*n_rows, img_size*n_cols))  # 全体の画像
    #  生成された画像を並べて一枚の画像にする
    for r in range(n_rows):
        for c in range(n_cols):
            g_img = g_imgs[r*n_cols + c].reshape(img_size, img_size)
            matrix_image[r*img_size : (r+1)*img_size, c*img_size: (c+1)*img_size] = g_img


# 学習
batch_half = batch_size // 2
loss_record = np.zeros((n_learn, 3))
acc_record = np.zeros((n_learn, 2))
for i in range(n_learn):
    # ノイズから画像を生成しDiscriminatorを訓練
    g_noise = np.random.normal(0, 1, (batch_half, n_noize))
    g_imgs = generator.predict(g_noise)
    loss_fake, acc_fake = discriminator.train_on_batch(g_imgs, np.zeros((batch_half, 1)))
    loss_record[i][0] = loss_fake
    acc_record[i][0] = acc_fake
    # 本物の画像を使ってDiscriminatorを訓練
    rand_ids = np.random.randint(len(x_train), size=batch_half)
    real_imgs = x_train[rand_ids, :]
    loss_real, acc_real = discriminator.train_on_batch(real_imgs, np.ones((batch_half, 1)))
    loss_record[i][1] = loss_real
    acc_record[i][1] = acc_real
    # 結合したモデルによりGeneratorを訓練する
    c_noise = np.random.normal(0, 1, (batch_size, n_noize))
    loss_comb = combined.train_on_batch(c_noise, np.ones((batch_size, 1)))
    loss_record[i][2] = loss_comb
    # 一定間隔で生成された画像を表示
    if i % interval == 0:
        generate_images(i)