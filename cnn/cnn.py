from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import numpy as np

# データセット読み込み
(x_train, t_train), (x_test, t_test) = cifar10.load_data()

# 各設定
batch_size = 32
epochs = 20
n_class = 10  # 10のクラスに分類

# one-hot表現に変換
t_train = to_categorical(t_train, n_class)
t_test = to_categorical(t_test, n_class)

# モデルの構築
model = Sequential()
# 畳み込み層 x2
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))  # フィルタ数32、サイズ3x3、ゼロパディング、入力画像のサイズ
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# Maxプーリング層
model.add(MaxPooling2D(pool_size=(2, 2)))
# 畳み込み層 x2
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
# Maxプーリング層
model.add(MaxPooling2D(pool_size=(2, 2)))
# 一次元の配列に変換
model.add(Flatten())
# 全結合層
model.add(Dense(256))
model.add(Activation('relu'))
# ドロップアウト：ランダムにセルをニューロンする → モデルの汎化性能の向上
model.add(Dropout(0.5))
# 全結合層（出力層）
model.add(Dense(n_class))
model.add(Activation('softmax'))
# モデルをコンパイル
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
x_train = x_train / 255  # 0から1の範囲に収める
x_test = x_test / 255
history = model.fit(x_train, t_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_test, t_test))

# 予測
n_image = 25
rand_idx = np.random.randint(0, len(x_test), n_image)
y_rand = model.predict(x_test[rand_idx])
predicted_class = np.argmax(y_rand, axis=1)