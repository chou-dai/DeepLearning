import numpy as np
from sklearn import datasets, preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model

iris = datasets.load_iris()

# ---------- データの前処理 ----------
# データの標準化のためのスケーラー
scaler = preprocessing.StandardScaler()
# 引数の配列データの平均と標準偏差を計算して記憶する
scaler.fit(iris.data)
# 標準化、データの平均を0、標準偏差を1に変換する 入力データ
input_data = scaler.transform(iris.data)
# target = ラベルをone-hot表現(0,1)に変換 正解データ
correct_data = np_utils.to_categorical(iris.target)

# ---------- 訓練データとテストデータを生成 ----------
# 75%が訓練用、25%がテスト用
input_train, input_test, correct_train, correct_test = train_test_split(input_data, correct_data, train_size=0.75)

# ---------- 4層モデルの構築 ----------
# 層を積み重ねるモデル
model = Sequential()
# 入力層ニューロン数4、中間層ニューロン数32の全結合層
model.add(Dense(32, input_dim=4))
# 活性化関数(ReLU)を追加
model.add(Activation('relu'))
# ニューロン数32の全結合層を追加
model.add(Dense(32))
# 活性化関数(ReLU)を追加
model.add(Activation('relu'))
# 出力層ニューロン数3の線結合層を追加
model.add(Dense(3))
# ソフトマックス関数を追加
model.add(Activation('softmax'))
# モデルをコンパイル
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- 学習 ----------
# 訓練用入力データと正解データでモデルを訓練する
history = model.fit(input_train, correct_train, epochs=30, batch_size=8)

# ---------- モデルの評価 ----------
# テストデータを用いて誤差と正解率の評価を行う
loss, accuracy = model.evaluate(input_test, correct_test)
print(loss, accuracy)

# ---------- 予測 ----------
model.predict(input_test)

# ---------- モデルの保存・読み込み ----------
model.save('model.h5')
new_model = load_model('model.h5')