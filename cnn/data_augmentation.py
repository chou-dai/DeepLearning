# データ拡張
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

(x_train, t_train), (x_test, t_test) = cifar10.load_data()
image = x_train[12]

def gen_image(generator: ImageDataGenerator):
    gen = generator.flow(image, batch_size=1)  # 変換された画像の生成
    gen_img = gen.next()[0].astype(np.uint8)  # 画像の取得  

# -20°から20°の範囲でランダムに回転を行う画像生成器
generator = ImageDataGenerator(rotation_range=20)
gen_image(generator)

# 画像サイズの半分の範囲でランダムにシフトする
generator = ImageDataGenerator(width_shift_range=0.5)
gen_image(generator)

# 画像サイズの半分の範囲でランダムにシフトする
generator = ImageDataGenerator(height_shift_range=0.5)
gen_image(generator)

# シアー強度の範囲を指定
generator = ImageDataGenerator(shear_range=20)
gen_image(generator)

# 拡大縮小する範囲を指定
generator = ImageDataGenerator(zoom_range=0.4)
gen_image(generator)

# 水平、垂直方向にランダムに反転
generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
gen_image(generator)



# CNNへの適応（学習）
generator = ImageDataGenerator(
           rotation_range=0.2,
           horizontal_flip=True)
generator.fit(x_train)

history = model.fit_generator(generator.flow(x_train, t_train, batch_size=batch_size),
                              epochs=epochs,
                              validation_data=(x_test, t_test))