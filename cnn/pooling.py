# プーリング関数の実装
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets

def im2col(img, flt_h, flt_w, out_h, out_w, stride):  # 入力画像、プーリング領域の高さ、幅、出力画像の高さ、幅、ストライド

    cols = np.zeros((flt_h*flt_w, out_h*out_w)) # 生成される行列のサイズ

    for h in range(out_h):
        h_lim = stride*h + flt_h  # h:プーリング領域の上端、h_lim:プーリング領域の下端
        for w in range(out_w):
            w_lim = stride*w + flt_w  # w:プーリング領域の左端、w_lim:プーリング領域の右端
            cols[:, h*out_w+w] = img[stride*h:h_lim, stride*w:w_lim].reshape(-1)

    return cols

digits = datasets.load_digits()
image = digits.data[0].reshape(8, 8)
img_h, img_w = image.shape  # 入力画像の高さ、幅
pool = 2  # プーリング領域のサイズ

out_h = img_h//pool  # 出力画像の高さ
out_w = img_w//pool  # 出力画像の幅

cols = im2col(image, pool, pool, out_h, out_w, pool)
image_out = np.max(cols, axis=0)  # Maxプーリング
image_out = image_out.reshape(out_h, out_w)