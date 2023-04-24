# 畳み込みの実装
import numpy as np


# チャンネル数1、バッチサイズ1、パディングなし、ストライド1
def im2col(img, flt_h, flt_w):  # 入力画像、フィルタの高さ、幅
    img_h, img_w = img.shape  # 入力画像の高さ、幅
    out_h = img_h - flt_h + 1  # 出力画像の高さ（パディング無し、ストライド1）
    out_w = img_w - flt_w + 1  # 出力画像の幅（パディング無し、ストライド1）

    cols = np.zeros((flt_h*flt_w, out_h*out_w)) # 生成される行列のサイズ

    for h in range(out_h):
        h_lim = h + flt_h  # h:フィルタがかかる領域の上端、h_lim:フィルタがかかる領域の下端
        for w in range(out_w):
            w_lim = w + flt_w  # w:フィルタがかかる領域の左端、w_lim:フィルタがかかる領域の右端
            cols[:, h*out_w+w] = img[h:h_lim, w:w_lim].reshape(-1)

    return cols



# 様々なバッチサイズ、チャンネル数、パディング幅、ストライドに対応
def im2col(images, flt_h, flt_w, stride, pad):
   
    n_bt, n_ch, img_h, img_w = images.shape
    out_h = (img_h - flt_h + 2*pad) // stride + 1  # 出力画像の高さ
    out_w = (img_w - flt_w + 2*pad) // stride + 1  # 出力画像の幅
    
    img_pad = np.pad(images, [(0,0), (0,0), (pad, pad), (pad, pad)], "constant")
    cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))

    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            cols[:, :, h, w, :, :] = img_pad[:, :, h:h_lim:stride, w:w_lim:stride]

    cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(n_ch*flt_h*flt_w, n_bt*out_h*out_w)
    return cols



# 以下畳み込みの適用例
img = np.array([[[[1, 2, 3, 4],  # 入力画像
                  [5, 6, 7, 8],
                  [9, 10,11,12],
                  [13,14,15,16]]]])

flt = np.array([[-1, 1, -1,],  # 縦の線を強調するフィルタ
                [-1, 1, -1,],
                [-1, 1, -1,]])
flt_h, flt_w = flt.shape
cols = im2col(img, flt_h, flt_w, 1, 1)  # 入力画像、フィルタの高さ、幅、ストライド、パディング幅

image_out = np.dot(flt, cols)  # 畳み込み