import numpy as np
import random
import math

# 给图片加上椒盐噪声，r为噪声比例
def salt(image, r):
    res = np.zeros(image.shape, np.uint8)
    thres = r / 2
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            temp = random.random()
            if temp <= r:
                if temp <= thres:
                    res[i][j] = 0
                else:
                    res[i][j] = 255
            else:
                res[i][j] = image[i][j]
    return res


# 给图片加上高斯噪声
def gasuss_noise(image, mean=0, var=0.001):
    temp = np.array(image / 255, dtype=float)

    noise = np.random.normal(mean, var ** 0.5, image.shape)

    out = temp + noise

    if out.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


# 计算图片PSNR
def calc_psnr(origin, after):
    diff = origin - after
    mse = np.mean(diff ** 2)

    return 10 * np.log10(255 * 255 / mse)

def median_blur(src,window_size):
    half_ws = window_size // 2
    m , n = src.shape
    padding_imge = np.pad(src,half_ws)
    res = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            origin_i , origin_j = i + half_ws , j + half_ws
            res[i][j] = np.median(padding_imge[origin_i - half_ws: origin_i + half_ws + 1, origin_j - half_ws: origin_j + half_ws + 1])
    return np.uint8(res)
