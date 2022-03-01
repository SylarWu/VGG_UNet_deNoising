import numpy as np
import random
import cv2
import time
import math
import matplotlib.pyplot as plt


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

# 计算图片PSNR
def calc_psnr(origin, after):
    diff = origin - after
    mse = np.mean(diff ** 2)
    return 10*math.log10(255*255/mse)


# 生成图像窗口里的序列
def generate_window(image, i, j, window_size):
    temp = []
    thres = window_size // 2
    a = i - thres
    while a <= i + thres:
        if a < 0:
            a += 1
            continue
        elif a >= image.shape[0]:
            break
        b = j - thres
        while b <= j + thres:
            if b < 0:
                b += 1
                continue
            elif b >= image.shape[1]:
                break
            temp.append(image[a][b])
            b += 1
        a += 1
    return np.array(temp)


# 标准中值过滤
def SMF(image, window_size):
    res = np.zeros(image.shape, np.uint8)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            # 在图像边界，直接取中值
            temp = generate_window(image, i, j, window_size)
            median = np.median(temp)
            res[i][j] = median
    return res


# 加权快速中值滤波算法
def method_1(image, window_size):
    res = np.zeros(image.shape, np.uint8)
    last_median = -1
    thres = window_size // 2
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            # 在图像边界，直接取中值
            if j - 1 - thres < 0 or j + thres >= res.shape[1]:
                temp = generate_window(image, i, j, window_size)
                last_median = np.median(temp)
                res[i][j] = last_median
            else:
                # 在图像内部，这时可以判断新的窗口取值是否与之前相等
                flag = True
                k = i - thres if i - thres >= 0 else 0
                while k <= i + thres:
                    if k >= res.shape[0]:
                        break
                    if image[k][j - 1 - thres] != image[k][j + thres]:
                        flag = False
                        break
                    k += 1
                # 如果新的模板与之前窗口取值相同，则直接取上个窗口中值
                if flag:
                    res[i][j] = last_median
                    continue
                temp = generate_window(image, i, j, window_size)
                last_median = np.median(temp)
                last_mean = np.mean(temp)
                res[i][j] = int(last_median * 0.3 + last_mean * 0.7)
    return res

def method_2(origin, image, max_window_size, count_iter = 100):

    def layer_a(image, loc, max_window_size, threshold):
        window_size = 3
        while window_size <= max_window_size:
            temp = generate_window(image, loc[0], loc[1], window_size)
            maximum = np.max(temp)
            minimum = np.min(temp)
            mid = np.median(temp)
            a1 = mid - minimum
            a2 = maximum - mid
            if a1 > threshold and a2 > threshold:
                return layer_b(image, loc, minimum, maximum, mid, threshold)
            window_size += 2
        return image[loc[0]][loc[1]]

    def layer_b(image, loc, minimum, maximum, mid, threshold):
        b1 = image[loc[0]][loc[1]] - minimum
        b2 = maximum - image[loc[0]][loc[1]]
        if b1 > threshold and b2 > threshold:
            return image[loc[0]][loc[1]]
        return mid

    res = np.copy(image)
    best_res = res
    max_psnr = 0
    threshold = 60
    count = 0
    while count <= count_iter:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                res[i][j] = layer_a(image, (i, j), max_window_size, threshold)
        threshold = threshold // 2
        temp_psnr = calc_psnr(origin, res)
        if np.absolute(temp_psnr - max_psnr) <= 1e-5:
            break
        if temp_psnr > max_psnr:
            max_psnr = temp_psnr
            best_res = np.copy(res)
        count += 1
    return best_res


img = cv2.imread("./123.jpg", cv2.IMREAD_GRAYSCALE)

window_size = 7
noising_ration = 0.25
cv2.imshow("origin", img)
# 加上椒盐噪声
after_salt = salt(img, noising_ration)
cv2.imshow("after_salt", after_salt)

# 标准中值过滤
before = time.time()
after_smf_filter = SMF(after_salt, window_size)
after = time.time()
#print("smf_runtime:", after - before)
cv2.imshow("after_smf", after_smf_filter)

# 加权快速中值过滤
before = time.time()
after_m1_filter = method_1(after_salt, window_size)
after = time.time()
#print("method1_runtime:", after - before)
cv2.imshow("after_m1", after_m1_filter)

# 加权自适应中值过滤
before = time.time()
after_m2_filter = method_2(img, after_salt, 9)
after = time.time()
#print("method2_runtime:", after - before)
cv2.imshow("after_m2", after_m2_filter)

print("after_salt_PSNR:", calc_psnr(img, after_salt))
print("smf_PSNR:", calc_psnr(img, after_smf_filter))
print("method1_PSNR:", calc_psnr(img, after_m1_filter))
print("method2_PSNR:", calc_psnr(img, after_m2_filter))

cv2.waitKey()
