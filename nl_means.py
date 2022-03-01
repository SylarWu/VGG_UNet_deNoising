import numpy as np
import cv2
import base
import time
import matplotlib.pyplot as plt


# 计算两个窗口的欧式距离，距离越小，则其权重越高
def distance_method1(src, loc1, loc2, window_size):
    res = 0.0
    half_ws = window_size // 2
    for i in range(-half_ws, half_ws + 1):
        for j in range(-half_ws, half_ws + 1):
            res += (src[loc1[0] + i][loc1[1] + j] - src[loc2[0] + i][loc2[1] + j]) * (
                    src[loc1[0] + i][loc1[1] + j] - src[loc2[0] + i][loc2[1] + j])
    return res


# 计算两个窗口的欧式距离，距离越小，则其权重越高
def distance_method2(window1, window2):
    temp = np.square(window1 - window2)
    return np.sum(temp)


# 根据当前像素模板窗口和搜索窗口中所有模板窗口比较相似度计算其相应权重，从而计算出当前像素值
# src padding后的图像
# origin_i,origin_j 计算出当前像素值的核心位置
# h_2 公式中的h值的平方
# sigma_2 当像素j的模板与核心像素的模板窗口距离小于两倍的sigma平方，则可认为该像素值权重最大
# half_ws 半个模板窗口大小
# half_search_ws 半个搜索窗口大小
# step 步长，默认为1，可以跳过一定像素加速计算
def deduce_pixel_value(src, origin_i, origin_j, h_2, sigma_2, half_ws, half_search_ws, step):
    # 当前核心像素的模板窗口
    focus_window = src[origin_i - half_ws: origin_i + half_ws + 1, origin_j - half_ws:origin_j + half_ws + 1]
    # 生成search_window_size*search_window_size尺寸的权重矩阵
    weight = np.zeros((half_search_ws * 2 + 1, half_search_ws * 2 + 1))
    # 归一化距离
    D = (2 * half_ws + 1) * (2 * half_ws + 1)
    # 在search window里计算每个像素值相对于当前像素的权重
    for i in range(origin_i - half_search_ws, origin_i + half_search_ws + 1, step):
        for j in range(origin_j - half_search_ws, origin_j + half_search_ws + 1, step):
            if i == origin_i and j == origin_j:
                continue
            # dist = distance_method1(src,(origin_i,origin_j),(i,j),window_size)
            # 使用numpy函数比起自己实现的方法计算速度快
            dist = distance_method2(focus_window, src[i - half_ws: i + half_ws + 1, j - half_ws:j + half_ws + 1]) / D
            weight[i - origin_i + half_search_ws][
                j - origin_j + half_search_ws] = np.exp(-max(dist - 2 * sigma_2, 0.0) / h_2)
    # 将权重归一化
    weight = (1.0 / np.sum(weight)) * weight
    # 返回权重乘以搜索窗口下各个像素的值，得到过滤后的像素值
    return np.sum(weight * src[origin_i - half_search_ws:origin_i + half_search_ws + 1,
                           origin_j - half_search_ws: origin_j + half_search_ws + 1])


# Non-local means去噪算法，默认搜索窗口大小为21*21，权重比较窗口大小为7*7
# h 为过滤程度，控制指数函数的衰减
def nl_means_denoising(src, h=10, window_size=7, search_window_size=21, step=1):
    # 未padding的原图像size
    m, n = src.shape
    # 初试化结果图像
    res = np.zeros((m, n))
    # 给原图像padding
    half_ws = window_size // 2
    half_search_ws = search_window_size // 2
    padding_size = half_ws + half_search_ws
    padding_img = np.pad(src, padding_size).astype(np.int32)
    h_2 = h ** 2
    sigma = h / 4
    sigma_2 = sigma ** 2
    for i in range(m):
        for j in range(n):
            res[i][j] = deduce_pixel_value(padding_img, i + padding_size, j + padding_size, h_2, sigma_2, half_ws,
                                           half_search_ws, step)
        if i % 10 == 0:
            print("进度：", ((i * n) / (m * n)) * 100, "%")
    return np.uint8(res)

# 读取展示原始图像
img = cv2.imread("./123.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("origin", img)

# 给图像加上均值为0，方差为0.0025的高斯噪声
after_noise = base.gasuss_noise(img,0,0.0025)
cv2.imshow("after_noise", after_noise)

# 使用非局部均值去噪算法为图像去噪
before = time.time()
after_nl_means = nl_means_denoising(after_noise)
after = time.time()
cv2.imshow("after_nl_means", after_nl_means)
print("非局部均值去噪算法用时：",(after - before),"s")



window_size = 7
# 使用中值滤波为图像去噪
after_median_blur = cv2.medianBlur(after_noise,window_size)
cv2.imshow("after_median_blur", after_median_blur)
# 使用均值滤波为图像去噪
after_blur = cv2.blur(after_noise,(window_size,window_size))
cv2.imshow("after_blur", after_blur)
# 使用高斯滤波为图像去噪
after_gaussian_blur = cv2.GaussianBlur(after_noise,(window_size,window_size),0)
cv2.imshow("after_gaussian_blur", after_gaussian_blur)
# 使用方框滤波为图像去噪
after_box_filter = cv2.boxFilter(after_noise,-1,(window_size,window_size))
cv2.imshow("after_box_filter", after_box_filter)
# 使用保留边缘信息的滤波
after_edge_preserving_filter = cv2.edgePreservingFilter(after_noise)
cv2.imshow("after_edge_preserving_filter", after_edge_preserving_filter)

X = ["加噪图像",
     "中值滤波",
     "均值滤波",
     "高斯滤波",
     "方框滤波",
     "保留边缘滤波",
     "非局部均值滤波"]
Y = [base.calc_psnr(img,after_noise),
     base.calc_psnr(img,after_median_blur),
     base.calc_psnr(img,after_blur),
     base.calc_psnr(img,after_gaussian_blur),
     base.calc_psnr(img,after_box_filter),
     base.calc_psnr(img,after_edge_preserving_filter),
     base.calc_psnr(img,after_nl_means)]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xlabel("不同算法")
plt.ylabel("PSNR")
plt.title("不同算法去噪效果")
plt.bar(X,Y,label = 'PSNR',color=['b','b','b','b','b','b','r'])
# 旋转X轴上标签
plt.xticks(X, X, rotation=30)
for a, b in zip(X, Y):
    plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=12)
plt.show()

cv2.waitKey()