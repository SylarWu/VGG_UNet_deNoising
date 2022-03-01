import numpy as np
import cv2
import base
import time
import matplotlib.pyplot as plt

# 得到两个矩阵在各个元素位置的最大值
def max_every_element(matrix1, matrix2, m, n):
    res = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            res[i][j] = max(matrix1[i][j], matrix2[i][j])
    return res


# 求出在偏置为(t1,t2)的情况下积分图
def integral_image_square_diff_2(src, half_search_ws, t1, t2):
    m, n = src.shape
    res = np.square(src[half_search_ws:m - half_search_ws, half_search_ws:n - half_search_ws] -
                    src[half_search_ws + t1:m - half_search_ws + t1, half_search_ws + t2:n - half_search_ws + t2])
    return res.cumsum(0).cumsum(1)


# 快速NL-means去噪算法在原有的NL-means去噪算法上使用积分图加速计算
def fast_nl_means_denoising(src, h=10, window_size=7, search_window_size=21):
    # 原图像行列数
    m, n = src.shape

    half_ws = window_size // 2
    half_search_ws = search_window_size // 2
    # 给原图像padding
    padding_size = half_ws + half_search_ws + 1
    padding_img = np.pad(src, padding_size).astype(np.float64)

    h_2 = (h ** 2)
    D = window_size ** 2

    length0 = m + 2 * half_search_ws
    length1 = n + 2 * half_search_ws
    padding_v = np.pad(src, half_search_ws).astype(np.float64)

    # 权重*像素矩阵，最后除以权重归一化矩阵
    average = np.zeros((m, n))
    # 权重归一化矩阵
    sweight = np.zeros((m, n))
    # 对于每个像素，其由于自己和自己最像，所以取在计算权重中，最大的那一个
    wmax = np.zeros((m, n))

    # 将偏移量作为最外层循环，即每次只需要在一个偏移方向上求取积分图像，从而加速算法
    for t1 in range(-half_search_ws, half_search_ws + 1):
        for t2 in range(-half_search_ws, half_search_ws + 1):
            if t1 == 0 and t2 == 0:
                continue
            # 得到(t1,t2)偏置下的积分图
            SI = integral_image_square_diff_2(padding_img, half_search_ws, t1, t2)
            # 根据积分图得到在该偏置下的像素与像素间的“距离”
            SqDist2 = SI[2 * half_ws + 1:-1, 2 * half_ws + 1:-1] + SI[0:-2 * half_ws - 2, 0:-2 * half_ws - 2] - \
                      SI[2 * half_ws + 1:-1, 0:-2 * half_ws - 2] - SI[0:-2 * half_ws - 2, 2 * half_ws + 1:-1]
            SqDist2 /= D * h_2
            w = np.exp(-SqDist2)
            v = padding_v[half_search_ws + t1:length0 - half_search_ws + t1,
                half_search_ws + t2:length1 - half_search_ws + t2]
            average += w * v
            wmax = max_every_element(wmax, w, m, n)
            sweight += w
            print("%.2f%%" % (2 * (t1 + half_search_ws) * half_search_ws + (t2 + half_search_ws) / (4 * half_search_ws * half_search_ws)))
    average += wmax * src[:, :]
    average /= wmax + sweight
    return np.uint8(average)


if __name__ == '__main__':
    #
    # # 读取展示原始图像
    # img = cv2.imread("./123.jpg", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("origin", img)
    # # 给图像加上均值为0，方差为0.0025的高斯噪声
    # after_noise = base.gasuss_noise(img,0,0.0025)
    # cv2.imshow("after_noise", after_noise)
    #
    # # 使用快速非局部均值去噪算法为图像去噪
    # before = time.time()
    # after_fast_nl_means = fast_nl_means_denoising(img)
    # after = time.time()
    # cv2.imshow("after_fast_nl_means", after_fast_nl_means)
    # print("快速非局部均值去噪算法用时：",(after - before),"s")
    #
    # window_size = 7
    # # 使用中值滤波为图像去噪
    # after_median_blur = cv2.medianBlur(after_noise,window_size)
    # cv2.imshow("after_median_blur", after_median_blur)
    # # 使用均值滤波为图像去噪
    # after_blur = cv2.blur(after_noise,(window_size,window_size))
    # cv2.imshow("after_blur", after_blur)
    # # 使用高斯滤波为图像去噪
    # after_gaussian_blur = cv2.GaussianBlur(after_noise,(window_size,window_size),0)
    # cv2.imshow("after_gaussian_blur", after_gaussian_blur)
    # # 使用方框滤波为图像去噪
    # after_box_filter = cv2.boxFilter(after_noise,-1,(window_size,window_size))
    # cv2.imshow("after_box_filter", after_box_filter)
    # # 使用保留边缘信息的滤波
    # after_edge_preserving_filter = cv2.edgePreservingFilter(after_noise)
    # cv2.imshow("after_edge_preserving_filter", after_edge_preserving_filter)
    #
    # X = ["加噪图像",
    #      "中值滤波",
    #      "均值滤波",
    #      "高斯滤波",
    #      "方框滤波",
    #      "保留边缘滤波",
    #      "快速非局部均值滤波"]
    # Y = [base.calc_psnr(img,after_noise),
    #      base.calc_psnr(img,after_median_blur),
    #      base.calc_psnr(img,after_blur),
    #      base.calc_psnr(img,after_gaussian_blur),
    #      base.calc_psnr(img,after_box_filter),
    #      base.calc_psnr(img,after_edge_preserving_filter),
    #      base.calc_psnr(img,after_fast_nl_means)]
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.xlabel("不同算法")
    # plt.ylabel("PSNR")
    # plt.title("不同算法去噪效果")
    # plt.bar(X,Y,label = 'PSNR',color=['b','b','b','b','b','b','r'])
    # # 旋转X轴上标签
    # plt.xticks(X, X, rotation=30)
    # for a, b in zip(X, Y):
    #     plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=12)
    # plt.show()
    #
    # cv2.waitKey()

    img = cv2.imread("D:/fw.png")
    origin = np.array(img)
    origin = np.transpose(origin, (2, 0, 1))

    origin[0] = fast_nl_means_denoising(origin[0])
    origin[1] = fast_nl_means_denoising(origin[1])
    origin[2] = fast_nl_means_denoising(origin[2])

    origin = np.transpose(origin, (1, 2, 0))
    cv2.imwrite("D:/fw_after.png",origin)