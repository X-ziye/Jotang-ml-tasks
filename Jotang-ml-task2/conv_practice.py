import numpy as np
from PIL import Image

# 1. 创建一个简单的 5x5 灰度图（模拟矩阵）
# (实践1 第2题：了解图像表示)
image = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

# 2. 定义几种卷积核
# 均值模糊 (3x3)
kernel_mean = np.ones((3, 3)) / 9.0
# 高斯模糊 (这里用简化的中心权重大）
kernel_gaussian = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16.0
# 边缘检测 (Sobel 水平方向)
kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# 3. 手写二维卷积（核心部分）
def my_conv2d(image, kernel, stride=1, padding=0):
    # 对图像进行填充
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    H, W = image.shape
    K_h, K_w = kernel.shape
    
    # 计算输出尺寸（核心数学公式）
    H_out = (H - K_h) // stride + 1
    W_out = (W - K_w) // stride + 1
    
    # 初始化输出矩阵
    output = np.zeros((H_out, W_out))
    
    # 滑动窗口
    for i in range(H_out):
        for j in range(W_out):
            # 取出当前窗口的切片
            window = image[i*stride : i*stride+K_h, j*stride : j*stride+K_w]
            # 对应位置相乘并求和
            output[i, j] = np.sum(window * kernel)
    
    return output

# 4. 测试卷积
print("--- 原图 ---")
print(image)
print("\n--- 均值模糊结果 ---")
print(my_conv2d(image, kernel_mean))
print("\n--- Sobel边缘检测结果 ---")
print(my_conv2d(image, kernel_sobel_x, padding=1))

# 实践1 第5题：用真实的图片测试
# 如果你手头有一张图，可以取消下面注释：
# img = np.array(Image.open('your_image.jpg').convert('L'))
# blurred = my_conv2d(img, kernel_gaussian)
# Image.fromarray(blurred).save('blurred.jpg')