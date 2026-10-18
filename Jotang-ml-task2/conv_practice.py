import numpy as np
from PIL import Image

# 1. 读取真实图片 (随便找一张图，命名为 test.jpg 放在同一目录)
img = np.array(Image.open('test.jpg').convert('RGB'))
print(f"Shape: {img.shape}, Dtype: {img.dtype}, Max: {img.max()}, Min: {img.min()}")
print(f"左上角第一个像素 RGB: {img[0, 0, :]}")
print("NumPy/PIL 是 HWC，PyTorch 是 CHW，需要通过 permute 转换。")

# 灰度图供卷积测试
gray_img = np.array(Image.open('test.jpg').convert('L'), dtype=np.float32)

# 2. 定义四种卷积核
k_mean = np.ones((3, 3)) / 9.0
# 高斯核 (5x5)
k_gaussian = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273.0
# 锐化核
k_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# Sobel
k_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
k_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# 3. 手写卷积 (增加 valid / same 逻辑)
def my_conv2d(image, kernel, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    H, W = image.shape
    K_h, K_w = kernel.shape
    H_out = H - K_h + 1
    W_out = W - K_w + 1
    output = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            output[i, j] = np.sum(image[i:i+K_h, j:j+K_w] * kernel)
    # 题目要求：处理越界值
    output = np.clip(output, 0, 255)
    return output

# 4. 运行并保存结果
Image.fromarray(my_conv2d(gray_img, k_mean, padding=1).astype(np.uint8)).save('out_mean.jpg')
Image.fromarray(my_conv2d(gray_img, k_gaussian, padding=2).astype(np.uint8)).save('out_gaussian.jpg')
Image.fromarray(my_conv2d(gray_img, k_sharpen, padding=1).astype(np.uint8)).save('out_sharpen.jpg')

# Sobel X+Y 合成
sobel_x = my_conv2d(gray_img, k_sobel_x, padding=1)
sobel_y = my_conv2d(gray_img, k_sobel_y, padding=1)
sobel_combined = np.clip(np.sqrt(sobel_x**2 + sobel_y**2), 0, 255)
Image.fromarray(sobel_combined.astype(np.uint8)).save('out_sobel_combined.jpg')

print("Valid 尺寸:", my_conv2d(gray_img, k_mean).shape)
print("Same 尺寸:", my_conv2d(gray_img, k_mean, padding=1).shape)