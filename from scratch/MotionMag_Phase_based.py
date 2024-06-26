import cv2
import numpy as np
import os
import gc
from tqdm import tqdm

def load_images_from_folder(folder, batch_size):
    filenames = sorted(os.listdir(folder))
    total_batches = (len(filenames) + batch_size - 1) // batch_size  # 计算总批次
    for i in tqdm(range(0, len(filenames), batch_size), total=total_batches, desc="图像加载批次"):
        batch_images = []
        for filename in filenames[i:i+batch_size]:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                batch_images.append(img)
        yield batch_images

# 创建高斯金字塔
def build_gaussian_pyramid(img, levels):
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

# 由高斯金字塔生成拉普拉斯金字塔
def laplacian_from_gaussian(gaussian_pyr):
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        L = cv2.subtract(gaussian_pyr[i], cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size))
        laplacian_pyr.append(L)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

# 由拉普拉斯金字塔重建图像
def reconstruct_from_laplacian_pyramid(lpyr):
    img = lpyr[-1].astype(np.float32) 
    for layer in reversed(lpyr[:-1]):
        img = cv2.pyrUp(img, dstsize=(layer.shape[1], layer.shape[0])).astype(np.float32)
        layer_float = layer.astype(np.float32)
        img = cv2.add(img, layer_float) 
    img = np.clip(img, 0, 255) 
    return img.astype(np.uint8) 

# 由局部幅度信息和非线性响应来调整相位放大的程度
def adaptive_phase_amplification(magnitude, phase, base_magnification):
    # 计算自适应放大因子，这里引入一个限制因子以减少放大的极端效果
    adaptive_factor = np.tanh(magnitude / np.max(magnitude) * base_magnification / 10)
    # 计算放大后的相位
    amplified_phase = phase + adaptive_factor * np.sign(phase)  # 保持原相位符号
    return amplified_phase

# 相位放大单个颜色通道
def phase_magnify(channel, magnification_factor, levels):
    g_pyr = build_gaussian_pyramid(channel, levels)
    l_pyr = laplacian_from_gaussian(g_pyr)
    
    # 处理每一层的相位
    for i in range(len(l_pyr)):
        complex_layer = np.fft.fft2(l_pyr[i].astype(np.float32))
        magnitude = np.abs(complex_layer)
        phase = np.angle(complex_layer)
        # 使用自适应相位放大
        magnified_phase = adaptive_phase_amplification(magnitude, phase, magnification_factor) 
        new_complex_layer = magnitude * np.exp(1j * magnified_phase)
        l_pyr[i] = np.fft.ifft2(new_complex_layer).real
        # 确保值在有效范围内
        l_pyr[i] = np.clip(l_pyr[i], 0, 255)  

    return reconstruct_from_laplacian_pyramid(l_pyr)

def phase_based_motion_magnification(images, magnification_factor, levels=3):
    output_images = []
    # 为每幅图像的每个颜色通道应用相位放大并重建
    for img in tqdm(images, desc="处理图像"):
        channels = cv2.split(img)
        magnified_channels = []
        for channel in channels:
            magnified_channel = phase_magnify(channel, magnification_factor, levels)
            magnified_channels.append(np.clip(magnified_channel, 0, 255).astype(np.uint8))
        magnified_image = cv2.merge(magnified_channels)
        output_images.append(magnified_image)
    return output_images

def save_images_to_folder(images, folder, magnification_factor, start_index=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 使用 start_index 作为初始索引
    for i, img in enumerate(tqdm(images, desc=f"保存放大系数为 {magnification_factor}x 的图像"), start=start_index):
        filename = os.path.join(folder, f"output_{magnification_factor}x_{i:04d}.jpg")
        cv2.imwrite(filename, img)
    return i + 1  # 返回最后一个保存的图像索引

# 视频帧图像文件夹
input_folder = r''
# 放大后的图像保存文件夹
base_output_folder = r''
batch_size = 900  # 可以根据实际情况调整批次大小
magnification_levels = [5, 10, 20, 40]

last_index = 0
for factor in magnification_levels:
    folder = os.path.join(base_output_folder, f"magnified_{factor}x")
    for images_batch in load_images_from_folder(input_folder, batch_size):
        magnified_images = phase_based_motion_magnification(images_batch, factor)
        last_index = save_images_to_folder(magnified_images, folder, factor, last_index)
        del images_batch, magnified_images
        gc.collect() 
    # 重置索引
    last_index = 0  