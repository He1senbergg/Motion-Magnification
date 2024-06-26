import os
import gc
import cv2
import numpy as np
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

def magnify_motion(images, magnification_factor):
    output_images = []
    prev_image = images[0]
    for i in range(1, len(images)):
        current_image = images[i]
        frame_diff = cv2.absdiff(current_image, prev_image)
        magnified_diff = cv2.multiply(frame_diff, np.array([magnification_factor], dtype=np.uint8))
        enhanced_image = cv2.add(current_image, magnified_diff)
        output_images.append(enhanced_image)
        prev_image = current_image
    return output_images

def save_images_to_folder(images, folder, factor, start_index=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 使用 start_index 作为初始索引
    for i, img in enumerate(tqdm(images, desc=f"保存放大系数为 {factor}x 的图像"), start=start_index):
        filename = os.path.join(folder, f"output_{factor}x_{i:04d}.jpg")
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
    output_folder = os.path.join(base_output_folder, f"magnified_{factor}x")
    for images_batch in load_images_from_folder(input_folder, batch_size):
        magnified_images = magnify_motion(images_batch, factor)
        last_index = save_images_to_folder(magnified_images, output_folder, factor, last_index)
        del images_batch, magnified_images
        gc.collect() 
    # 重置索引
    last_index = 0