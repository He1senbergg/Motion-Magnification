import os
import cv2

def extract_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频文件
    video_capture = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print("无法打开视频文件")
        return

    frame_count = 0

    while True:
        # 逐帧读取视频
        ret, frame = video_capture.read()

        # 如果没有读取到帧，退出循环
        if not ret:
            break

        # 构建每一帧图片的文件名
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")

        # 保存帧图片
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # 释放视频捕获对象
    video_capture.release()
    print(f"提取的帧数: {frame_count}")

# 调用函数，指定视频路径和输出文件夹
video_path = r''  # 替换为你的视频路径
output_folder = r''  # 替换为你希望保存帧图片的文件夹
extract_frames(video_path, output_folder)