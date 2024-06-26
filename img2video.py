import cv2
import os
import argparse

def parser():
    parser = argparse.ArgumentParser(description='Image to video')
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help='input folder')
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='output video file')
    return parser.parse_args()

def create_video_from_images(image_folder, output_video_file, amp=5, fps=30):
    images = [img for img in sorted(os.listdir(image_folder))]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        cv2.putText(img, 'amp_factor={}'.format(amp), (7, 37),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        video.write(img)

    video.release()  # 释放资源

if __name__ == "__main__":
    args = parser()
    # 指定输入文件夹
    base_dir = args.input
    # 指定输出文件夹
    output_folder = args.output

    # 为每个放大倍数生成视频文件
    for i in [5, 10, 20, 40]:
        # used for version "from scratch"
        # folder = os.path.join(base_dir, f"magnified_{i}x")
        # used for version "github"
        folder = os.path.join(base_dir, f"first5mins_amp{i}")
        video_name = f"output_video_{i}x.mp4"
        output_video_file = os.path.join(output_folder, video_name)
        create_video_from_images(folder, output_video_file, amp=i)
        print(f"已生成视频文件: {video_name}")