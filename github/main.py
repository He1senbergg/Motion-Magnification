import os
import cv2
import torch
from tqdm import tqdm
from config import Config
from magnet import MagNet
from callbacks import gen_state_dict
from data import get_gen_ABC, unit_postprocessing, numpy2cuda


# config
config = Config()
# 模型pth位置
weights_path = 'checkpoint/checkpoint.pth'
ep = int(weights_path.split('epoch')[-1].split('_')[0])
state_dict = gen_state_dict(weights_path)

model_test = MagNet().cuda()
model_test.load_state_dict(state_dict)
model_test.eval()
print("Loading weights:", weights_path)

# 原视频的位置
video_path = '' 
video_name = os.path.basename(video_path).split('.')[0]
video_format = '.' + os.path.basename(video_path).split('.')[-1]

# 保存帧图像的子文件夹名称
testset = 'first5mins'
# 保存的位置的子文件夹名称
dir_results = 'res_' + testset
# 保存的位置
base_dir = r''
dir_results = os.path.join(base_dir, dir_results)
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

# 保存帧图像的前置文件夹路径
config.data_dir = r''
data_loader = get_gen_ABC(config, mode='test_on_'+testset)
print('Number of test image couples:', data_loader.data_len)
vid_size = cv2.imread(data_loader.paths[0]).shape[:2][::-1]

for amp in [5, 10, 20, 40]:
    data_loader = get_gen_ABC(config, mode='test_on_'+testset)
    output_folder = os.path.join(dir_results, f"{testset}_amp{amp}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    frame_count = 0

    for idx_load in tqdm(range(0, data_loader.data_len, data_loader.batch_size), desc=f"Processing {amp}x"):
        batch_A, batch_B = data_loader.gen_test()
        amp_factor = numpy2cuda(amp)
        for _ in range(len(batch_A.shape) - len(amp_factor.shape)):
            amp_factor = amp_factor.unsqueeze(-1)
        with torch.no_grad():
            y_hats = model_test(batch_A, batch_B, 0, 0, amp_factor, mode='evaluate')
        for y_hat in y_hats:
            y_hat = unit_postprocessing(y_hat, vid_size=vid_size)
            filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(filename, cv2.cvtColor(y_hat, cv2.COLOR_RGB2BGR))
            frame_count += 1
            if frame_count >= data_loader.data_len:
                break
        if frame_count >= data_loader.data_len:
            break