import os
import time
import numpy as np

class Config(object):
    def __init__(self):

        # General
        self.epochs = 12
        # self.GPUs = '0'
        self.batch_size = 800    
        self.date = '0626'

        # Data
        self.data_dir = '../../../datasets/mm'
        self.dir_train = os.path.join(self.data_dir, 'train')
        self.dir_test = os.path.join(self.data_dir, 'test')
        self.dir_water = os.path.join(self.data_dir, 'train/train_vid_frames/val_water')
        self.dir_baby = os.path.join(self.data_dir, 'train/train_vid_frames/val_baby')
        self.dir_gun = os.path.join(self.data_dir, 'train/train_vid_frames/val_gun')
        self.dir_drone = os.path.join(self.data_dir, 'train/train_vid_frames/val_drone')
        self.dir_guitar = os.path.join(self.data_dir, 'train/train_vid_frames/val_guitar')
        self.dir_cattoy = os.path.join(self.data_dir, 'train/train_vid_frames/val_cattoy')
        self.dir_myself = os.path.join(self.data_dir, 'train/train_vid_frames/myself')
        self.frames_train = 'coco100000'      
        self.cursor_end = int(self.frames_train.split('coco')[-1])
        if os.path.exists(os.path.join(self.dir_train, 'train_mf.txt')):
          self.coco_amp_lst = np.loadtxt(os.path.join(self.dir_train, 'train_mf.txt'))[:self.cursor_end]
        else:
          print('Please load train_mf.txt if you want to do training.')
          self.coco_amp_lst = None
        self.videos_train = []
        self.load_all = False 

        # Training
        self.lr = 1e-4
        self.betas = (0.9, 0.999)
        self.batch_size_test = 1
        self.preproc = ['poisson']   # ['resize', ]
        self.pretrained_weights = ''

        # Callbacks
        self.num_val_per_epoch = 10
        self.save_dir = 'weights_date{}'.format(self.date)
        self.time_st = time.time()
        self.losses = []