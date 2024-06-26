# 使用顺序
## 1. `extract_video.py`
## 2. `from scratch` or `github` ->处理视频帧图像，进行运动放大
## 3. `img2video.py` 由处理好的视频帧图像，生成视频

# 代码文件介绍

## 一、最外层文件夹
- `extract_video.py`
  
    读入视频，转存为帧图像。

- `img2video.py`

    由处理好的图像，生成视频。

## 二、from scratch
- `MotionMag_Linear.py`

    线性方法
- `MotionMag_Phase_based.py`

    相位基方法

## 三、github

### 3-1 模型权重文件
- `github\checkpoint\checkpoint.pth`

### 3-2 代码文件
- **仅被调用文件**

  + `callbacks.py`

    读取模型权重

  + `config.py`

    读取配置信息

  + `data.py`

    加载数据信息

    + `magnet.py`

    定义模型

- **需要使用的文件**
  + `make_frameACB.py`

    用来将视频帧图像，转换为模型推理所需的数据集格式。

  + main.py`

    运行模型推理。