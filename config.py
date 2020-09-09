# -*- encoding: utf8 -*-
import os
import torch

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

# PROJECT_PATH = '/Users/gechen/PycharmProjects/FaceDemo/'
# 训练数据集
DATA_TRAIN = os.path.join(PROJECT_PATH, "FaceDemo/data/train")
# 验证数据集ss
DATA_TEST = os.path.join(PROJECT_PATH, "FaceDemo/data/test")
# 模型保存地址
DATA_MODEL = os.path.join(PROJECT_PATH, "FaceDemo/data/model")

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10