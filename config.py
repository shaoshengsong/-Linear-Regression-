import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 假设我们有这样的原始数据集
# y,x1,x2,x3 ...
# y是预测值
class Config:  #该配置用于回归模型
    #设置哪些列是feature列
    #设置哪些列是要预测的列

    # 数据参数
    feature_columns = [1, 2, 3]  # feature 都有哪些列，也就是'x1,x2,x3的索引
    label_columns = list([0]) #实例中 y的索引
    # 网络参数
    input_size = len(feature_columns)
    output_size = len(label_columns)
 
    # 训练参数
    phase="train" # or predict
    load_model=False

    train_data_rate = 0.9      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.1     # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 10
    learning_rate = 0.001
    epoch = 500                 # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 200                # 训练多少epoch，验证集没提升就停掉
    random_seed = 1            # 随机种子，保证可复现


    # 框架参数
    model_name="model.pth"


    # 路径参数
    train_data_path = "./data/train.csv"
    test_data_path = "./data/predict.csv"
    model_save_path = "./checkpoint/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True                  # 是否将config和训练过程记录到log
    do_figure_save = True
    do_train_visualized = False          # 训练loss可视化，pytorch用visdom 或者tensorboardX
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if phase=="train" and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + "/"
        os.makedirs(log_save_path)

    model=1 #模型类型