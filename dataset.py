import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import Config

class Dataset:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        print("all data number:",self.data_num )
        self.train_num = int(self.data_num * self.config.train_data_rate)
        print("train data number:",self.train_num)

        self.mean = np.mean(self.data, axis=0)              # 数据的均值和方差
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # 归一化，去量纲

    def read_data(self):                # 读取初始数据
        if self.config.phase=="train": # or predict
            init_data = pd.read_csv(self.config.train_data_path,header=0)#usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.test_data_path,header=0)
            
        #print(init_data.columns.tolist())
        return init_data.values, init_data.columns.tolist()     # .columns.tolist() 是获取列名

    def get_train_and_valid_data(self):
        print(self.data)
        feature_data = self.norm_data[0:self.train_num,1:1+len(self.config.feature_columns)]
        label_data = self.norm_data[0:self.train_num,self.config.label_columns]    

        #少量数据测试，原样输出确定数据划分正确
        # print("--------------------------------------")
        # print(label_data)
        # print(self.mean[0]) #y 预测列的均值和标准差，还原数据时用
        # print(self.std[0])

        # print(self.train_num)
        # print("--------------------------------------")
        # print(self.data[0:self.train_num,1:1+len(self.config.feature_columns)])#x
        # print("--------------------------------------")
        # print(self.data[0:self.train_num,self.config.label_columns]) #y
        # print("--------------------------------------")
      
      
        train_x = feature_data
        train_y = label_data


        train_x, train_y = np.array(train_x), np.array(train_y)
        #print(train_x.shape)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=False)   # 划分训练和验证集
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        
        #假设数据都在一个文件夹
        # feature_data = self.norm_data[self.train_num:,1:1+len(self.config.feature_columns)]
        # label_data = self.norm_data[self.train_num:,self.config.label_columns] 
        
        #取所有数据测试一下
        feature_data = self.norm_data[:,1:1+len(self.config.feature_columns)]
        label_data = self.norm_data[:,self.config.label_columns]   
        
        # ##少量数据测试，原样输出确定数据划分正确
        # print("--------------------------------------")
        # print(self.data[:,1:1+len(self.config.feature_columns)]) #y
        # print("--------------------------------------")
        # print(self.data[:,self.config.label_columns])#x
        # print("--------------------------------------")
        
        
        test_x=feature_data
      
        if return_label_data:       # 实际应用中的测试集是没有label数据的
            return np.array(test_x), label_data
        return np.array(test_x)


np.random.seed(Config.random_seed)  # 设置随机种子，保证可复现
data_g = Dataset(Config)