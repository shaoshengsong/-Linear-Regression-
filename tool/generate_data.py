# -*- coding: utf-8 -*-
'''
author:shaoshengsong
generat random data and write them to the file
随机生成数据，并写入文件中
我们生成的数据具有以下列
主键ID，时间，数值，其余的是特征
#'object_id','year_month_day','value','feature_1','feature_2','feature_3'
变成
object_id','year_month_day','value','feature_1','feature_2','feature_3','feature_month',feature_day
最终在程序中使用的是year_month_day变成多个feature列，还要加入一个label列
#pandas和numpy随机生成一张表格数据，包含生成日期，合并列，设置列标题，写入文件,使用matplotlib可视化等
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontManager

record_count=200 #记录条数
feature_count=3 #特征数

#数据一列或者几列的生成
data_id=pd.DataFrame(np.ones((record_count,1),dtype=int))  # object_id 相当于一张表的主键

data_feature = pd.DataFrame(np.random.uniform(10, 100, (record_count,feature_count)))  # feature

#  生成日期，从2019-12-31开始按天生成，共生成record_count个数据
#  1D，这里可以是M，按月生成3M,按3月生成。具体都可以是什么可以看这里
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
data_date = pd.DataFrame(pd.date_range('2019-12-31 00:00:00', periods=record_count, freq='1D'))
data_value=pd.DataFrame(np.random.uniform(200,300,(record_count,1)))

# 4种数据组成一张表，一列或者几列的连接
data_temp=pd.concat([data_id,data_date,data_value,data_feature],axis=1)

#设置列标题
data_temp.columns=(['object_id','year_month_day','value','feature_1','feature_2','feature_3'])
print(data_temp.head(10))
#将月和日也作为feature
data_temp['feature_month'] = data_temp['year_month_day'].dt.month
data_temp['feature_day'] = data_temp['year_month_day'].dt.day
data_temp['value']=pd.DataFrame(data_temp['feature_day'] * 10)#可以更改下标签 与时间相关

#打印头10条数据
print(data_temp.head(10))

#数据写入csv文件
data_temp.to_csv("./data/data5.csv", mode='w', index=False,header=True)


# df = data_temp.groupby('feature_day').size()#按个数统计
#
# #按日分组统计，计算ualue的和
# df =data_temp['value'].groupby(data_temp['feature_day']).sum()
#
# # Make the plot with pandas
# #可以是pie也可以是line,bar等
# df.plot(kind='line', subplots=True, figsize=(8, 8))
#
#
# #支持中文
# font = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf')
# plt.title("按时间统计", fontproperties=font)
# plt.xlabel("x轴", fontproperties=font)
# plt.ylabel("y轴", fontproperties=font)
# plt.show()

#中断退出
exit(0)



















chengji=[[100,95,100,99],[90,98,99,100],[88,95,98,88],[99,98,97,87],[96.5,90,96,85],[94,94,93,91],[91, 99, 92, 87], [85, 88, 85, 90], [90, 92, 99, 88], [90, 88, 89, 81], [85, 89, 89, 82], [95, 87, 86, 88], [90, 97, 97, 98], [80, 92, 89, 98], [80, 98, 85, 81], [98, 88, 95, 92]]
data=pd.DataFrame(chengji,columns=['语文','英语','数学','政治'])
print (data)
# data1=data[['数学','语文','英语','政治']]                               #排序
# data1=data1.reset_index(drop=True)                    #序列重建
# data1.index.names=['序号']                                 #序列重命名
# data1.index=data1.index+1                             #序列从1开始
# print (data1)
data=pd.DataFrame(chengji,columns=['语文','英语','数学','政治'],index=[i for i in range(1,len(chengji)+1)])
print (data)
data[['合计','平均']]=data.apply(lambda x: (x.sum(), x.sum()/4),axis=1,result_type='expand')
print (data[:])

data=pd.DataFrame(chengji,columns=['语文','英语','数学','政治'],index=[i for i in range(1,len(chengji)+1)])
print (data)
data[['合计','平均']]=data.apply(lambda x:('数据','月份'),axis=1,result_type='expand')
print (data[:])