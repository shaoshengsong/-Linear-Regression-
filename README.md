PyTorch Linear Regression
PyTorch实现线性回归

包括模型训练、保存、推理使用等

输入输出维度可以都是一维的   
torch.nn.Linear(1, 1)
输入可以多维输出可以是一维的
torch.nn.Linear(10, 1)

整个文件夹结构如下
├── checkpoint
│   └── model.pth
├── config.py
├── data
│   ├── predict.csv
│   └── train.csv
├── dataset.py
├── evaluator.py
├── figure
├── log
├── main.py
├── model
│   └── net_regression.py
├── tool
│   ├── generate_data.py
│   └── merge_csv.py
├── trainner.py
└── utils.py

数据集部分
数据集包括在data文件夹中
train.csv文件存储了训练集
predict.csv存储了测试集

数据格式如下
y	x1	x2	x3
290	11	607	37979
515	12	690	61651
165	12	212	66933
265	10	372	51775

列标题y表示要预测的列，x1、x2、x3表示特征列
如果数据集有特别的处理，可以更改dataset.py

配置说明
关于配置在config.py文件

设置哪些列是feature列
设置哪些列是要预测的列
```
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
```
训练
python main.py -p "train"

测试
python main.py -p "predict"
