import tensorflow as tf
import numpy as np
import pandas as pd

train_user=0
train_time=1


model_path=""
X_csv_train_file_path=""        #需要训练的features
y_csv_train_file_path=""        #需要训练的label
Y_prediction_file_path=""       #需要需要预测的数据路径
output_file_path=""
root_directory = "D:\\jd\\model"#训练模型后保存的根目录

#需要训练两个模型 一个是关于用户的，一个是关于日期的
#train_object=train_user
train_object=train_time

if train_object == train_user :
        model_path = root_directory+"\\User\\train.ckpt"
        X_csv_train_file_path = "../data/X_train_user.csv"
        y_csv_train_file_path = "../data/y_train_user.csv"
        Y_prediction_file_path = "../data/y_pred_user.csv"
        output_file_path = "../data/S_1.csv"


if train_object == train_time :
        model_path = root_directory+"\\Time\\train.ckpt"
        X_csv_train_file_path = "../data/X_train_time.csv"
        y_csv_train_file_path = "../data/y_train_time.csv"
        Y_prediction_file_path = "../data/y_pred_time.csv"
        output_file_path = "../data/S_2.csv"
tf.reset_default_graph()

feature_size=57#特征数
dataset_size=0
X1 = pd.read_csv(Y_prediction_file_path )
X1.fillna(0,inplace=True)
X=np.array(X1)

[dataset_size,feature_size]=X.shape
print(X.shape)

[dataset_size,feature_size]=X.shape
print(X.shape)
maximums, minimums, avgs = X.max(axis=0), X.min(axis=0), X.sum(
        axis=0) / X.shape[0]
for i in range(feature_size - 1):
        X[:, i] = (X[:, i] - avgs[i]) / (maximums[i] - minimums[i])
        
# 定义神经网络参数
w=tf.Variable(tf.random_normal([feature_size,1],stddev=1,seed=1))
b = tf.Variable(0.0, name="biases",dtype=tf.float32)
x=tf.placeholder(tf.float32,shape=(None,feature_size),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

# 定义前向传播
y=tf.add(tf.matmul( x,w) ,b)
# 定义损失函数和反向传播算法
#cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
#train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

cross_entropy = tf.reduce_mean(tf.square(y - y_))
#optimizer = tf.train.GradientDescentOptimizer(0.001)
optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy)

batch_size=100


with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print("W:", sess.run(w))  # 打印v1、v2的值和之前的进行对比
    print("b:", sess.run(b))


    result = sess.run(y, feed_dict={x: X})
    print(result)
    df = pd.DataFrame(result)
    df['ID'] = pd.Series(df.index+1)


    df.to_csv(output_file_path, index=False)