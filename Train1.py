
import tensorflow as tf
import numpy as np
import pandas as pd 
import random

tf.reset_default_graph()


train_user=0
train_time=1


model_path=""
X_csv_train_file_path=""
y_csv_train_file_path=""
Y_prediction_file_path=""

root_directory = "D:\\jd\\model"
#train_object=train_user
train_object=train_time

if train_object == train_user :
        model_path = root_directory+"\\User\\train.ckpt"
        X_csv_train_file_path = "../data/X_train_user.csv"
        y_csv_train_file_path = "../data/y_train_user.csv"
        Y_prediction_file_path = "../data/y_pred_user.csv"


if train_object == train_time :
        model_path = root_directory+"\\Time\\train.ckpt"
        X_csv_train_file_path = "../data/X_train_time.csv"
        y_csv_train_file_path = "../data/y_train_time.csv"
        Y_prediction_file_path = "../data/y_pred_time.csv"






feature_size=57
dataset_size=0
X1=pd.read_csv(X_csv_train_file_path,header=0)
X1.fillna(0,inplace=True)
X=np.array(X1)

[dataset_size,feature_size]=X.shape
print(X.shape)
maximums, minimums, avgs = X.max(axis=0), X.min(axis=0), X.sum(
        axis=0) / X.shape[0]
for i in range(feature_size - 1):
        X[:, i] = (X[:, i] - avgs[i]) / (maximums[i] - minimums[i])


Y1 = pd.read_csv(y_csv_train_file_path,header=0)
Y1.fillna(0,inplace=True)
Y=np.array(Y1)
Y=Y.reshape(-1,1)



print(Y.shape)
# 定义训练数据batch的大小
batch_size=100
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
optimizer = tf.train.AdamOptimizer(0.01)#查看不同的优化效果
train_step = optimizer.minimize(cross_entropy)

with tf.Session() as sess:
    saver = tf.train.Saver()
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    
    print('训练前网络参数的值为：')
    print(sess.run(w))
    print(sess.run(b))
    
    # 设定训练的轮数 准备做随机批量
    STEPS=200000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        #start=(i*batch_size)%dataset_size
        #end=min(start+batch_size,dataset_size)
        n=random.randint(0, dataset_size-batch_size)
        start=n
        end=n+batch_size
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 100==0:
            # 每隔一段时间计算交叉熵并输出
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("start:{}: After{} training step(s),cross entropy on all data is {}".
                  format(start,i,total_cross_entropy))
    
    print('训练后网络参数的值为：')
    print(sess.run(w))
    print(sess.run(b))

    save_path = saver.save(sess, model_path)

    print("complete")