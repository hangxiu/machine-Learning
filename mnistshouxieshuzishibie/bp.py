import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
# 1. MNIST 数据集
# 自动下载和安装这个数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 1. 回归模型实现
# x不是一个特定的值，而是一个占位符placeholder,
# 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder("float", [None, 784])
# 我们对图片像素值进行加权求和
# W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量
W = tf.Variable(tf.zeros([784, 10]))
# 加入一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量
# b的形状是[10]，所以我们可以直接把它加到输出上面
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# 3. 训练模型
y_ = tf.placeholder("float", [None, 10])
# 在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss）
# 交叉熵
# y 是我们预测的概率分布, y_是实际的分布（我们输入的one-hot vector)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# TensorFlow会用你选择的优化算法来不断地修改变量以降低成本
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 初始化我们创建的变量
init = tf.initialize_all_variables()
# 现在我们可以在一个Session里面启动我们的模型，并且初始化变量：
sess = tf.Session()
sess.run(init)

# 我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
for i in range(1000):
    batchxs, batchys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batchxs, y_: batchys})
#  4. 评估模型
# 让我们找出那些预测正确的标签
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 取平均值后得到 正確率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 最后，我们计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy,
               feed_dict={
                   x: mnist.test.images,
                   y_: mnist.test.labels
               }))
