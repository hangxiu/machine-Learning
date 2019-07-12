# import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time
import math
# 最大迭代次数
max_step = 3000
# 批处理的大小
batch_size = 128
# data_dir = 'c:'

# 初始化weigth函数， 并通过wl参数控制L2正则化大小


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        # L2正则化可用tf.contrib.layers.l2_regularizer(lambda)(w)实现，自带正则化参数
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# cifar10.maybe_download_and_extract()

# 此处的cifar10_input.distorted_inputs()和cifar10_input.inputs()函数
# 都是TensorFlow的操作operation，需要在会话中run来实际运行
# distorted_inputs()函数对数据进行了数据增强
image_train, labels_train = cifar10_input.distorted_inputs(
    batch_size=batch_size)

# 裁剪图片正中间的24*24大小的区块并进行数据标准化操作
image_test, lables_test = cifar10_input.inputs(eval_data=True,
                                               batch_size=batch_size)
# 定义placeholder
# 注意此处输入尺寸的第一个值应该是batch_size而不是None
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
labels_holder = tf.placeholder(tf.int32, [batch_size])

# 卷积层1，不对权重进行正则化
weight1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder,
                       weight1,
                       strides=[1, 1, 1, 1],
                       padding='SAME')

bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 卷积层2
weight2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))

norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME')
# 全连接层3
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss([dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
# 全连接层4
weight4 = variable_with_weight_loss([384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
# 全连接层5
weight5 = variable_with_weight_loss([192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.nn.relu(tf.matmul(local4, weight5) + bias5)

# 定义损失函数loss


def loss(logits, lables):
    labels = tf.cast(lables, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 定义loss
loss = loss(logits, labels_holder)
# 定义优化器
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, labels_holder, 1)

# 定义会话并开始迭代训练
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动图片数据增强的线程队列
tf.train.start_queue_runners()

# 迭代训练
for i in range(max_step):
    start_time = time.time()
    # 获取训练数据
    image_batch, label_batch = sess.run([image_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={
                                 image_holder: image_batch,
                                 labels_holder: label_batch
                             })
    # 计算每次迭代需要的时间
    duration = time.time() - start_time
    if i % 10 == 0:
        # 每秒处理的样本数
        examples_per_sec = batch_size / duration
        # 每批处理需要的时间
        sec_per_batch = float(duration)
        format_str = ('step %d, loss=%.2f (%.2f examples/sec;%.3f sec/batch)')
        print(format_str % (i, loss_value, examples_per_sec, sec_per_batch))

# 在测试集上测评准确率
num_examples = 10000

num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, laber_batch = sess.run([image_test, lables_test])
    predictions = sess.run([top_k_op],
                           feed_dict={
                               image_holder: image_batch,
                               labels_holder: laber_batch
                           })
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision = %.3f' % precision)
