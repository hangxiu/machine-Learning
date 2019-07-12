# import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

# 本节使用的数据集是CIFAR-10，这是一个经典的数据集，包含60000张32*32的彩色图像，其中训练集50000张，测试集10000张
# 一共标注为10类，每一类图片6000张。10类分别是 airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# 我们载入一些常用库，比如NumPy和time，并载入TensorFlow Models中自动下载、读取CIFAR-10数据的类

# 最大迭代次数
max_step = 3000
# 批处理的大小
batch_size = 128
# data_dir = 'c:'

# 初始化weigth函数， 并通过wl参数控制L2正则化大小


def variable_with_weight_loss(shape, stddev, wl):
    # 定义初始化weights的函数，和之前一样依然使用tf.truncated_normal截断的正太分布来初始化权值
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        # L2正则化可用tf.contrib.layers.l2_regularizer(lambda)(w)实现，自带正则化参数
        # 给weight加一个L2的loss，相当于做了一个L2的正则化处理
        # 在机器学习中，不管是分类还是回归任务，都可能因为特征过多而导致过拟合，一般可以通过减少特征或者惩罚不重要特征的权重来缓解这个问题
        # 但是通常我们并不知道该惩罚哪些特征的权重，而正则化就是帮助我们惩罚特征权重的，即特征的权重也会成为模型的损失函数的一部分
        # 我们使用w1来控制L2 loss的大小
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        # 我们使用tf.add_to_collection把weight loss统一存到一个collection，这个collection名为"losses"，它会在后面计算神经网络
        # 总体loss时被用上
        tf.add_to_collection('losses', weight_loss)
    return var


# cifar10.maybe_download_and_extract()

# 此处的cifar10_input.distorted_inputs()和cifar10_input.inputs()函数
# 都是TensorFlow的操作operation，需要在会话中run来实际运行
# distorted_inputs()函数对数据进行了数据增强
# 使用cifar10_input类中的distorted_inputs函数产生训练需要使用的数据，包括特征及其对应的label，这里是封装好的tensor，
# 每次执行都会生成一个batch_size的数量的样本。需要注意的是这里对数据进行了Data Augmentation数据增强
# 具体实现细节查看函数，其中数据增强操作包括随机水平翻转tf.image.random_flip_left_right()
# 随机剪切一块24*24大小的图片tf.random_crop，随机设置亮度和对比度，tf.image.random_brightness、tf.image.random_contrast
# 以及对数据进行标准化，白化 tf.image.per_image_standardization() 减去均值、除以方差，保证数据零均值，方差为1
image_train, labels_train = cifar10_input.distorted_inputs(
    batch_size=batch_size)

# 裁剪图片正中间的24*24大小的区块并进行数据标准化操作
# 生成测试数据，不过这里不需要进行太多处理，不需要对图片进行翻转或修改亮度、对比度，
# 不过需要裁剪图片正中间的24*24大小的区块。（因为训练的数据是24*24的，通过函数cifar10_input.distorted_inputs读进来时处理了）
# 并进行数据标准化操作
# 测试的是一批数据
image_test, lables_test = cifar10_input.inputs(eval_data=True,
                                               batch_size=batch_size)
# 定义placeholder
# 注意此处输入尺寸的第一个值应该是batch_size而不是None
# 因为batch_size在之后定义网络结构时被用到了，所以数据尺寸中的第一个值即样本条数需要被预先设定，而不能像以前那样设置为None
# 而数据尺寸中的图片尺寸为24*24即是剪裁后的大小，颜色通道数则设为3
# 这里写batch_size而不是None 因为后面代码中get_shape会拿到这里面的batch_size
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
# 用激活函数
# 使用尺寸3*3步长2*2的最大池化层处理数据，这里最大池化的尺寸和步长不一样，可以增加数据的丰富性
pool1 = tf.nn.max_pool(conv1,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME')
# 使用LRN对结果进行处理
# LRN最早见于Alex那篇用CNN参加ImageNet比赛的论文，Alex在论文中解释LRN层模仿了生物神经系统的"侧抑制(单边抑制)"机制，
# 对局部神经元的活动创建竞争环境，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
# Alex在ImageNet（上百万张图片）数据集上的实验表明，使用LRN后CNN在Top1的错误率可以降低1.4%，因此其在经典AlexNet中使用了LRN层
# LRN对ReLU这种没有上限边界的激活函数会比较有用，因为它会从附近的多个卷积核的响应中挑选比较大的反馈
# 但不适合Sigmoid这种有固定边界并且能抑制过大值得激活函数
# LRN对Relu配合较好，适合Alex架构                       
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 卷积层2
# 创建第二个卷积层
# 上面64个卷积核，即输出64个通道，所以本层卷积核尺寸的第三个维度即输入的通道数也需要调整为64
weight2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
# 这里的bias值全部初始化为0.1，而不是0.最后，调换了最大池化层和LRN层的顺序，先进行LRN层处理，再使用最大池化层
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))

norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME')
# 全连接层3
# 两个卷积层之后，是全连接层
# 先把第二个卷积层之后的输出结果flatten，使用tf.reshape函数将每个样本都变成一维向量，使用get_shape函数获取数据扁平化之后的长度
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
# 接着初始化权值，隐含节点384个，正太分布的标准差设为0.04，bias的值也初始化为0.1
# 注意这里我们希望这个全连接层不要过拟合，因此设了一个非零的weight loss值0.04，让这一层具有L2正则所约束。
weight3 = variable_with_weight_loss([dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
# 写0.1是为了Relu小于0时全为0，所以给0.1不至于成为死亡神经元
# 最后我们依然使用ReLU激活函数进行非线性化
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
# 全连接层4
#全连接的神经元384---192 是不断减少的，成倍减少 因为在不断的总结 卷积是不断地变宽，也是成倍的
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
