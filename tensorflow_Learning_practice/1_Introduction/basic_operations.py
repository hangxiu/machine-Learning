#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basic_operations.py
@Time    :   2019/07/15 16:04:10
@Author  :   jmh 
@Version :   1.0
@Contact :   mail@qq.com
@License :   
@Desc    :   None
'''

# here put the import lib
from __future__ import print_function
import tensorflow as tf

constant1 = tf.constant(2)
constant2 = tf.constant(3)


with tf.Session() as sess:
    print("constant1=2, constant2=3")
    print("Addition with constants: %i" % sess.run(constant1+constant2))
    print("Multiplication with constants: %i" % sess.run(constant1*constant2))

constant1 = tf.compat.v1.placeholder(tf.int16)
constant2 = tf.compat.v1.placeholder(tf.int16)

add = tf.add(constant1, constant2)
mul = tf.multiply(constant1, constant2)

with tf.Session() as sess:
    print("Addition with constants: %i" % sess.run(add, feed_dict={constant1: 2, constant2: 3}))
    print("Multiplication with constants: %i" % sess.run(mul, feed_dict={constant1: 2, constant2: 3}))


matrix1 = tf.constant([[3., 3.0]])
matrix2 = tf.constant([[2.],  [2.]])  

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)



