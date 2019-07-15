#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basic_eager_api.py
@Time    :   2019/07/15 16:40:30
@Author  :   jmh 
@Version :   1.0
@Contact :   mail@qq.com
@License :   
@Desc    :   None
'''

# here put the import lib
    
from __future__ import absolute_import, print_function, division
import numpy as np
import tensorflow as tf

import tensorflow.contrib.eager as tfe

print("Setting eager mode...")
tfe.enable_eager_execution()

a = tf.constant(2)
b = tf.constant(3)
print("a = %i" % a)
print("b = %i" % b)

c = a + b
print("a+b = %i" % c)
d = a*b
print("a*b = %i" % d)

a = tf.constant([[2., 1.], [1., 0.]], dtype=tf.float32)
print("Tensor \n a=%s" % a)

b = np.array([[3., 1.], [5., 1.]], dtype=np.float32)
print("NumpyArray \n b=%s" % b)

c = a + b
print("a + b = %s" % c) 

d = tf.matmul(a, b)
print("a * b = %s" % d)

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])


