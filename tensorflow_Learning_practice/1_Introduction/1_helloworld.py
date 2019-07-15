#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   helloworld.py
@Time    :   2019/07/15 15:50:36
@Author  :   jmh 
@Version :   1.0
@Contact :   mail@qq.com
@License :   
@Desc    :   None
'''

# here put the import lib

from __future__ import print_function

import tensorflow as tf

hello = tf.constant('Hello tensorflow')

sess = tf.Session()

print(sess.run(hello))

