# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:14:17 2024

@author: Arturo
"""
import tensorflow as tf

#==================================================
# Enable if you are use oneDNN (Improve performance with intel processors tensorflow 2.9 or later).
TF_ENABLE_ONEDNN_OPTS=1

x = tf.constant([[1.,2.,3.,
                  4.,5.,6.]])

print(x) #--------------------> Variable to return.
print(x.shape) #--------------> Shape of tensor.
print(x.dtype) #--------------> Type of variable.
#==================================================
