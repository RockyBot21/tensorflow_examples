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

print(x), print("\n") #--------------------> Variable to return.
print(x.shape), print("\n") #--------------> Shape of tensor.
print(x.dtype), print("\n") #--------------> Type of variable.
#=================================================
# Multiplication (Math operations)
print(5 * x), print("\n")

#=================================================
# Transpose variable
print(tf.transpose(x)), print("\n")
print(x @ tf.transpose(x)), print("\n")  #-----> Multiply each element & sum all.

#=================================================
# Concat several values or tensors.
print(tf.concat([x,x,x], axis=0)), print("\n")

#=================================================
# Sum all values in one value
print(tf.reduce_sum(x)), print("\n")

#=================================================
# Assign tensor variables
var = tf.Variable([0.0, 0.0, 0.0])
print(var), print("\n")

#=================================================
# Assig values to variable
var.assign([1,2,3])
print(var), print("\n")

#=================================================
# Assig values and add
var.assign_add([1,1,1])
print(var), print("\n")

#=================================================