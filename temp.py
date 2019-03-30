import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
data = np.reshape(np.arange(27), [1,3, 3, 3])
print(data)
#a= np.random.randn(3,4,3)
# m = tf.random.uniform((3,2))
m = tf.random.uniform((1,2,2,3))
index = [[[[0, 0,0],[0, 0,1]],[[0, 1,0],[0, 1,1]]]]
p = tf.gather_nd(data,index)
p =tf.scatter_nd(index,m,[1,3, 3, 3])
print(p,p.shape)
#z = tf.zeros((4,3))
#z[0,:] = m
