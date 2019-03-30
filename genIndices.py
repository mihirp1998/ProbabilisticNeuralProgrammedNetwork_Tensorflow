import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import ipdb
st = ipdb.set_trace

m =5
n=4
h = 2
w = 2
update = tf.random.uniform((1,h,w,64))
# tf.ran
h_val =np.arange(m,m+h)
w_val =np.arange(n,n+w)
indices = np.zeros((1,h,w,3))
for i_h in range(h):
	for i_w in range(w):
		indices[0,i_h,i_w,:] = np.array([0,h_val[i_h],w_val[i_w]])
indices = np.array(indices,np.int64)
# indices= np.array([[[[0, 0,0],[0, 0,1]],[[0, 1,0],[0, 1,1]]]])
# print(indices.dtype)
# print(m==indices)
print(indices)
# data = np.arange(36).reshape((1,3,3,4))
# p = tf.gather_nd(data,indices)
st()
p =tf.scatter_nd(indices,update,[1,16, 16, 64])

print(p)
