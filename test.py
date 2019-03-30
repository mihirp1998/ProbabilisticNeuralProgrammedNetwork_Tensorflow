import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import psutil as ps
import gc

tf.enable_eager_execution(config=tf.ConfigProto(log_device_placement=True))
for i in range (50000):
    w0=tfe.Variable(initial_value=np.ones((8,1)))
    print(ps.virtual_memory().percent)
    print(i,"iter")
