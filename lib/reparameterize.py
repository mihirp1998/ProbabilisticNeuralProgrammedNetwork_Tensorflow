# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import tensorflow as tf

class reparameterize(tf.keras.Model):
    def __init__(self):
        super(reparameterize, self).__init__()

    @tf.function
    def call(self, mu, logvar, sample_num=1, phase='training'):
        if phase == 'training':
            std = tf.math.exp(logvar*0.5)
            eps = tf.keras.backend.random_normal(shape=[1]+std.get_shape().as_list()[1:], mean=0., stddev=1.)
            return (eps*std)+mu
        else:
            raise ValueError('Wrong phase. Always assume training phase.')
        # elif phase == 'test':tf.math.exp(
        #  return mu
        # elif phase == 'generation':
        #  eps = Variable(logvar.data.new(logvar.size()).normal_())
        #  return eps
def run():
  x1 = tf.random.normal([16,64])
  x2 = tf.random.normal([16,64])
  import time
  s = time.time()
  sample(x1,x2)
  print(time.time() - s)
if __name__ =="__main__":
    sample = reparameterize()
    for i in range(20):
        run()
    print(len(sample.trainable_variables))