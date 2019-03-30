# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import tensorflow as tf

class reparameterize(object):
    def __init__(self):
        super(reparameterize, self).__init__()

    def __call__(self, mu, logvar, sample_num=1, phase='training'):
        if phase == 'training':
            std = tf.exp(logvar*0.5)
            eps = tf.keras.backend.random_normal(shape=[1]+std.get_shape().as_list()[1:], mean=0., stddev=1.)
            # eps = tf.Variable(std.data.new(std.size()).normal_())
            return (eps*std)+mu
        else:
            raise ValueError('Wrong phase. Always assume training phase.')
        # elif phase == 'test':tf.exp(
        #  return mu
        # elif phase == 'generation':
        #  eps = Variable(logvar.data.new(logvar.size()).normal_())
        #  return eps

if __name__ =="__main__":
    sample = reparameterize()
    sess = tf.Session()
    mu = tf.zeros((2,4))
    std = tf.ones((2,4))
    print(sess.run(sample(mu,std)))