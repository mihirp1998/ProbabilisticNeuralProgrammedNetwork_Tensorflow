import tensorflow as tf
from collections import OrderedDict
from instance_norm import InstanceNormalization
import time
from tensorflow.python.platform import tf_logging
tf_logging.set_verbosity(tf_logging.DEBUG)
import ipdb
st = ipdb.set_trace
# tf.set_random_seed(1)

class Combine_Vis(tf.keras.Model):
  def __init__(self, hiddim_v, hiddim_p=None, op='PROD'):
    super(Combine_Vis, self).__init__()
    self.op = op
    self.hiddim_v = hiddim_v
    if self.op == 'gPoE':
      self.gates_v = tf.keras.Sequential([tf.keras.layers.Conv2D(hiddim_v*4,3,1,padding="same"),
        InstanceNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Conv2D(hiddim_v*4,3,1,padding="same"),
        InstanceNormalization()])

  @tf.function
  def call(self, x1, x2):
    if self.op == 'gPoE':
        gates    = tf.keras.layers.Activation('sigmoid')(self.gates_v(tf.concat([x1[0], x1[1], x2[0], x2[1]], 3)))
        x1_mu_g  = gates[:,:,:,:self.hiddim_v]
        x1_var_g = gates[:,:,:,self.hiddim_v:2*self.hiddim_v]
        x2_mu_g  = gates[:,:,:,2*self.hiddim_v:3*self.hiddim_v]
        x2_var_g = gates[:,:,:,3*self.hiddim_v:4*self.hiddim_v]
        x1_updated = [x1_mu_g*x1[0],tf.math.log(x1_var_g + 1e-5) + x1[1]]
        x2_updated = [x2_mu_g*x2[0], tf.math.log(x2_var_g + 1e-5) + x2[1]]
        mlogvar1 = -x1_updated[1]
        mlogvar2 = -x2_updated[1]
        mu1      = x1_updated[0]
        mu2      = x2_updated[0]
        logvar   = -tf.math.log(tf.math.exp(mlogvar1) + tf.math.exp(mlogvar2))
        mu       = tf.math.exp(logvar)*(tf.math.exp(mlogvar1)*mu1 + tf.math.exp(mlogvar2)*mu2)
        return [mu, logvar]


class Combine_Pos(tf.keras.Model):
  def __init__(self,hiddim_p, op='PROD'):
    super(Combine_Pos, self).__init__()
    self.op = op
    self.hiddim_p = hiddim_p
    if self.op == 'gPoE':
      self.gates_p = tf.keras.Sequential([tf.keras.layers.Conv2D(hiddim_p*4,3,1,padding="same"),
        InstanceNormalization(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Conv2D(hiddim_p*4,3,1,padding="same"),
        InstanceNormalization()])

  @tf.function
  def call(self, x1, x2):
    if self.op == 'gPoE':
        gates    = tf.keras.layers.Activation('sigmoid')(self.gates_p(tf.concat([x1[0], x1[1], x2[0], x2[1]], 3)))
        x1_mu_g  = gates[:,:,:,:self.hiddim_p]
        x1_var_g = gates[:,:,:,self.hiddim_p:2*self.hiddim_p]
        x2_mu_g  = gates[:,:,:,2*self.hiddim_p:3*self.hiddim_p]
        x2_var_g = gates[:,:,:,3*self.hiddim_p:4*self.hiddim_p]

        x1_updated = [x1_mu_g*x1[0],tf.math.log(x1_var_g + 1e-5) + x1[1]]
        x2_updated = [x2_mu_g*x2[0], tf.math.log(x2_var_g + 1e-5) + x2[1]]

        mlogvar1 = -x1_updated[1]
        mlogvar2 = -x2_updated[1]
        mu1      = x1_updated[0]
        mu2      = x2_updated[0]

        logvar   = -tf.math.log(tf.math.exp(mlogvar1) + tf.math.exp(mlogvar2))
        mu       = tf.math.exp(logvar)*(tf.math.exp(mlogvar1)*mu1 + tf.math.exp(mlogvar2)*mu2)
        return [mu, logvar]

def run():
  x1 = [tf.random.normal([1,16,16,64]),tf.random.normal([1,16,16,64])]
  x2 = [tf.random.normal([1,16,16,64]),tf.random.normal([1,16,16,64])]
  import time
  s = time.time()
  cfunc(x1,x2,True)
  print(time.time() - s)

if __name__ == "__main__":
  cfunc= Combine_Vis(64,"gPoE")
  for i in range(20):
    run()
  print(len(cfunc.trainable_variables))
