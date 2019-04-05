import tensorflow as tf
# tf.set_random_seed(1)
import ipdb
st = ipdb.set_trace
from collections import OrderedDict
from instance_norm import InstanceNormalization
# one more advanced plan: predict an attention map a, then
# render the vw by a*x*y + (1-a)*y
from tensorflow.python.platform import tf_logging
tf_logging.set_verbosity(tf_logging.DEBUG)
class Describe_Vis(tf.keras.Model):
    def __init__(self,hiddim_v, op='CAT'):
        super(Describe_Vis, self).__init__()
        self.op = op
        self.hiddim_v = hiddim_v
        if op == 'CAT_gPoE':
            self.net1_mean_vis = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(hiddim_v, 3, 1,padding="same"),
                InstanceNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.Conv2D(hiddim_v, 3, 1,padding="same"),
                InstanceNormalization()]
            )

            self.net1_var_vis = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(hiddim_v, 3, 1, "same"),
                InstanceNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.Conv2D(hiddim_v, 3, 1, "same"),
                InstanceNormalization()]
            )

            self.gates_v = tf.keras.Sequential(
               [tf.keras.layers.Conv2D(hiddim_v * 4, 3, 1, "same"),
                InstanceNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.Conv2D(hiddim_v * 4, 3, 1, "same"),
                InstanceNormalization()]
            )

    # Q: in gpoe why do we do different actions on mean and variance
    @tf.function
    def call(self, x, y, lognormal=False):  # -> x describe y
        x_mean = self.net1_mean_vis(tf.concat([x[0], y[0]], 3))
        
        x_var = self.net1_var_vis(tf.concat([x[1], y[1]], 3))

        # gates
        gates = tf.keras.layers.Activation('sigmoid')(self.gates_v(tf.concat([x_mean, x_var, y[0], y[1]], 3)))

        x1_mu_g  = gates[:,:,:,:self.hiddim_v]
        x1_var_g = gates[:,:,:,self.hiddim_v:2*self.hiddim_v]
        x2_mu_g  = gates[:,:,:,2*self.hiddim_v:3*self.hiddim_v]
        x2_var_g = gates[:,:,:,3*self.hiddim_v:4*self.hiddim_v]

        x_mean = x1_mu_g * x_mean
        x_var = tf.math.log(x1_var_g + 1e-5) + x_var
        y_u0 = x2_mu_g * y[0]
        y_u1 = tf.math.log(x2_var_g + 1e-5) + y[1]
        y_u = [y_u0,y_u1]

        mlogvar1 = -x_var
        mlogvar2 = -y_u[1]
        mu1 = x_mean
        mu2 = y_u[0]

        y_var = -tf.math.log(tf.math.exp(mlogvar1) + tf.math.exp(mlogvar2))
        y_mean = tf.math.exp(y_var) * (tf.math.exp(mlogvar1) * mu1 + tf.math.exp(mlogvar2) * mu2)
        return [y_mean, y_var]


class Describe_Pos(tf.keras.Model):
    def __init__(self,hiddim_p, op='CAT'):
        super(Describe_Pos, self).__init__()
        self.op = op
        self.hiddim_p = hiddim_p

        if op == 'CAT_gPoE':
            self.net1_mean_pos = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(hiddim_p, 1,1, "same"),
                InstanceNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.Conv2D(hiddim_p, 1,1, "same"),
                InstanceNormalization()]
            )

            self.net1_var_pos = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(hiddim_p, 1,1, "same"),
                InstanceNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.Conv2D(hiddim_p, 1,1, "same"),
                InstanceNormalization()]
            )

            self.gates_p = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(hiddim_p * 4, 3, 1, "same"),
                InstanceNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.Conv2D(hiddim_p * 4, 3, 1, "same"),
                InstanceNormalization()]
            )
    # Q: in gpoe why do we do different actions on mean and variance
    @tf.function
    def call(self, x, y, lognormal=False):  # -> x describe y
        x_mean = self.net1_mean_pos(tf.concat([x[0], y[0]], 3))
        
        x_var = self.net1_var_pos(tf.concat([x[1], y[1]], 3))

        gates = tf.keras.layers.Activation('sigmoid')(self.gates_p(tf.concat([x_mean, x_var, y[0], y[1]], 3)))
        
        x1_mu_g  = gates[:,:,:,:self.hiddim_p]
        x1_var_g = gates[:,:,:,self.hiddim_p:2*self.hiddim_p]
        x2_mu_g  = gates[:,:,:,2*self.hiddim_p:3*self.hiddim_p]
        x2_var_g = gates[:,:,:,3*self.hiddim_p:4*self.hiddim_p]

        x_mean = x1_mu_g * x_mean
        x_var = tf.math.log(x1_var_g + 1e-5) + x_var
        y_u0 = x2_mu_g * y[0]
        y_u1 = tf.math.log(x2_var_g + 1e-5) + y[1]
        y_u = [y_u0,y_u1]
        mlogvar1 = -x_var
        mlogvar2 = -y_u[1]
        mu1 = x_mean
        mu2 = y_u[0]

        y_var = -tf.math.log(tf.math.exp(mlogvar1) + tf.math.exp(mlogvar2))
        y_mean = tf.math.exp(y_var) * (tf.math.exp(mlogvar1) * mu1 + tf.math.exp(mlogvar2) * mu2)
        return [y_mean, y_var]

def run():
  x1 = [tf.random.normal([1,16,16,64]),tf.random.normal([1,16,16,64])]
  x2 = [tf.random.normal([1,16,16,64]),tf.random.normal([1,16,16,64])]
  import time
  s = time.time()
  ct(x1,x2,True)
  print(time.time() - s)


if __name__ == "__main__":
  ct= Describe_Pos(hiddim_p=64,op="CAT_gPoE")
  # st()
  for i in range(20):
    run()
  print(len(ct.trainable_variables))

