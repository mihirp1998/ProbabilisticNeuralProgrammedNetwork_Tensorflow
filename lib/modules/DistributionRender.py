import tensorflow as tf
tf.set_random_seed(1)
from collections import OrderedDict


class DistributionRender(object):
  def __init__(self,name, hiddim):
    super(DistributionRender, self).__init__()
    self.name = name
    self.trainable_variables = OrderedDict()
    self.render_mean = tf.keras.Sequential([
                      tf.keras.layers.Conv2D(hiddim, 3, 1, "same"),
                      tf.keras.layers.Activation("elu"),
                      tf.keras.layers.Conv2D(hiddim, 3, 1, "same"),
                      tf.keras.layers.Activation("elu"),
                      tf.keras.layers.Conv2D(hiddim, 3, 1, "same")])

    self.render_var  = tf.keras.Sequential([
                      tf.keras.layers.Conv2D(hiddim, 3, 1, "same"),
                      tf.keras.layers.Activation("elu"),
                      tf.keras.layers.Conv2D(hiddim, 3, 1, "same"),
                      tf.keras.layers.Activation("elu"),
                      tf.keras.layers.Conv2D(hiddim, 3, 1, "same")])


  def __call__(self, x):
    # x = [mean, var]
    with tf.variable_scope(self.name):
      mean = self.render_mean(x[0])
      var = self.render_var(x[1])
      if "render_mean" not in self.trainable_variables:
        self.trainable_variables["render_mean"] = self.render_mean.trainable_variables 
      if "render_var" not in self.trainable_variables:
        self.trainable_variables["render_var"] = self.render_var.trainable_variables
      return mean,var 

if __name__ == "__main__":
  dr = DistributionRender("dib",10)
  print(dr([tf.zeros([1,16,16,3]),tf.zeros([1,16,16,3])]))
  print(len(dr.trainable_variables))
