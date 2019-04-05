import tensorflow as tf
# tf.set_random_seed(1)
from collections import OrderedDict


class DistributionRender(tf.keras.Model):
  def __init__(self, hiddim):
    super(DistributionRender, self).__init__()
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

  @tf.function
  def call(self, x):
    mean = self.render_mean(x[0])
    var = self.render_var(x[1])
    return mean,var 

def run():
  x1 = [tf.random.normal([1,16,16,3]),tf.random.normal([1,16,16,3])]
  import time
  s = time.time()
  dr(x1)
  print(time.time() - s)

if __name__ == "__main__":
  dr = DistributionRender(10)
  for i in range(20):
    run()
  print(len(dr.trainable_variables))
