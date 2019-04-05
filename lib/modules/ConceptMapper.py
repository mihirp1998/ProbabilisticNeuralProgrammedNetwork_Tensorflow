
import tensorflow as tf
# tf.set_random_seed(1)
import ipdb
st = ipdb.set_trace

from collections import OrderedDict

class ConceptMapper(tf.keras.Model):
  def __init__(self, CHW):
    super(ConceptMapper, self).__init__()
    C, H, W = CHW[0], CHW[1], CHW[2]
    self.mean_dictionary = tf.keras.layers.Dense(C*H*W, use_bias=False,name="mean_dictionary")
    self.std_dictionary  = tf.keras.layers.Dense(C*H*W, use_bias=False,name="std_dictionary")
    self.C, self.H, self.W = C, H, W

  @tf.function
  def call(self, x):
    word_mean = self.mean_dictionary(x)
    word_std  = self.std_dictionary(x)

    if self.H == 1 and self.W == 1:
      return [tf.reshape(word_mean,(-1, 1, 1, self.C)), \
              tf.reshape(word_std,(-1, 1, 1, self.C))]
    else:
      return [tf.reshape(word_mean,(-1, self.H, self.W, self.C)), \
              tf.reshape(word_std,(-1, self.H, self.W, self.C))]


def run():
  # x1 = [tf.random.normal([10,21]),tf.random.normal([10,21])] 

  x1 = tf.random.normal([10,21])
  import time
  s = time.time()
  c(x1)
  print(time.time() - s)

if __name__ == "__main__":
  c = ConceptMapper([64,16,16])
  for i in range(20):
    run()
  print(len(c.trainable_variables))



