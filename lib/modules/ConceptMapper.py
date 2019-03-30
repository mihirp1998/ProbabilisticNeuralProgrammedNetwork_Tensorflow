
import tensorflow as tf
tf.set_random_seed(1)
import ipdb
st = ipdb.set_trace

from collections import OrderedDict

class ConceptMapper(object):
  def __init__(self,name, CHW):
    super(ConceptMapper, self).__init__()
    self.name = name
    self.trainable_variables = OrderedDict()
    # st()
    C, H, W = CHW[0], CHW[1], CHW[2]
    self.mean_dictionary = tf.keras.layers.Dense(C*H*W, use_bias=False,name="mean_dictionary")
    self.std_dictionary  = tf.keras.layers.Dense(C*H*W, use_bias=False,name="std_dictionary")
    self.C, self.H, self.W = C, H, W

  def __call__(self, x):
    with tf.variable_scope(self.name):
      word_mean = self.mean_dictionary(x)
      if "word_mean" not in self.trainable_variables:
        self.trainable_variables["word_mean"] = self.mean_dictionary.trainable_variables

      word_std  = self.std_dictionary(x)
      if "word_std" not in self.trainable_variables:
        self.trainable_variables["word_std"] = self.std_dictionary.trainable_variables
      # st()
      if self.H == 1 and self.W == 1:
        return [tf.reshape(word_mean,(-1, 1, 1, self.C)), \
                tf.reshape(word_std,(-1, 1, 1, self.C))]
      else:
        return [tf.reshape(word_mean,(-1, self.H, self.W, self.C)), \
                tf.reshape(word_std,(-1, self.H, self.W, self.C))]

if __name__ == "__main__":
  c = ConceptMapper([64,16,16])
  print(c(tf.zeros((1,21))))


