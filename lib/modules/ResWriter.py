import _init_paths
from lib.ResidualModule import ResidualModule
import tensorflow as tf
# tf.set_random_seed(1)

from collections import OrderedDict
import itertools

class Writer(tf.keras.Model):
    def __init__(self, indim, hiddim, outdim, ds_times, normalize, nlayers=4):
        super(Writer, self).__init__()
        self.ds_times = ds_times
        if normalize == 'gate':
          ifgate = True
        else:
          ifgate = False

        self.decoder = ResidualModule(modeltype='decoder', indim=indim, hiddim=hiddim, outdim=hiddim,
                                      nres=self.ds_times, nlayers=nlayers, ifgate=ifgate, normalize=normalize)

        self.out_conv = tf.keras.layers.Conv2D(outdim, 3, 1, "same")

    @tf.function
    def call(self, x):
      out = self.decoder(x)
      out = self.out_conv(out)
      return out

def run():
  x1 = tf.random.normal([1, 30, 30, 3])
  import time
  s = time.time()
  write(x1)
  print(time.time() - s)
if __name__ == "__main__":
  write = Writer(3, 3, 3, nlayers=3, ds_times=3, normalize='gate')
  # print(write(tf.zeros([1, 30, 30, 3])))
  for i in range(20):
    run()
  print(len(write.trainable_variables))
