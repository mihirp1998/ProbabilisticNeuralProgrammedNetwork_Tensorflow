import _init_paths
from lib.ResidualModule import ResidualModule
import tensorflow as tf
# tf.set_random_seed(1)
from collections import OrderedDict
import itertools

class Reader(tf.keras.Model):
    def __init__(self,indim, hiddim, outdim, ds_times, normalize, nlayers=4):
        super(Reader, self).__init__()

        self.ds_times = ds_times
        # TODO normalization fixed to gates
        if normalize == 'gate':
          ifgate = True
        else:
          ifgate = False

        self.encoder = ResidualModule(modeltype='encoder', indim=indim, hiddim=hiddim, outdim=outdim,
                                      nres=self.ds_times, nlayers=nlayers, ifgate=ifgate, normalize=normalize)

    @tf.function
    def call(self, x):
        with tf.name_scope("reader") as scope:
            out = self.encoder(x)
        return out
def run():
  x1 = tf.random.normal([1, 30, 30, 10])
  import time
  s = time.time()
  read(x1)
  print(time.time() - s)
if __name__ == "__main__":
    read = Reader(10, 10, 10, nlayers=3, ds_times=3, normalize='gate')
    # read(tf.zeros([1, 30, 30, 10]))
    for i in range(20):
        run()
    print(len(read.trainable_variables))


