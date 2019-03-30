import _init_paths
from lib.ResidualModule import ResidualModule
import tensorflow as tf
tf.set_random_seed(1)

from collections import OrderedDict
import itertools

class Writer(object):
    def __init__(self,name, indim, hiddim, outdim, ds_times, normalize, nlayers=4):
        super(Writer, self).__init__()
        self.name = name
        self.ds_times = ds_times
        self.trainable_variables = OrderedDict()
        if normalize == 'gate':
          ifgate = True
        else:
          ifgate = False

        self.decoder = ResidualModule(modeltype='decoder', indim=indim, hiddim=hiddim, outdim=hiddim,
                                      nres=self.ds_times, nlayers=nlayers, ifgate=ifgate, normalize=normalize)

        self.out_conv = tf.keras.layers.Conv2D(outdim, 3, 1, "same")

    def __call__(self, x):
      with tf.variable_scope(self.name):
        out = self.decoder(x)
        if "decoder" not in self.trainable_variables:
          self.trainable_variables["decoder"] =  list(itertools.chain.from_iterable(list(self.decoder.trainable_variables.values())))
        # print(out,"outnot")
        out = self.out_conv(out)
        if "out_conv" not in self.trainable_variables:
          self.trainable_variables["out_conv"] = self.out_conv.trainable_variables
        # self.trainable_variables = self.trainable_variables + self.out_conv.trainable_variables
        return out


if __name__ == "__main__":
  write = Writer("plain", 3, 3, 3, nlayers=3, ds_times=3, normalize='gate')
  print(write(tf.zeros([1, 30, 30, 3])))
  variables_names = [v for v in tf.trainable_variables()]
  print(len(variables_names),"tf gen",variables_names[0],len(write.trainable_variables))
