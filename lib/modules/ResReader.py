import _init_paths
from lib.ResidualModule import ResidualModule
import tensorflow as tf
tf.set_random_seed(1)
from collections import OrderedDict
import itertools

class Reader(object):
    def __init__(self,name,indim, hiddim, outdim, ds_times, normalize, nlayers=4):
        super(Reader, self).__init__()

        self.ds_times = ds_times
        self.name = name
        # TODO normalization fixed to gates
        self.trainable_variables = OrderedDict()


        # normalize= "gate"
        if normalize == 'gate':
          ifgate = True
        else:
          ifgate = False

        self.encoder = ResidualModule(modeltype='encoder', indim=indim, hiddim=hiddim, outdim=outdim,
                                      nres=self.ds_times, nlayers=nlayers, ifgate=ifgate, normalize=normalize)

    def __call__(self, x):
      with tf.variable_scope(self.name):
        out = self.encoder(x)
        if "encoder" not in self.trainable_variables:
            self.trainable_variables["encoder"] = list(itertools.chain.from_iterable(list(self.encoder.trainable_variables.values())))
        return out

if __name__ == "__main__":
    read = Reader("plain", 10, 10, 10, nlayers=3, ds_times=3, normalize='gate')
    print(read(tf.zeros([1, 30, 30, 10])))
    variables_names = [v for v in tf.trainable_variables()]
    print(len(variables_names),"tf gen",variables_names[0],len(read.trainable_variables))
