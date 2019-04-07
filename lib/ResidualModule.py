import numpy as np
import tensorflow as tf
import keras
from collections import OrderedDict
import ipdb
st = ipdb.set_trace


class ResidualModule(tf.keras.Model):
    def __init__(self, modeltype, indim, hiddim, outdim, nlayers, nres, ifgate=False, nonlinear='elu', normalize='instance_norm'):
        super(ResidualModule, self).__init__()
        if ifgate:
          print('Using gated version.')
        if modeltype == 'encoder':
            self.model = self.encoder(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
        elif modeltype == 'decoder':
            self.model = self.decoder(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
        elif modeltype == 'plain':
            self.model = self.plain(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
        else:
            raise ('Uknown model type.')

    def encoder(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))
            layers.append(ResidualBlock('down', nonlinear, ifgate, hiddim, hiddim, normalize))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

        return tf.keras.Sequential(layers)

    def decoder(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))
            layers.append(ResidualBlock('up', nonlinear, ifgate, hiddim, hiddim, normalize))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

        return tf.keras.Sequential(layers)

    def plain(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

        return tf.keras.Sequential(layers)

       

    def call(self, x, training=False):
        val= self.model(x)
        return val



class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, resample, nonlinear, ifgate, indim, outdim, normalize):
        super(ResidualBlock, self).__init__()

        self.ifgate = ifgate
        self.indim = indim
        self.outdim = outdim
        self.resample = resample
        self.act_name = nonlinear
        if resample == 'down':
            convtype = 'sconv_d'
        elif resample == 'up':
            convtype = 'upconv'
        elif resample == None:
            convtype = 'sconv'

        self.shortflag = False
        if not (indim == outdim and resample == None):
            self.shortcut = self.conv(convtype, indim, outdim)
            self.shortflag = True

        if ifgate:
            self.conv1 = tf.keras.layers.Conv2D(outdim,3,1,padding="same")
            self.conv2 = tf.keras.layers.Conv2D(outdim,3,1,padding="same")
            self.c = tf.keras.activations.sigmoid
            self.g = tf.keras.activations.tanh
            self.conv3 = self.conv(convtype, outdim, outdim)
        
        elif normalize == 'batch_norm':
            self.resblock = tf.keras.Sequential(
                [self.conv('sconv', indim, outdim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(self.act_name),
                self.conv(convtype, outdim, outdim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(self.act_name)])
            # self.act = self.nonlinear(nonlinear)
        # elif normalize == 'batch_norm':
        #     self.resblock = nn.Sequential(
        #         self.conv('sconv', indim, outdim),
        #         nn.BatchNorm2d(outdim),
        #         self.nonlinear(nonlinear),
        #         self.conv(convtype, outdim, outdim),
        #         nn.BatchNorm2d(outdim),
        #         self.nonlinear(nonlinear)
        #     )
        # elif normalize == 'instance_norm':
        #     self.resblock = nn.Sequential(
        #         self.conv('sconv', indim, outdim),
        #         nn.InstanceNorm2d(outdim),
        #         self.nonlinear(nonlinear),
        #         self.conv(convtype, outdim, outdim),
        #         nn.InstanceNorm2d(outdim),
        #         self.nonlinear(nonlinear)


    def conv(self, name, indim, outdim, normalize=None):
        if name == 'sconv_d':
            if normalize == 'weight_norm':
              return tf.keras.layers.Conv2D(outdim,4,2,padding="same")
            else:
              return tf.keras.layers.Conv2D(outdim,4,2,padding="same")
        elif name == 'sconv':
            if normalize == 'weight_norm':
              return tf.keras.layers.Conv2D(outdim,3,1,padding="same")
            else:
              return tf.keras.layers.Conv2D(outdim,3,1,padding="same")
        elif name == 'upconv':
            if normalize == 'weight_norm':
              return tf.keras.layers.Conv2DTranspose(outdim,4,2,padding="same")
            else:
              return tf.keras.layers.Conv2DTranspose(outdim,4,2,padding="same")
        else:
            raise ("Unknown convolution type")

    def compute_output_shape(self, input_shape):
        return input_shape        

    def nonlinear(self,x,name):
        if name == 'elu':
            return tf.keras.backend.elu(x,1)
        elif name == 'relu':
            return tf.keras.activations.relu(x)

    def call(self, x):
        # print(self.ifgate,"val")
        if self.ifgate:
            conv1 = self.conv1(x)
            conv2 = self.conv2(x)
            c = self.c(conv1)
            g = self.g(conv2)
            gated = c * g
            conv3 = self.conv3(gated)
            res = self.nonlinear(conv3,self.act_name)
            if not (self.indim == self.outdim and self.resample == None):
                out = self.shortcut(x) + res
            else:
                out = x + res
        else:
            if self.shortflag:
                out = self.shortcut(x) + self.resblock(x)
            else:
                out = x + self.resblock(x)
        return out


if __name__ == "__main__":
    resMod = ResidualModule("encoder", 10, 10, 10, nlayers=3, nres=3, ifgate=False, nonlinear='elu', normalize='batch_norm')
    variables_names = [v.name for v in tf.trainable_variables()]
    print(variables_names)
    print(resMod(tf.zeros([1, 30, 30, 10]),training=True))
    variables_names = [v for v in tf.trainable_variables()]
    st()
    print(len(variables_names),"tf gen",variables_names[0],len(resMod.trainable_variables))
        # resMod = ResidualModule("plain", 10, 10, 10, nlayers=3, nres=3, ifgate=True, nonlinear='elu', normalize='instance_norm')
        # variables_names = [v.name for v in tf.trainable_variables()]
        # print(variables_names)
        # print(resMod(tf.zeros([1, 30, 30, 10])))
        # variables_names = [v.name for v in tf.trainable_variables()]
        # print(variables_names)

    # block = ResidualBlock("up", "elu", True, 10, 10, False)
    # my_seq = tf.keras.Sequential([block,block])
    # print(my_seq(tf.zeros([1, 30, 30, 10])))
