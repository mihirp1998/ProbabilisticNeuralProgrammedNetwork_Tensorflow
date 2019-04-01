import numpy as np
import tensorflow as tf
import keras
from collections import OrderedDict
import ipdb
st = ipdb.set_trace

tf.set_random_seed(1)

# @add_arg_scope
# def conv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
#     ''' convolutional layer '''
#     name = get_name('conv2d', counters)
#     with tf.variable_scope(name):
#         if init:
#             # data based initialization of parameters
#             V = tf.get_variable('V', filter_size+[int(x.get_shape()[-1]),num_filters], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
#             V_norm = tf.nn.l2_normalize(V.initialized_value(), [0,1,2])
#             x_init = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
#             m_init, v_init = tf.nn.moments(x_init, [0,1,2])
#             scale_init = init_scale/tf.sqrt(v_init + 1e-8)
#             g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
#             b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
#             x_init = tf.reshape(scale_init,[1,1,1,num_filters])*(x_init-tf.reshape(m_init,[1,1,1,num_filters]))
#             if nonlinearity is not None:
#                 x_init = nonlinearity(x_init)
#             return x_init

#         else:
#             V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
#             tf.assert_variables_initialized([V,g,b])

#             # use weight normalization (Salimans & Kingma, 2016)
#             W = tf.reshape(g,[1,1,1,num_filters])*tf.nn.l2_normalize(V,[0,1,2])

#             # calculate convolutional layer output
#             x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1]+stride+[1], pad), b)

#             # apply nonlinearity
#             if nonlinearity is not None:
#                 x = nonlinearity(x)
#             return x

# class ResidualModule(nn.Module):
#     def __init__(self, modeltype, indim, hiddim, outdim, nlayers, nres, ifgate=False, nonlinear='elu', normalize='instance_norm'):
#         super(ResidualModule, self).__init__()
#         if ifgate:
#           print('Using gated version.')
#         if modeltype == 'encoder':
#             self.model = self.encoder(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
#         elif modeltype == 'decoder':
#             self.model = self.decoder(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
#         elif modeltype == 'plain':
#             self.model = self.plain(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
#         else:
#             raise ('Uknown model type.')

#     def encoder(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
#         layers = []
#         layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

#         for i in range(0, nres):
#             for j in range(0, nlayers):
#                 layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))
#             layers.append(ResidualBlock('down', nonlinear, ifgate, hiddim, hiddim, normalize))

#         layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

#         return nn.Sequential(*layers)

#     def decoder(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
#         layers = []
#         layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

#         for i in range(0, nres):
#             for j in range(0, nlayers):
#                 layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))
#             layers.append(ResidualBlock('up', nonlinear, ifgate, hiddim, hiddim, normalize))

#         layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

#         return nn.Sequential(*layers)

#     def plain(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
#         layers = []
#         layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

#         for i in range(0, nres):
#             for j in range(0, nlayers):
#                 layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))

#         layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

class ResidualModule(object):
    def __init__(self, modeltype, indim, hiddim, outdim, nlayers, nres, ifgate=False, nonlinear='elu', normalize='instance_norm'):
        super(ResidualModule, self).__init__()
        self.trainable_variables = OrderedDict()
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

       

    def __call__(self, x, training=False):
        val= self.model(x)
        if "model" not in self.trainable_variables:
            self.trainable_variables["model"] = self.model.trainable_variables
        # print(len(self.model.trainable_variables),"inside keras",self.model.trainable_variables[0])
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
        #     )
        # elif normalize == 'no_norm':
        #     self.resblock = nn.Sequential(
        #         self.conv('sconv', indim, outdim),
        #         self.nonlinear(nonlinear),
        #         self.conv(convtype, outdim, outdim),
        #         self.nonlinear(nonlinear)
        #     )
        # elif normalize == 'weight_norm':
        #     self.resblock = nn.Sequential(
        #         self.conv('sconv', indim, outdim, 'weight_norm'),
        #         self.nonlinear(nonlinear),
        #         self.conv(convtype, outdim, outdim, 'weight_norm'),
        #         self.nonlinear(nonlinear)
        #     )

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
            # print("hello")
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
            # print("hello")
            if self.shortflag:
                st()
                out = self.shortcut(x) + self.resblock(x)
            else:
                st()
                out = x + self.resblock(x)
        # else:
        #     if self.shortflag:
        #         out = self.shortcut(x) + self.resblock(x)
        #     else:
        #         out = x + self.resblock(x)
        # print(out,"out")
        return out



# class ResidualBlock(nn.Module):
#     def __init__(self, resample, nonlinear, ifgate, indim, outdim, normalize):
#         super(ResidualBlock, self).__init__()

#         self.ifgate = ifgate
#         self.indim = indim
#         self.outdim = outdim
#         self.resample = resample

#         if resample == 'down':
#             convtype = 'sconv_d'
#         elif resample == 'up':
#             convtype = 'upconv'
#         elif resample == None:
#             convtype = 'sconv'

#         self.shortflag = False
#         if not (indim == outdim and resample == None):
#             self.shortcut = self.conv(convtype, indim, outdim)
#             self.shortflag = True

#         if ifgate:
#             self.conv1 = nn.Conv2d(indim, outdim, 3, 1, 1)
#             self.conv2 = nn.Conv2d(indim, outdim, 3, 1, 1)
#             self.c = nn.Sigmoid()
#             self.g = nn.Tanh()
#             self.conv3 = self.conv(convtype, outdim, outdim)
#             self.act = self.nonlinear(nonlinear)
#         elif normalize == 'batch_norm':
#             self.resblock = nn.Sequential(
#                 self.conv('sconv', indim, outdim),
#                 nn.BatchNorm2d(outdim),
#                 self.nonlinear(nonlinear),
#                 self.conv(convtype, outdim, outdim),
#                 nn.BatchNorm2d(outdim),
#                 self.nonlinear(nonlinear)
#             )
#         elif normalize == 'instance_norm':
#             self.resblock = nn.Sequential(
#                 self.conv('sconv', indim, outdim),
#                 nn.InstanceNorm2d(outdim),
#                 self.nonlinear(nonlinear),
#                 self.conv(convtype, outdim, outdim),
#                 nn.InstanceNorm2d(outdim),
#                 self.nonlinear(nonlinear)
#             )
#         elif normalize == 'no_norm':
#             self.resblock = nn.Sequential(
#                 self.conv('sconv', indim, outdim),
#                 self.nonlinear(nonlinear),
#                 self.conv(convtype, outdim, outdim),
#                 self.nonlinear(nonlinear)
#             )
#         elif normalize == 'weight_norm':
#             self.resblock = nn.Sequential(
#                 self.conv('sconv', indim, outdim, 'weight_norm'),
#                 self.nonlinear(nonlinear),
#                 self.conv(convtype, outdim, outdim, 'weight_norm'),
#                 self.nonlinear(nonlinear)
#             )

#     def conv(self, name, indim, outdim, normalize=None):
#         if name == 'sconv_d':
#             if normalize == 'weight_norm':
#               return weight_norm(nn.Conv2d(indim, outdim, 4, 2, 1))
#             else:
#               return nn.Conv2d(indim, outdim, 4, 2, 1)
#         elif name == 'sconv':
#             if normalize == 'weight_norm':
#               return weight_norm(nn.Conv2d(indim, outdim, 3, 1, 1))
#             else:
#               return nn.Conv2d(indim, outdim, 3, 1, 1)
#         elif name == 'upconv':
#             if normalize == 'weight_norm':
#               return weight_norm(nn.ConvTranspose2d(indim, outdim, 4, 2, 1))
#             else:
#               return nn.ConvTranspose2d(indim, outdim, 4, 2, 1)
#         else:
#             raise ("Unknown convolution type")

#     def nonlinear(self, name):
#         if name == 'elu':
#             return nn.ELU(1, True)
#         elif name == 'relu':
#             return nn.ReLU(True)

#     def forward(self, x):
#         if self.ifgate:
#             conv1 = self.conv1(x)
#             conv2 = self.conv2(x)
#             c = self.c(conv1)
#             g = self.g(conv2)
#             gated = c * g
#             conv3 = self.conv3(gated)
#             res = self.act(conv3)
#             if not (self.indim == self.outdim and self.resample == None):
#                 out = self.shortcut(x) + res
#             else:
#                 out = x + res
#         else:
#             if self.shortflag:
#                 out = self.shortcut(x) + self.resblock(x)
#             else:
#                 out = x + self.resblock(x)

#         return out


if __name__ == "__main__":
    with tf.variable_scope("bar",reuse=False): 
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
