import tensorflow as tf
tf.set_random_seed(1)
import keras
from collections import OrderedDict
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# one more advanced plan: predict an attention map a, then
# render the vw by a*x*y + (1-a)*y
class Describe(object):
    def __init__(self,name, hiddim_v, hiddim_p=None, op='CAT'):
        super(Describe, self).__init__()
        self.op = op
        self.hiddim_v = hiddim_v
        self.hiddim_p = hiddim_p
        self.name = name
        self.trainable_variables=OrderedDict()
        # if op == 'CAT' or op == 'CAT_PoE':
        #     self.net1_mean_vis = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
        #     )

        #     self.net1_var_vis = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
        #     )

        #     self.net1_mean_pos = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
        #     )

        #     self.net1_var_pos = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
        #     )
        # elif op == 'DEEP':
        #     self.net_vis = nn.Sequential(
        #         weight_norm(nn.Conv2d(4 * hiddim_v, hiddim_v, 3, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_v, 2 * hiddim_v, 3, 1, 1))
        #     )

        #     self.net_pos = nn.Sequential(
        #         weight_norm(nn.Conv2d(4 * hiddim_p, hiddim_p, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_p, 2 * hiddim_p, 1, 1))
        #     )

        # elif op == 'CAT_PROD':
        #     self.net1_mean_vis = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
        #     )

        #     self.net1_var_vis = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
        #     )

        #     self.net2_mean_vis = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
        #     )

        #     self.net2_var_vis = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
        #     )

        #     self.net1_mean_pos = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
        #     )

        #     self.net1_var_pos = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
        #     )

        #     self.net2_mean_pos = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
        #     )

        #     self.net2_var_pos = nn.Sequential(
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1)),
        #         nn.ELU(inplace=True),
        #         weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
        #     )
        if op == 'CAT_gPoE':
            self.net1_mean_vis = keras.Sequential(
                [keras.layers.Conv2D(hiddim_v, 3, 1,padding="same"),
                InstanceNormalization(),
                keras.layers.Activation('elu'),
                keras.layers.Conv2D(hiddim_v, 3, 1,padding="same"),
                InstanceNormalization()]
            )

            self.net1_var_vis = keras.Sequential(
                [keras.layers.Conv2D(hiddim_v, 3, 1, "same"),
                InstanceNormalization(),
                keras.layers.Activation('elu'),
                keras.layers.Conv2D(hiddim_v, 3, 1, "same"),
                InstanceNormalization()]
            )

            self.net1_mean_pos = keras.Sequential(
                [keras.layers.Conv2D(hiddim_p, 1,1, "same"),
                InstanceNormalization(),
                keras.layers.Activation('elu'),
                keras.layers.Conv2D(hiddim_p, 1,1, "same"),
                InstanceNormalization()]
            )

            self.net1_var_pos = keras.Sequential(
                [keras.layers.Conv2D(hiddim_p, 1,1, "same"),
                InstanceNormalization(),
                keras.layers.Activation('elu'),
                keras.layers.Conv2D(hiddim_p, 1,1, "same"),
                InstanceNormalization()]
            )
            self.gates_v = keras.Sequential(
               [keras.layers.Conv2D(hiddim_v * 4, 3, 1, "same"),
                InstanceNormalization(),
                keras.layers.Activation('elu'),
                keras.layers.Conv2D(hiddim_v * 4, 3, 1, "same"),
                InstanceNormalization()]
            )
            self.gates_p = keras.Sequential(
                [keras.layers.Conv2D(hiddim_p * 4, 3, 1, "same"),
                InstanceNormalization(),
                keras.layers.Activation('elu'),
                keras.layers.Conv2D(hiddim_p * 4, 3, 1, "same"),
                InstanceNormalization()]
            )
    # Q: in gpoe why do we do different actions on mean and variance

    def __call__(self, x, y, mode, lognormal=False):  # -> x describe y
        with tf.variable_scope(self.name):
            if mode == 'vis':
                # if self.op == 'CAT_PROD':
                #     x_mean = self.net1_mean_vis(torch.cat([x[0], y[0]], dim=1))
                #     x_var = self.net1_var_vis(torch.cat([x[1], y[1]], dim=1))

                #     if lognormal == True:
                #         x_mean = torch.exp(x_mean)

                #     y_mean = self.net2_mean_vis(x_mean * y[0])
                #     y_var = self.net2_var_vis(x_var * y[1])
                # elif self.op == 'CAT_PoE':
                #     # logvar = -log(exp(-logvar1) + exp(-logvar2))
                #     # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
                #     x_mean = self.net1_mean_vis(torch.cat([x[0], y[0]], dim=1))
                #     x_var = self.net1_var_vis(torch.cat([x[1], y[1]], dim=1))
                #     mlogvar1 = -x_var
                #     mlogvar2 = -y[1]
                #     mu1 = x_mean
                #     mu2 = y[0]

                #     y_var = -torch.log(torch.exp(mlogvar1) + torch.exp(mlogvar2))
                #     y_mean = torch.exp(y_var) * (torch.exp(mlogvar1) * mu1 + torch.exp(mlogvar2) * mu2)
                if self.op == 'CAT_gPoE':
                    # logvar = -log(exp(-logvar1) + exp(-logvar2))
                    # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
                    x_mean = self.net1_mean_vis(tf.concat([x[0], y[0]], 3))
                    if "net1_mean_vis" not in self.trainable_variables:
                        self.trainable_variables["net1_mean_vis"] = self.net1_mean_vis.trainable_variables

                    x_var = self.net1_var_vis(tf.concat([x[1], y[1]], 3))

                    if "net1_var_vis" not in self.trainable_variables:
                        self.trainable_variables["net1_var_vis"] = self.net1_var_vis.trainable_variables

                    # gates
                    gates = tf.keras.layers.Activation('sigmoid')(self.gates_v(tf.concat([x_mean, x_var, y[0], y[1]], 3)))
                    if "gates_v" not in self.trainable_variables:
                        self.trainable_variables["gates_v"] = self.gates_v.trainable_variables


                    x1_mu_g  = gates[:,:,:,:self.hiddim_v]
                    x1_var_g = gates[:,:,:,self.hiddim_v:2*self.hiddim_v]
                    x2_mu_g  = gates[:,:,:,2*self.hiddim_v:3*self.hiddim_v]
                    x2_var_g = gates[:,:,:,3*self.hiddim_v:4*self.hiddim_v]

                    x_mean = x1_mu_g * x_mean
                    x_var = tf.log(x1_var_g + 1e-5) + x_var
                    y[0] = x2_mu_g * y[0]
                    y[1] = tf.log(x2_var_g + 1e-5) + y[1]

                    mlogvar1 = -x_var
                    mlogvar2 = -y[1]
                    mu1 = x_mean
                    mu2 = y[0]

                    y_var = -tf.log(tf.exp(mlogvar1) + tf.exp(mlogvar2))
                    y_mean = tf.exp(y_var) * (tf.exp(mlogvar1) * mu1 + tf.exp(mlogvar2) * mu2)
                # elif self.op == 'CAT':
                #     y_mean = self.net1_mean_vis(torch.cat([x[0], y[0]], dim=1))
                #     y_var = self.net1_var_vis(torch.cat([x[1], y[1]], dim=1))
                # elif self.op == 'PROD':
                #     y_mean = x[0] * y[0]
                #     y_var = x[1] * y[1]
                # elif self.op == 'DEEP':
                #     gaussian_out = self.net_vis(torch.cat([x[0], x[1], y[0], y[1]], dim=1))
                #     y_mean = gaussian_out[:, :self.hiddim_v, :, :]
                #     y_var = gaussian_out[:, self.hiddim_v:, :, :]
                # else:
                #     raise ValueError('invalid operator name {} for Describe module'.format(self.op))

            elif mode == 'pos':
                # if self.op == 'CAT_PROD':
                #     x_mean = self.net1_mean_pos(torch.cat([x[0], y[0]], dim=1))
                #     x_var = self.net1_var_pos(torch.cat([x[1], y[1]], dim=1))

                #     y_mean = self.net2_mean_pos(x_mean * y[0])
                #     y_var = self.net2_var_pos(x_var * y[1])
                # elif self.op == 'CAT_PoE':
                #     # logvar = -log(exp(-logvar1) + exp(-logvar2))
                #     # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
                #     x_mean = self.net1_mean_pos(torch.cat([x[0], y[0]], dim=1))
                #     x_var = self.net1_var_pos(torch.cat([x[1], y[1]], dim=1))

                #     mlogvar1 = -x_var
                #     mlogvar2 = -y[1]
                #     mu1 = x_mean
                #     mu2 = y[0]

                #     y_var = -torch.log(torch.exp(mlogvar1) + torch.exp(mlogvar2))
                #     y_mean = torch.exp(y_var) * (torch.exp(mlogvar1) * mu1 + torch.exp(mlogvar2) * mu2)
                if self.op == 'CAT_gPoE':
                    # logvar = -log(exp(-logvar1) + exp(-logvar2))
                    # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
                    x_mean = self.net1_mean_pos(tf.concat([x[0], y[0]], 3))
                    
                    if "net1_mean_pos" not in self.trainable_variables:
                        self.trainable_variables["net1_mean_pos"] = self.net1_mean_pos.trainable_variables

                    x_var = self.net1_var_pos(tf.concat([x[1], y[1]], 3))

                    if "net1_var_pos" not in self.trainable_variables:
                        self.trainable_variables["net1_var_pos"] =  self.net1_var_pos.trainable_variables

                    # gates
                    gates = tf.keras.layers.Activation('sigmoid')(self.gates_p(tf.concat([x_mean, x_var, y[0], y[1]], 3)))
                    
                    if "gates_p" not in self.trainable_variables:
                        self.trainable_variables["gates_p"] = self.gates_p.trainable_variables

                    # x1_mu_g = gates[:, :self.hiddim_p, :, :]
                    # x1_var_g = gates[:, self.hiddim_p:2 * self.hiddim_p, :, :]
                    # x2_mu_g = gates[:, 2 * self.hiddim_p:3 * self.hiddim_p, :, :]
                    # x2_var_g = gates[:, 3 * self.hiddim_p:4 * self.hiddim_p, :, :]
                    x1_mu_g  = gates[:,:,:,:self.hiddim_p]
                    x1_var_g = gates[:,:,:,self.hiddim_p:2*self.hiddim_p]
                    x2_mu_g  = gates[:,:,:,2*self.hiddim_p:3*self.hiddim_p]
                    x2_var_g = gates[:,:,:,3*self.hiddim_p:4*self.hiddim_p]

                    x_mean = x1_mu_g * x_mean
                    x_var = tf.log(x1_var_g + 1e-5) + x_var
                    y[0] = x2_mu_g * y[0]
                    y[1] = tf.log(x2_var_g + 1e-5) + y[1]

                    mlogvar1 = -x_var
                    mlogvar2 = -y[1]
                    mu1 = x_mean
                    mu2 = y[0]

                    y_var = -tf.log(tf.exp(mlogvar1) + tf.exp(mlogvar2))
                    y_mean = tf.exp(y_var) * (tf.exp(mlogvar1) * mu1 + tf.exp(mlogvar2) * mu2)
                # elif self.op == 'CAT':
                #     y_mean = self.net1_mean_pos(torch.cat([x[0], y[0]], dim=1))
                #     y_var = self.net1_var_pos(torch.cat([x[1], y[1]], dim=1))
                # elif self.op == 'PROD':
                #     y_mean = x[0] * y[0]
                #     y_var = x[1] * y[1]
                # elif self.op == 'DEEP':
                #     gaussian_out = self.net_pos(torch.cat([x[0], x[1], y[0], y[1]], dim=1))
                #     y_mean = gaussian_out[:, :self.hiddim_p, :, :]
                #     y_var = gaussian_out[:, self.hiddim_p:, :, :]
                # else:
                #     raise ValueError('invalid operator name {} for Describe module'.format(self.op))

            else:
                raise ValueError('invalid mode {}'.format(mode))

        return [y_mean, y_var]


if __name__ == "__main__":
  c= Describe("describe",hiddim_v=64,hiddim_p=64,op="CAT_gPoE")
  x1 = [tf.zeros([1,16,16,64]),tf.zeros([1,16,16,64])]
  x2 = [tf.zeros([1,16,16,64]),tf.zeros([1,16,16,64])]
  print(c(x1,x2,'vis'))
  print(c(x1,x2,'pos'))
  print(len(c.trainable_variables))

