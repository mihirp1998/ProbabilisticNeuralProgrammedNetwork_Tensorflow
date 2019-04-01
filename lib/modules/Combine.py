import tensorflow as tf
from collections import OrderedDict
tf.set_random_seed(1)

class Combine(object):
  def __init__(self,name, hiddim_v, hiddim_p=None, op='PROD'):
    super(Combine, self).__init__()
    self.op = op
    self.hiddim_v = hiddim_v
    self.hiddim_p = hiddim_p
    self.name = name
    self.trainable_variables = OrderedDict()
    # if self.op == 'DEEP':
    #   self.net_vis = nn.Sequential(
    #                weight_norm(nn.Conv2d(4*hiddim_v, hiddim_v, 3, 1, 1)),
    #                nn.ELU(inplace=True),
    #                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1)),
    #                nn.ELU(inplace=True),
    #                weight_norm(nn.Conv2d(hiddim_v, 2*hiddim_v, 3, 1, 1))
    #              )
    #   self.net_pos = nn.Sequential(
    #                  weight_norm(nn.Conv2d(4*hiddim_p, hiddim_p, 1, 1)),
    #                  nn.ELU(inplace=True),
    #                  weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1)),
    #                  nn.ELU(inplace=True),
    #                  weight_norm(nn.Conv2d(hiddim_p, 2*hiddim_p, 1, 1))
    #                )

    # elif self.op == 'CAT':
    #   self.net_mean_vis = nn.Sequential(
    #                          weight_norm(nn.Conv2d(hiddim_v*2, hiddim_v, 3, 1, 1)),
    #                          nn.ELU(inplace=True),
    #                          weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
    #                       )
    #   self.net_var_vis  = nn.Sequential(
    #                          weight_norm(nn.Conv2d(hiddim_v*2, hiddim_v, 3, 1, 1)),
    #                          nn.ELU(inplace=True),
    #                          weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
    #                       )
    #   self.net_mean_pos = nn.Sequential(
    #                  weight_norm(nn.Conv2d(hiddim_p*2, hiddim_p, 1, 1)),
    #                  nn.ELU(inplace=True),
    #                  weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
    #                )
    #   self.net_var_pos = nn.Sequential(
    #                  weight_norm(nn.Conv2d(hiddim_p*2, hiddim_p, 1, 1)),
    #                  nn.ELU(inplace=True),
    #                  weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
    #                )
      # weight norm to be added
    if self.op == 'gPoE':
      self.gates_v = tf.keras.Sequential([tf.keras.layers.Conv2D(hiddim_v*4,3,1,padding="same"),
        tf.contrib.layers.instance_norm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Conv2D(hiddim_v*4,3,1,padding="same"),
        tf.contrib.layers.instance_norm()])

      self.gates_p = tf.keras.Sequential([tf.keras.layers.Conv2D(hiddim_v*4,3,1,padding="same"),
        tf.contrib.layers.instance_norm(),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Conv2D(hiddim_v*4,3,1,padding="same"),
        tf.contrib.layers.instance_norm()])

  def __call__(self, x1, x2, mode='vis'):
    # if self.op == 'PROD':
    #   return [x1[0]*x2[0], x1[1]*x2[1]]
    # elif self.op == 'PoE':
    #   # logvar = -log(exp(-logvar1) + exp(-logvar2))
    #   # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
    #   mlogvar1 = -x1[1]
    #   mlogvar2 = -x2[1]
    #   mu1      = x1[0]
    #   mu2      = x2[0]

    #   logvar   = -torch.log(torch.exp(mlogvar1) + torch.exp(mlogvar2))
    #   mu       = torch.exp(logvar)*(torch.exp(mlogvar1)*mu1 + torch.exp(mlogvar2)*mu2)
    #   return [mu, logvar]
    with tf.variable_scope(self.name):
      if self.op == 'gPoE':
        # logvar = -log(exp(-logvar1) + exp(-logvar2))
        # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
        # print(x1[0].dtype,"dtype")
        if mode == 'vis':
          gates    = tf.keras.layers.Activation('sigmoid')(self.gates_v(tf.concat([x1[0], x1[1], x2[0], x2[1]], 3)))
          x1_mu_g  = gates[:,:,:,:self.hiddim_v]
          x1_var_g = gates[:,:,:,self.hiddim_v:2*self.hiddim_v]
          x2_mu_g  = gates[:,:,:,2*self.hiddim_v:3*self.hiddim_v]
          x2_var_g = gates[:,:,:,3*self.hiddim_v:4*self.hiddim_v]
          if "gates_v" not in self.trainable_variables:
            self.trainable_variables["gates_v"] = self.gates_v.trainable_variables

        elif mode == 'pos':
          gates    = tf.keras.layers.Activation('sigmoid')(self.gates_p(tf.concat([x1[0], x1[1], x2[0], x2[1]], 3)))
          x1_mu_g  = gates[:,:,:,:self.hiddim_p]
          x1_var_g = gates[:,:,:,self.hiddim_p:2*self.hiddim_p]
          x2_mu_g  = gates[:,:,:,2*self.hiddim_p:3*self.hiddim_p]
          x2_var_g = gates[:,:,:,3*self.hiddim_p:4*self.hiddim_p]
          if "gates_p" not in self.trainable_variables:
            self.trainable_variables["gates_p"] =  self.gates_p.trainable_variables

        x1[0]    = x1_mu_g*x1[0]
        x1[1]    = tf.log(x1_var_g + 1e-5) + x1[1]
        x2[0]    = x2_mu_g*x2[0]
        x2[1]    = tf.log(x2_var_g + 1e-5) + x2[1]
        
        mlogvar1 = -x1[1]
        mlogvar2 = -x2[1]
        mu1      = x1[0]
        mu2      = x2[0]

        logvar   = -tf.log(tf.exp(mlogvar1) + tf.exp(mlogvar2))
        mu       = tf.exp(logvar)*(tf.exp(mlogvar1)*mu1 + tf.exp(mlogvar2)*mu2)
        return [mu, logvar]
    # elif self.op == 'ADD':
    #   return [x1[0] + x2[0], x1[1], x2[1]]
    # elif self.op == 'CAT':
    #   if mode == 'vis':
    #     return [self.net_mean_vis(torch.cat([x1[0], x2[0]], dim=1)), \
    #             self.net_var_vis(torch.cat([x1[1], x2[1]], dim=1))]
    #   elif mode == 'pos':
    #     return [self.net_mean_pos(torch.cat([x1[0], x2[0]], dim=1)), \
    #             self.net_var_pos(torch.cat([x1[1], x2[1]], dim=1))]
    # elif self.op == 'DEEP':
    #   if mode == 'vis':
    #     gaussian_out = self.net_vis(torch.cat([x1[0], x1[1], x2[0], x2[1]], dim=1))
    #     return [gaussian_out[:,:self.hiddim_v,:,:], gaussian_out[:,self.hiddim_v:,:,:]]
    #   elif mode == 'pos':
    #     gaussian_out = self.net_pos(torch.cat([x1[0], x1[1], x2[0], x2[1]], dim=1))
    #     return [gaussian_out[:,:self.hiddim_p,:,:], gaussian_out[:,self.hiddim_p:,:,:]]
    # else:
    #   print('Operator:', self.op)
    #   raise ValueError('Unknown operator for combine module.')

if __name__ == "__main__":
  c= Combine("combine",64,8,"gPoE")
  x1 = [tf.zeros([1,16,16,64]),tf.zeros([1,16,16,64])]
  x2 = [tf.zeros([1,16,16,64]),tf.zeros([1,16,16,64])]
  print(c(x1,x2,'vis'))
  print(len(c.trainable_variables))
