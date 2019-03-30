import _init_paths
from lib.BiKLD import BiKLD
from lib.reparameterize import reparameterize
import tensorflow as tf
from collections import OrderedDict
tf.set_random_seed(1)


class VAE(object):
    def __init__(self,name, indim, latentdim, half=False):
        super(VAE, self).__init__()
        self.name = name
        self.half = half
        self.trainable_variables= OrderedDict()
        if self.half is False:
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(latentdim * 2),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.Dense(latentdim * 2),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.Dense(latentdim * 2)],name="encoder"
            )
            self.mean = tf.keras.layers.Dense(latentdim,name="mean")
            self.logvar = tf.keras.layers.Dense(latentdim,name="logvar")
            self.bikld = BiKLD()

        dec_out = indim
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(latentdim * 2),
            tf.keras.layers.Activation('elu'),
            tf.keras.layers.Dense(latentdim * 2),
            tf.keras.layers.Activation('elu'),
           tf.keras.layers.Dense(dec_out)],name="decoder"
        )

        self.sampler = reparameterize()

    def __call__(self, x=None, prior=None):
        with tf.variable_scope(self.name):
            x = tf.convert_to_tensor(x)
            prior = [tf.reshape(prior[0],(1, -1)), tf.reshape(prior[1],(1, -1)) ]

            if self.half is False:
                encoding = self.encoder(x)
                if "encoder" not in self.trainable_variables:
                    self.trainable_variables["encoder"] =  self.encoder.trainable_variables

                mean, logvar = self.mean(encoding), self.logvar(encoding)

                if "mean" not in self.trainable_variables:
                    self.trainable_variables["mean"] = self.mean.trainable_variables

                if "logvar" not in self.trainable_variables:
                    self.trainable_variables["logvar"] = self.logvar.trainable_variables

                kld = self.bikld([mean, logvar], prior)
                z = self.sampler(mean, logvar)
            else:
                z = self.sampler(prior[0], prior[1])
                kld = 0

            decoding = self.decoder(z)
            if "decoder" not in self.trainable_variables:
                self.trainable_variables["decoder"] = self.decoder.trainable_variables

        return decoding, kld

    def generate(self, prior):
        prior = [prior[0].view(1, -1), prior[1].view(1, -1)]
        z = self.sampler(*prior)
        decoding = self.decoder(z)

        return decoding


if __name__ =="__main__":
    model = VAE("vae",6, 4,1)
    mean = tf.zeros((1, 4))
    var  = tf.zeros((1, 4))
    data = tf.zeros((1, 6))
    out, kld = model(data, [mean, var])
    print(out,kld)
    print(len(model.trainable_variables),len(tf.trainable_variables()))
'''
#Test case 0
model = VAE(6, 4).cuda()
mean = Variable(torch.zeros(16, 4)).cuda()
var  = Variable(torch.zeros(16, 4)).cuda()
data = Variable(torch.zeros(16, 6)).cuda()
out, kld = model(data, [mean, var])

#Test case 1
model = VAE(6, 4, 10).cuda()
mean = Variable(torch.zeros(16, 4)).cuda()
var  = Variable(torch.zeros(16, 4)).cuda()
data = Variable(torch.zeros(16, 6)).cuda()
condition = Variable(torch.zeros(16, 10)).cuda()
out, kld = model(data, [mean, var], condition)
'''
