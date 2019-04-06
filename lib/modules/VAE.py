import _init_paths
from lib.BiKLD import BiKLD
from lib.reparameterize import reparameterize
import tensorflow as tf
from collections import OrderedDict
# tf.set_random_seed(1)


class VAE(tf.keras.Model):
    def __init__(self, name_scope, indim, latentdim, half=False):
        super(VAE, self).__init__()
        self.half = half
        self.name_scope = name_scope
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

    @tf.function
    def call(self, x=None, prior=None):
        with tf.name_scope(self.name_scope) as scope:
            x_c = tf.convert_to_tensor(x)
            prior = [tf.reshape(prior[0],(1, -1)), tf.reshape(prior[1],(1, -1)) ]

            if self.half is False:
                encoding = self.encoder(x_c)
                mean, logvar = self.mean(encoding), self.logvar(encoding)
                kld = self.bikld([mean, logvar], prior)
                z = self.sampler(mean, logvar)
            else:
                z = self.sampler(prior[0], prior[1])
                kld = 0
            decoding = self.decoder(z)
        return decoding, kld

    @tf.function
    def generate(self, prior):
        prior = [tf.reshape(prior[0],(1, -1)), tf.reshape(prior[1],[1, -1])]
        z = self.sampler(*prior)
        decoding = self.decoder(z)
        return decoding


def run():
    mean = tf.zeros((10, 14))
    var  = tf.zeros((10, 14))
    data = tf.zeros((10, 16))
    import time
    s = time.time()
    model(data, [mean, var])
    print(time.time() - s)

if __name__ =="__main__":
    model = VAE(16, 14,1)
    for i in range(20):
        run()
    print(len(model.trainable_variables))
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
