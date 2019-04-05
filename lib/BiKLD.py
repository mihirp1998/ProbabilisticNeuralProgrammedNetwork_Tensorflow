import tensorflow as tf
import numpy as np
class BiKLD(tf.keras.Model):
    def __init__(self):
        super(BiKLD, self).__init__()

    @tf.function
    def call(self, q, p):
        q_mu, q_var = q[0], tf.math.exp(q[1])
        p_mu, p_var = p[0], tf.math.exp(p[1])
        kld = q_var / p_var - 1
        kld += tf.pow(p_mu - q_mu,2) / p_var
        kld += p[1] - q[1]
        kld = tf.reduce_sum(kld) / 2
        return kld
def run():
    x1 = tf.random.normal([20,64])
    x2 = tf.random.normal([20,64])
    import time
    s = time.time()
    bkld(x1,x2)
    print(time.time() - s)

if __name__ == "__main__":
    bkld = BiKLD()
    for i in range(20):
        run()
    print(len(bkld.trainable_variables))

