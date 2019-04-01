import tensorflow as tf
import numpy as np
class BiKLD(object):
    def __init__(self):
        super(BiKLD, self).__init__()

    def __call__(self, q, p):
        q_mu, q_var = q[0], tf.exp(q[1])
        p_mu, p_var = p[0], tf.exp(p[1])
        # print(p_var,"pvar")
        kld = q_var / p_var - 1
        kld += tf.pow(p_mu - q_mu,2) / p_var
        kld += p[1] - q[1]
        kld = tf.reduce_sum(kld) / 2
        return kld

if __name__ == "__main__":
    tf.enable_eager_execution()
    bkld = BiKLD()
    np.random.seed(1)
    a,b = (np.random.randn(2,4),np.random.randn(2,4))
    print(bkld(a,b))
