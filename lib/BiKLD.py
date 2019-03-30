import tensorflow as tf

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
    bkld = BiKLD()
    print(bkld(tf.zeros((2,4)),tf.zeros((2,4))))
