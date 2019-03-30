import tensorflow as tf

def lr_mult(alpha):
  @tf.custom_gradient
  def _lr_mult(x):
    print(x.name)
    def grad(dy):
      return dy * alpha * tf.ones_like(x)
    return x, grad
  return _lr_mult

x0 = tf.Variable(1.,name="x")
x1 = tf.Variable(1.,name="x1")
# loss = tf.square(x0) + tf.square(lr_mult(0.1)(x1))
loss = tf.square(x0) +tf.square(x1)
loss = lr_mult(0.1)(loss)
step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

for _ in range(5):
  sess.run([step])
  print(sess.run([x0, x1, loss]))
