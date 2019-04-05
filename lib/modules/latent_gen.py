import tensorflow as tf

class h_mean(tf.keras.Model):
	def __init__(self, latentdim):
		super(h_mean,self).__init__()
		self.h_mean_op = tf.keras.layers.Conv2D(latentdim,3,1,padding="same")
	
	@tf.function
	def call(self,x1):
		with tf.name_scope("h_mean") as scope:
			return self.h_mean_op(x1)

class h_var(tf.keras.Model):
	def __init__(self, latentdim):
		super(h_var,self).__init__()
		self.h_var_op = tf.keras.layers.Conv2D(latentdim,3,1,padding="same")
	
	@tf.function
	def call(self,x1):
		with tf.name_scope("h_var") as scope:
			return self.h_var_op(x1)