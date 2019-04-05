import tensorflow as tf
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from tensorflow.keras import backend as K
def _transform(X, affine_transformation, output_size):
	batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
	transformations = K.reshape(affine_transformation,
								shape=(batch_size, 2, 3))
	regular_grids = _make_regular_grids(batch_size,
											 *output_size)
	sampled_grids = K.batch_dot(transformations, regular_grids)
	interpolated_image = _interpolate(X, sampled_grids,
										   output_size)
	new_shape = (batch_size, output_size[0], 
				 output_size[1], num_channels)
	interpolated_image = K.reshape(interpolated_image, new_shape)
	
	return interpolated_image

def _make_regular_grids(batch_size, height, width):
	# making a single regular grid
	x_linspace = np.linspace(-1., 1., width)
	y_linspace = np.linspace(-1., 1., height)
	x_coordinates, y_coordinates = np.meshgrid(x_linspace,y_linspace)
	x_coordinates = K.flatten(x_coordinates)
	y_coordinates = K.flatten(y_coordinates)
	ones = K.ones_like(x_coordinates)
	grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)
	# repeating grids for each batch
	grid = K.flatten(grid)
	grids = tf.tile(grid, K.stack([batch_size]))
		
	return K.reshape(grids, (batch_size, 3, height * width))

def  _interpolate(image, sampled_grids, output_size):
	batch_size = K.shape(image)[0]
	height = K.shape(image)[1]
	width = K.shape(image)[2]
	num_channels = K.shape(image)[3]
	x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
	y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')
	x = .5 * (x + 1.0) * K.cast(height-1, dtype='float32')
	y = .5 * (y + 1.0) * K.cast(width-1, dtype='float32')
	x0 = K.cast(x, 'int32')
	x1 = x0 + 1
	y0 = K.cast(y, 'int32')
	y1 = y0 + 1
	max_x = int(K.int_shape(image)[1] - 1)
	max_y = int(K.int_shape(image)[2] - 1)
	x0 = K.clip(x0, 0, max_x)
	x1 = K.clip(x1, 0, max_x)
	y0 = K.clip(y0, 0, max_y)
	y1 = K.clip(y1, 0, max_y)
	pixels_batch = K.arange(0, batch_size) * (height * width)
	pixels_batch = K.expand_dims(pixels_batch, axis=-1)
	flat_output_size = output_size[0] * output_size[1]
	base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
	base = K.flatten(base)
	base_y0 = y0 * width
	base_y0 = base + base_y0
	base_y1 = y1 * width
	base_y1 = base_y1 + base
	indices_a = base_y0 + x0
	indices_b = base_y1 + x0
	print(x1.dtype,x.dtype,base_y0.dtype,base_y1.dtype)
	indices_c = base_y0 + x1
	indices_d = base_y1 + x1
	flat_image = K.reshape(image, shape=(-1, num_channels))
	flat_image = K.cast(flat_image, dtype='float32')
	pixel_values_a = K.gather(flat_image, indices_a)
	pixel_values_b = K.gather(flat_image, indices_b)
	pixel_values_c = K.gather(flat_image, indices_c)
	pixel_values_d = K.gather(flat_image, indices_d)
	x0 = K.cast(x0, 'float32')
	x1 = K.cast(x1, 'float32')
	y0 = K.cast(y0, 'float32')
	y1 = K.cast(y1, 'float32')
	area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
	area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
	area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
	area_d = K.expand_dims(((x - x0) * (y - y0)), 1)
	values_a = area_a * pixel_values_a
	values_b = area_b * pixel_values_b
	values_c = area_c * pixel_values_c
	values_d = area_d * pixel_values_d
	
	return values_a + values_b + values_c + values_d    

def np_to_torch(img):
	img = np.swapaxes(img, 0, 1) #w, h, 9
	img = np.swapaxes(img, 0, 2) #9, h, w
	return torch.from_numpy(img).float()

def torch_to_np(img):
	img = np.array(img)
	img = np.swapaxes(img, 1, 2) #w, h, 9
	img = np.swapaxes(img, 2, 3) #9, h, w
	return img
	# return torch.from_numpy(img).float()    

def stn(x,theta):
	# xs = .localization(x)
	# xs = xs.view(-1, 10 * 3 * 3)
	# theta = .fc_loc(xs)
	# theta = theta.view(-1, 2, 3)
	
	theta =torch.from_numpy(theta)
	size = torch.Size([x.size(0), x.size(1),80, 80])

	# print(theta.shape)
	grid = F.affine_grid(theta, size)
	x = F.grid_sample(x, grid)

	return x


def affine_grid_generator(height, width, theta):
	"""
	This function returns a sampling grid, which when
	used with the bilinear sampler on the input feature
	map, will create an output feature map that is an
	affine transformation [1] of the input feature map.
	Input
	-----
	- height: desired height of grid/output. Used
	  to downsample or upsample.
	- width: desired width of grid/output. Used
	  to downsample or upsample.
	- theta: affine transform matrices of shape (num_batch, 2, 3).
	  For each image in the batch, we have 6 theta parameters of
	  the form (2x3) that define the affine transformation T.
	Returns
	-------
	- normalized grid (-1, 1) of shape (num_batch, 2, H, W).
	  The 2nd dimension has 2 components: (x, y) which are the
	  sampling points of the original image for each point in the
	  target image.
	Note
	----
	[1]: the affine transformation allows cropping, translation,
		 and isotropic scaling.
	"""
	num_batch = tf.shape(theta)[0]

	# create normalized 2D grid
	x = tf.linspace(-1.0, 1.0, width)
	y = tf.linspace(-1.0, 1.0, height)
	x_t, y_t = tf.meshgrid(x, y)

	# flatten
	x_t_flat = tf.reshape(x_t, [-1])
	y_t_flat = tf.reshape(y_t, [-1])

	# reshape to [x_t, y_t , 1] - (homogeneous form)
	ones = tf.ones_like(x_t_flat)
	sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

	# repeat grid num_batch times
	sampling_grid = tf.math.expand_dims(sampling_grid, axis=0)
	sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

	# cast to float32 (required for matmul)
	theta = tf.cast(theta, 'float32')
	sampling_grid = tf.cast(sampling_grid, 'float32')

	# transform the sampling grid - batch multiply
	batch_grids = tf.matmul(theta, sampling_grid)
	# batch grid has shape (num_batch, 2, H*W)

	# reshape to (num_batch, H, W, 2)
	batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

	return batch_grids

def bilinear_sampler(img, x, y):
	"""
	Performs bilinear sampling of the input images according to the
	normalized coordinates provided by the sampling grid. Note that
	the sampling is done identically for each channel of the input.
	To test if the function works properly, output image should be
	identical to input image when theta is initialized to identity
	transform.
	Input
	-----
	- img: batch of images in (B, H, W, C) layout.
	- grid: x, y which is the output of affine_grid_generator.
	Returns
	-------
	- out: interpolated images according to grids. Same size as grid.
	"""
	H = tf.shape(img)[1]
	W = tf.shape(img)[2]
	max_y = tf.cast(H - 1, 'int32')
	max_x = tf.cast(W - 1, 'int32')
	zero = tf.zeros([], dtype='int32')

	# rescale x and y to [0, W-1/H-1]
	x = tf.cast(x, 'float32')
	y = tf.cast(y, 'float32')
	x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
	y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

	# grab 4 nearest corner points for each (x_i, y_i)
	x0 = tf.cast(tf.floor(x), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y), 'int32')
	y1 = y0 + 1

	# clip to range [0, H-1/W-1] to not violate img boundaries
	x0 = tf.clip_by_value(x0, zero, max_x)
	x1 = tf.clip_by_value(x1, zero, max_x)
	y0 = tf.clip_by_value(y0, zero, max_y)
	y1 = tf.clip_by_value(y1, zero, max_y)

	# get pixel value at corner coords
	Ia = get_pixel_value(img, x0, y0)
	Ib = get_pixel_value(img, x0, y1)
	Ic = get_pixel_value(img, x1, y0)
	Id = get_pixel_value(img, x1, y1)

	# recast as float for delta calculation
	x0 = tf.cast(x0, 'float32')
	x1 = tf.cast(x1, 'float32')
	y0 = tf.cast(y0, 'float32')
	y1 = tf.cast(y1, 'float32')

	# calculate deltas
	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	# add dimension for addition
	wa = tf.math.expand_dims(wa, axis=3)
	wb = tf.math.expand_dims(wb, axis=3)
	wc = tf.math.expand_dims(wc, axis=3)
	wd = tf.math.expand_dims(wd, axis=3)

	Ia = tf.cast(Ia, 'float32')
	Ib = tf.cast(Ib, 'float32')
	Ic = tf.cast(Ic, 'float32')
	Id = tf.cast(Id, 'float32')


	# print(wa.dtype,Ia.dtype,wb.dtype,Ib.dtype,Ic.dtype,wc.dtype,wd.dtype,Id.dtype,"dtypes")
	# compute output
	out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

	return out

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

if __name__ == "__main__":
	a = cv2.imread("temp.png")
	theta = np.array([[[0.707, -0.707, 0.], [0.707, 0.707, 0.]]], dtype=np.float64)
	a = np.expand_dims(a,0)
	print(a.shape)
	sess = tf.Session()
	way= "tfcontrib"
	if way == "torch":
		a = np_to_torch(a)
		a = stn(a,theta)
		a = torch_to_np(a)
		print(a.shape)
		print("max",a.max(),"min",a.min())
	elif way == "tf":
		t = _transform(a,theta,[80,80])
		print(t.shape)
		print("max",t.max(),"min",t.min())
		t = t.eval(session=sess)	
	elif way == "tfcontrib":
		batch_grids =affine_grid_generator(80, 80, theta)
		x_s = batch_grids[:, 0, :, :]
		y_s = batch_grids[:, 1, :, :]
		t = bilinear_sampler(a,x_s,y_s)
		t = t.eval(session=sess)	
		print(t.shape)
		print("max",t.max(),"min",t.min())	
	cv2.imshow("img",t[0]/255.0)
	cv2.waitKey(0)
	# stn()    