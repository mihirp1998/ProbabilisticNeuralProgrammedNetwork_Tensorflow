import numpy as np
import tensorflow as tf
# tf.set_random_seed(1)

class Transform(tf.keras.Model):
    def __init__(self, matrix='default'):
        super(Transform, self).__init__()

    def bilinear_sampler(self,img, x, y):
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
        Ia = self.get_pixel_value(img, x0, y0)
        Ib = self.get_pixel_value(img, x0, y1)
        Ic = self.get_pixel_value(img, x1, y0)
        Id = self.get_pixel_value(img, x1, y1)

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
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        Ia = tf.cast(Ia, 'float32')
        Ib = tf.cast(Ib, 'float32')
        Ic = tf.cast(Ic, 'float32')
        Id = tf.cast(Id, 'float32')


        # print(wa.dtype,Ia.dtype,wb.dtype,Ib.dtype,Ic.dtype,wc.dtype,wd.dtype,Id.dtype,"dtypes")
        # compute output
        out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

        return out

    def get_pixel_value(self,img, x, y):
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

    def affine_grid_generator(self,height, width, theta):
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
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
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

    @tf.function
    def call(self, x, hw, variance=False):
        if variance:
            x = tf.math.exp(x / 2.0)
        # size = torch.Size([x.size(0), x.size(1), int(hw[0]), int(hw[1])])

        # grid generation
        theta = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        theta = np.repeat(theta,int(x.shape[0]),0)
        # theta = theta.expand(x.shape[0], theta.shape[1], theta.shape[2])
        gridout = self.affine_grid_generator(hw[0],hw[1],theta)

        x_s = gridout[:, 0, :, :]
        y_s = gridout[:, 1, :, :]
        # bilinear sampling
        out = self.bilinear_sampler(x, x_s, y_s)
        if variance:
            out = tf.math.log(out) * 2.0
        return out

def run():
  x1 = tf.random.normal([1,32,32,3])
  import time
  s = time.time()
  t(x1,[60,60])
  print(time.time() - s)

if __name__ == "__main__":
    t = Transform()
    for i in range(20):
        run()
    print(len(t.trainable_variables))

