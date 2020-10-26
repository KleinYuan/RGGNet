import os
import tensorflow as tf
import tensorflow_probability as tfp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

###################################################################################################
#                          Variational Auto-Encoder                                               #
###################################################################################################


# Gaussian Encoder
def encoder(x, z_dim, is_training):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
    print("=======================     Encoder     ==========================")

    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        print(x)
        net = tf.layers.conv2d(x, 64, 4, 2, activation=None, name='conv1')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        net = tf.layers.conv2d(net, 128, 4, 2, activation=tf.nn.relu, name='conv2')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        net = tf.layers.conv2d(net, 64, 4, 2, activation=tf.nn.relu, name='conv3')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        net = tf.layers.conv2d(net, 32, 4, 2, activation=tf.nn.relu, name='conv4')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        net = tf.layers.flatten(net, name='flattened')
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='fc')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        gaussian_params = tf.layers.dense(net, 2*z_dim, activation=None, name='gaussian_params')
        print(gaussian_params)
        # The mean parameter is unconstrained
        mean = tf.identity(gaussian_params[:, :z_dim], 'mu')
        print(mean)
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = tf.identity(1e-6 + tf.nn.softplus(gaussian_params[:, z_dim:]), 'sigma')
        print(stddev)

    return mean, stddev


# Bernoulli decoder
def decoder(z, h, w, is_training):
    # b, h, w, c = self.x_cam.get_shape().as_list()
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    print("=======================     Decoder     ==========================")
    with tf.variable_scope("decoder"):
        net = tf.layers.dense(z, 1024, activation=tf.nn.relu, name='fc1')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        net = tf.layers.dense(net, int(32 * h/16 * w/16), activation=tf.nn.relu, name='fc2')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        net = tf.reshape(net, [tf.shape(net)[0], int(h/16), int(w/16), 32], name='reshape1')
        print(net)
        net = tf.layers.conv2d_transpose(net,  32, 4, 2, activation=tf.nn.relu, padding='same', name='deconv1')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        net = tf.layers.conv2d_transpose(net,  64, 4, 2, activation=tf.nn.relu, padding='same', name='deconv2')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        net = tf.layers.conv2d_transpose(net, 128, 4, 2, activation=tf.nn.relu, padding='same', name='deconv3')
        net = tf.layers.batch_normalization(net, training=is_training)
        print(net)
        out = tf.layers.conv2d_transpose(net, 5, 4, 2, activation=tf.nn.sigmoid, padding='same', name='out')
        print(out)

        return out


def sample(mean, stddev):
    z = mean + stddev * tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
    return z


def compute_ELBO(x, y, mu, sigma):
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
    marginal_likelihood = tf.reduce_sum(x * tf.log(1e-8 + y) + (1 - x) * tf.log(1e-8 + 1 - y), [1, 2])
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])
    neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)
    ELBO = -neg_loglikelihood - KL_divergence
    # tf.summary.scalar("ELBO", ELBO)
    tf.summary.scalar("KL_divergence", KL_divergence)
    tf.summary.scalar("neg_loglikelihood", neg_loglikelihood)
    tf.summary.scalar("ELBO", -ELBO)
    return -ELBO


def compute_KL(mu_hat, sigma_hat, mu_gt, sigma_gt):
    with tf.variable_scope('KL'):
        dist_q = tfp.distributions.Normal(loc=mu_gt, scale=sigma_gt)
        dist_p = tfp.distributions.Normal(loc=mu_hat, scale=sigma_hat)
        KL = tfp.distributions.kl_divergence(dist_p, dist_q, name='KL')
        KL = tf.reduce_mean(KL)
        tf.summary.scalar("KL_divergence", KL)
        return KL

def vae_test():
    import numpy as np
    H = 160
    W = 640
    z_dim = 1024
    x = tf.placeholder(tf.float32, shape=[12, H, W, 2], name='x_test')
    is_training = tf.placeholder(tf.bool)
    mean, std = encoder(x, z_dim, is_training)
    sample_z = sample(mean, std)
    out = decoder(sample_z, H, W, is_training)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        feed_dict = \
            {
                x: np.random.rand(*x.get_shape().as_list()).astype(np.float32)
            }
        print(sess.run(out, feed_dict=feed_dict).shape)


def tolerance_regularizer(inputs, vae_latent_dim, is_training):
    _, h, w, c = inputs.get_shape().as_list()
    with tf.variable_scope('Tolerance_Regularization'):
        print("    [Tolerance_Regularization]           ##################################")
        mu_hat, sigma_hat = encoder(x=inputs, z_dim=vae_latent_dim,
                                    is_training=is_training)
        z_hat = sample(mean=mu_hat, stddev=sigma_hat)
        vae_outputs = decoder(z=z_hat, h=h, w=w, is_training=is_training)
        tolerance_regularization = compute_ELBO(x=inputs, y=vae_outputs, mu=mu_hat, sigma=sigma_hat)
    return tolerance_regularization, vae_outputs

###################################################################################################
#                           Conditional 3D Spatial Transformer                                    #
###################################################################################################


def _reverse_cantor_pair(z, H, W):
    z = tf.cast(z, 'float32')
    w = tf.floor((tf.sqrt(8. * z + 1.) - 1.) / 2.0)
    t = (w ** 2 + w) / 2.0
    y = tf.clip_by_value(tf.expand_dims(z - t, 1), 0.0, H - 1)
    x = tf.clip_by_value(tf.expand_dims(w - y[:, 0], 1), 0.0, W - 1)

    return tf.concat([y, x], 1)


def _grid_transform(img, x, y, H, W, updated_indices=None):

    indices = tf.stack([y, x], 2)
    indices = tf.reshape(indices, (H * W, 2))
    values = tf.reshape(img, [-1])

    Y = indices[:, 0]
    X = indices[:, 1]
    Z = tf.cast((X + Y) * (X + Y + 1) / 2, tf.int32) + Y

    filtered, idx = tf.unique(tf.squeeze(Z))
    updated_values = tf.unsorted_segment_max(values, idx, tf.shape(filtered)[0])
    if updated_indices is None:
        updated_indices = _reverse_cantor_pair(filtered, H, W)
        updated_indices = tf.cast(updated_indices, 'int32')
        # updated_values have to be without duplicates, otherwise it will accumulate
        resolved_map = tf.scatter_nd(updated_indices, updated_values, (H, W))
    else:
        resolved_map = tf.scatter_nd(updated_indices, updated_values, (H, W))

    return resolved_map, updated_indices


def c_3dstn(depth_map_in, transform, R_rect, P_rect, H, W):

    print("        [TF Transform In]:                 {}".format(depth_map_in))
    print("        [TF Transform In]:                 {}".format(transform))
    print("        [TF Transform In]:                 {}".format(R_rect))
    print("        [TF Transform In]:                 {}".format(P_rect))

    pad = tf.ones(shape=[H, W, 1])
    # Step1: Normalize back
    depth_map_xyz = depth_map_in[:, :, :3]

    # Step2: Transform
    depth_map_xyz_hom = tf.concat([depth_map_xyz, pad], -1)
    depth_map_xyz_transformed = tf.einsum('fuc,ck->fuk', depth_map_xyz_hom, tf.transpose(transform))

    # Step2: Compute data on cam frame
    depth_map_xyz_transformed_rect = tf.einsum('fuc,ck->fuk', depth_map_xyz_transformed, tf.transpose(R_rect))
    depth_map_xyz_transformed_rect_hom = tf.concat([depth_map_xyz_transformed_rect, pad], -1)

    depth_map_xyz_cam_pre = tf.einsum('fuc,ck->fuk', depth_map_xyz_transformed_rect_hom, tf.transpose(P_rect))
    depth_map_x_cam = tf.div(depth_map_xyz_cam_pre[:, :, 0:1], depth_map_xyz_cam_pre[:, :, 2:3], name='depth_map_x_cam')
    depth_map_y_cam = tf.div(depth_map_xyz_cam_pre[:, :, 1:2], depth_map_xyz_cam_pre[:, :, 2:3], name='depth_map_y_cam')
    depth_map_z_cam = depth_map_xyz_cam_pre[:, :, 2:3]
    # (H, W)
    depth_map_x_cam = tf.squeeze(depth_map_x_cam)
    depth_map_y_cam = tf.squeeze(depth_map_y_cam)

    # (H, W)
    x = tf.cast(depth_map_x_cam, tf.int32)
    y = tf.cast(depth_map_y_cam, tf.int32)
    x_cliped = tf.clip_by_value(x, clip_value_min=0, clip_value_max=W - 1)
    y_cliped = tf.clip_by_value(y, clip_value_min=0, clip_value_max=H - 1)

    depth_map_next_i, updated_indices = _grid_transform(depth_map_in[:, :, 3:4], x=x_cliped, y=y_cliped, H=H, W=W)

    depth_map_next_x, _ = _grid_transform(depth_map_xyz_transformed[:, :, 0:1], x=x_cliped,
                                               y=y_cliped, H=H, W=W, updated_indices=updated_indices)
    depth_map_next_y, _ = _grid_transform(depth_map_xyz_transformed[:, :, 1:2], x=x_cliped,
                                               y=y_cliped, H=H, W=W, updated_indices=updated_indices)
    depth_map_next_z, _ = _grid_transform(depth_map_xyz_transformed[:, :, 2:3], x=x_cliped,
                                               y=y_cliped, H=H, W=W, updated_indices=updated_indices)

    depth_map_z_cam = tf.div(depth_map_z_cam, tf.reduce_max(depth_map_z_cam))
    depth_map_next_zcam, _ = _grid_transform(depth_map_z_cam, x=x_cliped, y=y_cliped, H=H, W=W, updated_indices=updated_indices)
    depth_map_next = tf.stack([depth_map_next_x, depth_map_next_y, depth_map_next_z, depth_map_next_i, depth_map_next_zcam], -1)

    print("        [TF Transform Out]:                 {}".format(depth_map_next))
    return depth_map_next


def batch_c_3dstn(tfs, x_dm, R_rects, P_rects, H, W):
    with tf.variable_scope('transformer'):
        pred_depth_map = tf.map_fn(
            lambda x: (
                c_3dstn(
                    x[0],
                    x[1],
                    x[2],
                    x[3],
                    H,
                    W
                )
            ),
            (
                x_dm,
                tfs,
                R_rects,
                P_rects,
                ),
            dtype=tf.float32)

        return pred_depth_map

###################################################################################################
#                          se3 to SE(3)                                                           #
###################################################################################################


def se3toSE3(se3):

    with tf.name_scope("Exponential_map"):

        u = se3[3:]
        omega = se3[:3]
        theta = tf.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])
        omega_cross = tf.stack([0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0])
        omega_cross = tf.reshape(omega_cross, [3, 3])
        A = tf.sin(theta)/theta
        B = (1.0 - tf.cos(theta))/(tf.pow(theta,2))
        C = (1.0 - A)/(tf.pow(theta,2))
        omega_cross_square = tf.matmul(omega_cross, omega_cross)
        R = tf.eye(3, 3) + A*omega_cross + B*omega_cross_square
        V = tf.eye(3, 3) + B*omega_cross + C*omega_cross_square
        Vu = tf.matmul(V, tf.expand_dims(u, 1))
        SE3 = tf.concat([R, Vu], 1)

        return SE3


def batch_se3toSE3(se3s):
    with tf.variable_scope('se3toSE3'):
        SE3s = tf.map_fn(lambda x: se3toSE3(x), se3s, dtype=tf.float32)
        return SE3s

########################################################################################################################
#                              Normal 3D Spatial Transformer (Auxiliary Experiments                                    #
#  Some codes are borrowed from : https://github.com/epiception/CalibNet/blob/master/code/common/all_transformer.py    #
########################################################################################################################


def batch_3dstn(tfs, x_dm, R_rects, P_rects, H, W):
    with tf.variable_scope('transformer'):
        pred_depth_map = tf.map_fn(
            lambda x: (
                n_3dstn(
                    x[0],
                    x[1],
                    x[2],
                    x[3],
                    H,
                    W
                )
            ),
            (
                x_dm,
                tfs,
                R_rects,
                P_rects,
                ),
            dtype=tf.float32)

        return pred_depth_map


def n_3dstn(depth_map_in, transform, R_rect, P_rect, H, W):

    batch_grids_z, transformed_depth_map_z, reprojected_grid_z = _3D_meshgrid_batchwise_diff(H, W, depth_map_in[:, :, 4:5], transform, R_rect, P_rect, reprojected_grid_in=None)
    x_all_z = tf.reshape(batch_grids_z[:,0], (H, W))
    y_all_z = tf.reshape(batch_grids_z[:,1], (H, W))
    pred_depth_map_z = _bilinear_sampling(transformed_depth_map_z, x_all_z, y_all_z, H, W)

    batch_grids_i, transformed_depth_map_i, _ = _3D_meshgrid_batchwise_diff(H, W, depth_map_in[:, :, 3:4], transform, R_rect, P_rect, reprojected_grid_in=reprojected_grid_z)
    x_all_i = tf.reshape(batch_grids_i[:,0], (H, W))
    y_all_i = tf.reshape(batch_grids_i[:,1], (H, W))
    pred_depth_map_i = _bilinear_sampling(transformed_depth_map_i, x_all_i, y_all_i, H, W)

    pred_depth_map = tf.concat([depth_map_in[:, :, 0:3], tf.expand_dims(pred_depth_map_i, -1), tf.expand_dims(pred_depth_map_z, -1)], -1)
    return pred_depth_map


def _3D_meshgrid_batchwise_diff(height, width, depth_img, transformation_matrix, tf_K_mat, small_transform, reprojected_grid_in=None):

    """
    Creates 3d sampling meshgrid
    """

    x_index = tf.linspace(-1.0, 1.0, width)
    y_index = tf.linspace(-1.0, 1.0, height)
    z_index = tf.range(0, width*height)

    x_t, y_t = tf.meshgrid(x_index, y_index)

    # flatten
    x_t_flat = tf.reshape(x_t, [1,-1])
    y_t_flat = tf.reshape(y_t, [1,-1])
    ZZ = tf.reshape(depth_img, [-1])

    zeros_target = tf.zeros_like(ZZ)
    mask = tf.not_equal(ZZ, zeros_target)
    ones = tf.ones_like(x_t_flat)

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)
    sampling_grid_2d_sparse = tf.transpose(tf.boolean_mask(tf.transpose(sampling_grid_2d), mask))
    ZZ_saved = tf.boolean_mask(ZZ, mask)
    ones_saved = tf.expand_dims(tf.ones_like(ZZ_saved), 0)

    projection_grid_3d = tf.matmul(tf.matrix_inverse(tf_K_mat), sampling_grid_2d_sparse*ZZ_saved)

    homog_points_3d = tf.concat([projection_grid_3d, ones_saved], 0)
    final_transformation_matrix = tf.matmul( tf.transpose(transformation_matrix),small_transform)[:3, :]
    warped_sampling_grid = tf.matmul(final_transformation_matrix, homog_points_3d)

    points_2d = tf.matmul(tf_K_mat, warped_sampling_grid[:3, :])

    Z = points_2d[2, :]

    mask_int = tf.cast(mask, 'int32')
    updated_indices = tf.expand_dims(tf.boolean_mask(mask_int * z_index, mask), 1)
    updated_Z = tf.scatter_nd(updated_indices, Z, tf.constant([width * height]))

    if reprojected_grid_in is None:
        x = tf.transpose(points_2d[0, :] / Z)
        updated_x = tf.scatter_nd(updated_indices, x, tf.constant([width * height]))
        neg_ones = tf.ones_like(updated_x) * -1.0
        updated_x_fin = tf.where(tf.equal(updated_Z, zeros_target), neg_ones, updated_x)

        y = tf.transpose(points_2d[1, :] / Z)
        updated_y = tf.scatter_nd(updated_indices, y, tf.constant([width * height]))
        updated_y_fin = tf.where(tf.equal(updated_Z, zeros_target), neg_ones, updated_y)

        reprojected_grid = tf.stack([updated_x_fin, updated_y_fin], 1)
    else:
        reprojected_grid = reprojected_grid_in

    transformed_depth = tf.reshape(updated_Z, (height, width))

    return reprojected_grid, transformed_depth, reprojected_grid_in


def reverse_all(z, H, W):

    """Reversing from cantor function indices to correct indices"""

    z = tf.cast(z, 'float32')
    w = tf.floor((tf.sqrt(8.*z + 1.) - 1.)/2.0)
    t = (w**2 + w)/2.0
    y = tf.clip_by_value(tf.expand_dims(z - t, 1), 0.0, H - 1)
    x = tf.clip_by_value(tf.expand_dims(w - y[:,0], 1), 0.0, W - 1)

    return tf.concat([y,x], 1)


def get_pixel_value(img, x, y, H, W):

    """Cantor pairing for removing non-unique updates and indices. At the time of implementation, unfixed issue with scatter_nd causes problems with int32 update values. Till resolution, implemented on cpu """

    indices = tf.stack([y, x], 2)
    indices = tf.reshape(indices, (H * W, 2))
    values = tf.reshape(img, [-1])

    Y = indices[:,0]
    X = indices[:,1]
    Z = tf.cast((X + Y) * (X + Y + 1) / 2, tf.int32) + Y

    filtered, idx = tf.unique(tf.squeeze(Z))
    updated_values  = tf.unsorted_segment_max(values, idx, tf.shape(filtered)[0])

    updated_indices = reverse_all(filtered, H, W)
    updated_indices = tf.cast(updated_indices, 'int32')
    resolved_map = tf.scatter_nd(updated_indices, updated_values, img.shape)

    return resolved_map


def _bilinear_sampling(img, x_func, y_func, H, W):

    """
    Sampling from input image and performing bilinear interpolation
    """

    max_y = tf.constant(H - 1, dtype=tf.int32)
    max_x = tf.constant(W - 1, dtype=tf.int32)

    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W/H/D]
    x = 0.5 * ((x_func + 1.0) * tf.cast(W - 1, 'float32'))
    y = 0.5 * ((y_func + 1.0) * tf.cast(H - 1, 'float32'))

    x = tf.clip_by_value(x, 0.0, tf.cast(max_x, 'float32'))
    y = tf.clip_by_value(y, 0.0, tf.cast(max_y, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    # find Ia, Ib, Ic, Id

    Ia = get_pixel_value(img, x0, y0, H, W)
    Ib = get_pixel_value(img, x0, y1, H, W)
    Ic = get_pixel_value(img, x1, y0, H, W)
    Id = get_pixel_value(img, x1, y1, H, W)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    loc = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return loc


if __name__ == "__main__":
    vae_test()

