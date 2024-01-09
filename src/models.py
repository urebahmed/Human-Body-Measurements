"""
Defines networks.

@Encoder_resnet
@Encoder_fc3_dropout

Helper:
@get_encoder_fn_separate
"""

import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from tensorflow.keras.applications import ResNet50

def Encoder_resnet(x, is_training=True, weight_decay=0.001, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    with tf.compat.v1.name_scope("Encoder_resnet", [x]):
        resnet = ResNet50(weights=None, include_top=False, input_tensor=x)
        net = layers.GlobalAveragePooling2D()(resnet.output)
        net = tf.squeeze(net, axis=[1, 2])
    variables = resnet.trainable_variables
    return net, variables


def Encoder_fc3_dropout(x,
                        num_output=85,
                        is_training=True,
                        reuse=False,
                        name="3D_module"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal: 
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    if reuse:
        print('Reuse is on!')
    with tf.compat.v1.variable_scope(name, reuse=reuse) as scope:
        net = layers.Dense(1024, activation='relu', name='fc1')(x)
        net = layers.Dropout(0.5, name='dropout1')(net)
        net = layers.Dense(1024, activation='relu', name='fc2')(net)
        net = layers.Dropout(0.5, name='dropout2')(net)
        small_xavier = initializers.VarianceScaling(scale=.01, mode='fan_avg', distribution='uniform')
        net = layers.Dense(num_output, activation=None, kernel_initializer=small_xavier, name='fc3')(net)

    variables = net.trainable_variables
    return net, variables


def get_encoder_fn_separate(model_type):
    """
    Retrieves diff encoder fn for image and 3D
    """
    encoder_fn = None
    threed_fn = None
    if 'resnet' in model_type:
        encoder_fn = Encoder_resnet
    else:
        print('Unknown encoder %s!' % model_type)
        exit(1)

    if 'fc3_dropout' in model_type:
        threed_fn = Encoder_fc3_dropout

    if encoder_fn is None or threed_fn is None:
        print('Dont know what encoder to use for %s' % model_type)
        import ipdb
        ipdb.set_trace()

    return encoder_fn, threed_fn




def Discriminator_separable_rotations(poses, shapes, weight_decay):
    """
    23 Discriminators on each joint + 1 for all joints + 1 for shape.
    To share the params on rotations, this treats the 23 rotation matrices
    as a "vertical image":
    Do 1x1 conv, then send off to 23 independent classifiers.

    Input:
    - poses: N x 23 x 1 x 9, NHWC ALWAYS!!
    - shapes: N x 10
    - weight_decay: float

    Outputs:
    - prediction: N x (1+23) or N x (1+23+1) if do_joint is on.
    - variables: tf variables
    """
    with tf.compat.v1.name_scope("Discriminator_sep_rotations", [poses, shapes]):
        with tf.compat.v1.variable_scope("D") as scope:
            poses = layers.Conv2D(32, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='D_conv1')(poses)
            poses = layers.Conv2D(32, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='D_conv2')(poses)
            theta_out = []
            for i in range(0, 23):
                theta_out.append(
                    layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="pose_out_j%d" % i)(poses[:, i, :, :]))
            theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))

            # Do shape on it's own:
            shapes = layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="shape_fc1")(shapes)
            shapes = layers.Dense(5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="shape_fc2")(shapes)
            shape_out = layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="shape_final")(shapes)

            # Compute joint correlation prior!
            nz_feat = 1024
            poses_all = layers.Flatten(name='vectorize')(poses)
            poses_all = layers.Dense(nz_feat, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="D_alljoints_fc1")(poses_all)
            poses_all = layers.Dense(nz_feat, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="D_alljoints_fc2")(poses_all)
            poses_all_out = layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="D_alljoints_out")(poses_all)
            out = tf.concat([theta_out_all, poses_all_out, shape_out], 1)

        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        return out, variables