import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import load_data as ld

real_labels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')

# Convolutional Layer 1
filter_size1 = 5  # Convolution filters are 5x5 pixels.
num_filters1 = 16  # There are 16 of these filters.

# Convolutional Layer 2
filter_size2 = 5  # Convolution filters are 5x5 pixels.
num_filters2 = 36  # There are 36 of these filters.

# Fully-connected layer
fc_size = 128  # Number of neurons in fully-connected layer.

# MNIST images are 28 pixels each dimension
img_size = 28

# Images are stored in one-dimensional arrays of this lenth
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays
img_shape = (img_size, img_size)

# Number of color channels for the images: 1 channel is gray-scale
num_channels = 1


def new_weights(shape, name):
    return tf.get_variable(name=name, shape=shape)


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,  # The previous layer
                   num_input_channels,  # Num. channels in previous layer
                   filter_size,  # Width and height of each filter
                   num_filters,  # Number of filters
                   weight_name,  # Name of weight
                   use_pooling=True):  # Use 2x2 max-pooling
    # Shape of the filter-weights for the convolution
    # determined by TensorFlow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape
    weights = new_weights(shape=shape, name=weight_name)

    # Create new biases, one for each filter
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    # Add the biases to the results of the convolution
    # A bias-value is added to each filter-channel
    layer += biases

    # Use pooling to down-sample the image resolution
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME")

    # Rectified Linear Unit (ReLU)
    # Calculates max(x, 0) for each pixel x
    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input,  # The previous layer
                 num_inputs,  # Num. inputs from prev. layer
                 num_outputs,  # Num. outputs
                 weight_name,  # Name of weight
                 use_relu=True):  # Use ReLU
    weights = new_weights(shape=[num_inputs, num_outputs], name=weight_name)
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 47], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels=num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            weight_name="weights_conv1",
                                            use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            weight_name="weights_conv2",
                                            use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         weight_name="weights_fc1",
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=47,
                         weight_name="weights_fc2",
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "/tmp/cnn_model.ckpt")


def predict(img):
    feed_dict = {x: [img]}
    pred_idx = sess.run(y_pred_cls, feed_dict=feed_dict)
    return real_labels[int(pred_idx[0])]


if __name__ == '__main__':
    data = ld.load_data("data/emnist-bymerge.mat")
    idx = int(np.argmax(data.test.labels[500]))
    # print("True: {}".format(real_labels[idx]))
    # print(data.test.display(500))
    # print("Predicted: {}".format(predict(data.test.images[500])))
