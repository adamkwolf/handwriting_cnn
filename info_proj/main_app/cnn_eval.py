'''
Created by Adam Wolf for INFO371 Data Mining
Using EMNIST Dataset: can be found here https://www.nist.gov/itl/iad/image-group/emnist-dataset
Loads the saved trained model and allows access as an API for getting predicted characters
from images. See consumers.py for django backend api calls.
'''

import tensorflow as tf

# real labels used for mapping argmax output to the actual characters
# real_labels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')  # for by_class dataset
# num_classes = 62

# real labels used for mapping argmax output to the actual characters
real_labels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')  # for by_merge dataset
num_classes = 47

################
# Hyper Params #
################

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
    '''
    :param shape: the shape of the variable
    :param name: the name of the variable to get from saved model
    :return: the saved variable
    '''
    return tf.get_variable(name=name, shape=shape)


def new_biases(length):
    '''
    :param length:  shape of bias
    :return: bias variable
    '''
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input, num_input_channels, filter_size, num_filters, weight_name, use_pooling=True):
    '''
    create a new convulational layer
    :param input: previous layer
    :param num_input_channels: number of channels of last layer
    :param filter_size: shape of each filter
    :param num_filters: number of filters to create
    :param weight_name: name of the weight for saving the model
    :param use_pooling: use max-pooling
    :return: convnet layer
    '''

    # Shape of the filter-weights for the convolution
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # create new weights of given shape
    weights = new_weights(shape=shape, name=weight_name)

    # create the new convnet flayer from tensorflow
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    # create biases for each filter
    biases = new_biases(length=num_filters)

    # add bias to the results of the convolution
    layer += biases

    # down-sample the image resolution
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    '''
    reshape a layer by number of features
    :param layer: the layer to reshape
    :return: the reshaped layer
    '''

    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, weight_name, use_relu=True):
    '''
    creates a new fully connected layer of size num_outputs
    :param input: previous layer
    :param num_inputs: number of inputs of the previous layer
    :param num_outputs: number of outputs
    :param weight_name: weight name
    :param use_relu: should use rectified linear unit on output
    :return: output layer
    '''
    weights = new_weights(shape=[num_inputs, num_outputs], name=weight_name)
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


# create the image placeholder
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# reshape the x variable with one channel and our dimensions of (28,28) based on the dataset
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# predicted value placeholder from fc layer output
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

# predicted class from softmax output
y_true_cls = tf.argmax(y_true, dimension=1)

# create both convnet layers
layer_conv1, weights_conv1 = new_conv_layer(x_image, num_channels, filter_size1, num_filters1, "weights_conv1", True)
layer_conv2, weights_conv2 = new_conv_layer(layer_conv1, num_filters1, filter_size2, num_filters2, "weights_conv2", True)

# flattened layer
layer_flat, num_features = flatten_layer(layer_conv2)

# reduce the size of the convnet features
layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, "weights_fc1", True)

# reduce the size of the first fc layer to our desired class size
layer_fc2 = new_fc_layer(layer_fc1, fc_size, num_classes, "weights_fc2", False)

# send the output of the fully connected layer into our softmax function
y_pred = tf.nn.softmax(layer_fc2)
# get the max value's index from softmax function
y_pred_cls = tf.argmax(y_pred, dimension=1)

# start up a new session and restore the trained model weights
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "info_proj/main_app/model/cnn_model.ckpt")


def predict(img):
    '''
    Given an image of shape (28,28), predict the character
    :param img: numpy image
    :return: the character label and the predicted confidence
    '''

    feed_dict = {x: [img]} # dictionary containing the images to send through the network
    pred_idx = sess.run(y_pred_cls, feed_dict)  # the predicted character's label index
    # pred_conf = sess.run(y_pred, feed_dict)
    return real_labels[int(pred_idx[0])]  # return t
