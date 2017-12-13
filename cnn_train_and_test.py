'''
Created by Adam Wolf for INFO371 Data Mining
Using EMNIST Dataset: can be found here https://www.nist.gov/itl/iad/image-group/emnist-dataset
Loads the saved trained model and allows access as an API for getting predicted characters
from images. See consumers.py for django backend api calls.
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import load_data as ld

# real labels used for mapping argmax output to the actual characters
real_labels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')  # for by_class dataset
num_classes = 62

# real labels used for mapping argmax output to the actual characters
# real_labels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')  # for by_merge dataset
# num_classes = 47

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


def plot_images(images, cls_true, cls_pred=None):
    '''
    given images, plot them on a matplotlib image
    :param images: the images to plot
    :param cls_true: the true value
    :param cls_pred: the predicted value
    '''
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # PLot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes
        if cls_pred is None:
            xlabel = "True: {0}".format(real_labels[cls_true[i]])
        else:
            xlabel = "True: {0}, Pred: {1}".format(real_labels[cls_true[i]], real_labels[cls_pred[i]])

        # Show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("stats/plot.png")  # save image here


def new_weights(shape, name):
    '''
    :param shape: the shape of the variable
    :param name: the name of the variable to save on the model
    :return: the saved variable
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)


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

############
# Training #
############

# compute the cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

# compute the cost
cost = tf.reduce_mean(cross_entropy)

# adam optimizer (fancy version of gradient descent)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# check if the prediction is correct compared to the true label
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# compute the accuracy of the pred vs true
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a new session and initialize the variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# batch size
train_batch_size = 10

# number of iterations performed so far
total_iterations = 0
test_batch_size = 256


def optimize(data, num_iterations):
    '''
    The optimization function for backtracking
    :param data: the data to run through the network and train
    :param num_iterations: the number of iterations
    '''
    global total_iterations

    # keep track of the start time to track training
    start_time = time.time()

    # get the next batch of images from the dataset, and the correct labels
    # for each iteration until complete
    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        # every 100 iterations compute the accuracy
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    # update the number of iterations
    total_iterations += num_iterations

    # record the end time
    end_time = time.time()

    # calculate the time difference
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(data, cls_pred, correct):
    '''
    plot 9 example errors from the data
    :param data: data to get incorrect predictions from for displaying
    :param cls_pred: predicted labels
    :param correct: correct labels
    '''
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


def plot_confusion_matrix(data, num_classes, cls_pred):
    '''
    Plots the confusion matrix from the testing dataset
    :param data: the testing dataset
    :param num_classes: the number of classes
    :param cls_pred: the predicted classes
    '''

    cls_true = data.test.cls
    # make a new confusion matrix
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    plt.figure(figsize=(10, 10))  # figure size
    plt.matshow(cm)  # show as matrix
    plt.colorbar()  # add the color bar for scale
    tick_marks = np.linspace(0, num_classes, num=num_classes)
    plt.xticks(tick_marks, real_labels, fontsize=6)
    plt.yticks(tick_marks, real_labels, fontsize=6)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("stats/confusion.png", dpi=600)


def print_test_accuracy(data, show_example_errors=False, show_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]

        feed_dict = {x: images, y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Creating Example Errors")
        plot_example_errors(data, cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Creating Confusion Matrix")
        plot_confusion_matrix(data, num_classes=num_classes, cls_pred=cls_pred)


def train_and_test(data):
    # create a new model saver from TF
    saver = tf.train.Saver()

    # run the optimizer
    optimize(data, num_iterations=200000)

    # print the output, and save the session in the directory below
    print("Finished Training and Testing")
    print_test_accuracy(data, show_example_errors=True, show_confusion_matrix=True)
    saver.save(session, "info_proj/main_app/model/cnn_model_byclass.ckpt")
    print("Saved model\nDONE")


def main():
    # use the byclass dataset by default, just change here for others
    data = ld.load_data("data/emnist-byclass.mat")
    data.test.cls = np.argmax(data.test.labels, axis=1)
    # start training and testing
    train_and_test(data)


if __name__ == '__main__':
    main()
