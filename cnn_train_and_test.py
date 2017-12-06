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


def plot_images(images, cls_true, cls_pred=None):
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

    plt.savefig("stats/plot.png")


def new_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)


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
    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape = [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]
    return layer_flat, num_features


def new_fc_layer(input,  # The previous layer
                 num_inputs,  # Num. inputs from prev. layer
                 num_outputs,  # Num. outputs
                 weight_name,  # Name of weight
                 use_relu=True):  # Use ReLU
    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs], name=weight_name)
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values
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

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

# Counter for total number of iterations performed so far
total_iterations = 0
test_batch_size = 256


def optimize(data, num_iterations):
    # Ensure we update the global variable rather than local
    global total_iterations

    # start-time used for printing time-usage below
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 1000 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(data, cls_pred, correct):
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(data, num_classes, cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    plt.figure(figsize=(10, 10))
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.linspace(0, num_classes, num=num_classes)
    plt.xticks(tick_marks, real_labels, fontsize=6)
    plt.yticks(tick_marks, real_labels, fontsize=6)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("stats/confusion.png", dpi=600)


def print_test_accuracy(data, show_example_errors=False,
                        show_confusion_matrix=False):
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
        plot_confusion_matrix(data, num_classes=47, cls_pred=cls_pred)


def train_and_test(data):
    saver = tf.train.Saver()
    optimize(data, num_iterations=40000)
    print("Finished Training and Testing")
    print_test_accuracy(data, show_example_errors=True, show_confusion_matrix=True)
    saver.save(session, "/tmp/cnn_model.ckpt")
    print("Saved model\nDONE")


def main():
    data = ld.load_data("data/emnist-bymerge.mat")
    data.test.cls = np.argmax(data.test.labels, axis=1)
    train_and_test(data)


if __name__ == '__main__':
    main()
