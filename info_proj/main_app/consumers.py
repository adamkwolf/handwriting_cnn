import json
import numpy as np
import skimage.measure
from . import cnn_eval as cnn
import math


def ws_connect(cnt):
    '''
    :return: send handshake back to ws connector
    '''
    cnt.reply_channel.send({"accept": True})


def display(img, threshold=0.5):
    '''
    Testing function used to display images in ascii text
    :param img: the matrix to display
    :param threshold: threshold value for discarding low values
    :return: returned the rendered ascii string
    '''
    render = ''
    for row in img:
        for col in row:
            if col > threshold:
                render += '##'
            else:
                render += '  '
        render += '\n'
    return render


def ws_receive(message):
    '''
    :param message: message from websocket where content is the image in the form of array
    :return: return the predicted character
    '''

    # get the message content (the image)
    text_ = message.content['text']

    # convert the content json to python datastructure
    content = json.loads(text_)

    # remove the color channels (we only want black and white)
    flat = content['data'][3::4]

    # reshape into matrix from flat array
    img = np.reshape(np.asarray(flat), [336, 336])

    # crop image
    img = crop_zeros(img)

    # padding width and height to make them even
    img = even_width_height(img)
    height, width = img.shape

    if height > width:  # pad width to make square
        img = pad_sides(img, 1, int((height - width) / 2), transpose=True)
    elif width > height:  # pad height to make square
        img = pad_sides(img, 0, int((width - height) / 2))

    img = add_padding_for_scaling(img)
    pooled = shrink_to_fit(img)
    prediction = get_prediction(pooled)
    message.reply_channel.send({"text": json.dumps({"pred": prediction})})


def crop_zeros(img):
    '''
    crop the image a shape with only image drawing data in it
    :param img: the image to crop
    :return: the cropped image
    '''

    # get the list of rows from the image that don't contain all zeros
    img = np.asarray([x for x in img if not all(v == 0 for v in x)])

    # get the list of columns from the image that don't contain all zeros
    # this is technically rows too but since its being transposed its actually the columns
    img = np.asarray([x for x in img.T if not all(v == 0 for v in x)]).T
    return img


def even_width_height(img):
    '''
    We need to even out the width and height pixel wise
    Example: if height is 227 pixels, padding it with a row of all 0s will change
    the shape to 228
    :param img: image to be padded
    :return: the padded image
    '''
    if len(img) % 2 != 0:
        img = np.concatenate((img, np.zeros(len(img[0]))[np.newaxis]), axis=0)
    if len(img[0]) % 2 != 0:
        img = np.concatenate((img, np.zeros(len(img))[np.newaxis].T), axis=1)
    return img


def pad_sides(img, axis, num_padding, transpose=False):
    '''
    We need to padd the sizes of the image in order to make it the correct
    dimension for pooling to (28,28)
    :param img: the image for padding
    :param axis: the axis to pad
    :param num_padding: the number of padding (zero rows) to add
    :param transpose: should be transposed?
    :return: the padded image
    '''
    # get the size of the image
    size = len(img[0]) if axis == 0 else len(img)

    # make an array of zeros based on the size to pad with
    zeros = np.zeros(size)[np.newaxis]

    # transpose the zeros if we are adding columns instead of rows
    if transpose:
        zeros = zeros.T

    # for the number of padding, evenly add to each of the left/right or top/bottom
    for _ in range(num_padding):
        img = np.concatenate((img, zeros), axis=axis)
        img = np.concatenate((zeros, img), axis=axis)
    return img


def shrink_to_fit(img):
    '''
    shrink the image to (28,28) by scaling it down (pooling)
    :param img: the image to pool
    :return: the pooled image to (28,28)
    '''
    # calculate amount of scaling
    scale = math.floor(len(img) / 28)
    # scale the image
    reduce = skimage.measure.block_reduce(img, (scale, scale))
    # convert back to np array
    pooled = np.asarray(reduce)
    return pooled


def get_prediction(img):
    '''
    get the prediction from the trained cnn model (cnn_eval.py)
    :param img: the image to send to the cnn eval
    :return: the predicted label
    '''
    print(display(img, 0.1))
    predicted = cnn.predict(img.flatten())
    print("Predicted {}".format(predicted))
    # print("Confidence {}".format(confidence))
    return predicted


def add_padding_for_scaling(img):
    '''
    Padding on the image to increase the width/height as a
    factor of 28 so it can easily be downscaled.
    :param img: the image to pad
    :return: the padded image
    '''
    # figure out how much padding is needed to be scaled to 28x28
    padding = int(28 - (len(img) % 28))
    # add padding for scaling
    for _ in range(padding):
        img = np.concatenate((img, np.zeros(len(img))[np.newaxis].T), axis=1)
        img = np.concatenate((img, np.zeros(len(img[0]))[np.newaxis]), axis=0)

    return img


def ws_disconnect(message):
    # not really used for anything
    print("disconnected")
