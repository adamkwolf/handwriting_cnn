import json
import numpy as np
import skimage.measure
from . import cnn_eval as cnn
import math


def ws_connect(message):
    message.reply_channel.send({"accept": True})


def display(img, threshold=0.5):
    render = ''
    for row in img:
        for col in row:
            if col > threshold:
                render += '@–'
            else:
                render += '––'
        render += '\n'
    return render


def ws_receive(message):
    text_ = message.content['text']
    content = json.loads(text_)
    flat = content['data'][3::4]  # remove color channels
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
    prediction, confidence = get_prediction(pooled)
    message.reply_channel.send({"text": json.dumps({"pred": prediction, "conf": confidence})})


def crop_zeros(img):
    img = np.asarray([x for x in img if not all(v == 0 for v in x)])
    img = np.asarray([x for x in img.T if not all(v == 0 for v in x)]).T
    return img


def even_width_height(img):
    if len(img) % 2 != 0:
        img = np.concatenate((img, np.zeros(len(img[0]))[np.newaxis]), axis=0)
    if len(img[0]) % 2 != 0:
        img = np.concatenate((img, np.zeros(len(img))[np.newaxis].T), axis=1)
    return img


def pad_sides(img, axis, num_padding, transpose=False):
    size = len(img[0]) if axis == 0 else len(img)
    zeros = np.zeros(size)[np.newaxis]

    if transpose:
        zeros = zeros.T

    for _ in range(num_padding):
        img = np.concatenate((img, zeros), axis=axis)
        img = np.concatenate((zeros, img), axis=axis)
    return img


def shrink_to_fit(img):
    # calculate amount of scaling
    scale = math.floor(len(img) / 28)
    # scale the image
    reduce = skimage.measure.block_reduce(img, (scale, scale))
    # convert back to np array
    pooled = np.asarray(reduce)
    return pooled


def get_prediction(pooled):
    print(display(pooled, 0.1))
    predicted, confidence = cnn.predict(pooled.flatten())
    print("Predicted {}".format(predicted))
    print("Confidence {}".format(confidence))
    return predicted, str(confidence)


def add_padding_for_scaling(img):
    # figure out how much padding is needed to be scaled to 28x28
    padding = int(28 - (len(img) % 28))
    # add padding for scaling
    for _ in range(padding):
        img = np.concatenate((img, np.zeros(len(img))[np.newaxis].T), axis=1)
        img = np.concatenate((img, np.zeros(len(img[0]))[np.newaxis]), axis=0)

    return img


def ws_disconnect(message):
    print("disconnected")
