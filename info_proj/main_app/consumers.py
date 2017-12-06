import json
import numpy as np
import skimage.measure
from . import cnn_eval as cnn


def ws_connect(message):
    message.reply_channel.send({"accept": True})


def display(img, threshold=0.5):
    render = ''
    for row in img:
        for col in row:
            if col > threshold:
                render += '@'
            else:
                render += '.'
        render += '\n'
    return render


def ws_receive(message):
    text_ = message.content['text']
    content = json.loads(text_)
    flat = content['data'][3::4]
    img = np.reshape(np.asarray(flat), [336, 336])
    pooled = np.asarray(skimage.measure.block_reduce(img, (12, 12), np.max))
    print(display(pooled, 0.1))
    predicted = cnn.predict(pooled.flatten())
    message.reply_channel.send({"text": predicted})


def ws_disconnect(message):
    print("disconnected")
