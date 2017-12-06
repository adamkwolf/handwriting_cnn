import numpy as np
from scipy.io import loadmat
import pickle
import Dataset as ds


def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    def rotate(img):
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    # load list structure from loadmat
    mat = loadmat(mat_file_path)

    # load char mapping
    mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('bin/mapping.p', 'wb'))

    # load training data
    if not max_:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images_u = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # load testing data
    if not max_:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images_u = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # reshape training data to be valid
    if verbose:
        _len = len(training_images_u)
    for i in range(len(training_images_u)):
        if verbose:
            print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1) / _len) * 100), end='\r')
            training_images_u[i] = rotate(training_images_u[i])
    if verbose:
        print('')

    # Reshape testing data to be valid
    if verbose:
        _len = len(testing_images_u)
    for i in range(len(testing_images_u)):
        if verbose:
            print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1) / _len) * 100), end='\r')
        testing_images_u[i] = rotate(testing_images_u[i])
    if verbose:
        print('')

    # convert type to float32
    training_images = np.asarray([t.flatten() for t in training_images_u.astype('float32')])
    testing_images = np.asarray([t.flatten() for t in testing_images_u.astype('float32')])

    # normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255
    nb_classes = len(mapping)

    testing_labels_r = []
    for label in testing_labels:
        n_label = np.zeros(shape=nb_classes)
        n_label[label] = 1
        testing_labels_r.append(n_label)

    training_labels_r = []
    for label in training_labels:
        n_label = np.zeros(shape=nb_classes)
        n_label[label] = 1
        training_labels_r.append(n_label)

    training_labels_r = np.asarray(training_labels_r)
    testing_labels_r = np.asarray(testing_labels_r)

    test = ds.DataSet(images=testing_images, labels=testing_labels_r)
    train = ds.DataSet(images=training_images, labels=training_labels_r)

    return ds.Datasets(train, test, None, nb_classes)
