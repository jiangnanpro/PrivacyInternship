import numpy as np

import os
import urllib.request
import gzip
import pickle
from zipfile import ZipFile

def emnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)
    if limit is not None:
        print("WARNING ONLY FIRST {} EMNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = np.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

        if n_labelled is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labelled)
        
        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)
        
        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]), np.copy(labelled))

        else:

            for i in range(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/emnist.zip'
    url = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

    if not os.path.isfile(filepath):
        print("Couldn't find EMNIST dataset in /tmp, downloading...")
        urllib.request.urlretrieve(url, filepath)
        with ZipFile('/tmp/emnist.zip', 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall('/tmp/emnist')
    
    train_num = 10000
    test_num = 3000
    train_images = load_emnist('/tmp/emnist/gzip/emnist-digits-train-images-idx3-ubyte.gz', num_images= train_num)
    train_labels = load_labels('/tmp/emnist/gzip/emnist-digits-train-labels-idx1-ubyte.gz', train_num)
    test_images = load_emnist('/tmp/emnist/gzip/emnist-digits-test-images-idx3-ubyte.gz', num_images=test_num)
    test_labels = load_labels('/tmp/emnist/gzip/emnist-digits-test-labels-idx1-ubyte.gz', test_num)

    return (
        emnist_generator((train_images, train_labels), batch_size, n_labelled),
        emnist_generator((test_images, test_labels), test_batch_size, n_labelled), 
        emnist_generator((test_images, test_labels), test_batch_size, n_labelled)
    )
    
def load_emnist(filepath, image_size=28, num_images=10000, flatten=False, grayscale=True):
    mnist_data = gzip.open(filepath, 'rb')
    mnist_data.seek(16) # skip over the first 16 bytes that correspond to the header
    buf = mnist_data.read(num_images * image_size * image_size)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    if flatten:
        data = data.reshape(num_images, image_size*image_size)
    else:
        data = data.reshape(num_images, image_size, image_size)
    if grayscale:
        data = data/254
    return data
    
def load_labels(labels_filepath, num=10000):
    mnist_data = gzip.open(labels_filepath, 'rb')
    mnist_data.seek(16) # skip over the first 16 bytes that correspond to the header
    buf = mnist_data.read(num)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int)
    return labels