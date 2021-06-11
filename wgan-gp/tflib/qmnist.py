import pickle

import numpy as np

def qmnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)
    if limit is not None:
        print("WARNING ONLY FIRST {} QMNIST DIGITS".format(limit))
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
                
        image_batches = images.reshape(-1, batch_size, int(images.shape[1]*images.shape[2]))
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]), np.copy(labelled))

        else:
            for i in range(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]))

    return get_epoch

def load(datapath, batch_size, test_batch_size, n_labelled=None, dev_num = 10000, test_num=30000):    
    train_images, reserved_images, train_labels, reserved_labels = load_qmnist_images_labels(datapath)
        
    dev_images = reserved_images[0:dev_num]
    dev_labels = reserved_labels[0:dev_num]
    
    test_images = reserved_images[dev_num:dev_num+test_num]
    test_labels = reserved_labels[dev_num:dev_num+test_num]
    
    return (
        qmnist_generator((train_images, train_labels), batch_size, n_labelled),
        qmnist_generator((dev_images, dev_labels), test_batch_size, n_labelled), 
        qmnist_generator((test_images, test_labels), test_batch_size, n_labelled)
    )
    
def load_qmnist_images_labels(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        x_private = pickle_data['x_private']
        x_reserved = pickle_data['x_reserved']
        y_private = pickle_data['y_private']
        y_reserved = pickle_data['y_reserved']

    return x_private, x_reserved, y_private[:,0], y_reserved[:,0]

