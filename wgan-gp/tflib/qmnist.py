import pickle

import numpy as np
from sklearn.decomposition import PCA

from tflib.utils import load_pretrained_model, grey2RGB, resize_image

def qmnist_generator(data, batch_size, n_labelled, limit=None, tabular=False):
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
        if tabular:
            image_batches = images.reshape(-1, batch_size, int(images.shape[1]))
        else:
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
    train_images, evaluator_images, train_labels, evaluator_labels = load_qmnist_images_labels(datapath)
        
    dev_images = evaluator_images[0:dev_num]
    dev_labels = evaluator_labels[0:dev_num]
    
    test_images = evaluator_images[dev_num:dev_num+test_num]
    test_labels = evaluator_labels[dev_num:dev_num+test_num]
    
    return (
        qmnist_generator((train_images/255, train_labels), batch_size, n_labelled),
        qmnist_generator((dev_images/255, dev_labels), test_batch_size, n_labelled), 
        qmnist_generator((test_images/255, test_labels), test_batch_size, n_labelled)
    )

def load_tabular(datapath, batch_size, test_batch_size, preprocessing='vgg19', pca_ncomp=None, n_labelled=None, dev_num = 10000, test_num=30000):    
    train_images, evaluator_images, train_labels, evaluator_labels = load_qmnist_images_labels(datapath)
    resize_height = 32
    resize_width = 32
    train_images = transform_qmnist(train_images/255, resize_width, resize_height)
        
    dev_images = transform_qmnist(evaluator_images[0:dev_num]/255, resize_width, resize_height)
    dev_labels = evaluator_labels[0:dev_num]
    
    test_images = transform_qmnist(evaluator_images[dev_num:dev_num+test_num]/255, resize_width, resize_height)
    test_labels = evaluator_labels[dev_num:dev_num+test_num]

    model = load_pretrained_model(preprocessing)
    n_features = model.output_shape[-1]

    train_images_tabular = model.predict(train_images).reshape(train_images.shape[0], n_features)
    dev_images_tabular = model.predict(dev_images).reshape(dev_images.shape[0], n_features)
    test_images_tabular = model.predict(test_images).reshape(test_images.shape[0], n_features)

    if pca_ncomp is not None:
        pca = PCA(n_components=pca_ncomp)
        train_images_tabular = pca.fit_transform(train_images_tabular)
        dev_images_tabular = pca.transform(dev_images_tabular)
        test_images_tabular = pca.transform(test_images_tabular)
        return (pca,
            qmnist_generator((train_images_tabular, train_labels), batch_size, n_labelled, tabular=True),
            qmnist_generator((dev_images_tabular, dev_labels), test_batch_size, n_labelled, tabular=True), 
            qmnist_generator((test_images_tabular, test_labels), test_batch_size, n_labelled, tabular=True)
        )
    else:    
        return(
            qmnist_generator((train_images_tabular, train_labels), batch_size, n_labelled, tabular=True),
            qmnist_generator((dev_images_tabular, dev_labels), test_batch_size, n_labelled, tabular=True), 
            qmnist_generator((test_images_tabular, test_labels), test_batch_size, n_labelled, tabular=True)
        )

def transform_qmnist(images, resized_width=32, resized_height=32):
    transform_images = []
    for image in images:
        image = grey2RGB(resize_image(image, resized_width, resized_height))
        transform_images.append(image)
    return np.array(transform_images)


def load_qmnist_attacker_evaluation_set(pickle_file, pos_size=1000, neg_size=10000):
    x_defender, x_evaluator, y_defender, y_evaluator = load_qmnist_images_labels(pickle_file)

    rng = np.random.RandomState(2021)
    pos_index_seq = rng.choice(range(len(x_defender)), size=pos_size, replace=False)
    neg_index_seq = rng.choice(range(len(x_evaluator)), size=neg_size, replace=False)

    pos_images = x_defender[pos_index_seq]
    pos_labels = y_defender[pos_index_seq]
    neg_images = x_evaluator[neg_index_seq]
    neg_labels = y_evaluator[neg_index_seq]
    return pos_images, neg_images, pos_labels, neg_labels
    
def load_qmnist_images_labels(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        x_defender = pickle_data['x_defender']
        x_evaluator = pickle_data['x_evaluator']
        y_defender = pickle_data['y_defender']
        y_evaluator = pickle_data['y_evaluator']

    return x_defender, x_evaluator, y_defender[:,0], y_evaluator[:,0]

def load_qmnist_attacker_images_labels(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        x_attacker = pickle_data['x_attacker']
        y_attacker = pickle_data['y_attacker']
    return x_attacker, y_attacker[:,0]

def load_qmnist_images_labels2(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        x_defender = pickle_data['x_defender']
        x_evaluator = pickle_data['x_reserved']
        y_defender = pickle_data['y_defender']
        y_evaluator = pickle_data['y_reserved']

    return x_defender, x_evaluator, y_defender[:,0], y_evaluator[:,0]
