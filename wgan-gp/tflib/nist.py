import os

import numpy as np

from tflib.utils import debinarize_image, resize_image, crop_with_bounding_box

def nist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)
    if limit is not None:
        print("WARNING ONLY FIRST {} NIST DIGITS".format(limit))
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

def load(datapath, batch_size, test_batch_size, n_labelled=None, hsf=4):
    
    train_num = 30000
    test_num = 10000
    
    with open(os.path.join(datapath, 'HSF_'+str(hsf)+'_images.npy'),'rb') as f:
        images = load_nist_images(np.load(f))
    with open(os.path.join(datapath,'HSF_'+str(hsf)+'_labels.npy'),'rb') as f:
        labels = np.load(f)
    
    train_images = images[:train_num]
    train_labels = images[:train_num]
    
    dev_images = images[train_num:train_num+test_num]
    dev_labels = images[train_num:train_num+test_num]
    
    test_images = images[train_num+test_num:train_num+2*test_num]
    test_labels = images[train_num+test_num:train_num+2*test_num]
    
    return (
        nist_generator((train_images, train_labels), batch_size, n_labelled),
        nist_generator((dev_images, dev_labels), test_batch_size, n_labelled), 
        nist_generator((test_images, test_labels), test_batch_size, n_labelled)
    )
    
def load_nist_images(images, num_images=None, resize=True, resize_width=28, resize_height=28):
    preprocessed_images=[]
    if num_images is None:
        num_images = images.shape[0]
    for img in range(num_images):
        unpack_image = np.unpackbits(images[img,:]).reshape((128,128)).astype('int16')*255
        unpack_image = np.abs(unpack_image-255).astype('uint8') # change: background to black and digit to white 
        cropped_image = crop_with_bounding_box(unpack_image,5)
        if np.unique(cropped_image).shape[0]>1:
            final_image = np.clip(debinarize_image(cropped_image),0,1)
            final_image = (final_image*255).astype('uint8')
            if resize:
                final_image = resize_image(final_image, resize_width, resize_height, interpolation=4)
            preprocessed_images.append(final_image.astype('uint8'))
    return np.array(preprocessed_images)
    
