import numpy

import os
import urllib.request
import gzip
import pickle
import cv2 as cv

def nist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print("WARNING ONLY FIRST {} NIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)
        
        #print(images.shape)
        #print(targets.shape)
        
        image_batches = images.reshape(-1, batch_size, int(images.shape[1]*images.shape[2]))
        target_batches = targets.reshape(-1, batch_size)
        
        #print(image_batches.shape)
        #print(target_batches.shape)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:
            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(datapath, batch_size, test_batch_size, n_labelled=None):
    
    train_num = 30000
    test_num = 10000
    
    hsf = 0
    with open(os.path.join(datapath, 'HSF_'+str(hsf)+'_images.npy'),'rb') as f:
        images = load_nist_images(numpy.load(f))
    with open(os.path.join(datapath,'HSF_'+str(hsf)+'_labels.npy'),'rb') as f:
        labels = numpy.load(f)
    
    train_images = images[:train_num]
    train_labels = images[:train_num]
    
    dev_images = images[train_num:train_num+test_num]
    dev_labels = images[train_num:train_num+test_num]
    
    test_images = images[train_num+test_num:train_num+2*test_num]
    test_labels = images[train_num+test_num:train_num+2*test_num]
    
    '''
    hsf = 0
    with open(os.path.join(datapath, 'HSF_'+str(hsf)+'_images.npy'),'rb') as f:
        train_images = numpy.load(f)
    train_images = load_nist_images(train_images, train_num)
    with open(os.path.join(datapath,'HSF_'+str(hsf)+'_labels.npy'),'rb') as f:
        train_labels = numpy.load(f)[:train_num]
   
   
    hsf = 1
    with open(os.path.join(datapath, 'HSF_'+str(hsf)+'_images.npy'),'rb') as f:
        test_images = numpy.load(f)
    test_images = load_nist_images(test_images, test_num)
    with open(os.path.join(datapath,'HSF_'+str(hsf)+'_labels.npy'),'rb') as f:
        test_labels = numpy.load(f)[:test_num]
        
    hsf = 2
    with open(os.path.join(datapath, 'HSF_'+str(hsf)+'_images.npy'),'rb') as f:
        dev_images = numpy.load(f)
    dev_images = load_nist_images(dev_images, test_num)
    with open(os.path.join(datapath,'HSF_'+str(hsf)+'_labels.npy'),'rb') as f:
        dev_labels = numpy.load(f)[:test_num]
    '''
    
    return (
        nist_generator((train_images, train_labels), batch_size, n_labelled),
        nist_generator((dev_images, dev_labels), test_batch_size, n_labelled), 
        nist_generator((test_images, test_labels), test_batch_size, n_labelled)
    )
    
def load_nist_images(images, num_images=None, resize=False, resize_width=28, resize_height=28):
    preprocessed_images=[]
    if num_images is None:
        num_images = images.shape[0]
    for img in range(num_images):
        unpack_image = numpy.unpackbits(images[img,:]).reshape((128,128))
        cropped_image = crop_image(unpack_image)
        final_image = debinarize_image(cropped_image)
        if resize:
            final_image = cv.resize(final_image, (resize_width,resize_height), interpolation=4)
        preprocessed_images.append(final_image)
    return numpy.array(preprocessed_images)
    
# Smooth the image to add non-binarity
def debinarize_image(image, kernel_size=(5,5), sigmaX=3, sigmaY=3):
    blur_image = cv.GaussianBlur(image, kernel_size, sigmaX=sigmaX, sigmaY=sigmaY, borderType = cv.BORDER_DEFAULT)
    return blur_image/numpy.max(blur_image)
    
def crop_image(img, left = 30, top = 30, right = 95, bottom = 95):
    return img[top:bottom, left:right]