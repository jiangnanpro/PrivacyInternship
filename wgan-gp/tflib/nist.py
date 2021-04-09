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
                
        image_batches = images.reshape(-1, batch_size, int(images.shape[1]*images.shape[2]))
        target_batches = targets.reshape(-1, batch_size)

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
    
    hsf = 4
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
        unpack_image = numpy.unpackbits(images[img,:]).reshape((128,128))*255
        unpack_image = numpy.abs(unpack_image-255) # change: background to black and digit to white 
        #cropped_image = crop_with_fixed_values(unpack_image)
        cropped_image = crop_with_bounding_box(unpack_image)
        if numpy.unique(cropped_image).shape[0]>1:
            final_image = debinarize_image(cropped_image)
            if resize:
                final_image = cv.resize(final_image, (resize_width,resize_height), interpolation=4)
            preprocessed_images.append(final_image)
    return numpy.array(preprocessed_images)
    
# Smooth the image to add non-binarity
def debinarize_image(image, kernel_size=(5,5), sigmaX=3, sigmaY=3):
    blur_image = cv.GaussianBlur(image, kernel_size, sigmaX=sigmaX, sigmaY=sigmaY, borderType = cv.BORDER_DEFAULT)
    return blur_image/numpy.max(blur_image)
    
def crop_with_fixed_values(img, left = 30, top = 30, right = 95, bottom = 95):
    return img[top:bottom, left:right]

def crop_with_bounding_box(black_white_image):
    cv.threshold(black_white_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU,black_white_image)
    contours, _ = cv.findContours(black_white_image, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    size = 0
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        # take largest bounding box 
        if h*w > size:
            size = h*w
            out_x = x
            out_y = y
            out_h = h
            out_w = w
    # Correct aspect-ratio for 1 digits
    epsilon = 0
    if (w/h) < 0.5:
        epsilon = (int(h/w)+(int((w/h)*15)))
    return black_white_image[out_y:out_y+out_h, out_x-epsilon:out_x+out_w+epsilon]