import os
import numpy as np
import fnmatch
import PIL.Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

NCOLS = 5

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def load_model_from_checkpoint(checkpoint_dir, saver, sess):
    print(" [*] Reading checkpoints...", checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        variables_in_checkpoint = tf.train.list_variables(os.path.join(checkpoint_dir, ckpt_name))
        print("Variables found in checkpoint file",variables_in_checkpoint)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False


def get_filepaths_from_dir(data_dir, ext):
    '''
    return all the file paths with extension 'ext' in the given directory 'data_dir'
    :param data_dir: the data directory
    :param ext: the extension type
    :return:
        path_list: list of file paths
    '''
    pattern = '*.' + ext
    path_list = []
    for d, s, fList in os.walk(data_dir):
        for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
                path_list.append(os.path.join(d, filename))
    return sorted(path_list)


def read_image(filepath, resolution=64, cx=89, cy=121):
    '''
    read,crop and scale an image given the path
    :param filepath:  the path of the image file
    :param resolution: desired size of the output image
    :param cx: x_coordinate of the crop center
    :param cy: y_coordinate of the crop center
    :return:
        image in range [-1,1] with shape (resolution,resolution,3)
    '''

    img = np.asarray(PIL.Image.open(filepath))
    shape = img.shape

    if shape == (resolution, resolution, 3):
        pass
    else:
        img = img[cy - 64: cy + 64, cx - 64: cx + 64]
        resize_factor = 128 // resolution
        img = img.astype(np.float32)
        while resize_factor > 1:
            img = (img[0::2, 0::2, :] + img[0::2, 1::2, :] + img[1::2, 0::2, :] + img[1::2, 1::2, :]) * 0.25
            resize_factor -= 1
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    img = img.astype(np.float32) / 255.
    img = img * 2 - 1.
    return img


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y: y + img_h, x: x + img_w] = images[idx]
    return grid


def convert_to_pil_image(image, drange=[0, 1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]  # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0)  # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0, 255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, format)


def save_image(image, filename, drange=[0, 1], quality=95):
    img = convert_to_pil_image(image, drange)
    if '.jpg' in filename:
        img.save(filename, "JPEG", quality=quality, optimize=True)
    else:
        img.save(filename)


def save_image_grid(images, filename, drange=[0, 1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)

def save_files(save_dir, file_name_list, array_list):
    '''
    save a list of array with the given name
    :param save_dir: the directory for saving the files
    :param file_name_list: the list of the file names
    :param array_list: the list of arrays to be saved
    '''
    assert len(file_name_list) == len(array_list)

    for i in range(len(file_name_list)):
        np.save(os.path.join(save_dir, file_name_list[i]), array_list[i], allow_pickle=False)

def inverse_transform(imgs):
    '''
    normalize the image to be of range [0,1]
    :param imgs: input images
    :return:
        images with value range [0,1]
    '''
    imgs = (imgs + 1.) / 2.
    return imgs


def visualize_gt(imgs, save_dir):
    '''
    visualize the ground truth images and save
    :param imgs: input images with value range [-1,1]
    :param save_dir: directory for saving the results
    '''
    plt.figure(1)
    num_imgs = len(imgs)
    imgs = np.clip(inverse_transform(imgs), 0., 1.)
    NROWS = int(np.ceil(float(num_imgs) / float(NCOLS)))
    for i in range(num_imgs):
        plt.subplot(NROWS, NCOLS, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'input.png'))
    plt.close()


def visualize_progress(imgs, loss, save_dir, counter):
    '''
    visualize the optimization results and save
    :param imgs: input images with value range [-1,1]
    :param loss: the corresponding loss values
    :param save_dir: directory for saving the results
    :param counter: number of the function evaluation
    :return:
    '''
    plt.figure(2)
    num_imgs = len(imgs)
    imgs = np.clip(inverse_transform(imgs), 0., 1.)
    NROWS = int(np.ceil(float(num_imgs) / float(NCOLS)))
    for i in range(num_imgs):
        plt.subplot(NROWS, NCOLS, i + 1)
        plt.imshow(imgs[i])
        plt.title('loss: %.4f' % loss[i], fontdict={'fontsize': 8, 'color': 'blue'})
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'output_%d.png' % counter))
    plt.close()


def visualize_samples(img_r01, save_dir):
    plt.figure(figsize=(20, 20))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(img_r01[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'samples.png'))