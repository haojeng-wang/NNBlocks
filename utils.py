import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
import random
from skimage import io
IMG_H = 144
IMG_W = 144
PAT_H = 112
PAT_W = 112
IMG_CHANNEL = 1
BATCH_SIZE = 64
num_epochs = 2000 
hist_bins=np.arange(0,256)

def globalQuant(img, lbnd, hbnd):
    [nx,ny]=img.shape
    img_data=img[:]
    hist=np.histogram(img_data[30:-30,30:-30],bins=hist_bins,normed=True)
    histcum=np.cumsum(hist[0])
    for k in hist_bins:
        if histcum[k]>lbnd:
            min_val=k
            break
    for k in hist_bins[-2:0:-1]:
        if histcum[k]< hbnd:
            max_val=k
            break
    img_GQ=np.minimum(max_val,np.maximum(min_val,img))
    
    img_GQ=(img_GQ-min_val)*255.0/(max_val-min_val)
    img_GQ=img_GQ.astype(np.uint8)
    return img_GQ

def cond(a, b, c, d):
    return tf.logical_or(tf.less(a + b, c),  tf.greater(a + b, d))

def body(a, b, c, d):
    # do some stuff with a, b
    a = tf.random_uniform([], 0, tf.to_int32(IMG_H-1) - PAT_H - 1, dtype=tf.int32)
    b = tf.random_uniform([], 0, tf.to_int32(IMG_W-1) - PAT_W - 1, dtype=tf.int32)
    
    return a, b, c, d
#%%
def input_parser_pairImgPatch(img_path, img_path_2, img_path_3):
    # convert the label to one-hot encoding
#    one_hot = tf.one_hot(label, NUM_CLASSES)
    #random.uniform(0.95, 1.05)*residual_sn
    
    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=None)
    img_file_2 = tf.read_file(img_path_2)
    img_decoded_2 = tf.image.decode_image(img_file_2, channels=None)
    img_file_3 = tf.read_file(img_path_3)
    img_decoded_3 = tf.image.decode_image(img_file_3, channels=None)
    
    img_decoded.set_shape([IMG_H,IMG_W,IMG_CHANNEL]) # Face
    img_decoded_2.set_shape([IMG_H,IMG_W,IMG_CHANNEL]) # Face
    img_decoded_3.set_shape([IMG_H,IMG_W,IMG_CHANNEL]) # Face
    
    ratio = tf.random_uniform([], 0.95, 1.05, dtype=tf.float32)
    img_decoded   = tf.image.resize_images(img_decoded  , (tf.to_int32(IMG_H*ratio), tf.to_int32(IMG_W*ratio)), method=2)
    img_decoded_2 = tf.image.resize_images(img_decoded_2, (tf.to_int32(IMG_H*ratio), tf.to_int32(IMG_W*ratio)), method=2)    #img_decoded.set_shape([IMG_H,IMG_W,IMG_CHANNEL]) # Face
    #img_decoded_3 = tf.image.resize_images(img_decoded_3, (tf.to_int32(IMG_H*ratio), tf.to_int32(IMG_W*ratio)), method=2)    #img_decoded.set_shape([IMG_H,IMG_W,IMG_CHANNEL]) # Face
    
    
    offset_height = tf.random_uniform([], 0, tf.to_int32(IMG_H*ratio) - 1 - PAT_H - 1, dtype=tf.int32)
    offset_width = tf.random_uniform([], 0, tf.to_int32(IMG_W*ratio)  - 1 - PAT_W - 1, dtype=tf.int32)
    
    img_decoded = tf.cast(img_decoded, tf.float32)
    img_decoded = tf.image.crop_to_bounding_box(img_decoded, offset_height, offset_width, PAT_H, PAT_W)
    img_decoded = img_decoded / 255.0
    

    
    #img_decoded_2.set_shape([IMG_H,IMG_W,IMG_CHANNEL]) # Face
    
    img_decoded_2 = tf.cast(img_decoded_2, tf.float32)
    img_decoded_2 = tf.image.crop_to_bounding_box(img_decoded_2, offset_height, offset_width, PAT_H, PAT_W)
    img_decoded_2 = img_decoded_2 / 255.0
   
    residual_sn = img_decoded_2 - img_decoded
 
    offset_height_2 = tf.random_uniform([], 0, tf.to_int32(IMG_H-1) - 1 - PAT_H - 1, dtype=tf.int32)
    offset_width_2 = tf.random_uniform([], 0, tf.to_int32(IMG_W-1)  - 1 - PAT_W - 1, dtype=tf.int32)
    comp_1 = tf.to_int32(0.1 * (IMG_H +IMG_W))
    comp_2 = tf.to_int32(0.2 * (IMG_H +IMG_W))
    
    '''     
    offset_height_2 = np.random.uniform() * (round(IMG_H*ratio-0.5) - PAT_H - 1)
    offset_width_2  = np.random.uniform() * (round(IMG_W*ratio-0.5) - PAT_W - 1)
    comp = offset_height_2 + offset_width_2
    comp1 = int(0.3 * (IMG_H +IMG_W) * ratio)
    comp2 = int(0.7 * (IMG_H +IMG_W) * ratio)
    #%%
    while((comp < comp1) or (comp > comp2)):
        offset_height_2 = np.random.uniform() * (round(IMG_H*ratio-0.5) - PAT_H - 1)
        offset_width_2  = np.random.uniform() * (round(IMG_W*ratio-0.5) - PAT_W - 1)
        comp = offset_height_2 + offset_width_2
    #%%
    
    offset_height_2 = tf.cast(offset_height_2, tf.int32)
    offset_width_2 = tf.cast(offset_width_2, tf.int32)
    '''
    
    offset_height_2, offset_width_2, comp_1, comp_2 = tf.while_loop(cond, body, [offset_height_2, offset_width_2, comp_1, comp_2])
    
    img_decoded_3 = tf.cast(img_decoded_3, tf.float32)
    img_decoded_3 = tf.image.crop_to_bounding_box(img_decoded_3, offset_height_2, offset_width_2, PAT_H, PAT_W)
    img_decoded_3 = img_decoded_3 / 255.0

    img_decoded_2 = tf.random_uniform([], 0.5, 2.0, tf.float32) * residual_sn + img_decoded_3

    #img_decoded_2 = tf.clip_by_value(img_decoded_2, -127.5/128, 127.5/128)

    return img_decoded_3, img_decoded_2


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


class train_data():
    def __init__(self, filepath='./data/image_clean_pat.npy'):
        print('train_data class obj init!')
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath='./data/image_clean_pat.npy'):
    return train_data(filepath=filepath)


def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    #im.save(filepath, 'bmp')
    io.imsave(filepath, im)


def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))
