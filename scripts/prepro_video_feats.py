'''
Load saliency feature map, extracted vgg16
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import h5py
import numpy as np
import skimage
import torch
from torch.autograd import Variable
from misc.model import EncoderCNN
import opts

def sample_frames(opt, video_path, train=True):
    # Video sampling
    # To reduce the computation, we extract num_frames/video at regular intervals
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frame_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame)
        frame_count += 1

    indices = np.linspace(0, frame_count, opt.num_frames, endpoint=False, dtype=int)
    frame_list = np.array(frame_list)[indices]
    return frame_list, frame_count


def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # Copy one channel gray image three times to from RGB image
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_ubyte(image).astype(np.float32)
    # Sub the mean value (RGB) on ILSVRC dataset
    image -= np.array([103.939, 116.779, 123.68])
    # Transfer BGR image to RGB image, as the input to caffe model is RGB format
    image = image[:, :, ::-1]
    return image


def extract_full_feature(opt, encoder):
    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)

    # Create hdf5 file to save video frame features
    if os.path.exists(opt.feature_h5_path):
        # If hdf5 file already exists, means it is processed or half-processed
        # Read with r+ (read and write) mode, for fear that overwrite the save data
        h5 = h5py.File(opt.feature_h5_path, 'r+')
        dataset_feats = h5[opt.feature_h5_feats]
        dataset_lens = h5[opt.feature_h5_lens]
    else:
        h5 = h5py.File(opt.feature_h5_path, 'w')
        dataset_feats = h5.create_dataset(opt.feature_h5_feats,
                                          (nvideos, opt.num_frames, opt.frame_feat_size),
                                          dtype='float32')
        dataset_lens = h5.create_dataset(opt.feature_h5_lens, (nvideos,), dtype='int')

    for i, video in enumerate(videos):
        print(video, end=' ')
        video_path = os.path.join(opt.video_root, video)
        # Sample video frames
        frame_list, frame_count = sample_frames(opt, video_path, train=True)
        print(frame_count)

        # Process the image, transform them to (batch, channel, height, width) format
        cropped_frame_list = np.array([preprocess_frame(x) for x in frame_list])
        cropped_frame_list = cropped_frame_list.transpose((0, 3, 1, 2))
        cropped_frame_list = Variable(torch.from_numpy(cropped_frame_list),
                                      volatile=True).cuda()

        # The feature shape of each video is num_frames x 4096
        # If the frame number if less than num_frames, then pad with 0
        feats = np.zeros((opt.num_frames, opt.frame_feat_size), dtype='float32')
        feats[:frame_count, :] = encoder(cropped_frame_list).data.cpu().numpy()
        dataset_feats[i] = feats
        dataset_lens[i] = frame_count


def main(opt):
    encoder = EncoderCNN(opt)
    encoder.eval()
    encoder.cuda()

    extract_full_feature(opt, encoder)


if __name__ == '__main__':
    opt = opts.parse_opt()
    main(opt)
