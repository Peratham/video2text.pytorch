#!/usr/bin/env python
# coding: utf-8
'''
Get the caption and attention vector of one video
'''

from __future__ import absolute_import
from __future__ import unicode_literals

import os
import pickle
import sys
import opts
import cv2
import h5py
import numpy as np
import torch
from misc import saliency as psal
from misc.model import DecoderRNN


def open_video(opt, video_path):
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
        # cv2.imshow('Video', frame)
        # cv2.waitKey(30)
        frame_list.append(frame)
        frame_count += 1
    indices = np.linspace(0, frame_count, opt.num_frames, endpoint=False, dtype=int)
    frame_list = np.array(frame_list)[indices]
    return frame_list


def sample(opt, vocab, video_feat, decoder, video_path, vid):
    # Create visualation directory for each video
    img_dir = os.path.join(opt.visual_dir, str(vid))
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    frame_list = open_video(video_path)
    if opt.use_cuda:
        video_feat = video_feat.cuda()
    video_feat = video_feat.unsqueeze(0)
    outputs, attens = decoder.sample(video_feat)
    words = []
    for i, token in enumerate(outputs.data.squeeze()):
        if token == vocab('<end>'):
            break
        word = vocab.idx2word[token]
        print(word)
        words.append(word)
        v, k = torch.topk(attens[i], 5)
        # pair = zip(v.data[0], k.data[0])
        # print(pair)
        selected_id = k.data[0][0]
        selected_frame = frame_list[selected_id]
        cv2.imshow('Attend', selected_frame)
        cv2.imwrite(os.path.join(img_dir, '%d_%d_%s.jpg' % (i, selected_id,
                                                            word)), selected_frame)

        # Calcuate the saliency map
        sal = psal.get_saliency_rbd(selected_frame).astype('uint8')
        cv2.imwrite(os.path.join(img_dir, '%d_%d_%s.jpg' % (i, selected_id,
                                                            'saliency')), sal)

        binary_sal = psal.binarise_saliency_map(sal, method='adaptive')
        I = binary_sal[:, :, np.newaxis]
        binary_mask = np.concatenate((I, I, I), axis=2)
        foreground_img = np.multiply(selected_frame, binary_mask).astype('uint8')
        cv2.imwrite(os.path.join(img_dir, '%d_%d_%s.jpg' % (i, selected_id,
                                                            'foreground')), foreground_img)

        k = cv2.waitKey(500)
        if k == ord('n'):
            return
    caption = ' '.join(words)
    print(caption)


if __name__ == '__main__':
    opt = opts.parse_opt()
    with open(opt.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    features = h5py.File(opt.feature_h5_path, 'r')[opt.feature_h5_feats]

    # Load the trained model
    decoder = DecoderRNN(opt.frame_size, opt.projected_size, opt.hidden_size,
                         opt.num_frames, opt.num_words, vocab)
    decoder.load_state_dict(torch.load(opt.best_decoder_pth_path))
    decoder.cuda()
    decoder.eval()

    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)

    if len(sys.argv) > 1:
        vid = int(sys.argv[1])
        video_path = os.path.join(opt.video_root, videos[vid])
        video_feat = torch.autograd.Variable(torch.from_numpy(features[vid]))
        sample(vocab, video_feat, decoder, video_path, vid)
    else:
        # selected_videos = [1412, 1420, 1425, 1466, 1484, 1554, 1821, 1830, 1841,
        #                    1848, 1849, 1850, 1882, 1884, 1931, 1934, 1937, 1944,
        #                    1949, 1950, 1951, 1962]
        # for vid in selected_videos:
        for vid in range(100):
            print(vid)
            video_path = os.path.join(opt.video_root, videos[vid])
            video_feat = torch.autograd.Variable(torch.from_numpy(features[vid]))
            sample(vocab, video_feat, decoder, video_path, vid)
