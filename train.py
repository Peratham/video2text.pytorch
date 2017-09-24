#!/usr/bin/env python
from __future__ import print_function

import os
from six.moves import cPickle
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
#from builtins import range
import tensorflow as tf
from torch.autograd import Variable

import opts
from evaluate import evaluate
from misc.data import get_train_loader
from misc.models import *
from misc.utils import *
from misc.caption import *
sys.path.append('./coco-caption/')
from pycocotools.coco import COCO


def load_vocabulary(opt):
    # Load vocabulary
    vocab_pkl_path = os.path.join(opt.feat_dir, opt.ds + '_vocab.pkl')
    with open(vocab_pkl_path, 'rb') as f:
        vocab = cPickle.load(f)
    vocab_size = len(vocab)
    return vocab, vocab_size

def main(opt):
    opt, infos, prev_epoch = utils.history_infos(opt)
    tf_summary_writer = tf.summary.FileWriter(opt.checkpoint_path + '/' + opt.id)

    vocab, vocab_size = load_vocabulary(opt)

    model, crit, optimizer, infos = build_models(opt, vocab, infos, {'split': 'train'})

    # Initialize data loader
    train_loader = get_train_loader(opt, opt.train_caption_pkl_path, opt.feature_h5_path, opt.batch_size)
    total_step = len(train_loader)

    # Prepare ground-truth for evaluation
    reference_json_path = '{0}.json'.format(opt.test_reference_txt_path)
    reference = COCO(reference_json_path)

    # Start training model
    best_CIDEr = 0
    loss_count = 0.0
    for epoch in range(opt.num_epochs):
        epsilon = max(0.6, opt.ss_factor / (opt.ss_factor + np.exp(epoch / opt.ss_factor)))
        print('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
        add_summary_value(tf_summary_writer, 'epsilon', epsilon, epoch)
        scheduled_sampling(opt, epoch, model) # Assign the scheduled sampling prob
        model.train()
        for i, (videos, captions, cap_lens, video_ids) in enumerate(train_loader, start=1):
            # Construct mini batch Variable
            videos = Variable(videos).cuda()
            targets = Variable(captions).cuda()

            optimizer.zero_grad()
            outputs, _ = model(videos, targets)
            # As we may can not sample a entire batch after one epoch
            # We need to re-calcuate the batch size
            loss = crit(vocab_size, captions, cap_lens, outputs, targets)
            add_summary_value(tf_summary_writer, 'loss', loss.data[0], epoch * total_step + i)
            loss_count += loss.data[0]
            loss.backward()
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
             
            if i % 10 == 0 or len(captions) < opt.batch_size:
                tokens, _ = model.sample(videos)
                tokens = tokens.data[0].squeeze()
                we = model.decode_tokens(tokens)
                gt = model.decode_tokens(captions[0].squeeze())
                #print('----------      [vid:%d]' % video_ids[0])
                #print('----------      WE: %s\nGT: %s' % (we, gt))
                #print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, opt.num_epochs, i, total_step, loss_count, np.exp(loss_count)))
                loss_count /= 10.0 if len(captions) == opt.batch_size else i % 10
                print('E[%d/%d],I[%d/%d]|Loss:%.4f|Plx:%5.4f|P:%s|G:%s|' % (prev_epoch+epoch, opt.num_epochs, i, total_step, loss_count, np.exp(loss_count), we, gt))
                loss_count = 0.0

        # Calcuate the performance on validation set
        blockPrint()
        model.eval()
        print('Evaluating current checkpoint to disk ... ...')
        metrics = evaluate(opt, vocab, model, opt.test_range, opt.test_prediction_txt_path, reference)
        enablePrint()
	
        print('Saving current checkpoint to disk ... ...')	
        opt.decoder_pth_path = os.path.join(opt.checkpoint_path + '/' + opt.id, opt.id + '_decoder.pth')
        opt.optimizer_pth_path = os.path.join(opt.checkpoint_path + '/' + opt.id, opt.id + '_optimizer.pth')
        torch.save(model.state_dict(), opt.decoder_pth_path)
        torch.save(optimizer.state_dict(), opt.optimizer_pth_path)
        infos['epoch'] = epoch+prev_epoch
        #infos['opt'] = opt
        with open(os.path.join(opt.checkpoint_path + '/' + opt.id, opt.id + '_best_infos.pkl'), 'wb') as f:
            cPickle.dump(infos, f)
        for k, v in metrics.items():
            add_summary_value(tf_summary_writer, k, v, epoch)
            print('%s: %.6f' % (k, v))
            if not os.path.exists(opt.checkpoint_path + '/' + opt.id):
                os.makedirs(opt.checkpoint_path + '/' + opt.id)
            if k == 'CIDEr' and v > best_CIDEr:
                opt.best_decoder_pth_path = os.path.join(opt.checkpoint_path + '/' + opt.id, opt.id + '_best_decoder.pth')
                opt.best_optimizer_pth_path = os.path.join(opt.checkpoint_path + '/' + opt.id, opt.id + '_best_optimizer.pth')
                # Backup the best model on Val set
		print('Saving best score (CIDEr) checkpoint to disk ... ...')	
                torch.save(model.state_dict(), opt.best_decoder_pth_path)
                torch.save(optimizer.state_dict(), opt.best_optimizer_pth_path)
                with open(os.path.join(opt.checkpoint_path + '/' + opt.id, opt.id + '_infos.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                best_CIDEr = v

        model.train()


'''
Main function: Start from here !!!
'''
opt = opts.parse_opt()
print(opt)
main(opt)
