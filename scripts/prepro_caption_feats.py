"""
Prepare text-related data sets, including:
1. splits the data set
2. transform the captions to tokens
3. prepare ground-truth
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import json
from six.moves import cPickle
from collections import Counter
import nltk
import torch
sys.path.append('/home/jxgu/github/TRECVID_VTT')
from misc.utils import create_reference_json, build_msvd_annotation
sys.path.append('../')
import opts

class Vocabulary(object):

    def __init__(self, opt):
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, w):
        # Add the new word to the vocabulary list
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.idx2word.append(w)
            self.nwords += 1

    def __call__(self, w):
        # Return the corresponding ID of each word
        if w not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[w]

    def __len__(self):
        # Get the number of words in vocabulary
        return self.nwords


def prepare_vocab(opt, sentences):
    '''
    Build the vocabulary according to the annotated captions.
    Drop those words which frequency is below the threshold
    '''
    counter = Counter()
    ncaptions = len(sentences)
    for i, row in enumerate(sentences):
        caption = row['caption']
        # Split the words with space
        # tokens = caption.lower().split(' ')
        # Split the words with nltk
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if i % 10000 == 0:
            print('[{}/{}] tokenized the captions.'.format(i, ncaptions))

    # Drop the low-frequency words
    threshold = 3
    words = [w for w, c in counter.items() if c >= threshold]
    # Start building the vocab
    vocab = Vocabulary(opt)
    for w in words:
        vocab.add_word(w)

    print('Vocabulary has %d words.' % len(vocab))
    with open(opt.vocab_pkl_path, 'wb') as f:
        cPickle.dump(vocab, f)
    print('Save vocabulary to %s' % opt.vocab_pkl_path)
    return vocab


def prepare_split(opt):
    '''
    Generate train, val, and test splits
    For MSVD dataset, we can split based on Vsubhashini's setting:
    train:1-1200, val:1201-1300, test:1301-1970
    '''
    split_dict = {}

    for i in range(*(opt.train_range)):
        split_dict[i] = 'train'
    for i in range(*(opt.val_range)):
        split_dict[i] = 'val'
    for i in range(*(opt.test_range)):
        split_dict[i] = 'test'

    # pprint.pprint(split_dict)

    return split_dict


def prepare_caption(opt, vocab, split_dict, anno_data):
    '''
    Transform captions to token index representation, and save to pickle
    Read json file with saved caption annotations
    Save each caption and its corresponding video ID
    Put back the caption word_id list and video_id list
    '''
    # Initialize dictionary
    captions = {'train': [], 'val': [], 'test': []}
    lengths = {'train': [], 'val': [], 'test': []}
    video_ids = {'train': [], 'val': [], 'test': []}

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for row in anno_data:
        caption = row['caption'].lower()
        words = nltk.tokenize.word_tokenize(caption)
        nw = len(words)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in xrange(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    count = 0
    for row in anno_data:
        caption = row['caption'].lower()
        video_id = int(row['video_id'][5:])
        if video_id in split_dict:
            split = split_dict[video_id]
        else:
            # If video_id does not exists in split_dict.
            # we regard it as 'test'
            split = 'test'
        words = nltk.tokenize.word_tokenize(caption)
        l = len(words) + 1  # Add the <end> token
        lengths[split].append(l)
        if l > opt.num_words:
            # We truncate the caption if its length exceeds the max length
            words = words[:opt.num_words - 1]  # The last token is for <end>
            count += 1
        # Transform captions to word_id representation
        tokens = []
        for word in words:
            tokens.append(vocab(word))
        tokens.append(vocab('<end>'))
        while l < opt.num_words:
            # We pad with <pad>(0) if less than threshold length
            tokens.append(vocab('<pad>'))
            l += 1
        captions[split].append(torch.LongTensor(tokens))
        video_ids[split].append(video_id)

    # Count how man captions exceed the maximum length
    print('There are %.3f%% too long captions' % (100 * float(count) / len(anno_data)))

    # Save the three splits: trian, val, and test respectively
    with open(opt.train_caption_pkl_path, 'wb') as f:
        cPickle.dump([captions['train'], lengths['train'], video_ids['train']], f)
        print('Save %d train captions to %s.' % (len(captions['train']),
                                                 opt.train_caption_pkl_path))
    with open(opt.val_caption_pkl_path, 'wb') as f:
        cPickle.dump([captions['val'], lengths['val'], video_ids['val']], f)
        print('Save %d val captions to %s.' % (len(captions['val']),
                                               opt.val_caption_pkl_path))
    with open(opt.test_caption_pkl_path, 'wb') as f:
        cPickle.dump([captions['test'], lengths['test'], video_ids['test']], f)
        print('Save %d test captions to %s.' % (len(captions['test']),
                                                opt.test_caption_pkl_path))


def prepare_gt(opt, anno_data):
    # Prepare the ground-truth
    print('Preparing ground-truth...')
    val_reference_txt = open(opt.val_reference_txt_path, 'w')
    test_reference_txt = open(opt.test_reference_txt_path, 'w')

    val_selected_range = range(*(opt.val_range))
    test_selected_range = range(*(opt.test_range))
    error_count = 0

    for row in anno_data:
        caption = row['caption'].lower()
        video_id = int(row['video_id'][5:])
        gt = '%d\t%s\n' % (video_id, caption)
        try:
            if video_id in val_selected_range:
                val_reference_txt.write(gt)
            elif video_id in test_selected_range:
                test_reference_txt.write(gt)
        except Exception as e:
            print(e)
            print(gt)
            error_count += 1

    if error_count > 0:
        print('Error count: %d\t' % error_count, end='')

    val_reference_txt.close()
    test_reference_txt.close()

    create_reference_json(opt.val_reference_txt_path)
    create_reference_json(opt.test_reference_txt_path)
    print('done!')


if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.ds == 'msvd':
        # Generate MSVD dataset annotation according to the
        # format of MSR-VTT dataset
        print('# Build MSVD dataset annotations:')
        if os.path.isfile('datasets/MSVD/annotations.json') == False:
            build_msvd_annotation(opt)

    # Read json file
    with open(opt.anno_json_path, 'r') as f:
        anno_json = json.load(f)
    anno_data = anno_json['sentences']

    print('\n# Build vocabulary')
    vocab = prepare_vocab(opt, anno_data)

    print('\n# Prepare dataset split')
    split_dict = prepare_split(opt)

    print('\n# Convert each caption to token index list')
    prepare_caption(opt, vocab, split_dict, anno_data)

    print('\n# Prepare ground-truth')
    prepare_gt(opt, anno_data)
