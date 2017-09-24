'''
Generate predict results on the given dataset split, and calcuate the scores.
'''

from __future__ import absolute_import
from __future__ import unicode_literals
from tqdm import tqdm
import pickle
import sys
import opts
import torch
from torch.autograd import Variable
from misc.data import get_eval_loader
from misc.models import *
from misc.utils import CocoResFormat

sys.path.append('./coco-caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def measure(prediction_txt_path, reference):
    # Transform predicted results(txt format) to the format required by evaluation function
    prediction_json_path = prediction_txt_path.replace('txt', 'json')
    crf = CocoResFormat()
    crf.read_file(prediction_txt_path, True)
    # Remove it first !!!
    os.remove(prediction_json_path)
    crf.dump_json(prediction_json_path)
    # crf.res is the transformed predict results
    cocoRes = reference.loadRes(prediction_json_path)
    #cocoRes = reference.loadRes(crf.res)
    cocoEval = COCOEvalCap(reference, cocoRes)

    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))
    return cocoEval.eval


def evaluate(opt, vocab, decoder, eval_range, prediction_txt_path, reference):
    # Load test dataset
    eval_loader = get_eval_loader(opt, eval_range, opt.feature_h5_path)

    result = {}
    eval_len = len(eval_loader.dataset.eval_list)
    with tqdm(total=eval_len) as pbar:
        for i, (videos, video_ids) in enumerate(eval_loader):
            pbar.update(1)
            # Create mini batch Variable
            videos = Variable(videos)

            if opt.use_cuda:
                videos = videos.cuda()

            outputs, attens = decoder.sample(videos)
            #outputs = outputs.data.squeeze(2)
            outputs = outputs.data
            for (tokens, vid) in zip(outputs, video_ids):
                s = decoder.decode_tokens(tokens)
                result[vid] = s
    
    # Remove it first !!!
    os.remove(prediction_txt_path)
    prediction_txt = open(prediction_txt_path, 'w')
    for vid, s in result.items():
        prediction_txt.write('%d\t%s\n' % (vid, s))  # Note that the video name of MSVD is start from 1
    print('Writing predictions to file ... ...')
    prediction_txt.close()

    print('Calculating scores for the generated results ... ...')
    metrics = measure(prediction_txt_path, reference)
    return metrics


if __name__ == '__main__':
    opt = opts.parse_opt()

    opt, infos, iteration, epoch = utils.history_infos(opt)
    with open(opt.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load pre-trained model
    model, crit, _, infos = build_models(opt, vocab, infos)
    reference_json_path = '{0}.json'.format(opt.test_reference_txt_path)
    reference = COCO(reference_json_path)
    evaluate(opt, vocab, model, opt.test_range, opt.test_prediction_txt_path, reference)
