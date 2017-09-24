import itertools
import os

import torch.optim as optim

from misc.cnn.EncoderCNN import *
from misc.lm.LangugeModel import *
from misc.Criterion import *

def build_lm(opt, vocab, split):
    opt.n_gpus = getattr(opt, 'n_gpus', 1)

    print('Loading with LanguageModel...')
    lm_model = LanguageModel(opt, vocab)
    if opt.n_gpus>1:
        print('Construct multi-gpu model ...')
        model = nn.DataParallel(lm_model, device_ids=opt.gpus, dim=0)
    else:
        model = lm_model
    crit = LanguageModelCriterion(opt)
    # Use GPU
    if opt.use_cuda:
        model.cuda()
        crit.cuda()
    else:
        raise AssertionError('Hey, get a GPU!!!')


    # check compatibility if training is continued from previously saved model
    print('Load the checkpoints')
    if len(opt.start_from) != 0:
        print("Load model from {}".format(opt.start_from))
        opt.id = os.path.basename(opt.start_from)
        decoder_ckp_path = os.path.join(opt.start_from, os.path.basename(opt.start_from) + '_best_decoder.pth')
        if os.path.isfile(decoder_ckp_path):
            model.load_state_dict(torch.load(decoder_ckp_path))

    if split == 'train':
        model.train()  # Assure in training mode
        model_parameters = itertools.ifilter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(model_parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)
        # Load the optimizer
        if len(opt.start_from) != 0:
            optim_ckp_path = os.path.join(opt.start_from, os.path.basename(opt.start_from) + '_best_optimizer.pth')
            if os.path.isfile(optim_ckp_path):
                optimizer.load_state_dict(torch.load(optim_ckp_path))
    else:
        model.eval()  # Assure in testing or validation mode
        optimizer = None
    return model, crit, optimizer

def build_models(opt, vocab, infos, model_kwargs):
    split = model_kwargs.get('split', 'train')
    model, crit, optimizer = build_lm(opt, vocab, split)
    
    # Pring training parameter
    print('Learning rate: %.4f' % opt.learning_rate)
    print('Batch size: %d' % opt.batch_size)

    return model, crit, optimizer, infos
