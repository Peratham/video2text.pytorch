import argparse
import datetime
import os
import time

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='debug')
    parser.add_argument('--start_from', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='save')

    parser.add_argument('--cnn_model', type=str, default='vgg16')  # vgg16, resnet101
    parser.add_argument('--cnn_weight', type=str, default='model/pytorch-resnet/resnet101.pth')
    # Training setting
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--ss_factor', type=int, default=24)
    parser.add_argument('--optim', type=str, default='adam') # rmsprop|sgd|sgdmom|adagrad|adam
    parser.add_argument('--learning_rate', type=float, default=4e-4) # 'learning rate'
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1)#at what iteration to start decaying learning rate? (-1 = dont) (in epoch)
    parser.add_argument('--learning_rate_decay_every', type=int, default=3) #every how many iterations thereafter to drop LR?(in epoch)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8) #every how many iterations thereafter to drop LR?(in epoch)
    parser.add_argument('--optim_alpha', type=float, default=0.8) #alpha for adam
    parser.add_argument('--optim_beta', type=float, default=0.999) #beta used for adam
    parser.add_argument('--optim_epsilon', type=float, default=1e-8) #epsilon that goes into denominator for smoothing
    parser.add_argument('--weight_decay', type=float, default=0) #weight_decay
    parser.add_argument('--grad_clip', type=float, default=0.1) #clip gradients at this value
    parser.add_argument('--drop_prob_lm', type=float, default=0.5) #strength of dropout in the Language Model RNN
    # SS
    parser.add_argument('--scheduled_sampling_start', type=int, default=0)#at what iteration to start decay gt probability
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5)#every how many iterations thereafter to gt probability
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05)#How much to update the prob
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.35) #Maximum scheduled sampling prob

    # Hyper-parameters
    parser.add_argument('--projected_size', type=int, default=512)  # 'image attention'
    parser.add_argument('--hidden_size', type=int, default=512)  # LSTM hidden size
    parser.add_argument('--frame_size', type=int, default=224)   # Frame size
    parser.add_argument('--frame_feat_size', type=int, default=4096) # Frame feature
    parser.add_argument('--frame_sample_rate', type=int, default=10)  # Video sample rate
    parser.add_argument('--num_frames', type=int, default=20)  # Number of frames
    parser.add_argument('--num_words', type=int, default=10)  # Number of words length

    parser.add_argument('--ds', type=str, default='msvd') #msr-vtt, msvd
    # MSR-VTT setting
    parser.add_argument('--msrvtt_video_root', type=str, default='./datasets/MSR-VTT/TrainValVideo/')
    parser.add_argument('--msrvtt_anno_json_path', type=str, default='./datasets/MSR-VTT/train_val_videodatainfo.json')

    # MSVD
    parser.add_argument('--msvd_video_root', type=str, default='./datasets/MSVD/youtube_videos_id')
    parser.add_argument('--msvd_csv_path', type=str, default='/home/jxgu/github/video2text_jxgu/pytorch/datasets/MSVD/MSR_Video_Description_Corpus.csv') # MSR_Video_Description_Corpus_refine
    parser.add_argument('--msvd_video_name2id_map', type=str, default='/home/jxgu/github/video2text_jxgu/pytorch/datasets/MSVD/youtube_mapping.txt')
    parser.add_argument('--msvd_anno_json_path', type=str, default='/home/jxgu/github/video2text_jxgu/pytorch/datasets/MSVD/annotations.json')

    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--feat_dir', type=str, default='feats')
    parser.add_argument('--visual_dir', type=str, default='visuals')
    parser.add_argument('--vgg_checkpoint', type=str, default='')
    parser.add_argument('--decoder_pth_path', type=str, default='')
    parser.add_argument('--best_decoder_pth_path', type=str, default='')
    parser.add_argument('--optimizer_pth_path', type=str, default='')
    parser.add_argument('--best_optimizer_pth_path', type=str, default='')

    parser.add_argument('--use_checkpoint', type=int, default=0)
    parser.add_argument('--use_cuda', type=int, default=1)  # Id identifying this run/job.
    parser.add_argument('--id', type=str, default='')
    # used in cross-val and appended when writing progress files'

    args = parser.parse_args()

    # Check if args are valid
    assert args.hidden_size > 0, "rnn_size should be greater than 0"

    if not os.path.exists(args.feat_dir):
        os.mkdir(args.feat_dir)
    if not os.path.exists(args.visual_dir):
        os.mkdir(args.visual_dir)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    args.cnn_weight = 'model/pytorch-resnet/resnet101.pth' if args.cnn_model == 'resnet101' else './models/vgg16-397923af.pth'

    msrvtt_video_sort_lambda = lambda x: int(x[5:-4])
    args.msrvtt_train_range = (0, 6512)
    args.msrvtt_val_range = (6513, 7009)
    args.msrvtt_test_range = (0, 2989)

    # msvd_video_sort_lambda = lambda x: int(x[3:-4])
    msvd_video_sort_lambda = lambda x: int(x[5:-4])
    args.msvd_train_range = (0, 1200)
    args.msvd_val_range = (1200, 1300)
    args.msvd_test_range = (1300, 1970)

    #
    args.video_root = args.msrvtt_video_root if args.ds=='msr-vtt' else args.msvd_video_root
    args.video_sort_lambda = msrvtt_video_sort_lambda if args.ds == 'msr-vtt' else msvd_video_sort_lambda
    args.anno_json_path = args.msrvtt_anno_json_path if args.ds == 'msr-vtt' else args.msvd_anno_json_path
    args.train_range = args.msrvtt_train_range if args.ds == 'msr-vtt' else args.msvd_train_range
    args.val_range = args.msrvtt_val_range if args.ds == 'msr-vtt' else args.msvd_val_range
    args.test_range = args.msrvtt_test_range if args.ds == 'msr-vtt' else args.msvd_test_range

    args.vocab_pkl_path = os.path.join(args.feat_dir, args.ds + '_vocab.pkl')
    args.caption_pkl_path = os.path.join(args.feat_dir, args.ds + '_captions.pkl')
    args.caption_pkl_base = os.path.join(args.feat_dir, args.ds + '_captions')
    args.train_caption_pkl_path = args.caption_pkl_base + '_train.pkl'
    args.val_caption_pkl_path = args.caption_pkl_base + '_val.pkl'
    args.test_caption_pkl_path = args.caption_pkl_base + '_test.pkl'

    args.sal_h5_path = os.path.join(args.feat_dir, args.ds + '_saliency.h5')
    args.sal_h5_dataset = 'sal'
    args.fore_h5_path = os.path.join(args.feat_dir, args.ds + '_foreground.h5')
    args.fore_h5_dataset = 'sal'
    args.back_h5_path = os.path.join(args.feat_dir, args.ds + '_background.h5')
    args.back_h5_dataset = 'back'
    args.full_h5_path = os.path.join(args.feat_dir, args.ds + '_videos.h5')
    args.full_h5_dataset = 'feats'
    args.feature_h5_path = os.path.join(args.feat_dir, args.ds + '_features.h5')
    args.feature_h5_feats = 'feats'
    args.feature_h5_lens = 'lens'

    args.val_reference_txt_path = os.path.join(args.result_dir, 'val_references.txt')
    args.val_prediction_txt_path = os.path.join(args.result_dir, 'val_predictions.txt')

    args.test_reference_txt_path = os.path.join(args.result_dir, 'test_references.txt')
    args.test_prediction_txt_path = os.path.join(args.result_dir, 'test_predictions.txt')

    # checkpoint parameters
    import datetime
    if len(args.start_from):
        last_name = os.path.basename(args.start_from)
        last_time = last_name[0:8]
        args.id = last_time + '_' + args.model_name + '_' + args.ds
        print('Load from last time :' + args.id)
    else:
        args.id = datetime.datetime.now().strftime("%m%d%H%M") + '_' + args.model_name + '_' + args.ds

    last_name = os.path.basename(args.start_from)
    last_time = last_name[0:8]

    time_format = '%m-%d_%X'
    current_time = time.strftime(time_format, time.localtime())
    env_tag = '%s_TA-RES_SS0.75' % (current_time)
    args.log_environment = os.path.join('logs', env_tag)  # tensorboard recoder enviroment

    return args

opt = parse_opt()
