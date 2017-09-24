from __future__ import print_function
import os
import sys
import json
import hashlib
import pandas as pd
from six.moves import cPickle
import tensorflow as tf
# Close print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Enable print
def enablePrint():
    sys.stdout = sys.__stdout__


class CocoAnnotations:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.img_dict = {}
        info = {
            "year": 2017,
            "version": '1',
            "description": 'Video CaptionEval',
            "contributor": 'Subhashini Venugopalan, Yangyu Chen',
            "url": 'https://github.com/vsubhashini/, https://github.com/Yugnaynehc/',
            "date_created": '',
        }
        licenses = [{"id": 1, "name": "test", "url": "test"}]
        self.res = {"info": info,
                    "type": 'captions',
                    "images": self.images,
                    "annotations": self.annotations,
                    "licenses": licenses,
                    }

    def read_multiple_files(self, filelist):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename)

    def get_image_dict(self, img_name):
        code = img_name.encode('utf8')
        image_hash = int(int(hashlib.sha256(code).hexdigest(), 16) % sys.maxsize)
        if image_hash in self.img_dict:
            assert self.img_dict[image_hash] == img_name, 'hash colision: {0}: {1}'.format(
                image_hash, img_name)
        else:
            self.img_dict[image_hash] = img_name
        image_dict = {"id": image_hash,
                      "width": 0,
                      "height": 0,
                      "file_name": img_name,
                      "license": '',
                      "url": img_name,
                      "date_captured": '',
                      }
        return image_dict, image_hash

    def read_file(self, filename):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                try:
                    assert len(id_sent) == 2
                    sent = id_sent[1]
                except Exception as e:
                    # print(line)
                    continue
                image_dict, image_hash = self.get_image_dict(id_sent[0])
                self.images.append(image_dict)

                self.annotations.append({
                    "id": len(self.annotations) + 1,
                    "image_id": image_hash,
                    "caption": sent,
                })

    def dump_json(self, outfile):
        self.res["images"] = self.images
        self.res["annotations"] = self.annotations
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(',', ': '))


def create_reference_json(reference_txt_path):
    output_file = '{0}.json'.format(reference_txt_path)
    crf = CocoAnnotations()
    crf.read_file(reference_txt_path)
    crf.dump_json(output_file)
    print('Created json references in %s' % output_file)


class CocoResFormat:

    def __init__(self):
        self.res = []
        self.caption_dict = {}

    def read_multiple_files(self, filelist, hash_img_name):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename, hash_img_name)

    def read_file(self, filename, hash_img_name):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                if len(id_sent) > 2:
                    id_sent = id_sent[-2:]
                assert len(id_sent) == 2
                sent = id_sent[1]

                if hash_img_name:
                    img_id = int(int(hashlib.sha256(id_sent[0].encode('utf8')).hexdigest(),
                                     16) % sys.maxsize)
                else:
                    img = id_sent[0].split('_')[-1].split('.')[0]
                    img_id = int(img)
                imgid_sent = {}

                if img_id in self.caption_dict:
                    assert self.caption_dict[img_id] == sent
                else:
                    self.caption_dict[img_id] = sent
                    imgid_sent['image_id'] = img_id
                    imgid_sent['caption'] = sent
                    self.res.append(imgid_sent)

    def dump_json(self, outfile):
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(',', ': '))

'''
Pre-training funcitons
'''
def history_infos(opt):
    infos = {}
    if len(opt.start_from) != 0:  # open old infos and check if models are compatible
        model_id = opt.start_from
        infos_id = model_id.replace('save/', '') + '_best_infos.pkl'
        with open(os.path.join(opt.start_from, infos_id)) as f:
            infos = cPickle.load(f)
            #saved_model_opt = infos['opt']
            #need_be_same = []
            #for checkme in need_be_same:
            #    assert vars(saved_model_opt)[checkme] == vars(opt)[
            #        checkme], "Command line argument and saved model disagree on '%s' " % checkme

    	epoch = infos.get('epoch', 0)
    else:
        epoch = 0

    return opt, infos, epoch

def build_msvd_annotation(opt):
    '''
    Follow the format of MSR-VTT dataset,create a json file including video
    info and caption annotations for MSVD. The reason why we need it similar to
    MSR-VTT format, is that we share one `prepare captions' code for all datasets
    '''
    # Firstly, get the name of each video according to the offical csv file
    video_data = pd.read_csv(opt.msvd_csv_path, sep=',', encoding='utf8')
    video_data = video_data[video_data['Language'] == 'English']
    # Only use videos with clean description
    # Fail, some videos does not have clean descriptions
    # video_data = video_data[video_data['Source'] == 'clean']
    video_data['VideoName'] = video_data.apply(lambda row: row['VideoID'] + '_' +
                                               str(row['Start']) + '_' +
                                               str(row['End']), axis=1)
    # Create the dictionary according to the mapping from video name to video_id provided by youtubeclips
    video_name2id = {}
    with open(opt.msvd_video_name2id_map, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name, vid = line.strip().split()
            # Extract video's digital id
            # Sub 1 is that the id starts from 1, but we start from 0 in post-processing
            # In practice, we process with sequential info, so sub 1 has no influence
            vid = int(vid[3:]) - 1
            # Transform vid to video+num_id format
            # Do not ask why, because MSR-VTT do like that, stupid!
            vid = 'video%d' % vid
            video_name2id[name] = vid
            static_link = False
            if static_link:
                raw_video_name = '/home/jxgu/github/video2text_jxgu/pytorch/datasets/MSVD/youtube_videos' + '/' + name + '.avi'
                id_video_name = '/home/jxgu/github/video2text_jxgu/pytorch/datasets/MSVD/youtube_videos_id' + '/' + vid + '.avi'
                os.system('ln -s '+ ' '+raw_video_name + ' '+id_video_name)
    # Create json file according to MSR-VTT data structure
    sents_anno = []
    not_use_video = []
    for name, desc in zip(video_data['VideoName'], video_data['Description']):
        if name not in video_name2id:
            if name not in not_use_video:
                #print('No use: %s' % name)
                not_use_video.append(name)
            not_use_video.append(name)
            continue
        # Be careful, one caption in the video:SKhmFSV-XB0 is NaN
        if type(desc) == float:
            print('Error annotation: %s\t%s' % (name, desc))
            continue
        d = {}
        # Then we filter all non-ascii character!
        desc = desc.encode('ascii', 'ignore').decode('ascii')
        # There are still many problems, some captions have a heap of '\n' or '\r\n'
        desc = desc.replace('\n', '')
        desc = desc.replace('\r', '')
        # Some captions are ended by period, while some does not have period
        # or even have multiple periods.
        # So we filter the period ('.') and the content exceeds one sentence
        # Note that the captions in MSR-VTT data set are not ended by period
        desc = desc.split('.')[0]

        d['caption'] = desc
        d['video_id'] = video_name2id[name]
        sents_anno.append(d)

    anno = {'sentences': sents_anno}
    with open(opt.msvd_anno_json_path, 'w') as f:
        json.dump(anno, f)

'''
Training funcitons
'''
def scheduled_sampling(opt, epoch, model):
    # Assign the scheduled sampling prob
    if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
        frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
        opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        model.ss_prob = opt.ss_prob

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def freeze_param(model):
    for param in model.parameters():
        param.requires_grad = False

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def update_lr(opt, epoch, optimizer):
    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
        decay_factor = opt.learning_rate_decay_rate ** frac
        opt.current_lr = opt.learning_rate * decay_factor
        set_lr(optimizer, opt.current_lr)  # set the decayed rate
    else:
        opt.current_lr = opt.learning_rate

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def record_summary(tf_summary_writer, uidx, t2i, i2t):
    for k, v in t2i.items():
        add_summary_value(tf_summary_writer, k, v, uidx)
    for k, v in i2t.items():
        add_summary_value(tf_summary_writer, k, v, uidx)
