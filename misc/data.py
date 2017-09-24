import pickle
import h5py
import torch
import torch.utils.data as data
import opts
class V2TDataset(data.Dataset):
    '''
    Video to Text dataset description class, used for loading and providing data.
    Support MSR-VTT and MSVD datasets
    Need the following input when constructing it:
    1. The pkl file containing text features
    2. The h5 file containing video frame information
    Provide the text and video h5 feature, and return the data according to the caption id
    '''

    def __init__(self, opt, cap_pkl, feature_h5):
        with open(cap_pkl, 'rb') as f:
            self.captions, self.lengths, self.video_ids = pickle.load(f)
        h5_file = h5py.File(feature_h5, 'r')
        self.video_feats = h5_file[opt.feature_h5_feats]

    def __getitem__(self, index):
        '''
        Return a training sample pair (including video frame features)
        Find the corresponding video according to the caption
        So we need the videos are sorted by ascending order (id)
        '''
        caption = self.captions[index]
        length = self.lengths[index]
        video_id = self.video_ids[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        return video_feat, caption, length, video_id

    def __len__(self):
        return len(self.captions)


class VideoDataset(data.Dataset):
    '''
    Class: only provides the dataloader for video features and its corresponding id
    We want to speed up the calculation of evaluation metric with this class
    '''
    def __init__(self, opt, eval_range, feature_h5):
        self.eval_list = tuple(range(*eval_range))
        h5_file = h5py.File(feature_h5, 'r')
        self.video_feats = h5_file[opt.feature_h5_feats]

    def __getitem__(self, index):
    #def __getindex__(self, index):
        '''
        Return a training sample pair (including video features and the corresponding ID)
        '''
        video_id = self.eval_list[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        return video_feat, video_id

    def __len__(self):
        return len(self.eval_list)


def train_collate_fn(data):
    '''
    Combining multi-training samples to a mini-batch
    '''
    # Sort the videos according to video length
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, captions, lengths, video_ids = zip(*data)

    # Combing video together (make 2D Tensor video frames into one 3D Tensor)
    videos = torch.stack(videos, 0)

    # Combine the captions together (make 1D Tensor words into on 2D Tensor)
    captions = torch.stack(captions, 0)
    return videos, captions, lengths, video_ids


def eval_collate_fn(data):
    '''
    Function used to combine multiple samples into one mini-batch
    '''
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, video_ids = zip(*data)

    # Combing video together (make 2D Tensor sequence into 3D Tensor)
    videos = torch.stack(videos, 0)

    return videos, video_ids


def get_train_loader(opt, cap_pkl, feature_h5, batch_size=50, shuffle=True, num_workers=2, pin_memory=True):
    v2t = V2TDataset(opt, cap_pkl, feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


def get_eval_loader(opt, cap_pkl, feature_h5, batch_size=1, shuffle=False, num_workers=1, pin_memory=False):
    vd = VideoDataset(opt, cap_pkl, feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    opt = opts.parse_opt()
    train_loader = get_train_loader(opt, opt.train_caption_pkl_path, opt.feature_h5_path)
    print(len(train_loader))
    d = next(iter(train_loader))
    print(d[0].size())
    print(d[1].size())
    print(len(d[2]))
