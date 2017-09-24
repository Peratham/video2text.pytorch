# PyTorch implementation of video captioning

## Requirements
### Pretrained Model
- VGG16 pretrained on ImageNet [PyTorch version]: https://download.pytorch.org/models/vgg16-397923af.pth
- Resnet-101 pretrained on ImageNet [PyTorch version]: https://github.com/ruotianluo/pytorch-resnet

### Datasets
- MSVD: https://www.microsoft.com/en-us/download/details.aspx?id=52422
- MSR-VTT: http://ms-multimedia-challenge.com/2017/dataset

**Obtain the dataset you need:**

* [MSR-VTT](http://ms-multimedia-challenge.com/dataset):
[train_val_videos.zip](http://202.38.69.241/static/resource/train_val_videos.zip),
[train_val_annotation.zip](http://202.38.69.241/static/resource/train_val_annotation.zip),
[test_videos.zip](http://202.38.69.241/static/resource/test_videos.zip),
[test_videodatainfo.json](http://ms-multimedia-challenge.com/static/resource/test_videodatainfo.json)

* [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/): flickr30k.tar.gz, flickr30k-images.tar

### Packages
```bash
torch, torchvision, numpy, scikit-image, nltk, h5py, pandas, future  # python2 only
tensorboard_logger  # for use tensorboard to view training loss
```
You can use:
```bash
sudo pip install -r requirements.txt
```
To install all the above packages.

## Usage
### Preparing Data
Firstly, we should make soft links to the dataset folder and pretrained models. For example:
```bash
mkdir datasets
ln -s YOUR_DATASET_PATH datasets/MSVD
mkdir models
ln -s YOUR_CNN_MODEL_PATH models/
```

Some details can be found in opts.py. Then we can:

1. Prepare video feature:
```bash
python scripits/prepro_video_feats.py
```

2. Prepare caption feature and dataset split:
```bash
python scripts/prepro_caption_feats.py
```

### Training and Testing
Before training the model, please make sure you can use GPU to accelerate computation in PyTorch. Some parameters, such as batch size and learning rate, can be found in args.py.

- Train:
```bash
python train.py
```

- Evaluate:
```bash
python evaluate.py
```

- Sample some examples:
```bash
python sample.py
```

### Related papers

1.[Supervising Neural Attention Models for Video Captioning by Human Gaze Data](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Supervising_Neural_Attention_CVPR_2017_paper.pdf)