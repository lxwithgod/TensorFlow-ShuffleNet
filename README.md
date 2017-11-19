# ShuffleNet in Tensorflow

## Introduction

This repository attempts to reproduce this amazing work by Xiangyu Zhang et al. : [ShuffleNet](https://arxiv.org/abs/1707.01083)

## Setup

- Python 3+
- Tensorflow 1.3+
- PyYAML 3+

## Train

1. Copy `conf/demo.yml` to `conf/config.yml`
1. Modify `conf/config.yml`
    - Name your network
    - Change `data_train_dir` to `path/to/tiny-image-net/train`
    - Change `data_test_dir` to `path/to/tiny-image-net/test`
    - Change other config items
1. Train with `python3 train.py --conf conf/config.xml`
1. Have a cup of coffee

## Evaluate

### Evaluate while training

```bash
python3 evaluate.py --conf conf/config.xml --watch
```

### Evaluate

```bash
python3 evaluate.py --conf conf/config.xml --file path/to/the/ckpt/file.ckpt
```

## Todo

- [ ] Debug
    - [ ] The loss is incorrect at present
    - [ ] Evaluation accuracy is incorrect at present
- [ ] Test all code
- [ ] Pre-trained model

## References

[paper] [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
