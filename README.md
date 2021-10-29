# PyTorch DeepLabV3
PyTorch implementation of DeepLabV3

## Results
* Training: PASCAL VOC 2012 trainaug set
* Evaluation: PASCAL VOC 2012 val set

| Backbone             | Output stride | mIoU | Configuration                                        |
|----------------------|:-------------:|:----:|------------------------------------------------------|
| ResNet101            | 16            | 77.8 | [configs/resnet101.yaml](configs/resnet101.yaml)     |
| MobileNetV2          | 16            | 70.1 | [configs/mobilenetV2.yaml](configs/mobilenetV2.yaml) |

## Requirements
* Python ≥ 3.6
* Install libraries: `pip install -r requirements.txt`

## Usage
### Data preparation
```bash
cd datasets/voc/

# Download standard PASCAL VOC 2012
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_11-May-2012.tar

# The original PASCAL VOC 2012 only labels 1,464 training images. Following the paper, we add additional
# annotations from Semantic Boundaries Dataset to augment the training set to 10,582 images.
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar zxf benchmark.tgz

# Generate CSV files that contain the paths to the images and annotations.
python prepare.py --voc VOCdevkit/VOC2012/ --sbd benchmark_RELEASE/
```

### Configuration
We use YAML for configuration management. See [configs/*.yaml](configs/) for examples.
You can modify the settings as needed.

### Training
```bash
python train.py --cfg <CONFIG_FILE> --logdir <LOG_DIRECTORY>

# For example, to train a model with ResNet101 backbone:
python train.py --cfg configs/resnet101.yaml --logdir runs/resnet101/exp0/
```
To visualize training progress using TensorBoard:
```bash
tensorboard --logdir <LOG_DIRECTORY>
```
An interrupted training can be resumed by:
```bash
# Run train.py with --resume to restore the latest saved checkpoint file in the log directory.
python train.py --cfg <CONFIG_FILE> --logdir <LOG_DIRECTORY> --resume
```
### Evaluation
```bash
python eval.py --cfg <CONFIG_FILE> --pth <LOG_DIRECTORY>/best.pth --csv datasets/voc/val.csv
```
