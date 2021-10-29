import os
import argparse
import glob
import shutil
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


def save_as_csv(filename, ids, image_dir, anno_dir):
    dataset = [
        [
            os.path.join(image_dir, id + '.jpg'),
            os.path.join(anno_dir, id + '.png')
        ]
        for id in ids
    ]
    df = pd.DataFrame(dataset, columns=['image', 'annotation'])
    df = df.applymap(os.path.abspath)
    print(df)
    df.to_csv(os.path.join(os.path.dirname(__file__), filename), index=False)


def main():
    parser = argparse.ArgumentParser(
    	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--voc', type=str, required=True,
                        help="path to VOCdevkit/VOC2012/")
    parser.add_argument('--sbd', type=str, required=True,
                        help="path to SBD root directory (benchmark_RELEASE/)")
    args = parser.parse_args()

    anno_dir = os.path.join(os.path.dirname(__file__), 'annotations/')
    image_dir = os.path.join(args.voc, 'JPEGImages/')

    shutil.rmtree(anno_dir, ignore_errors=True)
    os.mkdir(anno_dir)

    train_ids, val_ids = [], []

    # Parse VOC2012 annotations
    # Refer to: https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py
    for split in ['train', 'val']:
        with open(os.path.join(args.voc, 'ImageSets/Segmentation/%s.txt' % split)) as f:
            ids = [line.strip() for line in f.readlines()]
        for id in tqdm(ids, desc="VOC2012 %s" % split):
            anno = Image.open(os.path.join(args.voc, 'SegmentationClass/%s.png' % id))
            anno = Image.fromarray(np.array(anno))   # remove palette
            anno.save(os.path.join(anno_dir, id + '.png'))

            if split == 'train':
                train_ids.append(id)
            else:
                val_ids.append(id)
    save_as_csv('train.csv', train_ids, image_dir, anno_dir)
    save_as_csv('val.csv', val_ids, image_dir, anno_dir)

    # Parse the SBD annotations
    # Refer to: https://pytorch.org/vision/stable/_modules/torchvision/datasets/sbd.html#SBDataset
    mat_files = glob.glob(os.path.join(args.sbd, 'dataset/cls/*.mat'))
    mat_files.sort()
    num_extra = 0
    for mat_path in tqdm(mat_files, desc="SBD"):
        id = os.path.basename(os.path.splitext(mat_path)[0])
        if (id in train_ids) or (id in val_ids):
            continue

        num_extra += 1
        mat = sio.loadmat(mat_path)
        anno = Image.fromarray(mat['GTcls'][0]['Segmentation'][0])
        anno.save(os.path.join(anno_dir, id + '.png'))

        train_ids.append(id)
    print("Number of extra annotations:", num_extra)
    save_as_csv('trainaug.csv', train_ids, image_dir, anno_dir)


if __name__ == '__main__':
    main()
