import os
import csv
import math


import matplotlib.pyplot as plt
from PIL import Image



if __name__ == '__main__':
    total = 101
    error_labels_file = 'error_labels.txt'
    cats_dogs = []
    dogs_cats = []
    # 读取cleanlab预测出的错误标签
    with open(error_labels_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader, start=1):
            stem = row[0]
            given = int(row[1])
            p_given = float(row[2])
            pred = int(row[3])
            p_pred = float(row[4])
            name = stem + '.jpg'
            if given == 0 and pred == 1:
                cats_dogs.append((name, given, p_given, pred, p_pred))
            elif given == 1 and pred == 0:
                dogs_cats.append((name, given, p_given, pred, p_pred))
            else:
                raise ValueError("given and pred are the same label")
    num_cats_dogs = len(cats_dogs)
    num_dogs_cats = len(dogs_cats)
    ncols = 8

    # 展示原始标签为猫，cleanlab预测为狗的图像
    nrows = math.ceil(num_cats_dogs / ncols)
    plt.figure(figsize=(24,16))
    for i, (name, given, p_given, pred, p_pred) in enumerate(cats_dogs, start=1):
        path = os.path.join('data/train', '0', name)
        img = Image.open(path).convert("RGB")
        plt.subplot(nrows, ncols, i)
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.show()

    # 展示原始标签为狗，cleanlab预测为猫的图像
    nrows = math.ceil(num_dogs_cats / ncols)
    plt.figure(figsize=(24,16))
    for i, (name, given, p_given, pred, p_pred) in enumerate(dogs_cats, start=1):
        path = os.path.join('data/train', '1', name)
        img = Image.open(path).convert("RGB")
        plt.subplot(nrows, ncols, i)
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.show()
            