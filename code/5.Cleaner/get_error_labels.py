import os
import argparse
import sqlite3
from pathlib import Path

import torch
import cleanlab
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import metrics
from torchvision import datasets
from torch.utils.data import Dataset
from cleanlab import pruning
# pruning.MIN_NUM_PER_CLASS = 1


class MultiLabelDataset(Dataset):
    def __init__(self, root: str, df: pd.DataFrame, transform=None, target_transform=None):
        self.root = root
        paths = [os.path.join(self.root, name) for name in df.values[:, 0]]
        targets = np.array(df.values[:, 1:], dtype=np.float32)
        self.imgs = [(paths[i], targets[i]) for i in range(len(paths))]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        path, target = self.imgs[item]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=np.float32)
    return (metrics.precision_score(y_true=target, y_pred=pred, average='weighted'),
            metrics.recall_score(y_true=target, y_pred=pred, average='weighted'))


def gen_params(results):
    for one in results:
        yield one


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('-p', '--psx', type=str,
                        default=None, help='path to psx file')
    parser.add_argument('-db', '--database', type=str,
                        default=None, help='path to database')
    parser.add_argument('-t', '--table', type=str,
                        default=None, help='name of table')
    parser.add_argument('-o', '--output', type=str,
                        default=None, help='path to output file')
    parser.add_argument('-multi', '--multi', default=False,
                        action='store_true', help='multi')
    parser.add_argument('-csv', '--csv', type=str,
                        default=None, help='path to annotation file')
    args = parser.parse_args()
    psx = np.load(args.psx)
    if args.multi:
        df = pd.read_csv(args.csv)
        fields = df.columns
        imgs, labels = [list(z) for z in zip(
            *MultiLabelDataset(args.data, df).imgs)]
    else:
        imgs, labels = [list(z) for z in zip(
            *datasets.ImageFolder(args.data).imgs)]
        labels = np.array(labels, dtype=int)
    if args.multi:
        acc, recall = calculate_metrics(psx, labels)
        labels = np.array([[i for i in range(11) if label[i] == 1]
                           for label in labels], dtype=object)
        print("Overall accuracy: {:.2%} recall: {:.3%}".format(acc, recall))
    else:
        print('Overall accuracy: {:.2%}'.format(
            metrics.accuracy_score(labels, psx.argmax(axis=1))))
    label_errors_bool = cleanlab.pruning.get_noise_indices(
        s=labels,
        psx=psx,
        prune_method='both',
        sorted_index_method=None,
        multi_label=args.multi
    )
    print('Total:', label_errors_bool.sum())
    label_errors_idx = cleanlab.pruning.order_label_errors(
        label_errors_bool=label_errors_bool,
        psx=psx,
        labels=labels,
        sorted_index_method='normalized_margin',
    )
    if args.multi:
        probs = psx[label_errors_idx]
    else:
        prob_given = np.asarray([psx[i][j] for i, j in enumerate(labels)])[
            label_errors_idx]
        preds = np.argmax(psx, axis=1)
        prob_preds = np.asarray([psx[i][j] for i, j in enumerate(preds)])[
            label_errors_idx]
    # assert prob_given.shape[0] == prob_preds.shape[0], "check that given labels and predicted labels are the same size"
    results = []
    writer = open(args.output, 'w') if args.output is not None else None
    if args.multi:
        for i, (idx, prob) in enumerate(zip(label_errors_idx, probs)):
            stem = Path(imgs[idx]).stem
            s = '{},{}'.format(stem, ','.join(
                [str(round(v, 2)) for v in prob]))
            print(s)
            if writer is not None:
                writer.write(s + '\n')
            results.append((stem, *prob.tolist()))
    else:
        for idx, p_given, p_pred in zip(label_errors_idx, prob_given, prob_preds):
            stem = Path(imgs[idx]).stem
            s = '{},{},{:.3f},{},{:.3f}'.format(
                stem, labels[idx], p_given, preds[idx], p_pred)
            print(s)
            if writer is not None:
                writer.write(s + '\n')
            results.append(
                (stem, labels[idx].item(), p_given.item(), preds[idx].item(), p_pred.item()))
    if args.database is not None:
        conn = sqlite3.connect(args.database)
        c = conn.cursor()
        if args.table is not None:
            table = args.table
        else:
            table = 'test'
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='%s'" % table
        query = c.execute(sql).fetchall()
        if len(query) == 0:
            if args.multi:
                fields = df.columns.values[1:]
                sql = 'CREATE TABLE "{}" ("id" TEXT NOT NULL, {}, PRIMARY KEY("id"))'.format(
                    table, ','.join(['"{}" REAL'.format(v) for v in df.columns.values[1:].tolist()]))
            else:
                sql = 'CREATE TABLE "{}" ("id" TEXT NOT NULL, "given" INTEGER, "p_given" REAL, "pred" INTEGER, "p_pred" REAL, PRIMARY KEY("id"))'.format(
                    table)
            print(sql)
            c.execute(sql)
            conn.commit()
        if args.multi:
            sql = "INSERT OR REPLACE INTO {}({}) VALUES({})".format(
                table, ','.join(df.columns.values.tolist()), ','.join('?'*len(df.columns)))
        else:
            sql = "INSERT OR REPLACE INTO {}(id,given,p_given,pred,p_pred) VALUES(?,?,?,?,?)".format(
                table)
        c.executemany(sql, gen_params(results))
        conn.commit()
