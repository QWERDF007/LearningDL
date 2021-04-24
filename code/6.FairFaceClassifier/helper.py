""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
import logging.handlers
import numpy as np
import sklearn.metrics as metrics



class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt="[%(asctime)s] [%(name)s] [%(levelname)1.1s]: %(message)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)1.1s]: %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=np.float32)
    return (metrics.precision_score(y_true=target, y_pred=pred, average='weighted'),
            metrics.recall_score(y_true=target, y_pred=pred, average='weighted'))

