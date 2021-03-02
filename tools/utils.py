import os
import os.path as osp
import pickle

import torch


def write_log_train(path, name, log):
    file = open(osp.join(path, "train_" + name + ".log"), 'a')
    file.write(log)
    file.write('\n')
    file.close()


def write_log_valid(path, name, log):
    file = open(osp.join(path, "valid_" + name + ".log"), 'a')
    file.write(log)
    file.write('\n')
    file.close()


def save_model_torch(dir, filename, model, optimizer, epoch):
    makedirs(dir)
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, osp.join(dir, filename))


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5):
    # --- Decay learning rate --- #
    step = 50
    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            # print('Learning rate sets to {}.'.format(param_group['lr']))
    # else:
    #     for param_group in optimizer.param_groups:
    # print('Learning rate sets to {}.'.format(param_group['lr']))


def makedirs(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class AverageMeter(object):
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


if __name__ == '__main__':
    pass
