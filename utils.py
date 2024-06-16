import numpy as np
import cv2
#from skimage.measure import compare_psnr as psnr_metric
#from skimage.measure import compare_ssim as ssim_metric
import torch

from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from torch.autograd import Variable
from torch.optim import lr_scheduler

from numba import jit, float32, boolean, int32, float64

# 加载不同数据集的dataloader
def load_dataset(opt):
    if opt.dataset == 'mmnist_chaos':
        # MovingMNIST
        from data.movingmnist_chaos import MovingMNIST
        train_data = MovingMNIST(opt.data_root, True)
        valid_data = MovingMNIST(opt.data_root, False)
        test_data = MovingMNIST(opt.data_root, False)

    elif opt.dataset == 'mmnist':
        # MovingMNIST++
        from data.movingmnist import MovingMNIST
        train_data = MovingMNIST(opt.data_root, True)
        valid_data = MovingMNIST(opt.data_root, False)
        test_data = MovingMNIST(opt.data_root, False)

    elif opt.dataset == 'mmnistpp':
        # MovingMNIST++
        from data.mmnistPP import MovingMNISTpp
        train_data = MovingMNISTpp(opt.data_root, True)
        valid_data = MovingMNISTpp(opt.data_root, False)
        test_data = MovingMNISTpp(opt.data_root, False)

    elif opt.dataset == 'KNMI':
        # KNMI降水数据集 Unet
        from data.data_loader_precip import get_train_valid_loader,get_test_loader
        train_data, valid_data = get_train_valid_loader(opt.data_root,
                                                        batch_size=opt.batch_size,
                                                        random_seed=1337,
                                                        num_input_images=12,
                                                        num_output_images=6,
                                                        classification=False,
                                                        augment=False,
                                                        valid_size=0.1,
                                                        shuffle=True,
                                                        num_workers=opt.data_threads,
                                                        pin_memory=False)
        test_data = get_test_loader(opt.data_root,
                                    batch_size=opt.batch_size,
                                    num_input_images=12,
                                    num_output_images=6,
                                    classification=False,
                                    shuffle=True,
                                    num_workers=opt.data_threads,
                                    pin_memory=False)
    elif opt.dataset == 'NDVI':
        # 蓝藻数据集
        from data.ndvi import NdviData
        train_data = NdviData(opt.data_root, is_train=True, n_frames_input=opt.seq_len, n_frames_output=opt.pre_len)
        test_data = NdviData(opt.data_root, is_train=False, n_frames_input=opt.seq_len, n_frames_output=opt.pre_len)
        valid_data = NdviData(opt.data_root, is_train=False, n_frames_input=opt.seq_len, n_frames_output=opt.pre_len)

    else:
        raise NameError('Got unsupported dataset: {}'.format(opt.dataset))

    return train_data, valid_data, test_data


def sequence_input(seq, dtype):
    return [Variable(x.type(dtype), requires_grad=True) for x in seq]


def normalize_data(opt, dtype, sequence):
    if opt.dataset == 'mmnist_chaos' or opt.dataset == 'mmnistpp' or 'mmnist' or opt.dataset == 'KNMI' or opt.dataset == 'NDVI':
        sequence.transpose_(0, 1)  # T x B x C x H x W
        return sequence_input(sequence, dtype)
    else:
        raise NameError('Got unsupported dataset: {}'.format(opt.dataset))


def mse_metric(x1, x2):
    mse = np.square(x1 - x2).sum()
    return mse


def mae_metric(x1, x2):
    mae = np.sum(np.absolute(x1 - x2))
    return mae


def sharp_metric(x):
    x = np.transpose(x, [1, 2, 0])
    x = np.uint8(x * 255)
    sharp = np.max(cv2.convertScaleAbs(cv2.Laplacian(x, 3)))
    return sharp


def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]

    for t in range(T):
        pred[t] = np.maximum(pred[t], 0)
        pred[t] = np.minimum(pred[t], 1)
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    mae = np.zeros((bs, T))
    sharp = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])
            mae[i, t] = mae_metric(gt[t][i], pred[t][i])
            sharp[i, t] = sharp_metric(pred[t][i])
            x = np.uint8(gt[t][i] * 255)
            gx = np.uint8(pred[t][i] * 255)
            for c in range(gt[t][i].shape[0]):
                ssim[i, t] += ssim_metric(x[c], gx[c])
                psnr[i, t] += psnr_metric(x[c], gx[c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]

    return np.sum(mse, axis=0), np.sum(mae, axis=0), np.sum(ssim, axis=0), np.sum(psnr, axis=0), np.sum(sharp, axis=0)


def batch_mae_frame_float(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    mae = np.sum(np.absolute(x - y), axis=(1, 2), dtype=np.float32)
    return np.sum(mae, axis=0)


def batch_psnr(gen_frames, gt_frames):
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=(1, 2), dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.sum(psnr, axis=0)


@jit(int32(float32, float32))
def batch_confusion_matrix(gen_frames, gt_frames):
    pred = np.int32(gen_frames)
    gt = np.int32(gt_frames)
    height, width = pred.shape

    thresholds = [100, 150, 200]
    threshold_num = len(thresholds)
    ret = np.zeros(shape=(threshold_num, 4), dtype=np.int32)

    for m in range(height):
        for n in range(width):
            for k in range(threshold_num):
                bpred = pred[m][n] >= thresholds[k]
                btruth = gt[m][n] >= thresholds[k]

                ind = (1 - btruth) * 2 + (1 - bpred)
                ret[k][ind] += 1
                # The above code is the same as:
                # TP
                #ret[k][0] += bpred * btruth
                # FN
                #ret[k][1] += (1 - bpred) * btruth
                # FP
                #ret[k][2] += bpred * (1 - btruth)
                # TN
                #ret[k][3] += (1 - bpred) * (1 - btruth)

    return ret[:, 0], ret[:, 1], ret[:, 2], ret[:, 3]



def eval_seq_batch(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]

    ssim = np.zeros(T)
    psnr = np.zeros(T)
    mse = np.zeros(T)
    mae = np.zeros(T)
    sharp = np.zeros(T)
    for t in range(T):
        x = gt[t][:, 0, :, :]
        gx = pred[t][:, 0, :, :]
        gx = np.maximum(gx, 0)
        gx = np.minimum(gx, 1)

        mae[t] = batch_mae_frame_float(gx, x)
        mse_t = np.square(x - gx).sum()
        mse[t] = mse_t

        real_frm = np.uint8(x * 255)
        pred_frm = np.uint8(gx * 255)
        psnr[t] = batch_psnr(pred_frm, real_frm)

        for b in range(bs):
            sharp[t] += np.max(
                cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
            score, _ = ssim_metric(pred_frm[b], real_frm[b], full=True)
            ssim[t] += score

    return mse, mae, ssim, psnr, sharp

def detail_eval_seq_batch(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]

    ssim = np.zeros(T)
    psnr = np.zeros(T)
    mse = np.zeros(T)
    mae = np.zeros(T)
    sharp = np.zeros(T)

    tp_ = np.zeros(shape=3, dtype=np.float64)
    fn_ = np.zeros(shape=3, dtype=np.float64)
    fp_ = np.zeros(shape=3, dtype=np.float64)
    tn_ = np.zeros(shape=3, dtype=np.float64)

    for t in range(T):
        x = gt[t][:, 0, :, :]
        gx = pred[t][:, 0, :, :]
        gx = np.maximum(gx, 0)
        gx = np.minimum(gx, 1)

        mae[t] = batch_mae_frame_float(gx, x)
        mse_t = np.square(x - gx).sum()
        mse[t] = mse_t

        real_frm = np.uint8(x * 255)
        pred_frm = np.uint8(gx * 255)
        psnr[t] = batch_psnr(pred_frm, real_frm)

        #print("real_frm", real_frm.shape)
        #print("pred_frm", pred_frm.shape)

        for b in range(bs):
            sharp[t] += np.max(
                cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
            score, _ = ssim_metric(pred_frm[b], real_frm[b], full=True)
            ssim[t] += score

            tp, fn, fp, tn = batch_confusion_matrix(pred_frm[b], real_frm[b])
            tp_ += tp
            fn_ += fn
            fp_ += fp
            tn_ += tn

    return mse, mae, ssim, psnr, sharp, tp_, fn_, fp_, tn_

def reshape_patch(images, patch_size):
    bs = images.size(0)
    nc = images.size(1)
    height = images.size(2)
    weight = images.size(3)
    x = images.reshape(bs, nc, int(height / patch_size), patch_size, int(weight / patch_size), patch_size)
    x = x.transpose(2, 5)
    x = x.transpose(4, 5)
    x = x.reshape(bs, nc * patch_size * patch_size, int(height / patch_size), int(weight / patch_size))

    return x


def reshape_patch_back(images, patch_size):
    bs = images.size(0)
    nc = int(images.size(1) / (patch_size * patch_size))
    height = images.size(2)
    weight = images.size(3)
    x = images.reshape(bs, nc, patch_size, patch_size, height, weight)
    x = x.transpose(4, 5)
    x = x.transpose(2, 5)
    x = x.reshape(bs, nc, height * patch_size, weight * patch_size)

    return x


def get_scheduler(optimizer, opt, t_max):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net, name, verbose=True):
    """Print the total number of parameters in the network and (if verbose) network architecture
    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
    print('-----------------------------------------------')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss