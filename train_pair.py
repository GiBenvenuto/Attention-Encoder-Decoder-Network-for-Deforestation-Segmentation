# Imports

import cv2
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr
from CombinedLoss import CombinedLoss
from Models.Proposed_SegModels import Proposed

# Other
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import ceil
import time
import warnings
import psutil
from skimage import io, color
import rasterio
import pandas as pd

PATH_CLUSTER = ''
PATH_TO_DATASET = PATH_CLUSTER + 'Datasets/Dataset_Altamira/'

FP_MODIFIER = 10  # Tuning parameter, use 1 if unsure

BATCH_SIZE = 24
PATCH_SIDE = 96
N_EPOCHS = 50

NORMALISE_IMGS = True

TRAIN_STRIDE = int(PATCH_SIDE / 2) - 1

TYPE = 0  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

LOAD_TRAINED = False

DATA_AUG = True

L = 1024
N = 2

print('DEFINITIONS OK')


# Functions

def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""

    # crop if necesary
    I = I[:s[0], :s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])

    return np.pad(I, ((0, p0), (0, p1)), 'edge')



'''
def read_sentinel_img_diff1(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    bitwise = cv2.imread(path + "/pair/img_diff.png")
    #bitwise = cv2.cvtColor(bitwise, cv2.COLOR_RGB2GRAY)
    img1 = cv2.imread(path + "/pair2/img1.png")
    #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.imread(path + "/pair2/img2.png")
    #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    I = np.concatenate((img1, img2, bitwise), axis=-1)

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I
'''
def read_sentinel_img_diff2(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    bitwise = cv2.imread(path + "/pair/img_diff.png")
    bitwise = cv2.cvtColor(bitwise, cv2.COLOR_RGB2GRAY)
    img1 = cv2.imread(path + "/pair2/img1.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.imread(path + "/pair2/img2.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    I = np.stack((img1, img2, bitwise), axis=-1).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I







def read_sentinel_img_trio_diff(path, name):
    """Read cropped Sentinel-2 image pair and change map."""
    #     read images
    if TYPE == 0:
        # print(path)
        I1 = read_sentinel_img_diff2(path + name)
    elif TYPE == 1:
        print("Not implemented :(")
        exit(0)

    cm = io.imread(path + name + '/cm/cm_gt.png', as_gray=True) != 0

    return I1, cm


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)


class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train=0, patch_side=96, stride=None, use_all_bands=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride

        if train == 0:
            fname = 'train.txt'
        elif train == 1:
            fname = 'val.txt'
        else:
            fname = 'test.txt'

        print(path + fname)
        self.names = read_csv(path + fname).columns
        self.n_imgs = self.names.shape[0]

        n_pix = 0
        true_pix = 0

        # load images
        self.imgs_1 = {}
        # self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image

            I1, cm = read_sentinel_img_trio_diff(self.path, im_name)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            # self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm

            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()

            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name,
                                            [self.stride * i, self.stride * i + self.patch_side, self.stride * j,
                                             self.stride * j + self.patch_side],
                                            [self.stride * (i + 1), self.stride * (j + 1)])
                    self.patch_coords.append(current_patch_coords)

        #self.weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        # I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]

        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1 * np.array(label)).float()

        sample = {'I1': I1, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip(object):
    """Flip randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, label = sample['I1'], sample['label']

        if random.random() > 0.5:
            I1 = I1.numpy()[:, :, ::-1].copy()
            I1 = torch.from_numpy(I1)
            # I2 =  I2.numpy()[:,:,::-1].copy()
            # I2 = torch.from_numpy(I2)
            label = label.numpy()[:, ::-1].copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'label': label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, label = sample['I1'], sample['label']

        n = random.randint(0, 3)
        if n:
            I1 = sample['I1'].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            # I2 =  sample['I2'].numpy()
            # I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            # I2 = torch.from_numpy(I2)
            label = sample['label'].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'label': label}




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train(n_epochs=N_EPOCHS, save=True):
    t = np.linspace(1, n_epochs, n_epochs)

    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t
    epoch_train_change_accuracy = 0 * t
    epoch_train_nochange_accuracy = 0 * t
    epoch_train_precision = 0 * t
    epoch_train_recall = 0 * t
    epoch_train_Fmeasure = 0 * t
    epoch_test_loss = 0 * t
    epoch_test_accuracy = 0 * t
    epoch_test_change_accuracy = 0 * t
    epoch_test_nochange_accuracy = 0 * t
    epoch_test_precision = 0 * t
    epoch_test_recall = 0 * t
    epoch_test_Fmeasure = 0 * t

    #     mean_acc = 0
    #     best_mean_acc = 0
    fm = 0
    best_fm = 0

    lss = 1000
    best_lss = 1000

    plt.figure(num=1)
    plt.figure(num=2)
    plt.figure(num=3)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch_index in tqdm(range(n_epochs)):
        net.train()

        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        tot_count = 0
        tot_loss = 0
        tot_accurate = 0
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))

        #         for batch_index, batch in enumerate(tqdm(data_loader)):

        for batch in train_loader:
            I1 = Variable(batch['I1'].float())
            # I2 = Variable(batch['I2'].float())
            # print(I1)
            # print(I2)
            label = torch.squeeze(Variable(batch['label']))
            I1 = I1.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = net(I1)
            #print("label shape", label.shape)
            #print("output shape", output.shape)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                I1 = Variable(batch['I1'].float())
                label = torch.squeeze(Variable(batch['label']))
                I1 = I1.to(device)
                label = label.to(device)

                output = net(I1)
                loss = criterion(output, label.long())
                val_loss += loss.item()

        val_loss /= len(val_loader)  # Média da perda de validação

        # Atualizar o scheduler
        scheduler.step(val_loss)


        epoch_train_loss[epoch_index], epoch_train_accuracy[epoch_index], cl_acc, pr_rec = test(train_dataset)
        epoch_train_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_train_change_accuracy[epoch_index] = cl_acc[1]
        epoch_train_precision[epoch_index] = pr_rec[0]
        epoch_train_recall[epoch_index] = pr_rec[1]
        epoch_train_Fmeasure[epoch_index] = pr_rec[2]

        #         epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec = test(test_dataset)
        epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index], cl_acc, pr_rec = test(val_dataset)
        epoch_test_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_test_change_accuracy[epoch_index] = cl_acc[1]
        epoch_test_precision[epoch_index] = pr_rec[0]
        epoch_test_recall[epoch_index] = pr_rec[1]
        epoch_test_Fmeasure[epoch_index] = pr_rec[2]

        plt.figure(num=1)
        plt.clf()
        l1_1, = plt.plot(t[:epoch_index + 1], epoch_train_loss[:epoch_index + 1], label='Train loss')
        l1_2, = plt.plot(t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1], label='Test loss')
        plt.legend(handles=[l1_1, l1_2])
        plt.grid()
        #         plt.gcf().gca().set_ylim(bottom = 0)
        plt.gcf().gca().set_xlim(left=0)
        plt.title('Loss')
        # plt.clf()
        plt.gcf()

        plt.figure(num=2)
        plt.clf()
        l2_1, = plt.plot(t[:epoch_index + 1], epoch_train_accuracy[:epoch_index + 1], label='Train accuracy')
        l2_2, = plt.plot(t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1], label='Test accuracy')
        plt.legend(handles=[l2_1, l2_2])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Accuracy')
        # plt.clf()
        plt.gcf()

        plt.figure(num=3)
        plt.clf()
        l3_1, = plt.plot(t[:epoch_index + 1], epoch_train_nochange_accuracy[:epoch_index + 1],
                         label='Train accuracy: no change')
        l3_2, = plt.plot(t[:epoch_index + 1], epoch_train_change_accuracy[:epoch_index + 1],
                         label='Train accuracy: change')
        l3_3, = plt.plot(t[:epoch_index + 1], epoch_test_nochange_accuracy[:epoch_index + 1],
                         label='Test accuracy: no change')
        l3_4, = plt.plot(t[:epoch_index + 1], epoch_test_change_accuracy[:epoch_index + 1],
                         label='Test accuracy: change')
        plt.legend(handles=[l3_1, l3_2, l3_3, l3_4])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Accuracy per class')
        # plt.clf()
        plt.gcf()

        plt.figure(num=4)
        plt.clf()
        l4_1, = plt.plot(t[:epoch_index + 1], epoch_train_precision[:epoch_index + 1], label='Train precision')
        l4_2, = plt.plot(t[:epoch_index + 1], epoch_train_recall[:epoch_index + 1], label='Train recall')
        l4_3, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure[:epoch_index + 1], label='Train Dice/F1')
        l4_4, = plt.plot(t[:epoch_index + 1], epoch_test_precision[:epoch_index + 1], label='Test precision')
        l4_5, = plt.plot(t[:epoch_index + 1], epoch_test_recall[:epoch_index + 1], label='Test recall')
        l4_6, = plt.plot(t[:epoch_index + 1], epoch_test_Fmeasure[:epoch_index + 1], label='Test Dice/F1')
        plt.legend(handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 1)
        #         plt.gcf().gca().set_ylim(bottom = 0)
        #         plt.gcf().gca().set_xlim(left = 0)
        plt.title('Precision, Recall and F-measure')
        # plt.clf()
        plt.gcf()

        #         mean_acc = (epoch_test_nochange_accuracy[epoch_index] + epoch_test_change_accuracy[epoch_index])/2
        #         if mean_acc > best_mean_acc:
        #             best_mean_acc = mean_acc
        #             save_str = PATH_CLUSTER + 'net-best_epoch-' + str(epoch_index + 1) + '_acc-' + str(mean_acc) + '.pth.tar'
        #             torch.save(net.state_dict(), save_str)

        #         fm = pr_rec[2]
        fm = epoch_train_Fmeasure[epoch_index]
        if fm > best_fm:
            best_fm = fm
            save_str = PATH_CLUSTER + net_name + 'net-best_epoch-' + str(epoch_index + 1) + '_fm-' + str(
                fm) + '.pth.tar'
            torch.save(net.state_dict(), save_str)

        lss = epoch_train_loss[epoch_index]
        if lss < best_lss:
            best_lss = lss
            save_str = PATH_CLUSTER + net_name + 'net-best_epoch-' + str(epoch_index + 1) + '_loss-' + str(
                lss) + '.pth.tar'
            torch.save(net.state_dict(), save_str)

        #         print('Epoch loss: ' + str(tot_loss/tot_count))
        if save:
            im_format = 'png'
            #         im_format = 'eps'

            plt.figure(num=1)
            plt.savefig(PATH_CLUSTER + net_name + '-01-loss.' + im_format)

            plt.figure(num=2)
            plt.savefig(PATH_CLUSTER + net_name + '-02-accuracy.' + im_format)

            plt.figure(num=3)
            plt.savefig(PATH_CLUSTER + net_name + '-03-accuracy-per-class.' + im_format)

            plt.figure(num=4)
            plt.savefig(PATH_CLUSTER + net_name + '-04-prec-rec-fmeas.' + im_format)

    epochs = list(
        range(len(epoch_train_loss)))  # Supondo que a quantidade de epochs seja igual ao tamanho de qualquer lista
    data = {
        "Epoch": epochs,
        "Train Loss": epoch_train_loss,
        "Train Accuracy": epoch_train_accuracy,
        "Train No-Change Accuracy": epoch_train_nochange_accuracy,
        "Train Change Accuracy": epoch_train_change_accuracy,
        "Train Precision": epoch_train_precision,
        "Train Recall": epoch_train_recall,
        "Train F-Measure": epoch_train_Fmeasure,
        "Test Loss": epoch_test_loss,
        "Test Accuracy": epoch_test_accuracy,
        "Test No-Change Accuracy": epoch_test_nochange_accuracy,
        "Test Change Accuracy": epoch_test_change_accuracy,
        "Test Precision": epoch_test_precision,
        "Test Recall": epoch_test_recall,
        "Test F-Measure": epoch_test_Fmeasure,
    }

    # Cria o DataFrame
    df = pd.DataFrame(data)

    # Salva no arquivo CSV
    df.to_csv(PATH_CLUSTER + net_name + "_training_metrics.csv", index=False)

    print("Dados salvos em 'training_metrics.csv'.")

    out = {'train_loss': epoch_train_loss[-1],
           'train_accuracy': epoch_train_accuracy[-1],
           'train_nochange_accuracy': epoch_train_nochange_accuracy[-1],
           'train_change_accuracy': epoch_train_change_accuracy[-1],
           'test_loss': epoch_test_loss[-1],
           'test_accuracy': epoch_test_accuracy[-1],
           'test_nochange_accuracy': epoch_test_nochange_accuracy[-1],
           'test_change_accuracy': epoch_test_change_accuracy[-1]}

    print('pr_c, rec_c, f_meas, pr_nc, rec_nc')
    # print(pr_rec)

    return out


def test(dset):
    net.eval()
    tot_loss = 0
    tot_count = 0
    tot_accurate = 0

    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for img_index in dset.names:
        I1_full, cm_full = dset.get_img(img_index)

        s = cm_full.shape

        steps0 = np.arange(0, s[0], ceil(s[0] / N))
        steps1 = np.arange(0, s[1], ceil(s[1] / N))
        for ii in range(N):
            for jj in range(N):
                xmin = steps0[ii]
                if ii == N - 1:
                    xmax = s[0]
                else:
                    xmax = steps0[ii + 1]
                # ymin = jj
                ymin = steps1[jj]
                if jj == N - 1:
                    ymax = s[1]
                else:
                    ymax = steps1[jj + 1]
                I1 = I1_full[:, xmin:xmax, ymin:ymax]
                # I2 = I2_full[:, xmin:xmax, ymin:ymax]
                cm = cm_full[xmin:xmax, ymin:ymax]

                I1 = Variable(torch.unsqueeze(I1, 0).float())
                # I2 = Variable(torch.unsqueeze(I2, 0).float())
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0 * cm), 0).float())
                I1 = I1.to(device)
                cm = cm.to(device)

                output = net(I1)
                loss = criterion(output, cm.long())
                #         print(loss)
                tot_loss += loss.data * np.prod(cm.size())
                tot_count += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)

                c = (predicted.int() == cm.data.int())
                for i in range(c.size(1)):
                    for j in range(c.size(2)):
                        l = int(cm.data[0, i, j])
                        class_correct[l] += c[0, i, j]
                        class_total[l] += 1

                pr = (predicted.int() > 0).cpu().numpy()
                gt = (cm.data.int() > 0).cpu().numpy()

                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()

    net_loss = tot_loss / tot_count
    net_accuracy = 100 * (tp + tn) / tot_count

    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i], 0.00001)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_meas = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)

    pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]

    return net_loss, net_accuracy, class_accuracy, pr_rec




def save_test_results(dset, save_path):
    tempos = []
    net.eval()

    with torch.no_grad():  # Desativa cálculo de gradientes
        for name in tqdm(dset.names):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                I1, cm = dset.get_img(name)
                I1 = Variable(torch.unsqueeze(I1, 0).float()).to(device)

                t_start = time.time()
                out = net(I1)
                t_end = time.time()

                tempo_infer = t_end - t_start
                tempos.append(tempo_infer)

                _, predicted = torch.max(out.data, 1)
                I = 255 * np.squeeze(predicted.cpu().numpy())
                I = I.astype(np.uint8)

                save_file = f'{PATH_CLUSTER}{save_path}{net_name}-{name}.png'
                io.imsave(save_file, I)

    return tempos


def kappa(tp, tn, fp, fn):
    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (N * N)

    return (p0 - pe) / (1 - pe)



def verify_memory(process):
    # Obtém o uso de memória em bytes
    memory_info = process.memory_info()
    memory_used = memory_info.rss  # Uso de memória residente em bytes

    # Converte para megabytes
    memory_used_mb = memory_used / (1024 ** 2)

    print(f"Memória usada: {memory_used_mb:.2f} MB")


if __name__ == "__main__":
    cache_dir = PATH_CLUSTER + 'newcache'
    os.makedirs(cache_dir, exist_ok=True)
    os.chmod(cache_dir, 0o777)  # Garantir permissões de escrita

    os.environ['TORCH_HOME'] = cache_dir
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Global Variables' Definitions
    # Obtém o ID do processo atual
    pid = os.getpid()

    # Cria um objeto Process para o processo atual
    process = psutil.Process(pid)

    verify_memory(process)

    if DATA_AUG:
        data_transform = tr.Compose([RandomFlip(), RandomRot()])
    else:
        data_transform = None

    train_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=0, stride=TRAIN_STRIDE, transform=data_transform, patch_side=PATCH_SIDE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=1, stride=TRAIN_STRIDE, patch_side=PATCH_SIDE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    test_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=2, stride=TRAIN_STRIDE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    verify_memory(process)
    print('DATASETS OK')

    #print('\nUNET 5 com NDVI\n')

    # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

    print('\nProposed with Dice Loss\n')

    if TYPE == 0:
        net = Proposed(in_channels=3, out_channels=2)
        net_name = 'Proposed_Dice'
    elif TYPE == 1:
        net, net_name = Proposed(in_channels=3, out_channels=2), 'Proposed_Dice'

    net.to(device)

    criterion = CombinedLoss(alpha=0.1, beta=0.9, smooth=2.9, gamma=3.9)

    print('NETWORK OK')
    verify_memory(process)

    print('Number of trainable parameters:', count_parameters(net))

    if LOAD_TRAINED:
        if TYPE == 0:
            model_path = 'Models/Proposed.pth.tar'
        if TYPE == 1:
            model_path = 'Models/Proposed.pth.tar'
        net.eval()
        state_dict = torch.load(model_path, map_location=device)

        # Assuming net is your model
        net.load_state_dict(state_dict)
        print('LOAD OK')
        verify_memory(process)

    else:
        t_start = time.time()
        out_dic = train()
        t_end = time.time()
        print(out_dic)
        print(f'\n\nTraining time {net_name}:')
        elapsed_time = (t_end - t_start) / 60  # tempo em minutos
        print(f'{elapsed_time:.2f} minutes')

    if not LOAD_TRAINED:
        torch.save(net.state_dict(), PATH_CLUSTER + net_name + '.pth.tar')
        print('SAVE OK')

    tempo = save_test_results(test_dataset, 'Results/')
    print(f"Tempo médio de inferência: {np.mean(tempo):.4f} segundos")
    print(f"Tempo total de inferência: {np.sum(tempo) / 60:.2f} minutos")




