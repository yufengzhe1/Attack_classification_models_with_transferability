import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import input_diversity, get_gaussian_kernel, get_TI_kernel
from model import load_models, get_logits
from data import MyDataset, save_imgs, imgnormalize


variance = np.random.uniform(0.5, 5.5, 3)
neg_perturbations = - variance
liner_interval = np.append(variance, neg_perturbations)
liner_interval = np.append(liner_interval, [0.])
liner_interval = np.sort(liner_interval)
# liner_interval = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Transfer attack')
    parser.add_argument('--source_model', nargs="+", default=['resnet50', 'densenet161'], help='source models')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=500, help='Size of image')
    parser.add_argument('--max_iterations', type=int, default=40, help='Number of iterations')
    parser.add_argument('--lr', type=eval, default=1.0/255., help='Step size')
    parser.add_argument('--linf_epsilon', type=float, default=12, help='The maximum pixel value can be changed')
    parser.add_argument('--di', type=eval, default="True", help='Whether to use Input Diversity')
    parser.add_argument('--input_path', type=str, default='../input_dir', help='Path of input')
    parser.add_argument('--result_path', type=str, default='../output_dir', help='Path of images to be saved')
    args = parser.parse_args()
    return args


def run_attack(args):
    input_folder = os.path.join(args.input_path, 'images/')
    adv_img_save_folder = os.path.join(args.result_path, 'adv_images/')
    if not os.path.exists(adv_img_save_folder):
        os.makedirs(adv_img_save_folder)

    # Dataset, dev50.csv is the label file
    data_set = MyDataset(csv_path=os.path.join(args.input_path, 'dev50.csv'), path=input_folder)
    data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    norm = imgnormalize()  # standard imagenet normalize
    device = torch.device("cuda:0")
    source_models = load_models(args.source_model, device)  # load model, maybe several models

    seed_num = 0  # set seed
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True

    # gaussian_kernel: filter high frequency information of images
    gaussian_smoothing = get_gaussian_kernel(device, kernel_size=5, sigma=1, channels=3)

    print('Start attack......')
    for i, data in enumerate(data_loader, 0):
        start_t = time.time()
        X_ori, labels_gt, filenames = data
        X_ori = X_ori.to(device)
        labels_gt = labels_gt.to(device)

        # the noise
        delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
        X_ori = gaussian_smoothing(X_ori)  # filter high frequency information of images

        for t in range(args.max_iterations):
            g_temp = []
            for tt in range(len(liner_interval)):
                if args.di:  # use Input Diversity
                    X_adv = X_ori + delta
                    X_adv = input_diversity(X_adv)
                    # images interpolated to 224*224, adaptive standard networks and reduce computation time
                    X_adv = nn.functional.interpolate(X_adv, (224, 224), mode='bilinear', align_corners=False)
                else:
                    c = liner_interval[tt]
                    X_adv = X_ori + c * delta
                    X_adv = nn.functional.interpolate(X_adv, (224, 224), mode='bilinear', align_corners=False)
                # get ensemble logits
                logits = get_logits(X_adv, source_models, norm)
                loss = -nn.CrossEntropyLoss()(logits, labels_gt)
                loss.backward()

                grad = delta.grad.clone()
                # TI: smooth gradient
                grad = F.conv2d(grad, get_TI_kernel(), bias=None, stride=1, padding=(2,2), groups=3)
                g_temp.append(grad)

            # calculate the mean and cancel out the noise, retained the effective noise
            g = 0.0
            for j in range(len(liner_interval)):
                g += g_temp[j]
            g = g / float(len(liner_interval))
            delta.grad.zero_()

            delta.data = delta.data - args.lr * torch.sign(g)
            delta.data = delta.data.clamp(-args.linf_epsilon/255., args.linf_epsilon/255.)
            delta.data = ((X_ori+delta.data).clamp(0.0, 1.0)) - X_ori

        save_imgs(X_ori+delta, adv_img_save_folder, filenames)  # save adv images
        end_t = time.time()
        print('Attack batch: {}/{};   Time spent(seconds): {:.2f}'.format(i, len(data_loader), end_t-start_t))


if __name__ == '__main__':
    start_time = time.time()
    args = parse_arguments()
    run_attack(args)
    print('Total time(seconds):{:.3f}'.format(time.time()-start_time))
