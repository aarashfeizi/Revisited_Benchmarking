import argparse
import collections
import math
import os

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.top_model import *


import revisited_dataset
import tasks




def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('-mp', '--model_path', default='')
    parser.add_argument('-sp', '--save_path', default='embs/')
    parser.add_argument('-od', '--output_dim', default=2048, type=int)
    parser.add_argument('-bs', '--batch_size', default=16, type=int)
    parser.add_argument('-is', '--image_size', default=224, type=int)
    parser.add_argument('-ppth', '--project_path', default='./')
    parser.add_argument('-fe', '--feat_extractor', default='resnet50')
    parser.add_argument('-pool', '--pooling', default='spoc', choices=['spoc', 'gem', 'rmac', 'mac'])
    parser.add_argument('-mm', '--merge_method', default='diff-sim', choices=['diff-sim', 'diff', 'sim', 'concat'])
    parser.add_argument('-el', '--extra_layer', default=0, type=int)
    parser.add_argument('-n', '--normalize', default=False, action='store_true')

    parser.add_argument('-nw', '--num_workers', default=2, type=int)



    args = parser.parse_args()

    return args

def vector_merge_function(v1, v2, method='sim', normalize=True):
    if method == 'diff':
        ret = torch.pow((v1 - v2), 2)
        if normalize:
            ret = F.normalize(ret, p=2, dim=1)
        # ret = torch.div(ret, ret.max(dim=1, keepdim=True)[0])
        # return torch.div(ret, torch.sqrt(torch.sum(torch.pow(ret, 2))))
        return ret
    elif method == 'sim':

        if normalize:
            ret = F.normalize(v1, p=2, dim=1) * F.normalize(v2, p=2, dim=1)
        else:
            ret = v1 * v2
        # ret = torch.div(ret, ret.max(dim=1, keepdim=True)[0])
        # ret = F.normalize(ret, p=2, dim=1)
        return ret
        # return torch.div(ret, torch.sqrt(torch.sum(torch.pow(ret, 2))))
    elif method == 'diff-sim':
        diff_merged = torch.pow((v1 - v2), 2)
        if normalize:
            diff_merged = F.normalize(diff_merged, p=2, dim=1)
            sim_merged = F.normalize(v1, p=2, dim=1) * F.normalize(v2, p=2, dim=1)
        else:
            sim_merged = v1 * v2
        # diff_merged = torch.div(diff_merged, diff_merged.max(dim=1, keepdim=True)[0])
        # sim_merged = torch.div(sim_merged, sim_merged.max(dim=1, keepdim=True)[0])

        # diff_merged = F.normalize(diff_merged, p=2, dim=1)
        # sim_merged = F.normalize(sim_merged, p=2, dim=1)
        # ret1 = torch.div(diff_merged, torch.sqrt(torch.sum(torch.pow(diff_merged, 2))))
        # ret2 = torch.div(sim_merged, torch.sqrt(torch.sum(torch.pow(sim_merged, 2))))
        #
        # ret1 = torch.nn.BatchNorm1d(diff_merged)
        # ret2 = torch.nn.BatchNorm1d(sim_merged)

        return torch.cat([diff_merged, sim_merged], dim=1)
    elif method == 'concat':
        merged = torch.cat([v1, v2], dim=1)
        return merged
    else:
        raise Exception(f'Merge method {method} not implemented.')


def get_features(args, cfg):
    """

    Returns
    -------
    a dictionary with keys "Q" and "X", both values with shape (d, N)
    """

    import pdb
    model_path = args.model_path
    save_path = args.save_path

    net = top_module(args)

    # pretrained: /Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/savedmodels/PRETRAINED_model-bs10-1gpus-newheatmaps-diffsim-dsn_hotels-nor_200-fe_resnet50-pool_rmac-el_0-nn_3-bs_10-lrs_0.003-lrr_3e-06-m_1.0-loss_trpl-mm_diffsim-bco_1.0-igsz_224-time_2021-01-17_16-28-36-299758/model-epoch-20-val-acc-0.6875.pt

    model = load_model(args, net, model_path).ft_net

    if args.cuda:
        model = model.cuda()


    normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_list = [transforms.CenterCrop(args.image_size),
                      transforms.ToTensor(),
                      transforms.Normalize(**normalize_param)]

    transform = transforms.Compose(transform_list)

    x_oxf_dataset = revisited_dataset.Revisited_Dataset(cfg, 'im_fname', cfg['n'], transform=transform)
    q_oxf_dataset = revisited_dataset.Revisited_Dataset(cfg, 'qim_fname', cfg['nq'], transform=transform)

    x_dataloader = DataLoader(x_oxf_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True)
    q_dataloader = DataLoader(q_oxf_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True)

    task = tasks.TaskManager(args, save_path)

    x = task.run(model, dataloader=x_dataloader)



    q = task.run(model, dataloader=q_dataloader)


    d = {'X': x,
         'Q': q}
    print('DONE embeddings!!')
    return d


def load_model(args, net, best_model_path):
    if args.cuda:
        checkpoint = torch.load(best_model_path)
    else:
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    print('Loading model %s from epoch [%d]' % (best_model_path, checkpoint['epoch']))
    o_dic = checkpoint['model_state_dict']
    exp = True
    counter = 1
    exp_msg = ''
    while exp and counter < 4:
        try:
            net.load_state_dict(o_dic)
            exp = False
        except Exception as e:
            exp_msg = e
            counter += 1
            print(str(exp))
            new_o_dic = collections.OrderedDict()
            for k, v in o_dic.items():
                new_o_dic[k[7:]] = v
            o_dic = new_o_dic
    if exp:
        raise Exception(exp_msg)
    return net

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


# python example_evaluate.py -mp /Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/savedmodels/PRETRAINED_model-bs10-1gpus-newheatmaps-diffsim-dsn_hotels-nor_200-fe_resnet50-pool_rmac-el_0-nn_3-bs_10-lrs_0.003-lrr_3e-06-m_1.0-loss_trpl-mm_diffsim-bco_1.0-igsz_224-time_2021-01-17_16-28-36-299758/model-epoch-20-val-acc-0.6875.pt -pool rmac -nw 0