import argparse
import os
import torch
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import copy
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from reparam_module import ReparamModule

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--dataset', type=str, default='gzoo2', help='dataset')
parser.add_argument('--subset', type=str, default='imagenette', help='subset')
parser.add_argument('--model', type=str, default='ConvNet', help='model')
parser.add_argument('--num_experts', type=int, default=10, help='training iterations') # delete
parser.add_argument('--lr_teacher', type=float, default=0.001, help='learning rate for updating network parameters')
parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                    help='whether to use differentiable Siamese augmentation.')
parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                    help='differentiable Siamese augmentation strategy')
parser.add_argument('--data_path', type=str, default='data', help='dataset path') # delete
parser.add_argument('--buffer_path', type=str, default='/data/sbcaesar/galaxy_buffers', help='buffer path')
parser.add_argument('--train_epochs', type=int, default=30)
parser.add_argument('--zca', action='store_true')
parser.add_argument('--decay', action='store_true')
parser.add_argument('--mom', type=float, default=0, help='momentum')
parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--expert_num', type=int, default=0, help='l2 regularization')
parser.add_argument('--epoch_num', type=int, default=0)

args = parser.parse_args()

args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
student_net = ReparamModule(student_net)
student_net.train()
num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

expert_dir = os.path.join(args.buffer_path, 'gzoo2', args.model)
buffer = torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(0)))
print('doing expert', args.expert_num, 'epoch', args.epoch_num)
student_params = None
for i in range(2):
    expert_trajectory = buffer[i] # Expert Number
    starting_params = expert_trajectory[args.epoch_num] # Epoch Number
    if student_params is not None:
        student_params += torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
    else:
        student_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
student_params /= 2
student_params = student_params.requires_grad_(True)
criterion = nn.CrossEntropyLoss().to(args.device)
net = student_net

loss_avg, acc_avg, num_exp = 0, 0, 0
net = net.to(args.device)

net.eval()

for i_batch, datum in enumerate(testloader):
    img = datum[0].float().to(args.device)
    lab = datum[1].long().to(args.device)
    n_b = lab.shape[0]
    output = net(img, flat_param=student_params)
    loss = criterion(output, lab)

    acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

    loss_avg += loss.item() * n_b
    acc_avg += acc
    num_exp += n_b

loss_avg /= num_exp
acc_avg /= num_exp

print(loss_avg, acc_avg)

if __name__ == '__main__':
    exit(0)