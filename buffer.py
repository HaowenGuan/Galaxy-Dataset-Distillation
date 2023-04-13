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
from MSECrossEntropyLoss import MSECrossEntropyLoss
# This is needed to load galaxy dataset file
from gzoo2_dataset import GZooDataset, CustomDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    # Calculate the weight for loss function:
    class_count = torch.zeros(num_classes)
    dataset_count = 0
    for c in range(num_classes):
        class_count[c] = len(indices_class[c])
        dataset_count += len(indices_class[c])
    loss_weight = dataset_count / class_count
    loss_weight = loss_weight / torch.mean(loss_weight)
    print('Add weight to loss function', loss_weight)
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss(weight=loss_weight).to(args.device)
    # criterion = MSECrossEntropyLoss(weight=loss_weight).to(args.device)

    trajectories = []

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)

    total_train_cf, total_test_cf = np.array([[0] * num_classes for _ in range(num_classes)]), np.array([[0] * num_classes for _ in range(num_classes)])

    for it in range(0, args.num_experts):

        ''' Train synthetic data '''
        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        print(teacher_net)
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):

            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True)

            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, aug=False)

            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}\tAVG Train loss: {}\tAVG Test loss: {}".format(it, e, train_acc, test_acc, train_loss, test_loss))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()


        for dataset, name, count, total in [[trainloader, "train", len(dst_train), total_train_cf],[testloader, "test", len(dst_test), total_test_cf]]:
            total_acc = torch.zeros(num_classes)

            pred = []
            true = []
            for i_batch, datum in enumerate(dataset):
                img = datum[0].float().to(args.device)
                lab = datum[1].long().to(args.device)

                output = teacher_net(img)
                output = torch.argmax(output, 1)

                pred += output.tolist()
                true += lab.tolist()

                for i in range(lab.shape[0]):
                    if output[i] == lab[i]:
                        total_acc[lab[i]] += 1

            print(name, "set ACC of each class", total_acc / count)
        
            cf_matrix = confusion_matrix(true, pred)
            total += np.array(cf_matrix)
            print(cf_matrix)
            df_cm = pd.DataFrame(cf_matrix, index = [i for i in class_names],
                     columns = [i for i in class_names])
            plt.figure(figsize = (12,7))
            sn.heatmap(df_cm, annot=True, fmt='g')
            plt.title('Confusion Matrix Expert{} {}'.format(it,name))
            plt.xlabel("Prediction")
            plt.ylabel("True Label")
            plt.savefig('./cf_matrix_buffer/cf_expert{}_{}.png'.format(it,name))

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []

    # print total confusion matrix across all experts
    for name, cf_matrix in [["train", total_train_cf],["test", total_test_cf]]:
        cf_matrix = cf_matrix.tolist()
        for r in cf_matrix:
            t = sum(r)
            for i in range(len(r)):
                r[i] = round(r[i] / t, 3)
        df_cm = pd.DataFrame(cf_matrix, index=[i for i in class_names], columns=[i for i in class_names])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.title('Confusion Matrix total {}'.format(name))
        plt.xlabel("Prediction")
        plt.ylabel("True Label")
        plt.savefig('./cf_matrix_buffer/cf_total_{}.png'.format(name))


if __name__ == '__main__':
    import argparse
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=10, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='/data/sbcaesar/cifar10_buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()
    main(args)


