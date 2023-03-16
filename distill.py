import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    if args.wandb_name:
        wandb.init(sync_tensorboard=False,
                   project="DatasetDistillation",
                   job_type="CleanRepo",
                   name=args.wandb_name,
                   config=args,
                   )
    else:
        wandb.init(sync_tensorboard=False,
                   project="DatasetDistillation",
                   job_type="CleanRepo",
                   config=args,
                   )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1 and args.ipc != 1
    # args.distributed = False
    print("--------------------------------------------------------------", args.distributed)


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

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

    class_weights = []
    for c in range(num_classes):
        class_weights.append(1-len(indices_class[c])/len(labels_all))
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    print("Class Weights:", class_weights)
    class_weights = torch.tensor(class_weights).to(args.device)

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]


    ''' initialize the synthetic data '''
    # label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    label_syn = torch.tensor([np.ones(args.ipc, dtype=np.int_) * i for i in range(num_classes)], dtype=torch.long,
                             requires_grad=False, device=args.device).view(-1)
    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        if args.load_syn_image:
            image_syn = torch.load(os.path.join(".", "logged_files", args.dataset, args.load_syn_image))
        else:
            image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)


    # syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    # log_syn_lr = torch.log(syn_lr).detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    # optimizer_lr = torch.optim.SGD([log_syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    from torchvision import transforms
    blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 0.3)).to(args.device)

    # stage match trajectory #0 --------------------
    start_epoch_cap = args.init_epoch
    test_loss = []
    min_test = float('inf')
    min_test_idx = 0
    trending = False
    print(wandb.run.name)
    log_syn_lr_list = []
    optimizer_lr_list = []
    print(args.lr_teacher)
    if isinstance(args.lr_teacher, list):
        for i in range(start_epoch_cap):
            if i >= len(args.lr_teacher):
                i = -1
            syn_lr = torch.tensor(float(args.lr_teacher[i])).to(args.device)
            log_syn_lr = torch.log(syn_lr).detach().to(args.device).requires_grad_(True)
            optimizer_lr = torch.optim.SGD([log_syn_lr], lr=args.lr_lr, momentum=0.5)
            log_syn_lr_list.append(log_syn_lr)
            optimizer_lr_list.append(optimizer_lr)
    else:
        print("Warning, the lr_teacher should be a list")
        syn_lr = torch.tensor(args.lr_teacher).to(args.device)
        for _ in range(start_epoch_cap):
            log_syn_lr = torch.log(syn_lr).detach().to(args.device).requires_grad_(True)
            optimizer_lr = torch.optim.SGD([log_syn_lr], lr=args.lr_lr, momentum=0.5)
            log_syn_lr_list.append(log_syn_lr)
            optimizer_lr_list.append(optimizer_lr)
    # ----------------------------------------------

    for it in range(args.prev_iter, args.Iteration+1):
        # stage match trajectory #1 --------------------
        start_epoch = it % int(start_epoch_cap)
        syn_lr = torch.exp(log_syn_lr_list[start_epoch])
        test_syn_lr = torch.exp(log_syn_lr_list[-1])
        # ----------------------------------------------
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        # wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                
                total_train_cf, total_test_cf = np.array([[0] * num_classes for _ in range(num_classes)]), np.array([[0] * num_classes for _ in range(num_classes)])
                
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    # [Using the last epoch's lr to do evaluation]
                    # args.lr_net = torch.exp(log_syn_lr_list[-1]).item() * 5

                    # [Using the a list of lr to do evaluation]
                    args.lr_net = []
                    for i in log_syn_lr_list:
                        args.lr_net.append(torch.exp(i).item())

                    _, acc_train, acc_test, train_cf, test_cf = evaluate_synset(it, it_eval, net_eval, num_classes, image_syn_eval, label_syn_eval, dst_test, testloader, args, texture=args.texture)
                    total_train_cf += train_cf
                    total_test_cf += test_cf
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                # stage match trajectory #2 --------------------
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                #     start_epoch_cap = int(start_epoch_cap) # 重置stage累计
                # else:
                #     if it > 1000:
                #         start_epoch_cap += 0.5
                #         start_epoch_cap = min(start_epoch_cap, args.max_start_epoch)
                # ----------------------------------------------
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

                for name, cf_matrix in [["test", total_test_cf]]:
                    cf_matrix = cf_matrix.tolist()
                    for r in cf_matrix:
                        t = sum(r)
                        for i in range(len(r)):
                            r[i] = round(r[i] / t, 3)
                    df_cm = pd.DataFrame(cf_matrix, index=[i for i in class_names], columns=[i for i in class_names])
                    plt.figure(figsize=(12, 7))
                    sn.heatmap(df_cm, annot=True, fmt='g')
                    plt.title('Confusion Matrix Iteration{} {}'.format(it, name))
                    plt.xlabel("Prediction")
                    plt.ylabel("True Label")
                    plt.savefig('./cf_matrix_distill/cf_iteration_{}_{}.png'.format(it, name))
                    wandb.log({"cf_iteration": wandb.Image(plt)}, step=it)

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            print(it, 'lr_list --------------------------------')
            for lr in log_syn_lr_list:
                print(torch.exp(lr).item(), end =" ")
            with torch.no_grad():
                image_save = image_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

                # wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    # wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "Imalr_teachergeNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR_" + str(start_epoch): syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        # Using Next Epoch as Test Set------------------
        starting_params = expert_trajectory[start_epoch_cap]
        target_params = expert_trajectory[start_epoch_cap + args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        student_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn
        y_hat = label_syn.to(args.device)
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params, create_graph=True)[0]

            student_params = student_params - test_syn_lr * grad

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        test_grand_loss = float(param_loss.detach().cpu())
        test_loss.append(test_grand_loss)
        if test_loss[-1] < min_test:
            min_test = test_loss[-1]
            min_test_idx = it
        wandb.log({"Next_Epoch_Test_Loss": test_loss[-1]}, step=it)
        del param_loss, param_dist, test_grand_loss
        #-----------------------------------------------

        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)


        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr_list[start_epoch].zero_grad()

        grand_loss.backward()

        optimizer_img.step()
        if not args.fix_lr_teacher:
            optimizer_lr_list[start_epoch].step()

        wandb.log({"Grand_Loss_epoch_" + str(start_epoch): grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch}, step=it)

        # Detect overfitting and level up!
        if it % 10 == 0 and len(test_loss) >= 100:
            correlation = np.corrcoef(test_loss, list(range(len(test_loss))))[0, 1]
            three_sigma = 3 * np.sqrt(1 / (len(test_loss) - 2))
            if not trending:
                interval = len(test_loss) // 2
                trending = correlation < -three_sigma
                if trending:
                    print('trending------------', it, correlation)
                if it % 100 == 0:
                    print("trending?????", np.corrcoef(test_loss, list(range(len(test_loss)))))
                # if not trending and interval < it - min_test_idx:
                #     # If model didn't produce a global minimum on right half of test data, increment syn_steps
                #     args.syn_steps = min(args.syn_steps + 1, 50)
                #     min_test_idx = it
                #     wandb.log({"Synthetic_Step": args.syn_steps}, step=it)
                if not trending and len(test_loss) % 200 == 0:
                    # If model didn't produce a global minimum on right half of test data, decrease lr_img
                    args.lr_img *= 0.9
                    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
                    wandb.log({"Image Learning Rate": args.lr_img}, step=it)
            if trending and np.mean(test_loss[-(2 * interval):-interval]) <= np.mean(test_loss[-interval:]) or \
                    correlation > three_sigma:
                # reset
                test_loss = []
                min_test = float('inf')
                trending = False
                if start_epoch_cap + 1 <= args.max_start_epoch:
                    start_epoch_cap += 1
                    log_syn_lr = log_syn_lr_list[-1].clone().detach().to(args.device).requires_grad_(True)
                    optimizer_lr = torch.optim.SGD([log_syn_lr], lr=args.lr_lr, momentum=0.5)
                    log_syn_lr_list.append(log_syn_lr)
                    optimizer_lr_list.append(optimizer_lr)

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=4000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    # parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_teacher', '--list', nargs='+', help='<Required> Set flag', required=True)

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data.')
    parser.add_argument('--max_start_epoch', type=int, default=29, help='max epoch we can start at. It must be smaller than buffer epoch.')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--init_epoch', type=int, default=1, help="starting point of stage wise distillation")
    parser.add_argument('--fix_lr_teacher', action='store_true', help="Fix the lr_teach if you are confident")
    parser.add_argument('--prev_iter', type=int, default=1, help="Resume training start from previous iter")
    parser.add_argument('--wandb_name', type=str, default=None, help="Custom WanDB name")
    parser.add_argument('--load_syn_image', type=str, default=None, help="previous syn image")

    args = parser.parse_args()

    print(args)
    main(args)
