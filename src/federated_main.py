#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import os
import torch
# import tensorboardX.SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import yaml
import _pickle as pickle

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNEmnist
from utils import get_dataset, average_weights, exp_details, weighted_averages_n_samples, weighted_averages_n_classes
from datetime import datetime

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = None# SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device is: {device}')

    train_dataset, test_dataset, user_groups = get_dataset(args)
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M")
    # as requested in comment
    # user_groups = {'user_groups': user_groups}
    with open(f'user_groups_{current_time}.txt', 'w') as f:
        for userId in range(0, len(user_groups)):
            print(f'userId: {userId}', file=f)
            samples_ids = []
            samples_classes = []
            for s in user_groups[userId]:
                samples_ids.append(int(s))
            samples_ids.sort()

            for i in samples_ids:
                if args.dataset != 'cifar':
                    samples_classes.append(int(train_dataset.train_labels[i]))
                else:
                    samples_classes.append(int(train_dataset.targets[i]))
            samples_classes.sort()
            # print(f'samples_ids: {samples_ids}', file=f)
            print(f'samples_size: {len(samples_classes)}', file=f)
            print(f'samples_classes: {set(samples_classes)}', file=f)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'emnist-balanced':
            global_model = CNNEmnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    test_loss_list, test_accuracy_list = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        data, data_classes, samples_per_class = [], [], []
        classes = []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            user_data = len(user_groups[idx])
            data.append(user_data)

            # adding each client's data classes
            ids = []
            for k in user_groups[idx]:
                ids.append(int(k))
            user_samples_per_class = {i: 0 for i in range(47)}
            if args.dataset != 'cifar':
                for i in ids:
                    if int(train_dataset.train_labels[i]) not in classes:
                        classes.append(int(train_dataset.train_labels[i]))
                    user_samples_per_class[int(train_dataset.train_labels[i])] += 1
            else:
                user_data_classes = set(int(train_dataset.targets[i]) for i in ids)

            # data_classes.append(user_data_classes)
            samples_per_class.append(user_samples_per_class)
        # update global weights
        if args.avg_type == 'avg':
            global_weights = average_weights(local_weights)
        elif args.avg_type == 'avg_n_samples':
            global_weights = weighted_averages_n_samples(local_weights, data)
        else:
            # global_weights = weighted_averages_n_classes(local_weights, data_classes)
            global_weights = weighted_averages_n_classes(local_weights, samples_per_class, classes, data)

        previous_model = global_model
        previous_test_accuracy = test_accuracy_list[-1] if len(test_accuracy_list) > 0 else 0
        previous_test_loss = test_loss_list[-1] if len(test_loss_list) > 0 else 0
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(round(loss_avg, 3))

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(round((sum(list_acc) / len(list_acc)), 3))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        if test_acc > previous_test_accuracy:
            test_accuracy_list.append(test_acc)
            test_loss_list.append(test_loss)
        else:
            test_accuracy_list.append(previous_test_accuracy)
            test_loss_list.append(previous_test_loss)
            global_model = previous_model

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
            print("Test Accuracy: {:.2f}%".format(100 * test_acc))
            print(f'Test Loss: {round((test_loss / 100), 3)}')

    #######################   PLOTTING & args & results saving    ###################################

    matplotlib.use('Agg')

    if args.iid == 1:
        iidness = "iid"
    elif args.iid == 0:
        iidness = "noniid"
    else:
        iidness = "extreme"
    my_path = os.getcwd()
    full_path = '{}/../save/{}/{}/{}/{}/{}'.format(my_path, args.dataset, iidness,
                                                   args.avg_type, args.epochs, args.number_of_classes_of_half_of_user, current_time)
    os.makedirs(full_path)

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{full_path}/training_loss.pdf')

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{full_path}/training_accuracy.pdf')

    plt.figure()
    plt.title('Testing Loss vs Communication rounds')
    plt.plot(range(len(test_loss_list)), test_loss_list, color='r')
    plt.ylabel('Testing loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{full_path}/testing_loss.pdf')

    plt.figure()
    plt.title('Testing Accuracy vs Communication rounds')
    plt.plot(range(len(test_accuracy_list)), test_accuracy_list, color='k')
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{full_path}/testing_accuracy.pdf')

    # yaml file with all data and results
    data = dict(
        epochs=args.epochs,
        num_users=args.num_users,
        frac=args.frac,
        local_ep=args.local_ep,
        local_bs=args.local_bs,
        lr=args.lr,
        momentum=args.momentum,
        model=args.model,
        kernel_num=args.kernel_num,
        kernel_sizes=args.kernel_sizes,
        num_channels=args.num_channels,
        norm=args.norm,
        num_filters=args.num_filters,
        max_pool=args.max_pool,
        dataset=args.dataset,
        num_classes=args.num_classes,
        optimizer=args.optimizer,
        iid=args.iid,
        unequal=args.unequal,
        verbose=args.verbose,
        seed=args.seed,
        avg_type=args.avg_type,
        train_accuracy=train_accuracy,
        train_loss=train_loss,
        avg_train_accuracy=round(train_accuracy[-1], 3),
        avg_train_loss=round((train_loss[-1] / 100), 3),
        test_accuracy_list=test_accuracy_list,
        test_loss_list=test_loss_list,
        number_of_classes_of_half_of_user=args.number_of_classes_of_half_of_user
    )

    with open(f'{full_path}/data.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
