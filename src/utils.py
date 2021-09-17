#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, emnist_iid, emnist_noniid_unequal, emnist_noniid
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'emnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # Chose euqal splits for every user
                    user_groups = mnist_noniid(train_dataset, args.num_users, args.number_of_classes_per_user)

        elif args.dataset == 'emnist-balanced':
            data_dir = '../data/emnist-balanced/'
            train_dataset = datasets.EMNIST(data_dir, split="balanced", train=True, download=True,
                                            transform=apply_transform)

            test_dataset = datasets.EMNIST(data_dir, split="balanced", train=False, download=True,
                                           transform=apply_transform)
            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = emnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    user_groups = emnist_noniid_unequal(train_dataset, args.num_users, args.samples_distribution_type)
                else:
                    # Chose euqal splits for every user
                    user_groups = emnist_noniid(train_dataset, args.num_users, args.number_of_classes_of_half_of_user)
        else:
            data_dir = '../data/emnist-byclass/'
            train_dataset = datasets.EMNIST(data_dir, split="byclass", train=True, download=True,
                                            transform=apply_transform)

            test_dataset = datasets.EMNIST(data_dir, split="byclass", train=False, download=True,
                                           transform=apply_transform)
            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = emnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    user_groups = emnist_noniid_unequal(train_dataset, args.num_users, args.samples_distribution_type)
                else:
                    # Chose euqal splits for every user
                    user_groups = emnist_noniid(train_dataset, args.num_users, args.samples_distribution_type)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
        # print(f'{w_avg[key]} ,')
    return w_avg


def weighted_averages_n_samples(w, data):
    """
    Returns the average of the weights.
    """
    sum_of_data = sum(data)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = 0
        for i in range(0, len(w)):
            client_data = data[i]
            percentage = client_data / sum_of_data
            w_avg[key] += w[i][key] * percentage
    return w_avg


# Aidmar awesome algorithm
# def weighted_averages_n_classes(w, samples_per_class, classes, data):
#     """
#     Returns the average of the weights.
#     """
#
#     # get the total sample size
#     sum_of_data = sum(data)
#
#     user_power_per_samples = []
#
#     # for each user calculate the weighted averaging = w
#     for i in range(0, len(w)):
#         client_data = data[i]
#         user_power_per_samples.append(client_data / sum_of_data)
#
#     # this for calculating how many samples per class own by all users.
#     total_samples_per_class = [0] * 47
#     for user_id in range(0, len(w)):
#         for classs in range(0, 47):
#             user_map = samples_per_class[user_id]
#             total_samples_per_class[classs] += user_map[classs]
#     # weighting all classes depending on total samples for each class
#     class_power_per_samples = []
#     for classs in range(0, 47):
#         class_power_per_samples.append(total_samples_per_class[classs] / sum_of_data)
#
#     # calculating the sum of classes weights of each user
#     sum_of_classes_weights_per_user = []
#     for user_id in range(0, len(w)):
#         total_user_classes_power = 0
#         for classs in range(0, 47):
#             user_map = samples_per_class[user_id]
#             if user_map[classs] != 0.0:
#                 total_user_classes_power += class_power_per_samples[classs]
#         sum_of_classes_weights_per_user.append(total_user_classes_power)
#
#     # calculating the final power of each user
#
#     power_of_users = []
#     for user in range(0, len(w)):
#         power_of_users.append(user_power_per_samples[user] * sum_of_classes_weights_per_user[user])
#
#     # finally... do the averaging stuff
#     some_of_powers = sum(power_of_users)
#     final_power_of_users = []
#     for user in range(0, len(w)):
#         final_power_of_users.append(power_of_users[user] / some_of_powers)
#
#     w_avg = copy.deepcopy(w[0])
#     for key in w_avg.keys():
#         w_avg[key] = 0
#         for i in range(0, len(w)):
#             w_avg[key] += w[i][key] * final_power_of_users[i]
#     return w_avg

# my awesome algorithm
def weighted_averages_n_classes(w, samples_per_class, classes, data):
    """
    Returns the average of the weights.
    """
    # this for calculating how many samples per class own by all users.
    # this will be used in the next step to calculate the power of each user in this class.
    total_samples_per_class = [0] * 47
    power_of_users_per_class_table = []
    for user_id in range(0, len(w)):
        for classs in range(0, 47):
            user_map = samples_per_class[user_id]
            total_samples_per_class[classs] += user_map[classs]
    # calculating the power of user per class.
    for user_id in range(0, len(w)):
        power_of_user_per_class = {}
        for classs in range(0, 47):
            user_map = samples_per_class[user_id]
            if total_samples_per_class[classs] != 0:
                power_of_user_per_class[classs] = user_map[classs] / total_samples_per_class[classs]
            else:
                power_of_user_per_class[classs] = 0
        power_of_users_per_class_table.append(power_of_user_per_class)

    # calculating the power of each user, and the total power
    power_of_users = [0] * len(w)
    for user_id in range(0, len(w)):
        power_of_user_per_class = power_of_users_per_class_table[user_id]
        power_sum = 0
        for i in range(0, 47):
            power_sum += power_of_user_per_class[i]
        power_of_users[user_id] = power_sum

    total_power = sum(power_of_users)

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = 0
        for i in range(0, len(w)):
            percentage = power_of_users[i] / total_power
            w_avg[key] += w[i][key] * percentage
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
