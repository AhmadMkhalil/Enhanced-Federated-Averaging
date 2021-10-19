#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 5
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def emnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = 1200  # int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def emnist_noniid(dataset, num_users, number_of_classes_of_half_of_user):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 112,800 training imgs -->  1128 shards X 100 imgs/shard
    num_shards, num_imgs = 1128, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    if number_of_classes_of_half_of_user == 0:

        # divide and assign 12 shards/client
        for i in range(num_users):
            list_of_shards = []
            starting_point = i * 2 * 12
            for j in range(starting_point, starting_point + 12):
                list_of_shards.append(j)

            set_of_shards = set(list_of_shards)
            # rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            # idx_shard = list(set(idx_shard) - rand_set)
            for shard in set_of_shards:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[shard * num_imgs:(shard + 1) * num_imgs]), axis=0)

    elif number_of_classes_of_half_of_user == 1:
        # first half of the users
        # divide and assign one class for each one of the first half of users
        already_gotten_classes = []
        for i in range(23):
            rand_class = random.randint(0, 22)  # new
            while rand_class in already_gotten_classes:  # new
                rand_class = random.randint(0, 22)  # new
            already_gotten_classes.append(rand_class)  # new
            list_of_shards = []
            starting_point = rand_class * 2 * 12  # new
            for j in range(starting_point, starting_point + 12):
                list_of_shards.append(j)

            set_of_shards = set(list_of_shards)
            for shard in set_of_shards:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[shard * num_imgs:(shard + 1) * num_imgs]), axis=0)

        # second half of the users
        num_items = 1200
        all_idxs = [i for i in range(len(dataset))]

        # remove the already distributed idxs (and their classes to avoid overlapping)
        #### to be applied in all places
        list_of_removed_classes = []
        for j in range(0, 23):
            for i in set(dict_users[j]):
                list_of_removed_classes.append(int(dataset.targets[int(i)]))
        set_of_removed_classes = set(list_of_removed_classes)
        to_be_deleted = []
        for idd in idxs:
            if int(dataset.targets[idd]) in set_of_removed_classes:
                to_be_deleted.append(idd)
        all_idxs = list(set(all_idxs) - set(to_be_deleted))
        #### to be applied in all places

        # distribute the iid across all rest users
        for i in range(23, 47):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))

    elif number_of_classes_of_half_of_user == 2:
        # first half of the users
        # divide and assign two classes for each one of the first half of users
        dict_users[0] = np.concatenate((dict_users[0], idxs[0:600]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[2400:3000]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[3000:3600]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[4800:5400]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[5400:6000]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[7200:7800]), axis=0)
        dict_users[3] = np.concatenate((dict_users[3], idxs[7800:8400]), axis=0)
        dict_users[3] = np.concatenate((dict_users[3], idxs[9600:10200]), axis=0)
        dict_users[4] = np.concatenate((dict_users[4], idxs[10200:10800]), axis=0)
        dict_users[4] = np.concatenate((dict_users[4], idxs[12000:12600]), axis=0)
        dict_users[5] = np.concatenate((dict_users[5], idxs[12600:13200]), axis=0)
        dict_users[5] = np.concatenate((dict_users[5], idxs[14400:15000]), axis=0)
        dict_users[6] = np.concatenate((dict_users[6], idxs[15000:15600]), axis=0)
        dict_users[6] = np.concatenate((dict_users[6], idxs[16800:17400]), axis=0)
        dict_users[7] = np.concatenate((dict_users[7], idxs[17400:18000]), axis=0)
        dict_users[7] = np.concatenate((dict_users[7], idxs[19200:19800]), axis=0)
        dict_users[8] = np.concatenate((dict_users[8], idxs[19800:20400]), axis=0)
        dict_users[8] = np.concatenate((dict_users[8], idxs[21600:22200]), axis=0)
        dict_users[9] = np.concatenate((dict_users[9], idxs[22200:22800]), axis=0)
        dict_users[9] = np.concatenate((dict_users[9], idxs[24000:24600]), axis=0)
        dict_users[10] = np.concatenate((dict_users[10], idxs[24600:25200]), axis=0)
        dict_users[10] = np.concatenate((dict_users[10], idxs[26400:27000]), axis=0)
        dict_users[11] = np.concatenate((dict_users[11], idxs[27000:27600]), axis=0)
        dict_users[11] = np.concatenate((dict_users[11], idxs[28800:29400]), axis=0)
        dict_users[12] = np.concatenate((dict_users[12], idxs[29400:30000]), axis=0)
        dict_users[12] = np.concatenate((dict_users[12], idxs[31200:31800]), axis=0)
        dict_users[13] = np.concatenate((dict_users[13], idxs[31800:32400]), axis=0)
        dict_users[13] = np.concatenate((dict_users[13], idxs[33600:34200]), axis=0)
        dict_users[14] = np.concatenate((dict_users[14], idxs[34200:34800]), axis=0)
        dict_users[14] = np.concatenate((dict_users[14], idxs[36000:36600]), axis=0)
        dict_users[15] = np.concatenate((dict_users[15], idxs[36600:37200]), axis=0)
        dict_users[15] = np.concatenate((dict_users[15], idxs[38400:39000]), axis=0)
        dict_users[16] = np.concatenate((dict_users[16], idxs[39000:39600]), axis=0)
        dict_users[16] = np.concatenate((dict_users[16], idxs[40800:41400]), axis=0)
        dict_users[17] = np.concatenate((dict_users[17], idxs[41400:42000]), axis=0)
        dict_users[17] = np.concatenate((dict_users[17], idxs[43200:43800]), axis=0)
        dict_users[18] = np.concatenate((dict_users[18], idxs[43800:44400]), axis=0)
        dict_users[18] = np.concatenate((dict_users[18], idxs[45600:46200]), axis=0)
        dict_users[19] = np.concatenate((dict_users[19], idxs[46200:46800]), axis=0)
        dict_users[19] = np.concatenate((dict_users[19], idxs[48000:48800]), axis=0)
        dict_users[20] = np.concatenate((dict_users[20], idxs[48800:49400]), axis=0)
        dict_users[20] = np.concatenate((dict_users[20], idxs[50400:51000]), axis=0)
        dict_users[21] = np.concatenate((dict_users[21], idxs[51000:51600]), axis=0)
        dict_users[21] = np.concatenate((dict_users[21], idxs[52800:53600]), axis=0)
        dict_users[22] = np.concatenate((dict_users[22], idxs[53600:54200]), axis=0)
        dict_users[22] = np.concatenate((dict_users[22], idxs[55200:55800]), axis=0)

        # second half of the users
        num_items = 1200
        all_idxs = [i for i in range(len(dataset))]
        # remove the already distributed idxs
        #### to be applied in all places
        list_of_removed_classes = []
        for j in range(0, 23):
            for i in set(dict_users[j]):
                list_of_removed_classes.append(int(dataset.targets[int(i)]))
        set_of_removed_classes = set(list_of_removed_classes)
        to_be_deleted = []
        for idd in idxs:
            if int(dataset.targets[idd]) in set_of_removed_classes:
                to_be_deleted.append(idd)
        all_idxs = list(set(all_idxs) - set(to_be_deleted))
        #### to be applied in all places
        # distribute the iid across all rest users
        for i in range(23, 47):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))

    elif number_of_classes_of_half_of_user == 3:
        # first half of the users
        # divide and assign three classes for each one of the first half of users
        dict_users[0] = np.concatenate((dict_users[0], idxs[0:400]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[2400:2800]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[4800:5200]), axis=0)

        dict_users[1] = np.concatenate((dict_users[1], idxs[2800:3200]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[5200:5600]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[7200:7600]), axis=0)

        for i in range(2, 23):
            dict_users[i] = np.concatenate((dict_users[i], idxs[i * 2400 + 800:i * 2400 + 800 + 400]), axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 1) * 2400 + 400:(i + 1) * 2400 + 400 + 400]),
                                           axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 2) * 2400:(i + 2) * 2400 + 400]), axis=0)

        # second half of the users
        num_items = 1200
        all_idxs = [i for i in range(len(dataset))]
        # remove the already distributed idxs
        #### to be applied in all places
        list_of_removed_classes = []
        for j in range(0, 23):
            for i in set(dict_users[j]):
                list_of_removed_classes.append(int(dataset.targets[int(i)]))
        set_of_removed_classes = set(list_of_removed_classes)
        to_be_deleted = []
        for idd in idxs:
            if int(dataset.targets[idd]) in set_of_removed_classes:
                to_be_deleted.append(idd)
        all_idxs = list(set(all_idxs) - set(to_be_deleted))
        #### to be applied in all places
        # distribute the iid across all rest users
        for i in range(23, 47):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))

    elif number_of_classes_of_half_of_user == 4:
        # first half of the users
        # divide and assign three classes for each one of the first half of users
        dict_users[0] = np.concatenate((dict_users[0], idxs[0:300]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[2400:2700]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[4800:5100]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[7200:7500]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[2700:3000]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[5100:5400]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[7500:7800]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[9600:9900]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[5400:5700]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[7800:8100]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[9900:10200]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[12000:12300]), axis=0)

        for i in range(3, 23):
            dict_users[i] = np.concatenate((dict_users[i], idxs[i * 2400 + 900:i * 2400 + 900 + 300]), axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 1) * 2400 + 600:(i + 1) * 2400 + 600 + 300]),
                                           axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 2) * 2400 + 300:(i + 2) * 2400 + 300 + 300]),
                                           axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 3) * 2400:(i + 3) * 2400 + 300]),
                                           axis=0)

        # second half of the users
        num_items = 1200
        all_idxs = [i for i in range(len(dataset))]
        # remove the already distributed idxs
        #### to be applied in all places
        list_of_removed_classes = []
        for j in range(0, 23):
            for i in set(dict_users[j]):
                list_of_removed_classes.append(int(dataset.targets[int(i)]))
        set_of_removed_classes = set(list_of_removed_classes)
        to_be_deleted = []
        for idd in idxs:
            if int(dataset.targets[idd]) in set_of_removed_classes:
                to_be_deleted.append(idd)
        all_idxs = list(set(all_idxs) - set(to_be_deleted))
        #### to be applied in all places
        # distribute the iid across all rest users
        for i in range(23, 47):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))

    elif number_of_classes_of_half_of_user == 6:
        # first half of the users
        # divide and assign three classes for each one of the first half of users
        dict_users[0] = np.concatenate((dict_users[0], idxs[0:200]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[2400:2600]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[4800:5000]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[7200:7400]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[9600:9800]), axis=0)
        dict_users[0] = np.concatenate((dict_users[0], idxs[12000:12200]), axis=0)

        dict_users[1] = np.concatenate((dict_users[1], idxs[2600:2800]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[5000:5200]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[7400:7600]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[9800:10000]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[12200:12400]), axis=0)
        dict_users[1] = np.concatenate((dict_users[1], idxs[14400:14600]), axis=0)

        dict_users[2] = np.concatenate((dict_users[2], idxs[5200:5400]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[7600:7800]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[10000:10200]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[12400:12600]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[14600:14800]), axis=0)
        dict_users[2] = np.concatenate((dict_users[2], idxs[16800:17000]), axis=0)

        dict_users[3] = np.concatenate((dict_users[3], idxs[7800:8000]), axis=0)
        dict_users[3] = np.concatenate((dict_users[3], idxs[10200:10400]), axis=0)
        dict_users[3] = np.concatenate((dict_users[3], idxs[12600:12800]), axis=0)
        dict_users[3] = np.concatenate((dict_users[3], idxs[14800:15000]), axis=0)
        dict_users[3] = np.concatenate((dict_users[3], idxs[17000:17200]), axis=0)
        dict_users[3] = np.concatenate((dict_users[3], idxs[19200:19400]), axis=0)

        dict_users[4] = np.concatenate((dict_users[4], idxs[10400:10600]), axis=0)
        dict_users[4] = np.concatenate((dict_users[4], idxs[12800:13000]), axis=0)
        dict_users[4] = np.concatenate((dict_users[4], idxs[15000:15200]), axis=0)
        dict_users[4] = np.concatenate((dict_users[4], idxs[17200:17400]), axis=0)
        dict_users[4] = np.concatenate((dict_users[4], idxs[19400:19600]), axis=0)
        dict_users[4] = np.concatenate((dict_users[4], idxs[21600:21800]), axis=0)

        for i in range(5, 23):
            dict_users[i] = np.concatenate((dict_users[i], idxs[i * 2400 + 1000:i * 2400 + 1000 + 200]), axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 1) * 2400 + 800:(i + 1) * 2400 + 800 + 200]),
                                           axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 2) * 2400 + 600:(i + 2) * 2400 + 600 + 200]),
                                           axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 3) * 2400 + 400:(i + 3) * 2400 + 400 + 200]),
                                           axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 4) * 2400 + 200:(i + 4) * 2400 + 200 + 200]),
                                           axis=0)
            dict_users[i] = np.concatenate((dict_users[i], idxs[(i + 5) * 2400:(i + 5) * 2400 + 200]),
                                           axis=0)

        # second half of the users
        num_items = 1200
        all_idxs = [i for i in range(len(dataset))]
        # remove the already distributed idxs
        #### to be applied in all places
        list_of_removed_classes = []
        for j in range(0, 23):
            for i in set(dict_users[j]):
                list_of_removed_classes.append(int(dataset.targets[int(i)]))
        set_of_removed_classes = set(list_of_removed_classes)
        to_be_deleted = []
        for idd in idxs:
            if int(dataset.targets[idd]) in set_of_removed_classes:
                to_be_deleted.append(idd)
        all_idxs = list(set(all_idxs) - set(to_be_deleted))
        #### to be applied in all places
        # distribute the iid across all rest users
        for i in range(23, 47):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))

    return dict_users


def emnist_noniid_unequal(dataset, num_users, samples_distribution_type):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 112,800 training imgs --> 1128 shards X 100 imgs/shard
    num_shards, num_imgs = 1128, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(0, 47):
        if len(idx_shard) == 0:
            continue
        shard_size = 24
        if shard_size > len(idx_shard):
            shard_size = len(idx_shard)
        rand_set = set(idx_shard[0:shard_size])
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                axis=0)

    # The rest are iid (random samples)
    # num_items = 2400
    # all_idxs = [i for i in range(len(dataset))]
    # for i in range(47, num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items,
    #                                          replace=False))
    #     all_idxs = list(set(all_idxs) - dict_users[i])

    # The rest are non-iid (random shards)
    # divide and assign 2 shards/client
    idx_shard = [i for i in range(num_shards)]
    for i in range(47, num_users):
        rand_set = set(np.random.choice(idx_shard, 24, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
