import os
import numpy as np
from collections import defaultdict as ddict
from tqdm import tqdm
import random

data_name = ['FB15k-237', 'WN18RR']
num_client = [5, 10, 15, 20]
random.seed(123)

def read_triples(file_path):
    """
    Read in raw triples txt file and convert it into standard format

    Parameters
    ----------
    file_path

    Returns
    -------
    standard_triples
    """
    triples = []
    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triples.append((head, relation, tail))
        standard_triples = np.array(triples)
    return standard_triples

def load_data(file_path):

    print("load data from {}".format(file_path))
    train_triples = read_triples(os.path.join(file_path, 'train.txt'))
    print(os.path.join(file_path, 'train.txt'))
    valid_triples = read_triples(os.path.join(file_path, 'valid.txt'))
    test_triples = read_triples(os.path.join(file_path, 'test.txt'))
    print('num_train_triples:{}'.format(len(train_triples)))
    print('num_valid_triples:{}'.format(len(valid_triples)))
    print('num_test_triples:{}'.format(len(test_triples)))

    return train_triples, valid_triples, test_triples

def data_shuffle(train_triples, valid_triples, test_triples):
    """
    randomly shuffle triples to generate new federated dataset
    Parameters
    ----------
    train_triples:train dataset triples
    valid_triples: valid dataset triples
    test_triples: test dataset triples

    Returns
    -------

    """
    triples = np.concatenate((train_triples, valid_triples), axis = 0)
    triples = np.concatenate((triples, test_triples), axis = 0)
    np.random.shuffle(triples)
    return triples

def data_split(triples, num_client):
    """

    Parameters
    ----------
    triples: shuffled triples
    num_client: number of clients

    Returns
    -------

    """
    client_triples = np.array_split(triples, num_client)
    for idx, val in enumerate(client_triples):
        client_triples[idx] = client_triples[idx].tolist()
    return client_triples

def gen_fed_dataset(client_triples, output_path):
    with open(output_path, 'w') as f:
        for tri in client_triples:
            h = tri[0]
            r = tri[1]
            t = tri[2]
            f.write(h + '\t' + r + '\t' + t + '\n')

def separate_client(client_triples, num_client, dir_path):
    client_data = []

    for client_idx in tqdm(range(num_client)):
        all_triples = client_triples[client_idx]
        ent_freq = ddict(int)
        rel_freq = ddict(int)
        for tri in all_triples:
            h,r,t = tri
            ent_freq[h] += 1
            ent_freq[t] += 1
            rel_freq[r] += 1
        client_train_tri = []
        client_valid_tri = []
        client_test_tri = []

        random.shuffle(all_triples)
        for idx, tri in enumerate(all_triples):
            h,r,t = tri
            if ent_freq[h] >2 and ent_freq[t]>2 and rel_freq[r]>2:
                client_test_tri.append(tri)
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
            else:
                client_train_tri.append(tri)
            if len(client_test_tri) > int(len(all_triples)*0.2):
                break
        client_train_tri.extend(all_triples[idx+1:])
        random.shuffle(client_test_tri)
        test_len = len(client_test_tri)
        client_valid_tri = client_test_tri[:int(test_len/2)]
        client_test_tri = client_test_tri[int(test_len/2):]
        fed_dir_path = os.path.join(dir_path, r'Fed{}\client{}'.format(num_client, client_idx+1))
        if not os.path.exists(fed_dir_path):
            os.makedirs(fed_dir_path)
        train_output_path = os.path.join(dir_path, r'Fed{}\client{}\train.txt'.format(num_client, client_idx+1))
        valid_output_path = os.path.join(dir_path, r'Fed{}\client{}\valid.txt'.format(num_client, client_idx+1))
        test_output_path = os.path.join(dir_path, r'Fed{}\client{}\test.txt'.format(num_client, client_idx+1))
        gen_fed_dataset(client_train_tri, train_output_path)
        gen_fed_dataset(client_valid_tri, valid_output_path)
        gen_fed_dataset(client_test_tri, test_output_path)

for dataset in data_name:
    train_triples, valid_triples, test_triples = load_data('.\\data\\{}'.format(dataset))
    dir_path = r'D:\pythonProject\CompGCN\CompGCN-master\CompGCN-master\fed_data\{}'.format(dataset)
    for user_num in num_client:
        triples = data_shuffle(train_triples, valid_triples, test_triples)
        client_triples = data_split(triples, user_num)
        separate_client(client_triples, user_num, dir_path)





