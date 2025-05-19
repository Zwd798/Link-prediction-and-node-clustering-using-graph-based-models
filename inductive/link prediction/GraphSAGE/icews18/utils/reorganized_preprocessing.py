import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

import random

from torch_geometric.transforms import RandomLinkSplit

def canonicalize_edges(edge_list):
    return list(tuple(sorted(e)) for e in edge_list)

def random_link_split_homogenous(data):
    return RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        add_negative_train_samples=True,
        is_undirected=True,
        split_labels=True
    )(data)

def random_link_split_heterogenous(data, edge_types, rev_edge_types=None, add_negative_train_samples=True):
    return RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        add_negative_train_samples=add_negative_train_samples,
        edge_types=edge_types,
        rev_edge_types=rev_edge_types
    )(data)

def split_train_test_edges(data, remove_fraction, homogenous=True, edge_types=None):
    random.seed(42)

    if homogenous:
        train_data, val_data, test_data = random_link_split_homogenous(data)
    else:
        train_data, val_data, test_data = random_link_split_heterogenous(data, edge_types)

    original_edges = canonicalize_edges(train_data.pos_edge_label_index.t().tolist())
    
    test_edges = set(canonicalize_edges(test_data.pos_edge_label_index.t().tolist()))
    intersection_indices = [i for i, edge in enumerate(original_edges) if edge in test_edges]
    num_to_remove = int(len(intersection_indices) * remove_fraction)
    remove_indices = set(random.sample(intersection_indices, num_to_remove))
    filtered_edges = [edge for i, edge in enumerate(original_edges) if i not in remove_indices]

    train_data.pos_edge_label_index = torch.tensor(filtered_edges).t()

    val_edges = set(canonicalize_edges(val_data.pos_edge_label_index.t().tolist()))
    intersection_indices = [i for i, edge in enumerate(original_edges) if edge in val_edges]
    num_to_remove = int(len(intersection_indices) * remove_fraction)
    remove_indices = set(random.sample(intersection_indices, num_to_remove))
    filtered_edges = [edge for i, edge in enumerate(original_edges) if i not in remove_indices]

    train_data.pos_edge_label_index = torch.tensor(filtered_edges).t()
    train_data.pos_edge_label = train_data.pos_edge_label[:train_data.pos_edge_label_index.shape[1]]

    return train_data, val_data, test_data

def get_indices(data, df):
    head1, head2 = df.columns[:2]
    edge_tuples = list(map(tuple, data.pos_edge_label_index.t().tolist()))
    edge_set = set(edge_tuples)
    matching_indices = df[df.apply(
        lambda row: ((row[head1], row[head2]) in edge_set) or ((row[head2], row[head1]) in edge_set),
        axis=1
    )].index
    return matching_indices
    

def get_edges_and_indices(df, remove_fraction=1.0):
    head1,head2 = df.columns[:2]
    edge_index = torch.tensor(df[[head1, head2]].values.T, dtype=torch.long)
    data = Data(edge_index=edge_index)
    train_data, val_data, test_data = split_train_test_edges(data, remove_fraction)
    train_indices = get_indices(train_data, df)
    val_indices = get_indices(val_data, df)
    test_indices = get_indices(test_data, df)
    return train_data, val_data, test_data, train_indices, val_indices, test_indices

