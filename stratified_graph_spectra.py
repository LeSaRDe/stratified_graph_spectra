import logging
import math
import multiprocessing
import os
import threading
from os import walk
import sys
import time
import datetime
from itertools import combinations

from scipy import sparse, stats
import scipy
# from scipy.optimize import curve_fit
# from scipy.stats import poisson, norm
# from scipy.spatial.distance import jensenshannon
import numpy as np
# import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import default_rng

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn import preprocessing
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
import seaborn as sns

################################################################################
#   GLOBAL SETTINGS STARTS
################################################################################
# g_work_dir = '/scratch/mf3jh/data/papers/node_embed/'
g_work_dir = '/home/mf3jh/workspace/data/papers/node_embed/'


################################################################################
#   GLOBAL SETTINGS ENDS
################################################################################


################################################################################
#   SAMPLE GRAPHS STARTS
################################################################################
def gen_unweighted_symA_graph():
    '''
    symA is an undirected symmetric graph.
    '''
    logging.debug('[gen_unweighted_symA_graph] starts.')
    nx_symA = nx.Graph()
    nx_symA.add_edge('A', 'B')
    nx_symA.add_edge('A', 'C')
    nx_symA.add_edge('A', 'D')

    nx_symA.add_edge('B', 'E')
    nx_symA.add_edge('B', 'F')
    nx_symA.add_edge('E', 'F')

    nx_symA.add_edge('C', 'G')
    nx_symA.add_edge('C', 'H')
    nx_symA.add_edge('H', 'G')

    nx_symA.add_edge('D', 'I')
    nx_symA.add_edge('D', 'J')
    nx_symA.add_edge('I', 'J')

    nx_symA.add_edge('E', 'K')
    nx_symA.add_edge('G', 'L')
    nx_symA.add_edge('I', 'M')
    nx.write_gpickle(nx_symA, g_work_dir + 'uw_symA.pickle')
    logging.debug('[gen_unweighted_symA_graph] all done.')


def gen_weighted_symA_graph():
    logging.debug('[gen_weighted_symA_graph] starts.')
    nx_symA = nx.read_gpickle(g_work_dir + 'uw_symA.pickle')
    l_rand_weights = np.random.randint(low=1, high=20, size=len(nx_symA.edges()))
    for idx, edge in enumerate(nx_symA.edges()):
        nx_symA.edges[edge[0], edge[1]]['weight'] = l_rand_weights[idx]
    nx.write_gpickle(nx_symA, g_work_dir + 'w_symA.pickle')
    logging.debug('[gen_weighted_symA_graph] all done.')


def gen_unweighted_symA_K3_C1_graph():
    logging.debug('[gen_unweighted_symA_K3_C1_graph] starts.')
    nx_symA_K3_C1 = nx.Graph()

    nx_symA_K3_C1.add_edge('B', 'G')
    nx_symA_K3_C1.add_edge('B', 'H')
    nx_symA_K3_C1.add_edge('B', 'I')
    nx_symA_K3_C1.add_edge('B', 'J')

    nx_symA_K3_C1.add_edge('C', 'E')
    nx_symA_K3_C1.add_edge('C', 'F')
    nx_symA_K3_C1.add_edge('C', 'I')
    nx_symA_K3_C1.add_edge('C', 'J')

    nx_symA_K3_C1.add_edge('D', 'E')
    nx_symA_K3_C1.add_edge('D', 'F')
    nx_symA_K3_C1.add_edge('D', 'G')
    nx_symA_K3_C1.add_edge('D', 'H')
    nx.write_gpickle(nx_symA_K3_C1, g_work_dir + 'uw_symA_K3_C1.pickle')
    logging.debug('[gen_unweighted_symA_K3_C1_graph] all done.')


def gen_uw_symA_line_graph():
    logging.debug('[gen_uw_symA_line_graph] starts.')
    nx_symA = nx.read_gpickle(g_work_dir + 'uw_symA.pickle')
    nx_uw_symA_line = nx.line_graph(nx_symA)
    nx.write_gpickle(nx_uw_symA_line, g_work_dir + 'uw_symA_line.pickle')
    logging.debug('[gen_uw_symA_line_graph] all done.')


def gen_complete_graph(num_nodes):
    logging.debug('[gen_complete_graph] starts.')
    nx_complete_graph = nx.complete_graph(num_nodes)
    nx.write_gpickle(nx_complete_graph, g_work_dir + 'complete_graph.pickle')
    logging.debug('[gen_complete_graph] all done.')

def gen_star_graph(num_nodes):
    logging.debug('[gen_star_graph] starts.')
    nx_star_graph = nx.star_graph(num_nodes)
    nx.write_gpickle(nx_star_graph, g_work_dir + 'star_graph.pickle')
    logging.debug('[gen_star_graph] all done.')

def gen_ring_graph(num_nodes):
    logging.debug('[gen_ring_graph] starts.')
    nx_ring_graph = nx.Graph()
    for i in range(num_nodes):
        if i == 0:
            continue
        else:
            nx_ring_graph.add_edge(i, i-1)
    nx_ring_graph.add_edge(0, num_nodes-1)
    nx.write_gpickle(nx_ring_graph, g_work_dir + 'ring_graph.pickle')
    logging.debug('[gen_ring_graph] all done.')

def gen_8_graph(num_nodes):
    logging.debug('[gen_8_graph] starts.')
    nx_8_graph = nx.Graph()
    num_nodes_ring_1 = math.floor(num_nodes / 2)
    for i in range(num_nodes_ring_1):
        if i == 0:
            continue
        else:
            nx_8_graph.add_edge(i, i-1)
    nx_8_graph.add_edge(0, num_nodes_ring_1-1)

    for j in range(num_nodes_ring_1 - 1, num_nodes):
        if j == num_nodes_ring_1 - 1:
            continue
        else:
            nx_8_graph.add_edge(j, j-1)
    nx_8_graph.add_edge(num_nodes - 1, num_nodes_ring_1-1)
    nx.write_gpickle(nx_8_graph, g_work_dir + '8_graph.pickle')
    logging.debug('[gen_8_graph] all done.')

def gen_turan_graph(num_nodes, num_part):
    logging.debug('[gen_turan_graph] starts.')
    nx_turan_graph = nx.turan_graph(num_nodes, num_part)
    nx.write_gpickle(nx_turan_graph, g_work_dir + 'turan_graph.pickle')
    logging.debug('[gen_turan_graph] all done.')

def gen_grid_2d_graph(m, n):
    logging.debug('[gen_grid_2d_graph] starts.')
    nx_grid_2d_graph = nx.grid_2d_graph(m, n)
    nx.write_gpickle(nx_grid_2d_graph, g_work_dir + 'grid_2d_graph.pickle')
    logging.debug('[gen_grid_2d_graph] all done.')

def gen_weighted_karate_club_graph():
    logging.debug('[gen_weighted_karate_club_graph] starts.')
    nx_unweighted_karate = nx.karate_club_graph()
    logging.debug('[gen_weighted_karate_club_graph] load in nx_unweighted_karate: %s' % nx.info(nx_unweighted_karate))

    l_rand_weights = np.random.randint(low=1, high=20, size=len(nx_unweighted_karate.edges()))
    for idx, edge in enumerate(nx_unweighted_karate.edges()):
        nx_unweighted_karate.edges[edge[0], edge[1]]['weight'] = l_rand_weights[idx]
    nx.write_gpickle(nx_unweighted_karate, 'weighted_karate_club.pickle')
    logging.debug('[gen_weighted_karate_club_graph] all done.')


def draw_graph(nx_graph, unweighted=False):
    plt.figure(1, figsize=(20, 20), tight_layout={'pad': 1, 'w_pad': 5, 'h_pad': 5, 'rect': None})
    pos = nx.spring_layout(nx_graph, k=0.8)
    # pos = nx.spectral_layout(nx_graph)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.10
    plt.xlim(x_min - x_margin, x_max + x_margin)
    d_node_labels = {node[0]: node[0] for node in nx_graph.nodes(data=True)}
    nx.draw_networkx_nodes(nx_graph, pos, node_size=20)
    nx.draw_networkx_labels(nx_graph, pos, labels=d_node_labels, font_size=30, font_color='r', font_weight='semibold')
    num_edges = nx_graph.number_of_edges()
    # edge_colors = range(10, num_edges + 10)
    l_edges = nx_graph.edges()
    if unweighted:
        edge_colors = np.asarray([1.0] * len(nx_graph.edges()))
    else:
        edge_colors = np.asarray([nx_graph.edges[edge[0], edge[1]]['weight'] for edge in l_edges])
        edge_colors = (edge_colors - np.min(edge_colors)) / (np.max(edge_colors) - np.min(edge_colors))
    drawn_edges = nx.draw_networkx_edges(nx_graph,
                                         pos,
                                         edgelist=l_edges,
                                         width=6,
                                         edge_color=edge_colors,
                                         edge_cmap=plt.get_cmap('jet'),
                                         edge_vmin=0.5,
                                         edge_vmax=1.0,
                                         arrows=True,
                                         arrowsize=40)
    plt.show()
    plt.clf()
    plt.close()


def manual_node_embed(ne_type='ideal'):
    if ne_type == 'ideal':
        np_embed = np.asarray([[0.2, 0.2, 0.2],  # A
                               [1.0, 0.0, 1.0],  # B
                               [0.0, 1.0, 1.0],  # C
                               [1.0, 1.0, 0.0],  # D
                               [1.9659, 0.0, 1.2588],  # E
                               [1.2588, 0.0, 1.9659],  # F
                               [0.0, 1.9659, 1.2588],  # G
                               [0.0, 1.2588, 1.9659],  # H
                               [1.9659, 1.2588, 0.0],  # I
                               [1.2588, 1.9659, 0.0],  # J
                               [2.2247, 0.0, 2.2247],  # K
                               [0.0, 2.2247, 2.2247],  # L
                               [2.2247, 2.2247, 0.0],  # M
                               ])
    elif ne_type == '2_hop_noise':
        # >>> move E, G and I close to A
        np_embed = np.asarray([[0.2, 0.2, 0.2],  # A
                               [1.0, 0.0, 1.0],  # B
                               [0.0, 1.0, 1.0],  # C
                               [1.0, 1.0, 0.0],  # D
                               [0.3, 0.2, 0.3],  # E
                               [1.2588, 0.0, 1.9659],  # F
                               [0.2, 0.3, 0.3],  # G
                               [0.0, 1.2588, 1.9659],  # H
                               [0.3, 0.3, 0.2],  # I
                               [1.2588, 1.9659, 0.0],  # J
                               [2.2247, 0.0, 2.2247],  # K
                               [0.0, 2.2247, 2.2247],  # L
                               [2.2247, 2.2247, 0.0],  # M
                               ])
    elif ne_type == 'unsym_2_hop_noise':
        # >>> move G, H I and J close to A
        np_embed = np.asarray([[0.2, 0.2, 0.2],  # A
                               [1.0, 0.0, 1.0],  # B
                               [0.0, 1.0, 1.0],  # C
                               [1.0, 1.0, 0.0],  # D
                               [1.9659, 0.0, 1.2588],  # E
                               [1.2588, 0.0, 1.9659],  # F
                               [0.2, 0.3, 0.3],  # G
                               [0.3, 0.4, 0.4],  # H
                               [0.3, 0.3, 0.2],  # I
                               [0.4, 0.4, 0.3],  # J
                               [2.2247, 0.0, 2.2247],  # K
                               [0.0, 2.2247, 2.2247],  # L
                               [2.2247, 2.2247, 0.0],  # M
                               ])
    elif ne_type == 'sym_2_hop_noise':
        # >>> move E, F, G, H I and J close to A
        np_embed = np.asarray([[0.2, 0.2, 0.2],  # A
                               [1.0, 0.0, 1.0],  # B
                               [0.0, 1.0, 1.0],  # C
                               [1.0, 1.0, 0.0],  # D
                               [0.2, 0.2, 0.2],  # E
                               [0.2, 0.2, 0.2],  # F
                               [0.2, 0.2, 0.2],  # G
                               [0.2, 0.2, 0.2],  # H
                               [0.2, 0.2, 0.2],  # I
                               [0.2, 0.2, 0.2],  # J
                               [2.2247, 0.0, 2.2247],  # K
                               [0.0, 2.2247, 2.2247],  # L
                               [2.2247, 2.2247, 0.0],  # M
                               ])
    elif ne_type == 'unsym_2_hop_noise_EF':
        np_embed = np.asarray([[0.2, 0.2, 0.2],  # A
                               [1.0, 0.0, 1.0],  # B
                               [0.0, 1.0, 1.0],  # C
                               [1.0, 1.0, 0.0],  # D
                               [0.2, 0.2, 0.2],  # E
                               [0.2, 0.2, 0.2],  # F
                               [0.0, 1.9659, 1.2588],  # G
                               [0.0, 1.2588, 1.9659],  # H
                               [1.9659, 1.2588, 0.0],  # I
                               [1.2588, 1.9659, 0.0],  # J
                               [2.2247, 0.0, 2.2247],  # K
                               [0.0, 2.2247, 2.2247],  # L
                               [2.2247, 2.2247, 0.0],  # M
                               ])
    elif ne_type == 'sym_3_hop_noise_KLM':
        np_embed = np.asarray([[0.2, 0.2, 0.2],  # A
                               [1.0, 0.0, 1.0],  # B
                               [0.0, 1.0, 1.0],  # C
                               [1.0, 1.0, 0.0],  # D
                               [0.2, 0.2, 0.2],  # E
                               [0.2, 0.2, 0.2],  # F
                               [0.0, 1.9659, 1.2588],  # G
                               [0.0, 1.2588, 1.9659],  # H
                               [1.9659, 1.2588, 0.0],  # I
                               [1.2588, 1.9659, 0.0],  # J
                               [0.2, 0.2, 0.2],  # K
                               [0.2, 0.2, 0.2],  # L
                               [0.2, 0.2, 0.2],  # M
                               ])
    elif ne_type == 'uw_symA_pulse_B':
        np_embed = np.ones((13, 3))
        np_embed[1] = np.asarray([1, 0, -1])
    elif ne_type == 'uw_symA_pulse_BCD':
        np_embed = np.zeros((13, 3))
        np_embed = np_embed + 0.00001
        np_embed[1] += np.asarray([1, 0, 0]) #B
        np_embed[2] += np.asarray([0, 1, 0]) #C
        np_embed[3] += np.asarray([0, 0, 1]) #D
        np_embed[5] += np.asarray([-1, 0, 0]) #F
        np_embed[6] += np.asarray([-1, 0, 0]) #G
        np_embed[9] += np.asarray([-1, 0, 0]) #J
        np_embed[4] += np.asarray([0, -1, 0]) #E
        np_embed[7] += np.asarray([0, -1, 0]) #H
        np_embed[8] += np.asarray([0, -1, 0]) #I
    elif ne_type == 'uw_symA_high_freq':
        np_embed = np.tile([0.00001, 0.00001, -1], (13, 1))
        # np_embed = np_embed + 0.00001
        np_embed[1] = np.asarray([0.00001, 0.00001, 1]) #B
        np_embed[2] = np.asarray([0.00001, 0.00001, 1]) #C
        np_embed[3] = np.asarray([0.00001, 0.00001, 1]) #D
        np_embed[10] = np.asarray([-1, 0.00001, 0.00001]) #K
        np_embed[11] = np.asarray([-1, 0.00001, 0.00001]) #L
        np_embed[12] = np.asarray([-1, 0.00001, 0.00001]) #M
    return np_embed


def manual_pw_distance_mat(df_spectral_seq, nx_graph, pw_dist_type):
    num_nodes = nx.number_of_nodes(nx_graph)
    np_pw_dist = np.zeros((num_nodes, num_nodes))

    if pw_dist_type == 'zero_tv':
        '''
        all adjacent nodes have no distance, and all non-adjacent nodes have the max distance.
        '''
        np_pw_dist = np.ones((num_nodes, num_nodes))
        np.fill_diagonal(np_pw_dist, 0.0)
        for K, Srec in df_spectral_seq.iterrows():
            df_EADMs = Srec['df_EADMs']
            EADM_mask = df_EADMs.iloc[0]['EADM_mask']
            EADM_mask_row = [item[0] for item in EADM_mask]
            EADM_mask_col = [item[1] for item in EADM_mask]
            if K == 1:
                pw_dist_val = 0.0
                np_pw_dist[EADM_mask_row, EADM_mask_col] = pw_dist_val
            else:
                continue
    elif pw_dist_type == 'all_one':
        '''
        every node is not similar to any other node.
        '''
        np_pw_dist = np.ones((num_nodes, num_nodes))
    elif pw_dist_type == 'single_pulse':
        '''
        A pulse at the first node, i.e. the first node is dissimilar to any one else, and all others are all similar.
        '''
        np_pw_dist[0] = 1.0
        np_pw_dist[0][0] = 0.0
    # elif pw_dist_type == 'uw_symA_pulse_B':
    #     '''
    #     Single pulse at B in uw_symA.
    #     '''
    #     np_pw_dist[1] = 1.0
    #     np_pw_dist[1][1] = 0.0
    elif pw_dist_type == 'uw_symA_ideal_cluster':
        '''
        A is not similar to any one; B, E, F and K are similar and dissimilar to others;
        C, H, G and L are similar and dissimilar to others; D, I, J and M are similar and dissimilar to others.
        '''
        l_nodes = list(nx_graph.nodes())
        np_pw_dist = np.ones((num_nodes, num_nodes))
        # >>> A
        np_pw_dist[l_nodes.index('A')] = 1.0
        np_pw_dist[l_nodes.index('A')][l_nodes.index('B')] = 0.5
        np_pw_dist[l_nodes.index('A')][l_nodes.index('C')] = 0.5
        np_pw_dist[l_nodes.index('A')][l_nodes.index('D')] = 0.5
        # >>> B
        np_pw_dist[l_nodes.index('B')][l_nodes.index('E')] = 0.0
        np_pw_dist[l_nodes.index('B')][l_nodes.index('F')] = 0.0
        np_pw_dist[l_nodes.index('B')][l_nodes.index('K')] = 0.0
        # >>> C
        np_pw_dist[l_nodes.index('C')][l_nodes.index('G')] = 0.0
        np_pw_dist[l_nodes.index('C')][l_nodes.index('H')] = 0.0
        np_pw_dist[l_nodes.index('C')][l_nodes.index('L')] = 0.0
        # >>> D
        np_pw_dist[l_nodes.index('D')][l_nodes.index('I')] = 0.0
        np_pw_dist[l_nodes.index('D')][l_nodes.index('J')] = 0.0
        np_pw_dist[l_nodes.index('D')][l_nodes.index('M')] = 0.0
        # >>> E
        np_pw_dist[l_nodes.index('E')][l_nodes.index('B')] = 0.0
        np_pw_dist[l_nodes.index('E')][l_nodes.index('F')] = 0.0
        np_pw_dist[l_nodes.index('E')][l_nodes.index('K')] = 0.0
        # >>> F
        np_pw_dist[l_nodes.index('F')][l_nodes.index('B')] = 0.0
        np_pw_dist[l_nodes.index('F')][l_nodes.index('E')] = 0.0
        np_pw_dist[l_nodes.index('F')][l_nodes.index('K')] = 0.0
        # >>> G
        np_pw_dist[l_nodes.index('G')][l_nodes.index('C')] = 0.0
        np_pw_dist[l_nodes.index('G')][l_nodes.index('H')] = 0.0
        np_pw_dist[l_nodes.index('G')][l_nodes.index('L')] = 0.0
        # >>> H
        np_pw_dist[l_nodes.index('H')][l_nodes.index('C')] = 0.0
        np_pw_dist[l_nodes.index('H')][l_nodes.index('G')] = 0.0
        np_pw_dist[l_nodes.index('H')][l_nodes.index('L')] = 0.0
        # >>> I
        np_pw_dist[l_nodes.index('I')][l_nodes.index('D')] = 0.0
        np_pw_dist[l_nodes.index('I')][l_nodes.index('J')] = 0.0
        np_pw_dist[l_nodes.index('I')][l_nodes.index('M')] = 0.0
        # >>> J
        np_pw_dist[l_nodes.index('J')][l_nodes.index('D')] = 0.0
        np_pw_dist[l_nodes.index('J')][l_nodes.index('I')] = 0.0
        np_pw_dist[l_nodes.index('J')][l_nodes.index('M')] = 0.0
        # >>> K
        np_pw_dist[l_nodes.index('K')][l_nodes.index('B')] = 0.0
        np_pw_dist[l_nodes.index('K')][l_nodes.index('E')] = 0.0
        np_pw_dist[l_nodes.index('K')][l_nodes.index('F')] = 0.0
        # >>> L
        np_pw_dist[l_nodes.index('L')][l_nodes.index('C')] = 0.0
        np_pw_dist[l_nodes.index('L')][l_nodes.index('G')] = 0.0
        np_pw_dist[l_nodes.index('L')][l_nodes.index('H')] = 0.0
        # >>> M
        np_pw_dist[l_nodes.index('M')][l_nodes.index('D')] = 0.0
        np_pw_dist[l_nodes.index('M')][l_nodes.index('I')] = 0.0
        np_pw_dist[l_nodes.index('M')][l_nodes.index('J')] = 0.0

        np.fill_diagonal(np_pw_dist, 0.0)
    elif pw_dist_type == 'uw_symA_pulse_B_heat':
        np_sig_filtered = np.load(
            '/home/mf3jh/workspace/data/papers/node_embed/classic_gsp/uw_symA_B_pulse_heat/np_sig_filtered.npy')
        sp_sig_uxuy = sparse.coo_matrix(
            np.triu(np.matmul(np_sig_filtered.reshape(-1, 1), np_sig_filtered.reshape(-1, 1).T), k=1))
        sp_sig_uxsqr_add_uysqr = sparse.coo_matrix(
            np.triu(np.asarray([np.power(np_sig_filtered, 2)] * len(np_sig_filtered))
                    + np.power(np_sig_filtered, 2).reshape(-1, 1), k=1))
        sp_sig_ux_minus_uy_sqr = np.round(sp_sig_uxsqr_add_uysqr - 2 * sp_sig_uxuy, decimals=12)
        sp_adj_diff_mat = np.sqrt(sp_sig_ux_minus_uy_sqr)
        np_pw_dist = sp_adj_diff_mat.toarray()

    elif pw_dist_type == 'uw_symA_pulse_B':
        np_sig = np.load('/home/mf3jh/workspace/data/papers/node_embed/classic_gsp/uw_symA_B_pulse_heat/np_sig.npy')
        sp_sig_uxuy = sparse.coo_matrix(np.triu(np.matmul(np_sig.reshape(-1, 1), np_sig.reshape(-1, 1).T), k=1))
        sp_sig_uxsqr_add_uysqr = sparse.coo_matrix(np.triu(np.asarray([np.power(np_sig, 2)] * len(np_sig))
                                                           + np.power(np_sig, 2).reshape(-1, 1), k=1))
        sp_sig_ux_minus_uy_sqr = np.round(sp_sig_uxsqr_add_uysqr - 2 * sp_sig_uxuy, decimals=12)
        sp_adj_diff_mat = np.sqrt(sp_sig_ux_minus_uy_sqr)
        np_pw_dist = sp_adj_diff_mat.toarray()
    return np_pw_dist


def gen_rand_graphs(rand_graph_type='rand', num_rand_graphs=100, num_nodes=50, edge_density=0.05,
                    graph_name_prefix=None, save_ret=True, save_path=None, job_id=None):
    logging.debug('[gen_rand_graphs] starts.')

    l_ret_graphs = []
    if rand_graph_type == 'rand':
        idx = 0
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        while len(l_ret_graphs) < num_rand_graphs:
            sp_A = sparse.rand(num_nodes, num_nodes, density=edge_density, dtype=np.float32)
            np_A = sp_A.toarray()
            np_A = np_A + np_A.T
            np.fill_diagonal(np_A, 0)
            nx_graph = nx.from_scipy_sparse_matrix(sp_A)
            if not nx.is_connected(nx_graph):
                continue
            else:
                if save_ret:
                    if graph_name_prefix is None:
                        graph_name = 'rand_graphs#' + now + str(idx)
                    else:
                        graph_name = graph_name_prefix + now + str(idx)
                    l_ret_graphs.append((graph_name, nx_graph))
                else:
                    l_ret_graphs.append((None, nx_graph))
            idx += 1
        if save_ret:
            df_rand_graphs = pd.DataFrame(l_ret_graphs, columns=['graph_name', 'nx_graph'])
            df_rand_graphs = df_rand_graphs.set_index('graph_name')
            pd.to_pickle(df_rand_graphs, save_path + str(job_id) + '#rand_graphs.pickle')
        logging.debug('[gen_rand_graphs] all done with %s rand graphs.' % str(len(l_ret_graphs)))
    elif rand_graph_type == 'sbm':
        idx = 0
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        num_block = np.random.choice([i for i in range(2, 11)])
        l_block_sizes = [math.floor(num_nodes / num_block)] * (num_block - 1)
        l_block_sizes += [num_nodes - np.sum(l_block_sizes)]
        while len(l_ret_graphs) < num_rand_graphs:
            np_A = np.random.uniform(low=0.01, high=0.2, size=[num_block, num_block])
            np_A = (np_A + np_A.T) / 2.0
            nx_graph = nx.stochastic_block_model(l_block_sizes, np_A)
            if not nx.is_connected(nx_graph):
                continue
            else:
                if save_ret:
                    if graph_name_prefix is None:
                        graph_name = 'sbm_graphs#' + now + str(idx)
                    else:
                        graph_name = graph_name_prefix + now + str(idx)
                    l_ret_graphs.append((graph_name, nx_graph))
                else:
                    l_ret_graphs.append((None, nx_graph))
            idx += 1
        if save_ret:
            df_rand_graphs = pd.DataFrame(l_ret_graphs, columns=['graph_name', 'nx_graph'])
            df_rand_graphs = df_rand_graphs.set_index('graph_name')
            pd.to_pickle(df_rand_graphs, save_path + str(job_id) + '#rand_graphs.pickle')
        logging.debug('[gen_rand_graphs] all done with %s sbm graphs.' % str(len(l_ret_graphs)))
    return [item[1] for item in l_ret_graphs]


def gen_rand_sigs(nx_graph, sig_type='uniform', max_sig_val=1.0, min_sig_val=-1.0):
    if sig_type == 'uniform':
        np_sig = np.random.uniform(low=min_sig_val, high=max_sig_val, size=nx.number_of_nodes(nx_graph))
    elif sig_type == 'rand_pulse':
        np_sig = np.zeros(nx.number_of_nodes(nx_graph))
        np_sig[np.random.choice([i for i in range(nx.number_of_nodes(nx_graph))])] = 1.0
    return np_sig


def compute_pw_dist_from_sig(np_sig):
    sp_sig_uxuy = sparse.coo_matrix(np.triu(np.matmul(np_sig.reshape(-1, 1), np_sig.reshape(-1, 1).T), k=1))
    sp_sig_uxsqr_add_uysqr = sparse.coo_matrix(np.triu(np.asarray([np.power(np_sig, 2)] * len(np_sig))
                                                       + np.power(np_sig, 2).reshape(-1, 1), k=1))
    sp_sig_ux_minus_uy_sqr = np.round(sp_sig_uxsqr_add_uysqr - 2 * sp_sig_uxuy, decimals=12)
    sp_adj_diff_mat = np.sqrt(sp_sig_ux_minus_uy_sqr)
    np_pw_dist = sp_adj_diff_mat.toarray()
    return np_pw_dist


def gen_rand_node_embed(nx_graph, embed_dim, node_embed_type='rand'):
    if node_embed_type == 'rand':
        # return np.random.randn(nx.number_of_nodes(nx_graph), embed_dim)
        return np.random.uniform(low=-1.0, high=1.0, size=(nx.number_of_nodes(nx_graph), embed_dim))


################################################################################
#   SAMPLE GRAPHS ENDS
################################################################################


################################################################################
#   GSP STARTS
################################################################################
def graph_laplacian(nx_graph, use_norm=False):
    '''
    nx_graph is better to be connected.
    '''
    if use_norm:
        L = nx.linalg.normalized_laplacian_matrix(nx_graph)
    else:
        L = nx.linalg.laplacian_matrix(nx_graph)
    return L


def graph_eigs(L, use_sparse=False, top_M_eigs=None):
    if not use_sparse:
        if type(L) != np.ndarray:
            L = L.toarray()
        eig_vals, eig_vecs = np.linalg.eigh(L)
    else:
        eig_vals, eig_vecs = sparse.linalg.eigs(L, k=top_M_eigs)
    eig_vals = eig_vals.astype(np.float64)
    eig_vals = np.round(eig_vals, decimals=12).astype(np.float64)
    eig_vecs = eig_vecs.astype(np.float64).T
    l_eigs = sorted(zip(eig_vals, eig_vecs), key=lambda k: k[0])
    return l_eigs


def draw_graph_eigs(l_eigs, nx_graph):
    fig, axes = plt.subplots(ncols=1, nrows=len(l_eigs), figsize=(10, 30))
    for idx, (eig_val, eig_vec) in enumerate(l_eigs):
        axes[idx].grid(True)
        axes[idx].set_title(r'$\lambda$ = %s' % np.round(eig_val, decimals=3), fontsize=10)
        axes[idx].set_yticks(np.arange(-1, 1, step=0.1))
        markerline, stemlines, baselineaxes = axes[idx].stem(list(nx_graph.nodes()), eig_vec)
    plt.tight_layout(pad=1.0)
    plt.show()


def graph_fourier(U, F):
    '''
    U is the matrix of eigenvectors in rows.
    F is the matrix of signal vectors in rows.
    graph Fourier transformation is F * U.T
    return transformed signal vectors in columns.
    '''
    F_hat = np.matmul(F, U.T)
    F_hat = np.round(F_hat, decimals=12)
    return F_hat


def graph_inverse_fourier(U, F_hat):
    F = np.matmul(F_hat, U)
    F = np.round(F, decimals=12)
    return F


def classic_gsp_on_stratified_graphs(d_spec_seq, graph_sig_1, graph_sig_2=None, sig_1_label=None, sig_2_label=None,
                                     save_ret=True, save_path=None, save_img=True, show_img=False):
    logging.debug('[classif_gsp_on_stratified_graphs] starts.')

    # graph_sig_1 = preprocessing.normalize(graph_sig_1.reshape(1, -1))[0]
    # graph_sig_2 = preprocessing.normalize(graph_sig_2.reshape(1, -1))[0]

    d_graph_sig_hat = dict()
    l_rec = []
    for K, Srec in d_spec_seq.iterrows():
        l_eigs = Srec['eigs']
        l_eig_vals = [eig[0] for eig in l_eigs]
        np_U = np.stack([eig[1] for eig in l_eigs])
        graph_sig_hat_1 = graph_fourier(np_U, graph_sig_1)
        if graph_sig_2 is not None:
            graph_sig_hat_2 = graph_fourier(np_U, graph_sig_2)
        else:
            graph_sig_hat_2 = None
        d_graph_sig_hat[K] = (l_eig_vals, graph_sig_hat_1, graph_sig_hat_2)
        l_rec.append((K, l_eig_vals, graph_sig_hat_1, graph_sig_hat_2))
    l_K = list(d_spec_seq.index)

    df_ret = pd.DataFrame(l_rec, columns=['K', 'eig_vals', 'graph_sig_hat_1', 'graph_sig_hat_2'])
    df_ret = df_ret.set_index('K')
    if save_ret:
        pd.to_pickle(df_ret, save_path + 'classic_gsp.pickle')

    if not save_img and not show_img:
        logging.debug('[classif_gsp_on_stratified_graphs] All done.')
        return df_ret

    sig_hat_1_max = np.max(np.abs([item[1] for item in d_graph_sig_hat.values()]))
    sig_hat_1_min = np.min(np.abs([item[1] for item in d_graph_sig_hat.values()]))
    sig_hat_2_max = np.max(np.abs([item[2] for item in d_graph_sig_hat.values()]))
    sig_hat_2_min = np.min(np.abs([item[2] for item in d_graph_sig_hat.values()]))
    vmin = np.min([sig_hat_1_min, sig_hat_2_min])
    vmax = np.max([sig_hat_1_max, sig_hat_2_max])

    img_name = 'classic_gsp_on_sgs'
    fig_width = len(d_graph_sig_hat) * 3
    fig, axes = plt.subplots(ncols=1, nrows=len(d_graph_sig_hat), figsize=(10, fig_width))
    # fig.suptitle(img_name, fontsize=15,  fontweight='semibold')
    idx = 0
    for K in l_K:
        l_eig_vals = d_graph_sig_hat[K][0]
        graph_sig_hat_1 = np.abs(d_graph_sig_hat[K][1])
        graph_sig_hat_2 = np.abs(d_graph_sig_hat[K][2])
        # graph_sig_hat_1 = d_graph_sig_hat[K][1]
        # graph_sig_hat_2 = d_graph_sig_hat[K][2]

        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        sig_2_linefmt = 'tab:orange'
        sig_2_marker_fmt = 'x'
        axes[idx].stem(graph_sig_hat_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt, label=sig_1_label)
        if graph_sig_hat_2 is not None:
            axes[idx].stem(graph_sig_hat_2, linefmt=sig_2_linefmt, markerfmt=sig_2_marker_fmt, label=sig_2_label)
        axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
        axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.1, 0.1)])
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    # plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()

    logging.debug('[classif_gsp_on_stratified_graphs] All done.')
    return df_ret


################################################################################
#   GSP ENDS
################################################################################


################################################################################
#   NODE EMBEDDING GSP ANALYSIS STARTS
################################################################################
def eigenvec_adj_diff_mats(l_eigs, sp_mask, val_weighted=True, exp_ret=True, rm_dc=False):
    '''
    Since the eigenvector associated to the smallest eigenvalue (i.e. 0) is a constant vector,
    then the adjacent differences computed upon this vector are all zero.
    This all-zero vector may not be helpful in the analysis. Thus, we may simply remove this eigenvector from
    our consideration. And if we do not remove it, then we have to make up for the all-zero adjacent difference
    vector, e.g. apply e^x to all adjacent difference vectors to uplift zeros to ones.

    EADM_i ~ (lambda_i, u_i), EADM_i = [theta_xy]
    'val_weighted':
        True => theta_xy = |u_i(x) - u_i(y)| * |u_i(x)| * |u_i(y)| = |u_i(x) - u_i(y)| * |u_i(x) * u_i(y)|
                This method is consistent with the way computing the inner product between the signal and an eigenvector,
                i.e. when an element in the eigenvector is small, then it would not contribute to the inner product significantly,
                and in this method, |u_i(x) - u_i(y)| is weighted by both u_i(x) and u_i(y).

    'exp_ret':
        True => theta_xy=EXP(theta_xy) doing this will make all differences non-trivial, otherwise the small ones
                may not contribute to the downstream computation significantly.
    '''
    sp_u_mask = sparse.triu(sp_mask, k=1).astype(np.bool).astype(np.int32)
    sp_u_mask = sparse.coo_matrix(sp_u_mask)
    l_rec = []
    num_eig = len(l_eigs)
    for idx, eig in enumerate(l_eigs):
        eigenval = eig[0]
        eigenvec = eig[1]
        # >>> COMPUTE |u_i(x) - u_i(y)| <<<
        # >>> [[u(x) - u(y)]^2]^0.5 = [u(x)^2 + u(y)^2 - 2*u(x)u(y)]^0.5
        # >>> known: NORM[u]=1.0 => |u(x)| in [0, 1] => |u(x) - u(y)| in [0, 2]
        sp_u_uxuy = sparse.coo_matrix(np.triu(np.matmul(eigenvec.reshape(-1, 1), eigenvec.reshape(-1, 1).T), k=1))
        sp_u_uxuy = sp_u_uxuy.multiply(sp_mask)
        sp_u_uxsqr_add_uysqr = sparse.coo_matrix(
            np.triu(np.asarray([np.power(eigenvec, 2)] * num_eig) + np.power(eigenvec, 2).reshape(-1, 1), k=1))
        sp_u_uxsqr_add_uysqr = sp_u_uxsqr_add_uysqr.multiply(sp_mask)
        # >>> guarantee every element is non-negative
        sp_u_ux_minus_uy_sqr = np.round(sp_u_uxsqr_add_uysqr - 2 * sp_u_uxuy, decimals=12)
        sp_adj_diff_mat = np.sqrt(sp_u_ux_minus_uy_sqr)

        if np.count_nonzero(sp_adj_diff_mat.data) <= 0:
            sp_adj_diff_mat = sp_u_mask

        if exp_ret:
            sp_adj_diff_mat.data = np.exp(sp_adj_diff_mat.data)
            sp_adj_diff_mat = (sp_u_mask - sp_adj_diff_mat.astype(np.bool).astype(np.int32)) + sp_adj_diff_mat

        if val_weighted:
            sp_u_uxuy = np.abs(sp_u_uxuy)
            sp_adj_diff_mat = sp_adj_diff_mat.multiply(sp_u_uxuy)
        else:
            pass

        sp_adj_diff_mat = sparse.coo_matrix(sp_adj_diff_mat)
        if not np.isfinite(sp_adj_diff_mat.toarray()).all():
            raise Exception('[eigenvec_adj_diff_mats] invalid sp_u_pw_ele_diff_sqr: %s'
                            % sp_adj_diff_mat.toarray())
        l_rec.append((eigenval, eigenvec, sp_adj_diff_mat,
                      sorted(list(zip(sp_u_mask.row, sp_u_mask.col)), key=lambda k: k[0]), sp_u_uxuy))
    if rm_dc:
        l_rec = l_rec[1:]
    df_EADMs = pd.DataFrame(l_rec,
                            columns=['eigval', 'eigvec', 'eigvec_sp_EADM', 'EADM_mask', 'eigvec_sp_EADM_weights'])
    return df_EADMs


def basis_for_eigenvec_adj_diff_mats(df_EADMs, nx_graph, decomp='svd', norm_FEADMs=True):
    '''
    (1) flatten each EADM to a vector
    (2) compute a basis for the flattened EADMs
    '''
    EADM_mask = df_EADMs.iloc[0]['EADM_mask']
    EADM_mask_row = [item[0] for item in EADM_mask]
    EADM_mask_col = [item[1] for item in EADM_mask]

    # >>> get the dimensions of the basis
    l_node = list(nx_graph.nodes())
    l_basis_dim = [(l_node[EADM_mask_row[i]], l_node[EADM_mask_col[i]]) for i in range(len(EADM_mask))]

    l_FEADMs = []
    for _, EADM_rec in df_EADMs.iterrows():
        sp_EADM = EADM_rec['eigvec_sp_EADM']
        # TODO
        # is there any better way to extract the flattened vector without reconstructing the matrix?
        l_FEADMs.append(sp_EADM.toarray()[EADM_mask_row, EADM_mask_col])

    np_FEADMs = np.stack(l_FEADMs)
    if norm_FEADMs:
        # >>> by normalizing FEADMs, we may eliminate the "stratches" of singulars.
        np_FEADMs = preprocessing.normalize(np_FEADMs)

    if decomp == 'qr':
        # >>> use col vec for each diff mat for the convenience of computing QR
        # >>> Q and R are all col vectors
        np_FEADMs_Q, np_FEADMs_R = np.linalg.qr(np_FEADMs.T)
        # >>> convert Q and R to row vectors
        np_FEADMs_Q = np_FEADMs_Q.T
        np_FEADMs_R = np_FEADMs_R.T
        return (np_FEADMs_Q, np_FEADMs_R), l_basis_dim, np_FEADMs
    elif decomp == 'svd':
        np_FEADMs_U, np_FEADMs_S, np_FEADMs_Vh = np.linalg.svd(np_FEADMs, full_matrices=False)
        return (np_FEADMs_Vh, np_FEADMs_U, np_FEADMs_S), l_basis_dim, np_FEADMs


def incidence_mat_from_nodes_and_edges(l_nodes, l_edges):
    '''
    # TODO
    Only for unweighted graphs temporarily. Needs to be extended to weighted graphs.
    '''
    I = scipy.sparse.lil_matrix((len(l_nodes), len(l_edges)))
    I_oriented = scipy.sparse.lil_matrix((len(l_nodes), len(l_edges)))
    node_index = {node: i for i, node in enumerate(l_nodes)}
    for ei, e in enumerate(l_edges):
        (u, v) = e[:2]
        ui = node_index[u]
        vi = node_index[v]
        I[ui, ei] = 1
        I[vi, ei] = 1
        I_oriented[ui, ei] = 1
        I_oriented[vi, ei] = -1
    return I.asformat("coo"), I_oriented.asformat("coo")


def graph_spectra_sequence(nx_graph, save_path, decomp='svd', adj_only=False, val_weighted=True,
                           exp_EADMs=True, max_k=None, unweighted=True,
                           use_norm_L=False, top_M_eigs=None, rm_dc=False, norm_FEADMs=True):
    '''
    The sequence is computed based on [B_0, B_1,..., B_(max_k-1)].
    B_0 = A
    B_k = DIAG_0[DELTA(A^(k+1)) -* SUM([B_0,...,B_(k-1)])]
    DIAG_0: make the diagonal all 0
    DELTA: make non-zero elements all 1
    -*: non-negative subtraction, e.g. 1-1=0, 0-1=0, 1-0=1
    For each B_i:
    (1) compute eigenvalues and eigenvectors of L_i constructed on B_i
    (2) compute the adjacent difference matrices for the eigenvectors (EADMs)
    (3) compute a basis for the flattened EADMs (FEADMs) by using QR
    '''
    logging.debug('[graph_spectra_sequence] starts.')
    timer_start = time.time()

    if max_k is None:
        max_k = nx.diameter(nx_graph)

    sp_A = nx.linalg.adjacency_matrix(nx_graph)
    np_A = sp_A.toarray()
    l_np_Ak = [np_A]
    if max_k >= 2:
        for k in range(2, max_k + 1):
            np_Ak = np.matmul(np_A, l_np_Ak[-1])
            l_np_Ak.append(np_Ak)

    if unweighted:
        l_np_Ak = [np_Ak.astype(np.bool).astype(np.int32) for np_Ak in l_np_Ak]
    logging.debug('[graph_spectra_sequence] l_np_Ak done with %s Ak in %s secs.'
                  % (len(l_np_Ak), time.time() - timer_start))

    l_np_Bk = [l_np_Ak[0]]
    if unweighted:
        for i in range(1, len(l_np_Ak)):
            np_Bk = l_np_Ak[i] - np.cumsum(np.stack(l_np_Bk), axis=0)[-1]
            np.fill_diagonal(np_Bk, 0)
            np_Bk[np.where(np_Bk < 0)] = 0
            l_np_Bk.append(np_Bk)
    logging.debug('[graph_spectra_sequence] l_np_Bk done with %s Bk in %s secs.'
                  % (len(l_np_Bk), time.time() - timer_start))

    d_connected_comp_sizes = {i: [] for i in range(len(l_np_Bk))}
    for idx, np_Bk in enumerate(l_np_Bk):
        nx_sg = nx.from_numpy_array(np_Bk)
        for comp in nx.connected_components(nx_sg):
            d_connected_comp_sizes[idx].append(len(list(comp)))

    l_Sk = []
    for np_Bk in l_np_Bk:
        np_Lk = np.diag(np.sum(np_Bk, axis=1)) - np_Bk
        if use_norm_L:
            np_norm_Dk = np.diag(np.power(np.sum(np_Bk, axis=1), -0.5))
            np_Lk = np.matmul(np_norm_Dk, np.matmul(np_Lk, np_norm_Dk))
        l_eigs_k = graph_eigs(np_Lk, use_sparse=False, top_M_eigs=top_M_eigs)
        l_Sk.append(l_eigs_k)
    logging.debug('[graph_spectra_sequence] l_Sk done with %s spectra in %s secs.'
                  % (len(l_Sk), time.time() - timer_start))

    l_node = list(nx_graph.nodes())
    l_Srec = []
    for idx, l_eigs_k in enumerate(l_Sk):
        sp_u_Bk = sparse.triu(sparse.coo_matrix(l_np_Bk[idx]), k=1)
        if adj_only:
            sp_mask = sp_u_Bk
        else:
            l_node_idx = [i for i in range(nx.number_of_nodes(nx_graph))]
            l_mask_coord = list(combinations(l_node_idx, 2))
            mask_row = [item[0] for item in l_mask_coord]
            mask_col = [item[1] for item in l_mask_coord]
            sp_mask = sparse.coo_matrix(([1] * len(l_mask_coord), (mask_row, mask_col)),
                                        shape=(len(l_node_idx), len(l_node_idx)), dtype=np.int32)
        df_EADMs_k = eigenvec_adj_diff_mats(l_eigs_k, sp_mask, val_weighted=val_weighted, exp_ret=exp_EADMs,
                                            rm_dc=rm_dc)
        tp_FEADM_decomp, l_basis_dim, np_FEADMs = basis_for_eigenvec_adj_diff_mats(df_EADMs_k, nx_graph, decomp,
                                                                                   norm_FEADMs=norm_FEADMs)
        if decomp == 'qr':
            np_FEADM_Basis = tp_FEADM_decomp[0]
            np_FEADM_Embed = tp_FEADM_decomp[1]
            np_FEADM_Singulars = None
        elif decomp == 'svd':
            np_FEADM_Basis = tp_FEADM_decomp[0]
            np_FEADM_Embed = tp_FEADM_decomp[1]
            np_FEADM_Singulars = tp_FEADM_decomp[2]
        else:
            raise Exception('[graph_spectra_sequence] Invalid decomp type: %s' % str(decomp))

        l_adj_edges = [(l_node[sp_u_Bk.row[i]], l_node[sp_u_Bk.col[i]]) for i in range(len(sp_u_Bk.data))]

        # >>> Incidence matrix for Bk
        sp_Ik, sp_Ik_oriented = incidence_mat_from_nodes_and_edges(l_node, l_adj_edges)
        # >>> Adjacency for L(Bk) = Ik.T @ Ik - 2 * ID
        sp_Bk_ln = sp_Ik.T @ sp_Ik - 2 * sparse.diags([1] * sp_Ik.shape[1])
        # >>> Degree for L(Bk)
        if np.asarray(np.sum(sp_Bk_ln, axis=1)).shape[0] == 1:
            sp_Dk_ln = sparse.diags(np.asarray(np.sum(sp_Bk_ln, axis=1))[0])
        else:
            sp_Dk_ln = sparse.diags(np.squeeze(np.asarray(np.sum(sp_Bk_ln, axis=1))))
        # >>> Laplacian for L(Bk)
        sp_Lk_ln = sp_Dk_ln - sp_Bk_ln
        # >>> Eigenvalues and Eigenvectors for L(Bk)
        l_eigs_ln_k = graph_eigs(sp_Lk_ln.toarray(), use_sparse=False)
        # >>> Fourier transform for FEADMs
        np_ln_U_k = np.stack([eig[1] for eig in l_eigs_ln_k])
        np_FEADMs_hat = graph_fourier(np_ln_U_k, np_FEADMs)

        l_Srec.append((idx + 1, l_eigs_k, df_EADMs_k, np_FEADMs, np_FEADM_Basis, np_FEADM_Embed, np_FEADM_Singulars,
                       l_basis_dim, l_adj_edges, sp_Ik, sp_Ik_oriented, sp_Bk_ln, l_eigs_ln_k, np_FEADMs_hat,
                       d_connected_comp_sizes[idx]))
    df_Srec = pd.DataFrame(l_Srec, columns=['K', 'eigs', 'df_EADMs', 'FEADMs', 'FEADM_Basis', 'FEADM_Embeds',
                                            'FEADM_Singulars', 'FEADM_dims', 'l_edges', 'sp_ln_inc', 'sp_ln_inc_oriented',
                                            'sp_ln_adj', 'ln_eigs', 'ln_FEADMs_ft', 'connected_comp_sizes'])
    df_Srec = df_Srec.set_index('K')
    if save_path is not None:
        pd.to_pickle(df_Srec, save_path)
    logging.debug('[graph_spectra_sequence] all done with %s spectral recs in %s secs.'
                  % (len(df_Srec), time.time() - timer_start))
    return df_Srec


class LN_to_VX(nn.Module):
    def __init__(self, num_ln_eigs, num_ln_edges, num_vx_eigs, num_vx_vertices):
        super(LN_to_VX, self).__init__()
        self.m_ln_hidden = th.nn.Linear(in_features=num_ln_edges, out_features=num_vx_eigs, bias=False)
        self.m_inner_convert = th.nn.Linear(in_features=num_ln_eigs, out_features=num_ln_eigs)
        self.m_vx_hidden = th.nn.Linear(in_features=num_ln_eigs, out_features=num_vx_vertices, bias=False)

    def forward(self, th_ln_U):
        # th_ln_to_int = self.m_ln_hidden(th_ln_U)
        # th_ln_to_int = th_ln_to_int.T
        # th_inner = self.m_inner_convert(th_ln_to_int)
        # th_rec_vx_U = self.m_vx_hidden(th_inner)

        #>>> th_ln_U = L(U).T, where L(U) is a col mat
        #>>> th_ln_to_int = L(U).T @ H_ln.T
        th_ln_to_int = self.m_ln_hidden(th_ln_U)
        #>>> U.T = [L(U).T @ H_ln.T].T @ H_vx.T = H_ln @ L(U) @ H_vx.T, where U is also a col mat
        th_rec_vx_U = self.m_vx_hidden(th_ln_to_int.T)
        return th_rec_vx_U


def convert_ln_eigs_to_vx_eigs(df_spec_seq, loss_threshold=None, max_epoch=500, use_cuda=False, save_ret=True,
                               save_path=None):
    logging.debug('[convert_ln_eigs_to_vx_eigs] starts.')

    l_rec = []
    for K, Srec in df_spec_seq.iterrows():
        l_vx_eigs = Srec['eigs']
        l_ln_eigs = Srec['ln_eigs']

        np_vx_U = np.stack([eig[1] for eig in l_vx_eigs])
        np_ln_U = np.stack([eig[1] for eig in l_ln_eigs])

        th_vx_U = th.from_numpy(np_vx_U).type(th.float32)
        th_ln_U = th.from_numpy(np_ln_U).type(th.float32)
        ln_to_vx_model = LN_to_VX(num_ln_eigs=th_ln_U.shape[0], num_ln_edges=th_ln_U.shape[1],
                                  num_vx_eigs=th_vx_U.shape[0], num_vx_vertices=th_vx_U.shape[1])

        if use_cuda:
            th_vx_U = th_vx_U.to('cuda')
            th_ln_U = th_ln_U.to('cuda')
            ln_to_vx_model = ln_to_vx_model.to('cuda')

        th_vx_U.requires_grad = False
        th_ln_U.requires_grad = False

        optimizer = th.optim.Adagrad(ln_to_vx_model.parameters())
        loss_fn = th.nn.MSELoss()

        for epoch in range(max_epoch):
            optimizer.zero_grad()
            th_rec_vx_U = ln_to_vx_model(th_ln_U)
            loss = loss_fn(th_rec_vx_U, th_vx_U)
            logging.debug('[convert_ln_eigs_to_vx_eigs] Epoch %s: K=%s, loss=%s' % (epoch, K, loss.item()))
            if loss_threshold is not None and loss <= loss_threshold:
                break
            loss.backward()
            optimizer.step()

        l_rec.append((K, ln_to_vx_model.state_dict(), loss.item()))
    df_ln_to_vx = pd.DataFrame(l_rec, columns=['K', 'ln_to_vx_model', 'mse_loss'])
    df_ln_to_vx = df_ln_to_vx.set_index('K')
    if save_ret:
        pd.to_pickle(df_ln_to_vx, save_path + 'df_ln_to_vx_eigs_convert.pickle')
    logging.debug('[convert_ln_eigs_to_vx_eigs] All done.')
    return df_ln_to_vx


def draw_graph_spectra_sequence(df_spectral_seq, nx_graph, graph_name, save_folder, line_graph=False,
                                save_img=False, show_img=True):
    logging.debug('[draw_graph_spectra_sequence] starts.')
    timer_start = time.time()

    l_nodes = list(nx_graph.nodes())
    if line_graph:
        l_node_str = [node[0] + node[1] for node in l_nodes]
    else:
        l_node_str = l_nodes
    nx_graph_frame = nx.Graph()
    nx_graph_frame.add_edges_from(list(combinations(l_nodes, 2)))
    nx_graph_pos = nx.spring_layout(nx_graph_frame, k=0.8)

    for K, Srec in df_spectral_seq.iterrows():
        l_eigs = Srec['eigs']
        df_EADMs = Srec['df_EADMs']
        np_FEADMs = Srec['FEADMs']
        np_FEADM_basis = Srec['FEADM_Basis']
        np_FEADM_Embeds = Srec['FEADM_Embeds']
        np_FEADM_Singulars = Srec['FEADM_Singulars']
        l_basis_dims = Srec['FEADM_dims']
        if line_graph:
            l_basis_dims = [l_node_str[l_nodes.index(item[0])] + l_node_str[l_nodes.index(item[1])] for item in
                            l_basis_dims]
        l_edges = Srec['l_edges']
        l_edge_str = [edge[0] + edge[1] for edge in l_edges]
        sp_ln_inc = Srec['sp_ln_inc']
        sp_ln_adj = Srec['sp_ln_adj']
        l_ln_eigs = Srec['ln_eigs']
        np_FEADMs_ft = Srec['ln_FEADMs_ft']

        np_eff_FEADM_Singulars = np.round(np_FEADM_Singulars, decimals=12)
        num_eff_FEADM_Singulars = np.count_nonzero(np_eff_FEADM_Singulars)
        np_eff_FEADM_Singulars = np_eff_FEADM_Singulars[:num_eff_FEADM_Singulars]

        np_eff_FEADM_Embeds = np_FEADM_Embeds[:, :num_eff_FEADM_Singulars]

        np_FEADMs_norm = preprocessing.normalize(np_FEADMs)
        np_FEADM_sim = np.matmul(np_FEADMs_norm, np_FEADMs_norm.T)

        # nx_graph_frame.clear_edges()
        # nx_graph_frame.add_edges_from(l_edges)

        K_folder = save_folder + 'K%s' % str(K) + '/'
        if not os.path.exists(K_folder):
            os.mkdir(K_folder)

        # >>> draw the graph @ K
        graph_img_name = graph_name + '@K%s.PNG' % str(K)
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
        fig.suptitle('Graph @ K=%s' % str(K), fontsize=40, fontweight='semibold')
        x_values, y_values = zip(*nx_graph_pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.10
        plt.xlim(x_min - x_margin, x_max + x_margin)
        d_node_labels = {node[0]: node[0] for node in nx_graph.nodes(data=True)}
        nx.draw_networkx_nodes(nx_graph, nx_graph_pos, node_size=20)
        nx.draw_networkx_labels(nx_graph, nx_graph_pos, labels=d_node_labels, font_size=30, font_color='r',
                                font_weight='semibold')
        # l_edges = nx_graph_frame.edges()
        drawn_edges = nx.draw_networkx_edges(nx_graph_frame,
                                             nx_graph_pos,
                                             edgelist=l_edges,
                                             width=2,
                                             edge_color='blue')
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.95)
        if save_img:
            plt.savefig(K_folder + graph_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw eigenvalues and eigenvectors
        eig_img_name = graph_name + '#eigs@K%s.PNG' % str(K)
        max_eig_val = np.max([eig[1] for eig in l_eigs])
        min_eig_val = np.min([eig[1] for eig in l_eigs])
        img_height = math.ceil(len(l_eigs) / 10) * 24
        fig, axes = plt.subplots(ncols=1, nrows=len(l_eigs), figsize=(10, img_height))
        fig.suptitle('Eigenvalues & Eigenvectors @ K=%s' % str(K), fontsize=20, fontweight='semibold')
        for idx, (eig_val, eig_vec) in enumerate(l_eigs):
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eig_val, decimals=3), fontsize=10)
            markerline, stemlines, baselineaxes = axes[idx].stem(l_node_str, eig_vec)
            axes[idx].set_xticks([i for i in range(len(l_node_str))])
            axes[idx].set_yticks(np.arange(min_eig_val, max_eig_val, step=0.1))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_img:
            plt.savefig(K_folder + eig_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw EADMs
        max_EADM_val = np.max([np.max(eadm) for eadm in df_EADMs['eigvec_sp_EADM']])
        min_EADM_val = np.min([np.min(eadm) for eadm in df_EADMs['eigvec_sp_EADM']])
        eadms_img_name = graph_name + '#EADMs@K%s.PNG' % str(K)
        fig, axes = plt.subplots(ncols=1, nrows=len(df_EADMs), figsize=(5, 50))
        fig.suptitle('EADMs @ K=%s' % str(K), fontsize=15, fontweight='semibold')
        for idx, EADM_rec in df_EADMs.iterrows():
            eigenval = EADM_rec['eigval']
            eigenvec = EADM_rec['eigvec']
            sp_EADM = EADM_rec['eigvec_sp_EADM']
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eigenval, decimals=3), fontsize=10)
            axes[idx].set_xticks([i for i in range(len(eigenvec))])
            axes[idx].set_xticklabels(l_node_str)
            axes[idx].set_yticks([i for i in range(len(eigenvec))])
            axes[idx].set_yticklabels(l_node_str)
            # axes[idx].matshow(sp_u_pw_ele_diff.toarray())
            pos = axes[idx].imshow(sp_EADM.toarray(), vmin=min_EADM_val, vmax=max_EADM_val, cmap='viridis')
            divider = make_axes_locatable(axes[idx])
            cax = divider.append_axes("right", size="3%", pad=0.1)
            fig.colorbar(pos, ax=axes[idx], cax=cax)
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.97)
        if save_img:
            plt.savefig(K_folder + eadms_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw EADM Weights
        max_EADM_w_val = np.max([np.max(eadm) for eadm in df_EADMs['eigvec_sp_EADM_weights']])
        min_EADM_w_val = np.min([np.min(eadm) for eadm in df_EADMs['eigvec_sp_EADM_weights']])
        eadms_img_name = graph_name + '#EADM_Weights@K%s.PNG' % str(K)
        fig, axes = plt.subplots(ncols=1, nrows=len(df_EADMs), figsize=(5, 50))
        fig.suptitle('EADM Weights @ K=%s' % str(K), fontsize=15, fontweight='semibold')
        for idx, EADM_rec in df_EADMs.iterrows():
            eigenval = EADM_rec['eigval']
            eigenvec = EADM_rec['eigvec']
            sp_EADM_weights = EADM_rec['eigvec_sp_EADM_weights']
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eigenval, decimals=3), fontsize=10)
            axes[idx].set_xticks([i for i in range(len(eigenvec))])
            axes[idx].set_xticklabels(l_node_str)
            axes[idx].set_yticks([i for i in range(len(eigenvec))])
            axes[idx].set_yticklabels(l_node_str)
            # axes[idx].matshow(sp_u_pw_ele_diff.toarray())
            pos = axes[idx].imshow(sp_EADM_weights.toarray(), vmin=min_EADM_w_val, vmax=max_EADM_w_val, cmap='viridis')
            divider = make_axes_locatable(axes[idx])
            cax = divider.append_axes("right", size="3%", pad=0.1)
            fig.colorbar(pos, ax=axes[idx], cax=cax)
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.97)
        if save_img:
            plt.savefig(K_folder + eadms_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw FEADMs
        feadms_img_name = graph_name + '#FEADMs@K%s.PNG' % str(K)
        figsize_width = math.ceil(np_FEADMs.shape[1] / 40) * 10
        max_FEADM_val = np.max(np_FEADMs)
        min_FEADM_val = np.min(np_FEADMs)
        FEADM_val_step = np.round((max_FEADM_val - min_FEADM_val) / 10, decimals=2)
        fig, axes = plt.subplots(ncols=1, nrows=len(np_FEADMs), figsize=(figsize_width, 40))
        fig.suptitle('FEADM @ K=%s' % str(K), fontsize=20, fontweight='semibold')
        for idx, feadm in enumerate(np_FEADMs):
            axes[idx].grid(True)
            axes[idx].set_title(r'FEADM @ $\lambda$ = %s' % np.round(l_eigs[idx][0], decimals=3), fontsize=10)
            markerline, stemlines, baselineaxes = axes[idx].stem([item[0] + item[1] for item in l_basis_dims],
                                                                 feadm, bottom=min_FEADM_val)
            axes[idx].set_yticks(np.arange(min_FEADM_val, max_FEADM_val, step=FEADM_val_step))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_img:
            plt.savefig(K_folder + feadms_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw Pairwise Cosine Similarities between FEADMs
        eadmsim_img_name = graph_name + '#FEADM_Similarities@K%s.PNG' % str(K)
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        fig.suptitle('FEADM Similarities @ K=%s' % str(K), fontsize=15, fontweight='semibold')
        l_eigvals = [np.round(item[0], decimals=3) for item in l_eigs]
        axes.grid(True)
        pos = axes.imshow(np_FEADM_sim, vmin=0, vmax=1, cmap='viridis')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        axes.set_xticks([i for i in range(len(l_eigvals))])
        axes.set_xticklabels(l_eigvals)
        axes.set_yticks([i for i in range(len(l_eigvals))])
        axes.set_yticklabels(l_eigvals)
        fig.colorbar(pos, ax=axes, cax=cax)
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.98)
        if save_img:
            plt.savefig(K_folder + eadmsim_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw FEADM Basis
        feadmb_img_name = graph_name + '#FEADM_Basis@K%s.PNG' % str(K)
        figsize_width = math.ceil(np_FEADM_basis.shape[1] / 40) * 10
        max_FEADM_basis_val = np.max(np_FEADM_basis)
        min_FEADM_basis_val = np.min(np_FEADM_basis)
        FEADM_basis_val_step = np.round((max_FEADM_basis_val - min_FEADM_basis_val) / 10, decimals=2)
        fig, axes = plt.subplots(ncols=1, nrows=len(np_FEADM_basis), figsize=(figsize_width, 40))
        fig.suptitle('FEADM Basis (Top %s effective) @ K=%s' % (num_eff_FEADM_Singulars, K), fontsize=20,
                     fontweight='semibold')
        for idx, bc_vec in enumerate(np_FEADM_basis):
            axes[idx].grid(True)
            axes[idx].set_title('Basic Component %s' % str(idx), fontsize=10)
            markerline, stemlines, baselineaxes = axes[idx].stem([item[0] + item[1] for item in l_basis_dims], bc_vec)
            axes[idx].set_yticks(np.arange(min_FEADM_basis_val, max_FEADM_basis_val, step=FEADM_basis_val_step))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_img:
            plt.savefig(K_folder + feadmb_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw Effective FEADM Embeds
        feadme_img_name = graph_name + '#EF_FEADM_Embeds@K%s.PNG' % str(K)
        max_FEADM_Embed_val = np.max(np_eff_FEADM_Embeds)
        min_FEADM_Embed_val = np.min(np_eff_FEADM_Embeds)
        FEADM_Embed_val_step = np.round((max_FEADM_Embed_val - min_FEADM_Embed_val) / 10, decimals=2)
        fig, axes = plt.subplots(ncols=1, nrows=len(np_eff_FEADM_Embeds), figsize=(10, 40))
        fig.suptitle('Effective FEADM Embeds @ K=%s' % str(K), fontsize=20, fontweight='semibold')
        for idx, embed_vec in enumerate(np_eff_FEADM_Embeds):
            axes[idx].grid(True)
            axes[idx].set_title(r'Effective Embed for FEADM @ $\lambda$ = %s' % np.round(l_eigs[idx][0], decimals=3),
                                fontsize=10)
            markerline, stemlines, baselineaxes = axes[idx].stem(
                ['bc#' + str(i) for i in range(len(np_eff_FEADM_Singulars))], embed_vec)
            axes[idx].set_yticks(np.arange(min_FEADM_Embed_val, max_FEADM_Embed_val, step=FEADM_Embed_val_step))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_img:
            plt.savefig(K_folder + feadme_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw Effective FEADM Singulars
        if np_FEADM_Singulars is not None:
            feadmsin_img_name = graph_name + '#EF_FEADM_Singulars@K%s.PNG' % str(K)
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
            fig.suptitle('Effective FEADM Singulars (Top %s) @ K=%s' % (num_eff_FEADM_Singulars, K), fontsize=20,
                         fontweight='semibold')
            axes.grid(True)
            # axes.set_title(r'Singulars for FEADM Basis', fontsize=10)
            markerline, stemlines, baselineaxes = axes.stem(['bc#' + str(i) for i in range(len(np_FEADM_basis))],
                                                            np_FEADM_Singulars)
            # axes[idx].set_yticks(np.arange(-1, 1, step=0.1))
            plt.tight_layout(pad=1.0)
            plt.subplots_adjust(top=0.90)
            if save_img:
                plt.savefig(K_folder + feadmsin_img_name, format='PNG')
            if show_img:
                plt.show()
            plt.clf()

        # >>> draw Effective FEADM Singular-Weighted Embeds
        if np_FEADM_Singulars is not None:
            feadmswe_img_name = graph_name + '#EF_FEADM_Singular_Weighted_Embeds@K%s.PNG' % str(K)
            np_eff_FEADM_SWEmbeds = np_eff_FEADM_Embeds * np_eff_FEADM_Singulars
            max_FEADM_SWEmbed_val = np.max(np_eff_FEADM_SWEmbeds)
            min_FEADM_SWEmbed_val = np.min(np_eff_FEADM_SWEmbeds)
            FEADM_SWEmbed_val_step = np.round((max_FEADM_SWEmbed_val - min_FEADM_SWEmbed_val) / 10, decimals=2)
            fig, axes = plt.subplots(ncols=1, nrows=len(np_eff_FEADM_SWEmbeds), figsize=(10, 40))
            fig.suptitle('Effective FEADM Singular Weighted Embeds @ K=%s' % str(K), fontsize=20, fontweight='semibold')
            for idx, embed_vec in enumerate(np_eff_FEADM_SWEmbeds):
                axes[idx].grid(True)
                axes[idx].set_title(
                    r'Effective Singular Weighted Embed for FEADM @ $\lambda$ = %s' % np.round(l_eigs[idx][0],
                                                                                               decimals=3), fontsize=10)
                markerline, stemlines, baselineaxes = axes[idx].stem(
                    ['bc#' + str(i) for i in range(len(np_eff_FEADM_Singulars))], embed_vec)
                axes[idx].set_yticks(
                    np.arange(min_FEADM_SWEmbed_val, max_FEADM_SWEmbed_val, step=FEADM_SWEmbed_val_step))
            plt.tight_layout(pad=1.0)
            plt.subplots_adjust(top=0.96)
            if save_img:
                plt.savefig(K_folder + feadmswe_img_name, format='PNG')
            if show_img:
                plt.show()
            plt.clf()

        # >>> draw Line Adjacency
        np_ln_adj = sp_ln_adj.toarray()
        max_ln_adj_val = np.max(np_ln_adj)
        min_ln_adj_val = np.min(np_ln_adj)
        ln_adj_img_name = graph_name + '#ln_adj@K%s.PNG' % str(K)
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
        fig.suptitle('Line Adjacency @ K=%s' % str(K), fontsize=15, fontweight='semibold')
        axes.grid(True)
        # axes[idx].set_title(r'$\lambda$ = %s' % np.round(eigenval, decimals=3), fontsize=10)
        axes.set_xticks([i for i in range(len(l_edge_str))])
        axes.set_xticklabels(l_edge_str)
        axes.set_yticks([i for i in range(len(l_edge_str))])
        axes.set_yticklabels(l_edge_str)
        pos = axes.imshow(np_ln_adj, vmin=min_ln_adj_val, vmax=max_ln_adj_val, cmap='viridis')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(pos, ax=axes, cax=cax)
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.97)
        if save_img:
            plt.savefig(K_folder + ln_adj_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw line eigenvalues and eigenvectors
        ln_eig_img_name = graph_name + '#ln_eigs@K%s.PNG' % str(K)
        max_ln_eigs_val = np.max([eig[1] for eig in l_ln_eigs])
        min_ln_eigs_val = np.min([eig[1] for eig in l_ln_eigs])
        img_height = math.ceil(len(l_ln_eigs) / 10) * 24
        fig, axes = plt.subplots(ncols=1, nrows=len(l_ln_eigs), figsize=(10, img_height))
        fig.suptitle('Line Eigenvalues & Eigenvectors @ K=%s' % str(K), fontsize=20, fontweight='semibold')
        for idx, (eig_val, eig_vec) in enumerate(l_ln_eigs):
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eig_val, decimals=3), fontsize=10)
            markerline, stemlines, baselineaxes = axes[idx].stem(l_edge_str, eig_vec)
            axes[idx].set_yticks(np.arange(min_ln_eigs_val, max_ln_eigs_val, step=0.1))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_img:
            plt.savefig(K_folder + ln_eig_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw Fourier transform for FEADMs in line graph
        ln_FEADMs_ft_img_name = graph_name + '#ln_FEADMs_ft@K%s.PNG' % str(K)
        max_ln_FEADMs_ft_val = np.max(np_FEADMs_ft)
        min_ln_FEADMs_ft_val = np.min(np_FEADMs_ft)
        img_height = math.ceil(len(np_FEADMs_ft) / 10) * 24
        fig, axes = plt.subplots(ncols=1, nrows=len(np_FEADMs_ft), figsize=(10, img_height))
        fig.suptitle('Fourier Transformed FEADMs in Line Graph @ K=%s' % str(K), fontsize=20, fontweight='semibold')
        for idx, FEADM_ft in enumerate(np_FEADMs_ft):
            axes[idx].grid(True)
            axes[idx].set_title(r'Original $\lambda$ = %s' % np.round(l_eigs[idx][0], decimals=3), fontsize=10)
            markerline, stemlines, baselineaxes = axes[idx].stem(FEADM_ft)
            axes[idx].set_xticks([i for i in range(len(l_ln_eigs))])
            axes[idx].set_xticklabels(np.round([eig[0] for eig in l_ln_eigs], decimals=3))
            axes[idx].set_yticks(np.arange(min_ln_FEADMs_ft_val, max_ln_FEADMs_ft_val, step=0.1))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_img:
            plt.savefig(K_folder + ln_FEADMs_ft_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw Pairwise Cosine Similarities of Fourier transformed FEADMs in line graph
        l_eigvals = [np.round(item[0], decimals=3) for item in l_eigs]
        np_pwc_ln_FEADMs_ft = np.matmul(np_FEADMs_ft, np_FEADMs_ft.T)
        pwc_ln_FEADMs_ft_img_name = graph_name + '#ln_FEADMs_ft_Similarities@K%s.PNG' % str(K)
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        fig.suptitle('Fourier Transformed FEADMs in Line Graph Similarities @ K=%s' % str(K),
                     fontsize=20, fontweight='semibold')
        axes.grid(True)
        pos = axes.imshow(np_pwc_ln_FEADMs_ft, vmin=0, vmax=1, cmap='viridis')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(pos, ax=axes, cax=cax)
        axes.set_xticks([i for i in range(len(l_eigvals))])
        axes.set_xticklabels(l_eigvals)
        axes.set_yticks([i for i in range(len(l_eigvals))])
        axes.set_yticklabels(l_eigvals)
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_img:
            plt.savefig(K_folder + pwc_ln_FEADMs_ft_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw FEADM Embeds without DC component
        # feadme_nodc_img_name = graph_name + '#FEADM_Embeds_nodc@K%s.PNG' % str(K)
        # fig, axes = plt.subplots(ncols=1, nrows=len(np_FEADM_Embeds), figsize=(10, 40))
        # fig.suptitle('FEADM Embeds (without DC) @ K=%s' % str(K), fontsize=20, fontweight='semibold')
        # for idx, embed_vec in enumerate(np_FEADM_Embeds):
        #     axes[idx].grid(True)
        #     axes[idx].set_title(r'Embed for FEADM (without DC) @ $\lambda$ = %s' % np.round(l_eigs[idx][0], decimals=3), fontsize=10)
        #     markerline, stemlines, baselineaxes = axes[idx].stem(['bc#' + str(i) for i in range(1, len(np_FEADM_basis))], embed_vec[1:])
        #     # axes[idx].set_yticks(np.arange(-1, 1, step=0.1))
        # plt.tight_layout(pad=1.0)
        # plt.subplots_adjust(top=0.96)
        # plt.savefig(K_folder + feadme_nodc_img_name, format='PNG')
        # # plt.show()
        # plt.clf()
        # logging.debug('[draw_graph_spectra_sequence] done for K=%s in %s secs.' % (K, time.time() - timer_start))

    logging.debug('[draw_graph_spectra_sequence] all done in %s secs.' % str(time.time() - timer_start))


def analyze_vector_signal_against_spectral_sequence(np_vec_sig, df_spectral_seq, save_ret=False, save_path=None,
                                                    np_pw_dist=None, exp_pw_d=False, w_sig_FADM=False,
                                                    norm_sig_FADM=True):
    logging.debug('[analyze_vector_signal_against_spectral_sequence] starts.')
    timer_start = time.time()

    if np_vec_sig is not None and np_pw_dist is None:
        np_vec_sig = preprocessing.normalize(np_vec_sig)
        np_sig_pwd = 1.0 - np.matmul(np_vec_sig, np_vec_sig.T)
    elif np_pw_dist is not None:
        np_sig_pwd = np_pw_dist

    if exp_pw_d:
        # >>> to address the numeric issue caused by zeros and very small values
        np_sig_pwd = np.exp(np_sig_pwd)

    np_sig_pwd = np.round(np_sig_pwd, decimals=12)

    l_analysis_rec = []
    for K, Srec in df_spectral_seq.iterrows():
        l_eigs = Srec['eigs']
        df_EADMs = Srec['df_EADMs']
        np_FEADM_basis = Srec['FEADM_Basis']
        np_FEADM_Embeds = Srec['FEADM_Embeds']
        np_FEADM_Singulars = Srec['FEADM_Singulars']
        l_basis_dims = Srec['FEADM_dims']
        np_FEADMs = Srec['FEADMs']
        l_ln_eigs = Srec['ln_eigs']
        np_ln_U = np.stack([eig[1] for eig in l_ln_eigs])
        l_edges = Srec['l_edges']
        np_FEADMs_ft = Srec['ln_FEADMs_ft']

        EADM_mask = df_EADMs.iloc[0]['EADM_mask']
        EADM_mask_row = [item[0] for item in EADM_mask]
        EADM_mask_col = [item[1] for item in EADM_mask]

        l_EADM_weights = []
        for _, EADM_rec in df_EADMs.iterrows():
            EADM_weights = EADM_rec['eigvec_sp_EADM_weights']
            EADM_weights = np.squeeze(np.asarray(EADM_weights[EADM_mask_row, EADM_mask_col]))
            l_EADM_weights.append(EADM_weights)
        np_EADM_weights = np.stack(l_EADM_weights)

        np_sig_pwd_FADM = np_sig_pwd[EADM_mask_row, EADM_mask_col]
        np_sig_pwd_FADM = np.stack([np_sig_pwd_FADM] * len(l_eigs))
        if w_sig_FADM:
            np_sig_pwd_FADM = np_sig_pwd_FADM * np_EADM_weights
        if norm_sig_FADM:
            np_sig_pwd_FADM = preprocessing.normalize(np_sig_pwd_FADM)

        # >>> Compare signal to eigenvectors in the FEADM space
        np_FEADM_Singulars = np.round(np_FEADM_Singulars, decimals=12)
        num_eff_singulars = np.count_nonzero(np_FEADM_Singulars)
        np_eff_FEADM_Singulars = np_FEADM_Singulars[:num_eff_singulars]
        np_eff_FEADM_Embeds = np_FEADM_Embeds[:, :num_eff_singulars]
        np_sig_pwd_FEADM_Embed = np.matmul(np_sig_pwd_FADM, np_FEADM_basis[:num_eff_singulars, :].T)

        np_sig_pwd_FADM_Embed_vs_FEADM_Embeds = \
            np.einsum('ij..., ij...->i',
                      np_sig_pwd_FEADM_Embed / np_eff_FEADM_Singulars,
                      np_eff_FEADM_Embeds)
        np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds = \
            np.einsum('ij..., ij...->i',
                      np_sig_pwd_FEADM_Embed,
                      np_eff_FEADM_Embeds * np_eff_FEADM_Singulars)

        # >>> Compare signal to eigenvectors in the line graph eigenspace
        np_sig_pwd_FEADM_hat = graph_fourier(np_ln_U, np_sig_pwd_FADM[0])
        np_sig_pwd_FEADM_hat_vs_FEADMs_hat = np.matmul(np_sig_pwd_FEADM_hat, np_FEADMs_ft.T)

        l_analysis_rec.append((K, l_eigs, l_ln_eigs, np_sig_pwd_FEADM_Embed[0], np_sig_pwd_FEADM_hat,
                               np_sig_pwd_FADM_Embed_vs_FEADM_Embeds,
                               np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds,
                               np_sig_pwd_FEADM_hat_vs_FEADMs_hat))
        logging.debug('[analyze_vector_signal_against_spectral_sequence] all done with K=%s in %s secs.'
                      % (K, time.time() - timer_start))

    df_analysis = pd.DataFrame(l_analysis_rec, columns=['K', 'eigs', 'ln_eigs',
                                                        'sig_pwd_FEADM_Embed', 'sig_pwd_ln_FEADM_ft',
                                                        'sig_pwd_FADM_Embed_vs_FEADM_Embeds',
                                                        'w_sig_pwd_FADM_Embed_vs_FEADM_Embeds',
                                                        'sig_pwd_ln_FEADM_ft_vs_ln_FEADMs_ft'])
    df_analysis = df_analysis.set_index('K')
    if save_ret:
        pd.to_pickle(df_analysis, save_path)
    logging.debug('[analyze_vector_signal_against_spectral_sequence] all done with %s recs in %s secs.'
                  % (len(df_analysis), time.time() - timer_start))
    return df_analysis


def draw_vector_signal_analysis_against_spectral_sequence_over_epoch(df_analysis_collection, df_spectral_seq,
                                                                     nx_graph, param_str,
                                                                     init_only=False,
                                                                     save_ret=True, show_img=False, save_path_fmt=None):
    logging.debug('[draw_vector_signal_analysis_against_spectral_sequence_over_epoch] start.')
    timer_start = time.time()

    d_sig_vs_eigvecs_by_K = dict()
    d_w_sig_vs_eigvecs_by_K = dict()
    d_ln_sig_vs_eigvecs_by_K = dict()
    d_metrics_by_epoch = dict()
    l_metric_names = None
    for epoch, df_analysis_rec in df_analysis_collection.iterrows():
        l_metric_names = df_analysis_rec['metric_names']
        l_metric_vals = df_analysis_rec['metric_vals']
        if l_metric_names is not None and l_metric_vals is not None:
            for metric_idx, metric_name in enumerate(l_metric_names):
                if metric_name not in d_metrics_by_epoch:
                    d_metrics_by_epoch[metric_name] = [l_metric_vals[metric_idx]]
                else:
                    d_metrics_by_epoch[metric_name].append(l_metric_vals[metric_idx])

        df_analysis = df_analysis_rec['df_analysis']
        for K, ana_rec in df_analysis.iterrows():
            np_sig_pwd_FADM_Embed_vs_FEADM_Embeds = ana_rec['sig_pwd_FADM_Embed_vs_FEADM_Embeds']
            np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds = ana_rec['w_sig_pwd_FADM_Embed_vs_FEADM_Embeds']
            np_sig_pwd_FEADM_hat_vs_FEADMs_hat = ana_rec['sig_pwd_ln_FEADM_ft_vs_ln_FEADMs_ft']
            if K not in d_sig_vs_eigvecs_by_K:
                d_sig_vs_eigvecs_by_K[K] = [np_sig_pwd_FADM_Embed_vs_FEADM_Embeds]
            else:
                d_sig_vs_eigvecs_by_K[K].append(np_sig_pwd_FADM_Embed_vs_FEADM_Embeds)
            if K not in d_w_sig_vs_eigvecs_by_K:
                d_w_sig_vs_eigvecs_by_K[K] = [np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds]
            else:
                d_w_sig_vs_eigvecs_by_K[K].append(np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds)
            if K not in d_ln_sig_vs_eigvecs_by_K:
                d_ln_sig_vs_eigvecs_by_K[K] = [np_sig_pwd_FEADM_hat_vs_FEADMs_hat]
            else:
                d_ln_sig_vs_eigvecs_by_K[K].append(np_sig_pwd_FEADM_hat_vs_FEADMs_hat)

    df_analysis_init = df_analysis_collection.loc[0]['df_analysis']
    df_analysis_final = df_analysis_collection.loc[np.max(df_analysis_collection.index)]['df_analysis']

    np_ne_init = df_analysis_collection.loc[0]['np_ne']
    if not init_only:
        np_ne_init = preprocessing.normalize(np_ne_init)
        sp_pw_dc_init = sparse.coo_matrix(np.triu(1.0 - np.matmul(np_ne_init, np_ne_init.T), k=1))
        np_ne_final = df_analysis_collection.loc[np.max(df_analysis_collection.index)]['np_ne']
        np_ne_final = preprocessing.normalize(np_ne_final)
        sp_pw_dc_final = sparse.coo_matrix(np.triu(1.0 - np.matmul(np_ne_final, np_ne_final.T), k=1))
    else:
        if np_ne_init is not None:
            np_ne_init = preprocessing.normalize(np_ne_init)
            sp_pw_dc_init = sparse.coo_matrix(np.triu(1.0 - np.matmul(np_ne_init, np_ne_init.T), k=1))
        else:
            np_pw_dist = df_analysis_collection.loc[0]['np_pw_dist']
            if np_pw_dist is not None:
                sp_pw_dc_init = sparse.coo_matrix(np.triu(np_pw_dist, k=1))

    l_nodes = list(nx_graph.nodes())

    if init_only:
        # >>> draw matching between signal and eigenvectors
        img_name = param_str + ' Initial vs Eigenvectors'
        fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
        fig.suptitle(img_name, fontsize=15, fontweight='semibold')
        idx = 0
        for K in d_sig_vs_eigvecs_by_K:
            np_sig_pwd_FADM_Embed_vs_FEADM_Embeds = np.abs(
                df_analysis_init.loc[K]['sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
            l_eigs = df_analysis_collection.iloc[0]['df_analysis'].loc[K]['eigs']
            l_eigvals = [eig[0] for eig in l_eigs]
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=15)
            init_linefmt = 'tab:blue'
            init_marker_fmt = 'o'
            axes[idx].stem([i for i in range(len(l_eigvals))], np_sig_pwd_FADM_Embed_vs_FEADM_Embeds,
                           linefmt=init_linefmt,
                           markerfmt=init_marker_fmt)
            axes[idx].set_xticks([i for i in range(len(l_eigvals))])
            axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
            axes[idx].legend()
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.94)
        if save_ret:
            plt.savefig(save_path_fmt.format('init_vs_eigs'), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw weighted matching between signal and eigenvectors
        img_name = param_str + ' Weighted Initial vs Eigenvectors'
        fig, axes = plt.subplots(ncols=1, nrows=len(d_w_sig_vs_eigvecs_by_K), figsize=(10, 20))
        fig.suptitle(img_name, fontsize=15, fontweight='semibold')
        idx = 0
        for K in d_w_sig_vs_eigvecs_by_K:
            np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds = np.abs(
                df_analysis_init.loc[K]['w_sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
            l_eigs = df_analysis_collection.iloc[0]['df_analysis'].loc[K]['eigs']
            l_eigvals = [eig[0] for eig in l_eigs]
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=15)
            init_linefmt = 'tab:blue'
            init_marker_fmt = 'o'
            axes[idx].stem([i for i in range(len(l_eigvals))], np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds,
                           linefmt=init_linefmt,
                           markerfmt=init_marker_fmt)
            axes[idx].set_xticks([i for i in range(len(l_eigvals))])
            axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
            axes[idx].legend()
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.94)
        if save_ret:
            plt.savefig(save_path_fmt.format('w_init_vs_eigs'), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw matching between signal and eigenvectors in line graph
        img_name = param_str + ' Initial vs Eigenvectors in Line Graph'
        fig, axes = plt.subplots(ncols=1, nrows=len(d_ln_sig_vs_eigvecs_by_K), figsize=(10, 20))
        fig.suptitle(img_name, fontsize=15, fontweight='semibold')
        idx = 0
        for K in d_ln_sig_vs_eigvecs_by_K:
            np_sig_pwd_FEADM_hat_vs_FEADMs_hat = np.abs(df_analysis_init.loc[K]['sig_pwd_ln_FEADM_ft_vs_ln_FEADMs_ft'])
            l_eigs = df_analysis_collection.iloc[0]['df_analysis'].loc[K]['eigs']
            l_eigvals = [eig[0] for eig in l_eigs]
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=15)
            init_linefmt = 'tab:blue'
            init_marker_fmt = 'o'
            axes[idx].stem([i for i in range(len(l_eigvals))], np_sig_pwd_FEADM_hat_vs_FEADMs_hat,
                           linefmt=init_linefmt,
                           markerfmt=init_marker_fmt)
            axes[idx].set_xticks([i for i in range(len(l_eigvals))])
            axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
            axes[idx].legend()
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.94)
        if save_ret:
            plt.savefig(save_path_fmt.format('ln_init_vs_eigs'), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw FADMs for initial pairwise cosine distances
        img_name = param_str + ' Initial FADMs'
        fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
        fig.suptitle(img_name, fontsize=10, fontweight='semibold')
        idx = 0
        for K in d_sig_vs_eigvecs_by_K:
            df_EADMs = df_spectral_seq.loc[K]['df_EADMs']
            EADM_mask = df_EADMs.iloc[0]['EADM_mask']
            EADM_mask_row = [item[0] for item in EADM_mask]
            EADM_mask_col = [item[1] for item in EADM_mask]
            FADM_init = sp_pw_dc_init.toarray()[EADM_mask_row, EADM_mask_col]
            l_xlabels = [l_nodes[item[0]] + l_nodes[item[1]] for item in EADM_mask]
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=10)
            init_linefmt = 'tab:blue'
            init_marker_fmt = 'o'
            axes[idx].stem([i for i in range(len(FADM_init))], FADM_init,
                           linefmt=init_linefmt,
                           markerfmt=init_marker_fmt)
            axes[idx].set_xticks([i for i in range(len(l_xlabels))])
            axes[idx].set_xticklabels(l_xlabels)
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.95)
        if save_ret:
            plt.savefig(save_path_fmt.format('init#FADMs'), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw FADM-Embeds for initial pairwise distances
        img_name = param_str + ' Initial FADM Embeds'
        fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
        fig.suptitle(img_name, fontsize=10, fontweight='semibold')
        idx = 0
        for K in d_sig_vs_eigvecs_by_K:
            np_FADM_Embed_init = df_analysis_init.loc[K]['sig_pwd_FEADM_Embed']
            np_FEADM_Singulars = df_spectral_seq.loc[K]['FEADM_Singulars']
            num_eff_FEADM_Singulars = np.count_nonzero(np.round(np_FEADM_Singulars, decimals=12))
            np_eff_FADM_Embed_init = np.abs(
                np_FADM_Embed_init[:num_eff_FEADM_Singulars] / np_FEADM_Singulars[:num_eff_FEADM_Singulars])
            l_xlabels = ['bc#' + str(i) for i in range(len(np_eff_FADM_Embed_init))]
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=10)
            init_linefmt = 'tab:blue'
            init_marker_fmt = 'o'
            axes[idx].stem([i for i in range(len(np_eff_FADM_Embed_init))], np_eff_FADM_Embed_init,
                           linefmt=init_linefmt,
                           markerfmt=init_marker_fmt)
            axes[idx].set_xticks([i for i in range(len(l_xlabels))])
            axes[idx].set_xticklabels(l_xlabels)
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.95)
        if save_ret:
            plt.savefig(save_path_fmt.format('init#FADM_Embeds'), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw Singular-Weighted FADM-Embeds for initial pairwise distances
        img_name = param_str + ' Initial SWFADM Embeds'
        fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
        fig.suptitle(img_name, fontsize=10, fontweight='semibold')
        idx = 0
        for K in d_sig_vs_eigvecs_by_K:
            np_FADM_Embed_init = df_analysis_init.loc[K]['sig_pwd_FEADM_Embed']
            np_FEADM_Singulars = df_spectral_seq.loc[K]['FEADM_Singulars']
            num_eff_FEADM_Singulars = np.count_nonzero(np.round(np_FEADM_Singulars, decimals=12))
            np_eff_FADM_Embed_init = np.abs(np_FADM_Embed_init[:num_eff_FEADM_Singulars])
            l_xlabels = ['bc#' + str(i) for i in range(len(np_eff_FADM_Embed_init))]
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=10)
            init_linefmt = 'tab:blue'
            init_marker_fmt = 'o'
            axes[idx].stem([i for i in range(len(np_eff_FADM_Embed_init))], np_eff_FADM_Embed_init,
                           linefmt=init_linefmt,
                           markerfmt=init_marker_fmt)
            axes[idx].set_xticks([i for i in range(len(l_xlabels))])
            axes[idx].set_xticklabels(l_xlabels)
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.95)
        if save_ret:
            plt.savefig(save_path_fmt.format('init#FADM_SWEmbeds'), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw Fourier transform in line graph for initial pairwise distances
        img_name = param_str + ' Initial Line Graph Fourier Transform'
        fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
        fig.suptitle(img_name, fontsize=10, fontweight='semibold')
        idx = 0
        for K in d_sig_vs_eigvecs_by_K:
            l_ln_eigs = df_analysis_collection.iloc[0]['df_analysis'].loc[K]['ln_eigs']
            np_FADM_hat_init = np.abs(df_analysis_init.loc[K]['sig_pwd_ln_FEADM_ft'])
            l_xlabels = np.round([eig[0] for eig in l_ln_eigs], decimals=3)
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=10)
            init_linefmt = 'tab:blue'
            init_marker_fmt = 'o'
            axes[idx].stem([i for i in range(len(np_FADM_hat_init))], np_FADM_hat_init,
                           linefmt=init_linefmt,
                           markerfmt=init_marker_fmt)
            axes[idx].set_xticks([i for i in range(len(l_xlabels))])
            axes[idx].set_xticklabels(l_xlabels)
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.95)
        if save_ret:
            plt.savefig(save_path_fmt.format('init#ln_FADM_ft'), format='PNG')
        if show_img:
            plt.show()
        plt.clf()
        plt.close()
        return

    # >>> draw spectra over epoch and K
    num_epoch = len(df_analysis_collection)
    l_epoch = [i for i in range(num_epoch)]
    for K in d_sig_vs_eigvecs_by_K:
        l_eigs = df_analysis_collection.iloc[0]['df_analysis'].loc[K]['eigs']
        l_eigvals = [eig[0] for eig in l_eigs]
        num_eig = len(l_eigvals)
        np_sig_vs_eigvecs_over_epoch_at_K = np.abs(np.stack(d_sig_vs_eigvecs_by_K[K]))
        np_w_sig_vs_eigvecs_over_epoch_at_K = np.abs(np.stack(d_w_sig_vs_eigvecs_by_K[K]))
        np_ln_sig_vs_eigvecs_over_epoch_at_K = np.abs(np.stack(d_ln_sig_vs_eigvecs_by_K[K]))

        # >>> draw np_sig_vs_eigvecs_over_epoch_at_K
        img_name = 'Signal vs Eigenvectors @K=%s' % K
        fig, axes = plt.subplots(ncols=1, nrows=num_eig + len(l_metric_names), figsize=(10, 40))
        fig.suptitle(img_name, fontsize=20, fontweight='semibold')
        xtick_stride = math.ceil(num_epoch / 25)

        # >>> draw metrics
        for idx in range(len(l_metric_names)):
            metric_name = l_metric_names[idx]
            l_metric_vals = d_metrics_by_epoch[metric_name]
            axes[idx].grid(True)
            axes[idx].set_title(metric_name, fontsize=15)
            plot_color = 'tab:blue'
            linewidth = 2
            axes[idx].plot(l_epoch[1:], l_metric_vals, color=plot_color, linewidth=linewidth)
            axes[idx].set_xticks(np.arange(0, num_epoch + xtick_stride, step=xtick_stride))

        for idx in range(len(l_metric_names), num_eig + len(l_metric_names)):
            eig_idx = idx - len(l_metric_names)
            np_sig_vs_eigvecs_over_epoch_at_K_per_eigval = np_sig_vs_eigvecs_over_epoch_at_K[:, eig_idx]
            eigval = l_eigvals[eig_idx]
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eigval, decimals=3), fontsize=15)
            plot_color = 'tab:blue'
            linewidth = 2
            if np.abs(np.max(np_sig_vs_eigvecs_over_epoch_at_K_per_eigval)
                      - np.min(np_sig_vs_eigvecs_over_epoch_at_K_per_eigval)) < 0.1:
                plot_color = 'tab:grey'
                linewidth = 1
            axes[idx].plot(l_epoch, np_sig_vs_eigvecs_over_epoch_at_K_per_eigval, color=plot_color, linewidth=linewidth)
            axes[idx].set_xticks(np.arange(0, num_epoch + xtick_stride, step=xtick_stride))
            # axes[idx].set_yticks(np.arange(0.0, 1.1, step=0.1))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_ret:
            plt.savefig(save_path_fmt.format('sig_vs_eig@K=%s' % K), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw np_w_sig_vs_eigvecs_over_epoch_at_K
        img_name = 'Weighted Signal vs Eigenvectors @K=%s' % K
        fig, axes = plt.subplots(ncols=1, nrows=num_eig + len(l_metric_names), figsize=(10, 40))
        fig.suptitle(img_name, fontsize=20, fontweight='semibold')
        xtick_stride = math.ceil(num_epoch / 25)

        # >>> draw metrics
        for idx in range(len(l_metric_names)):
            metric_name = l_metric_names[idx]
            l_metric_vals = d_metrics_by_epoch[metric_name]
            axes[idx].grid(True)
            axes[idx].set_title(metric_name, fontsize=15)
            plot_color = 'tab:blue'
            linewidth = 2
            axes[idx].plot(l_epoch[1:], l_metric_vals, color=plot_color, linewidth=linewidth)
            axes[idx].set_xticks(np.arange(0, num_epoch + xtick_stride, step=xtick_stride))

        for idx in range(len(l_metric_names), num_eig + len(l_metric_names)):
            eig_idx = idx - len(l_metric_names)
            np_w_sig_vs_eigvecs_over_epoch_at_K_per_eigval = np_w_sig_vs_eigvecs_over_epoch_at_K[:, eig_idx]
            eigval = l_eigvals[eig_idx]
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eigval, decimals=3), fontsize=15)
            plot_color = 'tab:blue'
            linewidth = 2
            if np.abs(np.max(np_w_sig_vs_eigvecs_over_epoch_at_K_per_eigval)
                      - np.min(np_w_sig_vs_eigvecs_over_epoch_at_K_per_eigval)) < 0.1:
                plot_color = 'tab:grey'
                linewidth = 1
            axes[idx].plot(l_epoch, np_w_sig_vs_eigvecs_over_epoch_at_K_per_eigval, color=plot_color,
                           linewidth=linewidth)
            axes[idx].set_xticks(np.arange(0, num_epoch + xtick_stride, step=xtick_stride))
            # axes[idx].set_yticks(np.arange(0.0, 1.1, step=0.1))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_ret:
            plt.savefig(save_path_fmt.format('w_sig_vs_eig@K=%s' % K), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

        # >>> draw np_ln_sig_vs_eigvecs_over_epoch_at_K
        img_name = 'Line Graph Fourier Transformed Signal vs Eigenvectors @K=%s' % K
        fig, axes = plt.subplots(ncols=1, nrows=num_eig + len(l_metric_names), figsize=(10, 40))
        fig.suptitle(img_name, fontsize=20, fontweight='semibold')
        xtick_stride = math.ceil(num_epoch / 25)

        # >>> draw metrics
        for idx in range(len(l_metric_names)):
            metric_name = l_metric_names[idx]
            l_metric_vals = d_metrics_by_epoch[metric_name]
            axes[idx].grid(True)
            axes[idx].set_title(metric_name, fontsize=15)
            plot_color = 'tab:blue'
            linewidth = 2
            axes[idx].plot(l_epoch[1:], l_metric_vals, color=plot_color, linewidth=linewidth)
            axes[idx].set_xticks(np.arange(0, num_epoch + xtick_stride, step=xtick_stride))

        for idx in range(len(l_metric_names), num_eig + len(l_metric_names)):
            eig_idx = idx - len(l_metric_names)
            np_ln_sig_vs_eigvecs_over_epoch_at_K_per_eigval = np_ln_sig_vs_eigvecs_over_epoch_at_K[:, eig_idx]
            eigval = l_eigvals[eig_idx]
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eigval, decimals=3), fontsize=15)
            plot_color = 'tab:blue'
            linewidth = 2
            if np.abs(np.max(np_ln_sig_vs_eigvecs_over_epoch_at_K_per_eigval)
                      - np.min(np_ln_sig_vs_eigvecs_over_epoch_at_K_per_eigval)) < 0.1:
                plot_color = 'tab:grey'
                linewidth = 1
            axes[idx].plot(l_epoch, np_ln_sig_vs_eigvecs_over_epoch_at_K_per_eigval, color=plot_color,
                           linewidth=linewidth)
            axes[idx].set_xticks(np.arange(0, num_epoch + xtick_stride, step=xtick_stride))
            # axes[idx].set_yticks(np.arange(0.0, 1.1, step=0.1))
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.96)
        if save_ret:
            plt.savefig(save_path_fmt.format('ln_sig_vs_eig@K=%s' % K), format='PNG')
        if show_img:
            plt.show()
        plt.clf()

    # >>> draw sig_pwd_FADM_Embed_vs_FEADM_Embeds for initial and final
    img_name = 'Initial & Final Signal vs Eigenvectors'
    fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in d_sig_vs_eigvecs_by_K:
        np_sig_pwd_FADM_Embed_vs_FEADM_Embeds_init = np.abs(
            df_analysis_init.loc[K]['sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
        np_sig_pwd_FADM_Embed_vs_FEADM_Embeds_final = np.abs(
            df_analysis_final.loc[K]['sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
        l_eigs = df_analysis_collection.iloc[0]['df_analysis'].loc[K]['eigs']
        l_eigvals = [eig[0] for eig in l_eigs]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        init_linefmt = 'tab:blue'
        final_linefmt = 'tab:orange'
        init_marker_fmt = 'o'
        final_marker_fmt = 'o'
        axes[idx].stem([i for i in range(len(l_eigvals))], np_sig_pwd_FADM_Embed_vs_FEADM_Embeds_init,
                       linefmt=init_linefmt,
                       markerfmt=init_marker_fmt,
                       label='Initial')
        axes[idx].stem([i for i in range(len(l_eigvals))], np_sig_pwd_FADM_Embed_vs_FEADM_Embeds_final,
                       linefmt=final_linefmt,
                       markerfmt=final_marker_fmt,
                       label='Final')
        axes[idx].set_xticks([i for i in range(len(l_eigvals))])
        axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_ret:
        plt.savefig(save_path_fmt.format('sig_vs_eig@init#final'), format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # >>> draw w_sig_pwd_FADM_Embed_vs_FEADM_Embeds for initial and final
    img_name = 'Initial & Final Weighted Signal vs Eigenvectors'
    fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in d_sig_vs_eigvecs_by_K:
        np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds_init = np.abs(
            df_analysis_init.loc[K]['w_sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
        np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds_final = np.abs(
            df_analysis_final.loc[K]['w_sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
        l_eigs = df_analysis_collection.iloc[0]['df_analysis'].loc[K]['eigs']
        l_eigvals = [eig[0] for eig in l_eigs]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        init_linefmt = 'tab:blue'
        final_linefmt = 'tab:orange'
        init_marker_fmt = 'o'
        final_marker_fmt = 'o'
        axes[idx].stem([i for i in range(len(l_eigvals))], np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds_init,
                       linefmt=init_linefmt,
                       markerfmt=init_marker_fmt,
                       label='Initial')
        axes[idx].stem([i for i in range(len(l_eigvals))], np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds_final,
                       linefmt=final_linefmt,
                       markerfmt=final_marker_fmt,
                       label='Final')
        axes[idx].set_xticks([i for i in range(len(l_eigvals))])
        axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_ret:
        plt.savefig(save_path_fmt.format('w_sig_vs_eig@init#final'), format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # >>> draw ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds for initial and final
    img_name = 'Initial & Final Weighted Signal vs Eigenvectors'
    fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in d_sig_vs_eigvecs_by_K:
        np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds_init = np.abs(
            df_analysis_init.loc[K]['sig_pwd_ln_FEADM_ft_vs_ln_FEADMs_ft'])
        np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds_final = np.abs(
            df_analysis_final.loc[K]['sig_pwd_ln_FEADM_ft_vs_ln_FEADMs_ft'])
        l_eigs = df_analysis_collection.iloc[0]['df_analysis'].loc[K]['eigs']
        l_eigvals = [eig[0] for eig in l_eigs]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        init_linefmt = 'tab:blue'
        final_linefmt = 'tab:orange'
        init_marker_fmt = 'o'
        final_marker_fmt = 'o'
        axes[idx].stem([i for i in range(len(l_eigvals))], np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds_init,
                       linefmt=init_linefmt,
                       markerfmt=init_marker_fmt,
                       label='Initial')
        axes[idx].stem([i for i in range(len(l_eigvals))], np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds_final,
                       linefmt=final_linefmt,
                       markerfmt=final_marker_fmt,
                       label='Final')
        axes[idx].set_xticks([i for i in range(len(l_eigvals))])
        axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_ret:
        plt.savefig(save_path_fmt.format('ln_sig_vs_eig@init#final'), format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # >>> draw pairwise cosine distances for initial and final
    img_name = param_str + ' Initial & Final Cosine Distances'
    fig, axes = plt.subplots(ncols=2, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
    fig.suptitle(img_name, fontsize=10, fontweight='semibold')
    idx = 0
    for K in d_sig_vs_eigvecs_by_K:
        # df_EADMs = df_spectral_seq.loc[K]['df_EADMs']
        # EADM_mask = df_EADMs.iloc[0]['EADM_mask']
        # sp_EADM_mask = sparse.coo_matrix(([1] * len(EADM_mask), ([item[0] for item in EADM_mask], [item[1] for item in EADM_mask])), shape=sp_pw_dc_init.shape)
        # np_ADM_init = sp_pw_dc_init.multiply(sp_EADM_mask).toarray()
        # np_ADM_final = sp_pw_dc_final.multiply(sp_EADM_mask).toarray()
        np_ADM_init = sp_pw_dc_init.toarray()
        np_ADM_final = sp_pw_dc_final.toarray()
        row_idx = idx
        col_idx = 0
        # >>> draw init
        axes[row_idx][col_idx].grid(True)
        axes[row_idx][col_idx].set_title('Initial K = %s' % K, fontsize=10)
        axes[row_idx][col_idx].set_xticks([i for i in range(len(l_nodes))])
        axes[row_idx][col_idx].set_xticklabels(l_nodes)
        axes[row_idx][col_idx].set_yticks([i for i in range(len(l_nodes))])
        axes[row_idx][col_idx].set_yticklabels(l_nodes)
        pos = axes[row_idx][col_idx].imshow(np_ADM_init, cmap='viridis')
        divider = make_axes_locatable(axes[row_idx][col_idx])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(pos, ax=axes[row_idx][col_idx], cax=cax)
        # >>> draw final
        col_idx = 1
        axes[row_idx][col_idx].grid(True)
        axes[row_idx][col_idx].set_title('Final K = %s' % K, fontsize=10)
        axes[row_idx][col_idx].set_xticks([i for i in range(len(l_nodes))])
        axes[row_idx][col_idx].set_xticklabels(l_nodes)
        axes[row_idx][col_idx].set_yticks([i for i in range(len(l_nodes))])
        axes[row_idx][col_idx].set_yticklabels(l_nodes)
        pos = axes[row_idx][col_idx].imshow(np_ADM_final, cmap='viridis')
        divider = make_axes_locatable(axes[row_idx][col_idx])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        fig.colorbar(pos, ax=axes[row_idx][col_idx], cax=cax)
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_ret:
        plt.savefig(save_path_fmt.format('init#final#pwdc'), format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # >>> draw FADMs for initial and final pairwise cosine distances
    img_name = param_str + ' Initial & Final FADMs'
    fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
    fig.suptitle(img_name, fontsize=10, fontweight='semibold')
    idx = 0
    for K in d_sig_vs_eigvecs_by_K:
        df_EADMs = df_spectral_seq.loc[K]['df_EADMs']
        EADM_mask = df_EADMs.iloc[0]['EADM_mask']
        EADM_mask_row = [item[0] for item in EADM_mask]
        EADM_mask_col = [item[1] for item in EADM_mask]
        FADM_init = sp_pw_dc_init.toarray()[EADM_mask_row, EADM_mask_col]
        FADM_final = sp_pw_dc_final.toarray()[EADM_mask_row, EADM_mask_col]
        l_xlabels = [l_nodes[item[0]] + l_nodes[item[1]] for item in EADM_mask]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=10)
        init_linefmt = 'tab:blue'
        final_linefmt = 'tab:orange'
        init_marker_fmt = 'o'
        final_marker_fmt = 'o'
        axes[idx].stem([i for i in range(len(FADM_init))], FADM_init,
                       linefmt=init_linefmt,
                       markerfmt=init_marker_fmt,
                       label='Initial')
        axes[idx].stem([i for i in range(len(FADM_final))], FADM_final,
                       linefmt=final_linefmt,
                       markerfmt=final_marker_fmt,
                       label='Final')
        axes[idx].set_xticks([i for i in range(len(l_xlabels))])
        axes[idx].set_xticklabels(l_xlabels)
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_ret:
        plt.savefig(save_path_fmt.format('init#final#FADMs'), format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # >>> draw FADM-Embeds for initial and final pairwise cosine distances
    img_name = param_str + ' Initial & Final FADM Embeds'
    fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
    fig.suptitle(img_name, fontsize=10, fontweight='semibold')
    idx = 0
    for K in d_sig_vs_eigvecs_by_K:
        np_FADM_Embed_init = df_analysis_init.loc[K]['sig_pwd_FEADM_Embed']
        np_FADM_Embed_final = df_analysis_final.loc[K]['sig_pwd_FEADM_Embed']
        np_FEADM_Singulars = df_spectral_seq.loc[K]['FEADM_Singulars']
        num_eff_FEADM_Singulars = np.count_nonzero(np.round(np_FEADM_Singulars, decimals=12))
        np_eff_FADM_Embed_init = np.abs(
            np_FADM_Embed_init[:num_eff_FEADM_Singulars] / np_FEADM_Singulars[:num_eff_FEADM_Singulars])
        np_eff_FADM_Embed_final = np.abs(
            np_FADM_Embed_final[:num_eff_FEADM_Singulars] / np_FEADM_Singulars[:num_eff_FEADM_Singulars])
        l_xlabels = ['bc#' + str(i) for i in range(len(np_eff_FADM_Embed_final))]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=10)
        init_linefmt = 'tab:blue'
        final_linefmt = 'tab:orange'
        init_marker_fmt = 'o'
        final_marker_fmt = 'o'
        axes[idx].stem([i for i in range(len(np_eff_FADM_Embed_final))], np_eff_FADM_Embed_init,
                       linefmt=init_linefmt,
                       markerfmt=init_marker_fmt,
                       label='Initial')
        axes[idx].stem([i for i in range(len(np_eff_FADM_Embed_final))], np_eff_FADM_Embed_final,
                       linefmt=final_linefmt,
                       markerfmt=final_marker_fmt,
                       label='Final')
        axes[idx].set_xticks([i for i in range(len(l_xlabels))])
        axes[idx].set_xticklabels(l_xlabels)
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_ret:
        plt.savefig(save_path_fmt.format('init#final#FADM_Embeds'), format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # >>> draw Singular-Weighted FADM-Embeds for initial and final pairwise cosine distances
    img_name = param_str + ' Initial & Final SWFADM Embeds'
    fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
    fig.suptitle(img_name, fontsize=10, fontweight='semibold')
    idx = 0
    for K in d_sig_vs_eigvecs_by_K:
        np_FADM_Embed_init = df_analysis_init.loc[K]['sig_pwd_FEADM_Embed']
        np_FADM_Embed_final = df_analysis_final.loc[K]['sig_pwd_FEADM_Embed']
        np_FEADM_Singulars = df_spectral_seq.loc[K]['FEADM_Singulars']
        num_eff_FEADM_Singulars = np.count_nonzero(np.round(np_FEADM_Singulars, decimals=12))
        np_eff_FADM_Embed_init = np.abs(np_FADM_Embed_init[:num_eff_FEADM_Singulars])
        np_eff_FADM_Embed_final = np.abs(np_FADM_Embed_final[:num_eff_FEADM_Singulars])
        l_xlabels = ['bc#' + str(i) for i in range(len(np_eff_FADM_Embed_final))]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=10)
        init_linefmt = 'tab:blue'
        final_linefmt = 'tab:orange'
        init_marker_fmt = 'o'
        final_marker_fmt = 'o'
        axes[idx].stem([i for i in range(len(np_eff_FADM_Embed_final))], np_eff_FADM_Embed_init,
                       linefmt=init_linefmt,
                       markerfmt=init_marker_fmt,
                       label='Initial')
        axes[idx].stem([i for i in range(len(np_eff_FADM_Embed_final))], np_eff_FADM_Embed_final,
                       linefmt=final_linefmt,
                       markerfmt=final_marker_fmt,
                       label='Final')
        axes[idx].set_xticks([i for i in range(len(l_xlabels))])
        axes[idx].set_xticklabels(l_xlabels)
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_ret:
        plt.savefig(save_path_fmt.format('init#final#FADM_SWEmbeds'), format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # >>> draw Line Graph Fourier Transformed FADMs for initial and final pairwise cosine distances
    img_name = param_str + ' Initial & Final Line Graph Fourier'
    fig, axes = plt.subplots(ncols=1, nrows=len(d_sig_vs_eigvecs_by_K), figsize=(10, 20))
    fig.suptitle(img_name, fontsize=10, fontweight='semibold')
    idx = 0
    for K in d_sig_vs_eigvecs_by_K:
        l_ln_eigs = df_analysis_init.loc[K]['ln_eigs']
        np_FADM_ln_ft_init = np.abs(df_analysis_init.loc[K]['sig_pwd_ln_FEADM_ft'])
        np_FADM_ln_ft_final = np.abs(df_analysis_final.loc[K]['sig_pwd_ln_FEADM_ft'])
        l_xlabels = np.round([eig[0] for eig in l_ln_eigs], decimals=3)
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=10)
        init_linefmt = 'tab:blue'
        final_linefmt = 'tab:orange'
        init_marker_fmt = 'o'
        final_marker_fmt = 'o'
        axes[idx].stem([i for i in range(len(np_FADM_ln_ft_init))], np_FADM_ln_ft_init,
                       linefmt=init_linefmt,
                       markerfmt=init_marker_fmt,
                       label='Initial')
        axes[idx].stem([i for i in range(len(np_FADM_ln_ft_final))], np_FADM_ln_ft_final,
                       linefmt=final_linefmt,
                       markerfmt=final_marker_fmt,
                       label='Final')
        axes[idx].set_xticks([i for i in range(len(l_xlabels))])
        axes[idx].set_xticklabels(l_xlabels)
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_ret:
        plt.savefig(save_path_fmt.format('init#final#ln_FADM_ft'), format='PNG')
    if show_img:
        plt.show()
    plt.clf()
    plt.close()


def draw_spec_seq_good_vs_bad(good_folder, bad_folder, save_img=True, show_img=False, save_folder=None):
    d_eigvals_at_K = dict()
    d_ln_eigvals_at_K = dict()
    d_sig_vs_eig_at_K_good = dict()
    d_w_sig_vs_eig_at_K_good = dict()
    d_ln_sig_vs_eig_at_K_good = dict()
    d_metrics_good = dict()
    d_sig_SWFADM_good = dict()
    d_sig_ln_ft_good = dict()
    for (dirpath, dirname, filenames) in walk(good_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:15] != 'ne_spec_seq_run':
                continue
            df_analysis_collection = pd.read_pickle(dirpath + '/' + filename)
            max_epoch = np.max(df_analysis_collection.index)
            df_analysis_final = df_analysis_collection.iloc[max_epoch]['df_analysis']
            l_metric_names = df_analysis_collection.iloc[max_epoch]['metric_names']
            l_metric_vals = df_analysis_collection.iloc[max_epoch]['metric_vals']
            for metric_idx, metric_name in enumerate(l_metric_names):
                if metric_name not in d_metrics_good:
                    d_metrics_good[metric_name] = [l_metric_vals[metric_idx]]
                else:
                    d_metrics_good[metric_name].append(l_metric_vals[metric_idx])
            for K, ana_rec in df_analysis_final.iterrows():
                l_eigs = ana_rec['eigs']
                l_eigvals = [item[0] for item in l_eigs]
                if K not in d_eigvals_at_K:
                    d_eigvals_at_K[K] = l_eigvals
                l_ln_eigs = ana_rec['ln_eigs']
                l_ln_eigvals = [item[0] for item in l_ln_eigs]
                if K not in d_ln_eigvals_at_K:
                    d_ln_eigvals_at_K[K] = l_ln_eigvals
                np_sig_pwd_FADM_Embed_vs_FEADM_Embeds = np.abs(ana_rec['sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
                np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds = np.abs(ana_rec['w_sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
                np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds = np.abs(ana_rec['sig_pwd_ln_FEADM_ft_vs_ln_FEADMs_ft'])
                np_sig_SWFADM = ana_rec['sig_pwd_FEADM_Embed']
                np_sig_ln_ft = ana_rec['sig_pwd_ln_FEADM_ft']
                if K not in d_sig_vs_eig_at_K_good:
                    d_sig_vs_eig_at_K_good[K] = [np_sig_pwd_FADM_Embed_vs_FEADM_Embeds]
                else:
                    d_sig_vs_eig_at_K_good[K].append(np_sig_pwd_FADM_Embed_vs_FEADM_Embeds)
                if K not in d_w_sig_vs_eig_at_K_good:
                    d_w_sig_vs_eig_at_K_good[K] = [np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds]
                else:
                    d_w_sig_vs_eig_at_K_good[K].append(np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds)
                if K not in d_ln_sig_vs_eig_at_K_good:
                    d_ln_sig_vs_eig_at_K_good[K] = [np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds]
                else:
                    d_ln_sig_vs_eig_at_K_good[K].append(np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds)
                if K not in d_sig_SWFADM_good:
                    d_sig_SWFADM_good[K] = [np_sig_SWFADM]
                else:
                    d_sig_SWFADM_good[K].append(np_sig_SWFADM)
                if K not in d_sig_ln_ft_good:
                    d_sig_ln_ft_good[K] = [np_sig_ln_ft]
                else:
                    d_sig_ln_ft_good[K].append(np_sig_ln_ft)
    for K in d_sig_vs_eig_at_K_good:
        l_sig_vs_eig = d_sig_vs_eig_at_K_good[K]
        sig_vs_eig_mean = np.mean(l_sig_vs_eig, axis=0)
        sig_vs_eig_std = np.std(l_sig_vs_eig, axis=0)
        d_sig_vs_eig_at_K_good[K] = (sig_vs_eig_mean, sig_vs_eig_std)
    for K in d_w_sig_vs_eig_at_K_good:
        l_sig_vs_eig = d_w_sig_vs_eig_at_K_good[K]
        sig_vs_eig_mean = np.mean(l_sig_vs_eig, axis=0)
        sig_vs_eig_std = np.std(l_sig_vs_eig, axis=0)
        d_w_sig_vs_eig_at_K_good[K] = (sig_vs_eig_mean, sig_vs_eig_std)
    for K in d_ln_sig_vs_eig_at_K_good:
        l_sig_vs_eig = d_ln_sig_vs_eig_at_K_good[K]
        sig_vs_eig_mean = np.mean(l_sig_vs_eig, axis=0)
        sig_vs_eig_std = np.std(l_sig_vs_eig, axis=0)
        d_ln_sig_vs_eig_at_K_good[K] = (sig_vs_eig_mean, sig_vs_eig_std)
    for K in d_sig_SWFADM_good:
        l_sig_SWFADM = d_sig_SWFADM_good[K]
        sig_SWFADM_mean = np.mean(l_sig_SWFADM, axis=0)
        sig_SWFADM_std = np.std(l_sig_SWFADM, axis=0)
        d_sig_SWFADM_good[K] = (sig_SWFADM_mean, sig_SWFADM_std)
    for K in d_sig_ln_ft_good:
        l_sig_ln_ft = d_sig_ln_ft_good[K]
        sig_ln_ft_mean = np.mean(l_sig_ln_ft, axis=0)
        sig_ln_ft_std = np.std(l_sig_ln_ft, axis=0)
        d_sig_ln_ft_good[K] = (sig_ln_ft_mean, sig_ln_ft_std)
    for metric_name in d_metrics_good:
        l_metric_vals = d_metrics_good[metric_name]
        metric_mean = np.mean(l_metric_vals)
        metric_std = np.mean(l_metric_vals)
        d_metrics_good[metric_name] = (metric_mean, metric_std)

    d_sig_vs_eig_at_K_bad = dict()
    d_w_sig_vs_eig_at_K_bad = dict()
    d_ln_sig_vs_eig_at_K_bad = dict()
    d_metrics_bad = dict()
    d_sig_SWFADM_bad = dict()
    d_sig_ln_ft_bad = dict()
    for (dirpath, dirname, filenames) in walk(bad_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:15] != 'ne_spec_seq_run':
                continue
            df_analysis_collection = pd.read_pickle(dirpath + '/' + filename)
            max_epoch = np.max(df_analysis_collection.index)
            df_analysis_final = df_analysis_collection.iloc[max_epoch]['df_analysis']
            l_metric_names = df_analysis_collection.iloc[max_epoch]['metric_names']
            l_metric_vals = df_analysis_collection.iloc[max_epoch]['metric_vals']
            for metric_idx, metric_name in enumerate(l_metric_names):
                if metric_name not in d_metrics_bad:
                    d_metrics_bad[metric_name] = [l_metric_vals[metric_idx]]
                else:
                    d_metrics_bad[metric_name].append(l_metric_vals[metric_idx])
            for K, ana_rec in df_analysis_final.iterrows():
                np_sig_pwd_FADM_Embed_vs_FEADM_Embeds = np.abs(ana_rec['sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
                np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds = np.abs(ana_rec['w_sig_pwd_FADM_Embed_vs_FEADM_Embeds'])
                np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds = np.abs(ana_rec['sig_pwd_ln_FEADM_ft_vs_ln_FEADMs_ft'])
                np_sig_SWFADM = ana_rec['sig_pwd_FEADM_Embed']
                np_sig_ln_ft = ana_rec['sig_pwd_ln_FEADM_ft']
                if K not in d_sig_vs_eig_at_K_bad:
                    d_sig_vs_eig_at_K_bad[K] = [np_sig_pwd_FADM_Embed_vs_FEADM_Embeds]
                else:
                    d_sig_vs_eig_at_K_bad[K].append(np_sig_pwd_FADM_Embed_vs_FEADM_Embeds)
                if K not in d_w_sig_vs_eig_at_K_bad:
                    d_w_sig_vs_eig_at_K_bad[K] = [np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds]
                else:
                    d_w_sig_vs_eig_at_K_bad[K].append(np_w_sig_pwd_FADM_Embed_vs_FEADM_Embeds)
                if K not in d_ln_sig_vs_eig_at_K_bad:
                    d_ln_sig_vs_eig_at_K_bad[K] = [np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds]
                else:
                    d_ln_sig_vs_eig_at_K_bad[K].append(np_ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds)
                if K not in d_sig_SWFADM_bad:
                    d_sig_SWFADM_bad[K] = [np_sig_SWFADM]
                else:
                    d_sig_SWFADM_bad[K].append(np_sig_SWFADM)
                if K not in d_sig_ln_ft_bad:
                    d_sig_ln_ft_bad[K] = [np_sig_ln_ft]
                else:
                    d_sig_ln_ft_bad[K].append(np_sig_ln_ft)
    for K in d_sig_vs_eig_at_K_bad:
        l_sig_vs_eig = d_sig_vs_eig_at_K_bad[K]
        sig_vs_eig_mean = np.mean(l_sig_vs_eig, axis=0)
        sig_vs_eig_std = np.std(l_sig_vs_eig, axis=0)
        d_sig_vs_eig_at_K_bad[K] = (sig_vs_eig_mean, sig_vs_eig_std)
    for K in d_w_sig_vs_eig_at_K_bad:
        l_sig_vs_eig = d_w_sig_vs_eig_at_K_bad[K]
        sig_vs_eig_mean = np.mean(l_sig_vs_eig, axis=0)
        sig_vs_eig_std = np.std(l_sig_vs_eig, axis=0)
        d_w_sig_vs_eig_at_K_bad[K] = (sig_vs_eig_mean, sig_vs_eig_std)
    for K in d_ln_sig_vs_eig_at_K_bad:
        l_sig_vs_eig = d_ln_sig_vs_eig_at_K_bad[K]
        sig_vs_eig_mean = np.mean(l_sig_vs_eig, axis=0)
        sig_vs_eig_std = np.std(l_sig_vs_eig, axis=0)
        d_ln_sig_vs_eig_at_K_bad[K] = (sig_vs_eig_mean, sig_vs_eig_std)
    for K in d_sig_SWFADM_bad:
        l_sig_SWFADM = d_sig_SWFADM_bad[K]
        sig_SWFADM_mean = np.mean(l_sig_SWFADM, axis=0)
        sig_SWFADM_std = np.std(l_sig_SWFADM, axis=0)
        d_sig_SWFADM_bad[K] = (sig_SWFADM_mean, sig_SWFADM_std)
    for K in d_sig_ln_ft_bad:
        l_sig_ln_ft = d_sig_ln_ft_bad[K]
        sig_ln_ft_mean = np.mean(l_sig_ln_ft, axis=0)
        sig_ln_ft_std = np.std(l_sig_ln_ft, axis=0)
        d_sig_ln_ft_bad[K] = (sig_ln_ft_mean, sig_ln_ft_std)
    for metric_name in d_metrics_bad:
        l_metric_vals = d_metrics_bad[metric_name]
        metric_mean = np.mean(l_metric_vals)
        metric_std = np.mean(l_metric_vals)
        d_metrics_bad[metric_name] = (metric_mean, metric_std)

    # draw sig_pwd_FADM_Embed_vs_FEADM_Embeds
    fig, axes = plt.subplots(ncols=1, nrows=len(d_eigvals_at_K) + 1, figsize=(10, 30))
    fig.suptitle('Sig vs Eig', fontsize=15, fontweight='semibold')
    idx = 0
    good_plot_color = 'tab:blue'
    bad_plot_color = 'tab:orange'
    axes[idx].grid(True)
    axes[idx].set_title('Metrics', fontsize=15)
    l_metric_names = list(d_metrics_good.keys())
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_good[key][0] for key in l_metric_names],
                       [d_metrics_good[key][1] for key in l_metric_names],
                       marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_bad[key][0] for key in l_metric_names],
                       [d_metrics_bad[key][1] for key in l_metric_names],
                       marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
    axes[idx].set_xticks([i for i in range(len(l_metric_names))])
    axes[idx].set_xticklabels(l_metric_names)
    axes[idx].legend()
    idx += 1
    for K in d_eigvals_at_K:
        l_eigvals = d_eigvals_at_K[K]
        (good_eigvalmags_mean, good_eigvalmags_std) = d_sig_vs_eig_at_K_good[K]
        (bad_eigvalmags_mean, bad_eigvalmags_std) = d_sig_vs_eig_at_K_bad[K]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        axes[idx].errorbar([i for i in range(len(good_eigvalmags_mean))], good_eigvalmags_mean, good_eigvalmags_std,
                           marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
        axes[idx].errorbar([i for i in range(len(bad_eigvalmags_mean))], bad_eigvalmags_mean, bad_eigvalmags_std,
                           marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
        axes[idx].set_xticks([i for i in range(len(l_eigvals))])
        axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_img:
        plt.savefig(save_folder + 'sig_vs_eig.PNG', format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # draw w_sig_pwd_FADM_Embed_vs_FEADM_Embeds
    fig, axes = plt.subplots(ncols=1, nrows=len(d_eigvals_at_K) + 1, figsize=(10, 30))
    fig.suptitle('Weighted Sig vs Eig', fontsize=15, fontweight='semibold')
    idx = 0
    good_plot_color = 'tab:blue'
    bad_plot_color = 'tab:orange'
    axes[idx].grid(True)
    axes[idx].set_title('Metrics', fontsize=15)
    l_metric_names = list(d_metrics_good.keys())
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_good[key][0] for key in l_metric_names],
                       [d_metrics_good[key][1] for key in l_metric_names],
                       marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_bad[key][0] for key in l_metric_names],
                       [d_metrics_bad[key][1] for key in l_metric_names],
                       marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
    axes[idx].set_xticks([i for i in range(len(l_metric_names))])
    axes[idx].set_xticklabels(l_metric_names)
    axes[idx].legend()
    idx += 1
    for K in d_eigvals_at_K:
        l_eigvals = d_eigvals_at_K[K]
        (good_eigvalmags_mean, good_eigvalmags_std) = d_w_sig_vs_eig_at_K_good[K]
        (bad_eigvalmags_mean, bad_eigvalmags_std) = d_w_sig_vs_eig_at_K_bad[K]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        axes[idx].errorbar([i for i in range(len(good_eigvalmags_mean))], good_eigvalmags_mean, good_eigvalmags_std,
                           marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
        axes[idx].errorbar([i for i in range(len(bad_eigvalmags_mean))], bad_eigvalmags_mean, bad_eigvalmags_std,
                           marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
        axes[idx].set_xticks([i for i in range(len(l_eigvals))])
        axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_img:
        plt.savefig(save_folder + 'w_sig_vs_eig.PNG', format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # draw ln_sig_pwd_FADM_Embed_vs_FEADM_Embeds
    fig, axes = plt.subplots(ncols=1, nrows=len(d_eigvals_at_K) + 1, figsize=(10, 30))
    fig.suptitle('Line Graph Fourier Transformed Sig vs Eig', fontsize=15, fontweight='semibold')
    idx = 0
    good_plot_color = 'tab:blue'
    bad_plot_color = 'tab:orange'
    axes[idx].grid(True)
    axes[idx].set_title('Metrics', fontsize=15)
    l_metric_names = list(d_metrics_good.keys())
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_good[key][0] for key in l_metric_names],
                       [d_metrics_good[key][1] for key in l_metric_names],
                       marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_bad[key][0] for key in l_metric_names],
                       [d_metrics_bad[key][1] for key in l_metric_names],
                       marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
    axes[idx].set_xticks([i for i in range(len(l_metric_names))])
    axes[idx].set_xticklabels(l_metric_names)
    axes[idx].legend()
    idx += 1
    for K in d_eigvals_at_K:
        l_eigvals = d_eigvals_at_K[K]
        (good_eigvalmags_mean, good_eigvalmags_std) = d_ln_sig_vs_eig_at_K_good[K]
        (bad_eigvalmags_mean, bad_eigvalmags_std) = d_ln_sig_vs_eig_at_K_bad[K]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        axes[idx].errorbar([i for i in range(len(good_eigvalmags_mean))], good_eigvalmags_mean, good_eigvalmags_std,
                           marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
        axes[idx].errorbar([i for i in range(len(bad_eigvalmags_mean))], bad_eigvalmags_mean, bad_eigvalmags_std,
                           marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
        axes[idx].set_xticks([i for i in range(len(l_eigvals))])
        axes[idx].set_xticklabels(np.round(l_eigvals, decimals=3))
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_img:
        plt.savefig(save_folder + 'ln_sig_vs_eig.PNG', format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # draw sig_SWFADM
    fig, axes = plt.subplots(ncols=1, nrows=len(d_eigvals_at_K) + 1, figsize=(10, 30))
    fig.suptitle('Singular Weighted FADM Embeds', fontsize=15, fontweight='semibold')
    idx = 0
    good_plot_color = 'tab:blue'
    bad_plot_color = 'tab:orange'
    axes[idx].grid(True)
    axes[idx].set_title('Metrics', fontsize=15)
    l_metric_names = list(d_metrics_good.keys())
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_good[key][0] for key in l_metric_names],
                       [d_metrics_good[key][1] for key in l_metric_names],
                       marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_bad[key][0] for key in l_metric_names],
                       [d_metrics_bad[key][1] for key in l_metric_names],
                       marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
    axes[idx].set_xticks([i for i in range(len(l_metric_names))])
    axes[idx].set_xticklabels(l_metric_names)
    axes[idx].legend()
    idx += 1
    for K in d_eigvals_at_K:
        (good_sig_SWFADM_mean, good_sig_SWFADM_std) = d_sig_SWFADM_good[K]
        (bad_sig_SWFADM_mean, bad_sig_SWFADM_std) = d_sig_SWFADM_bad[K]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        axes[idx].errorbar([i for i in range(len(good_sig_SWFADM_mean))], good_sig_SWFADM_mean, good_sig_SWFADM_std,
                           marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
        axes[idx].errorbar([i for i in range(len(bad_sig_SWFADM_mean))], bad_sig_SWFADM_mean, bad_sig_SWFADM_std,
                           marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
        axes[idx].set_xticks([i for i in range(len(good_sig_SWFADM_mean))])
        axes[idx].set_xticklabels(['bc#' + str(i) for i in range(len(good_sig_SWFADM_mean))])
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_img:
        plt.savefig(save_folder + 'sig_SWFADM.PNG', format='PNG')
    if show_img:
        plt.show()
    plt.clf()

    # draw sig_ln_ft
    fig, axes = plt.subplots(ncols=1, nrows=len(d_eigvals_at_K) + 1, figsize=(10, 30))
    fig.suptitle('Line Graph Fourier Transformed FADMs', fontsize=15, fontweight='semibold')
    idx = 0
    good_plot_color = 'tab:blue'
    bad_plot_color = 'tab:orange'
    axes[idx].grid(True)
    axes[idx].set_title('Metrics', fontsize=15)
    l_metric_names = list(d_metrics_good.keys())
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_good[key][0] for key in l_metric_names],
                       [d_metrics_good[key][1] for key in l_metric_names],
                       marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
    axes[idx].errorbar([i for i in range(len(l_metric_names))],
                       [d_metrics_bad[key][0] for key in l_metric_names],
                       [d_metrics_bad[key][1] for key in l_metric_names],
                       marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
    axes[idx].set_xticks([i for i in range(len(l_metric_names))])
    axes[idx].set_xticklabels(l_metric_names)
    axes[idx].legend()
    idx += 1
    for K in d_eigvals_at_K:
        l_ln_eigvals = d_ln_eigvals_at_K[K]
        (good_sig_ln_ft_mean, good_sig_ln_ft_std) = d_sig_ln_ft_good[K]
        (bad_sig_ln_ft_mean, bad_sig_ln_ft_std) = d_sig_ln_ft_bad[K]
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        axes[idx].errorbar([i for i in range(len(good_sig_ln_ft_mean))], good_sig_ln_ft_mean, good_sig_ln_ft_std,
                           marker='o', mfc=good_plot_color, capsize=2, capthick=1, label='good')
        axes[idx].errorbar([i for i in range(len(bad_sig_ln_ft_mean))], bad_sig_ln_ft_mean, bad_sig_ln_ft_std,
                           marker='o', mfc=bad_plot_color, capsize=2, capthick=1, label='bad')
        axes[idx].set_xticks([i for i in range(len(l_ln_eigvals))])
        axes[idx].set_xticklabels(np.round(l_ln_eigvals, decimals=3))
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.95)
    if save_img:
        plt.savefig(save_folder + 'sig_ln_ft.PNG', format='PNG')
    if show_img:
        plt.show()
    plt.clf()
    plt.close()


def draw_vector_signal_analysis_against_spectral_sequence(df_analysis, save_path):
    '''
    TODO
    Not finished yet
    '''
    logging.debug('[draw_vector_signal_analysis_against_spectral_sequence] starts.')
    timer_start = time.time()

    l_sig_dc_FEADM_Embed = []
    l_w_sig_dc_FADM_Embed_vs_FEADM_Embeds = []
    for K, ana_rec in df_analysis.iterrows():
        l_eigs = ana_rec['eigs']
        np_sig_dc_FEADM_Embed = ana_rec['sig_dc_FEADM_Embed']
        l_sig_dc_FEADM_Embed.append((K, np_sig_dc_FEADM_Embed))
        np_w_sig_dc_FADM_Embed_vs_FEADM_Embeds = ana_rec['w_sig_dc_FADM_Embed_vs_FEADM_Embeds']
        l_w_sig_dc_FADM_Embed_vs_FEADM_Embeds.append((K, np_w_sig_dc_FADM_Embed_vs_FEADM_Embeds))

    # >>> draw sig_dc_FEADM_Embed
    # feadme_img_name = graph_name + '#FEADM_Embeds@K%s.PNG' % str(K)
    # fig, axes = plt.subplots(ncols=1, nrows=len(np_FEADM_Embeds), figsize=(10, 40))
    # fig.suptitle('FEADM Embeds @ K=%s' % str(K), fontsize=20, fontweight='semibold')
    # for idx, embed_vec in enumerate(np_FEADM_Embeds):
    #     axes[idx].grid(True)
    #     axes[idx].set_title(r'Embed for FEADM @ $\lambda$ = %s' % np.round(l_eigs[idx][0], decimals=3), fontsize=10)
    #     markerline, stemlines, baselineaxes = axes[idx].stem(['bc#' + str(i) for i in range(len(np_FEADM_basis))], embed_vec)
    #     # axes[idx].set_yticks(np.arange(-1, 1, step=0.1))
    # plt.tight_layout(pad=1.0)
    # plt.subplots_adjust(top=0.96)
    # plt.savefig(K_folder + feadme_img_name, format='PNG')
    # # plt.show()
    # plt.clf()


def ln_vx(np_vec_sig, df_spectral_seq, df_ln_to_vx_eig_convert, np_pw_dist=None, save_ret=False):
    # logging.debug('[ln_vx] starts.')
    timer_start = time.time()

    if np_vec_sig is not None and np_pw_dist is None:
        np_vec_sig = preprocessing.normalize(np_vec_sig)
        np_sig_pwd = 1.0 - np.matmul(np_vec_sig, np_vec_sig.T)
    elif np_pw_dist is not None:
        np_sig_pwd = np_pw_dist
    np_sig_pwd = np.round(np_sig_pwd, decimals=12)

    l_analysis_rec = []
    for K, Srec in df_spectral_seq.iterrows():
        l_eigs = Srec['eigs']
        l_eig_vals = [eig[0] for eig in l_eigs]
        np_vx_U = np.stack([eig[1] for eig in l_eigs])
        l_ln_eigs = Srec['ln_eigs']
        np_ln_U = np.stack([eig[1] for eig in l_ln_eigs])
        df_EADMs = Srec['df_EADMs']
        EADM_mask = df_EADMs.iloc[0]['EADM_mask']
        EADM_mask_row = [item[0] for item in EADM_mask]
        EADM_mask_col = [item[1] for item in EADM_mask]
        np_sig_pwd_FADM = np_sig_pwd[EADM_mask_row, EADM_mask_col]

        np_sig_pwd_ln_ft = graph_fourier(np_ln_U, np_sig_pwd_FADM)

        th_ln_to_vx_model = LN_to_VX(num_ln_eigs=np_ln_U.shape[0], num_ln_edges=np_ln_U.shape[1],
                                     num_vx_eigs=len(l_eigs), num_vx_vertices=len(l_eigs[0][1]))
        th_ln_to_vx_model.load_state_dict(df_ln_to_vx_eig_convert.loc[K]['ln_to_vx_model'])
        np_sig_pwd_ln_ft = np.abs(np_sig_pwd_ln_ft)
        np_w_ln_U = np_ln_U * np_sig_pwd_ln_ft.reshape(-1, 1)
        th_rec_w_vx_U = th_ln_to_vx_model(th.from_numpy(np_w_ln_U).type(th.float32))
        np_sig_pwd_rec_vx_ft = np.einsum('ij..., ij...->i', th_rec_w_vx_U.detach().numpy(), np_vx_U)
        l_analysis_rec.append((K, np_sig_pwd_FADM, l_eig_vals, np_sig_pwd_rec_vx_ft))

    df_analysis = pd.DataFrame(l_analysis_rec, columns=['K', 'sig_pwd_FADM', 'eig_vals', 'sig_pwd_rec_vx_ft'])
    df_analysis = df_analysis.set_index('K')
    if save_ret:
        pd.to_pickle(df_analysis, save_path + 'ln_vx_only.pickle')
    # logging.debug('[ln_vx] all done with %s recs in %s secs.' % (len(df_analysis), time.time() - timer_start))
    return df_analysis


def stratified_graph_spectra_and_transformations(np_vec_sig, df_spectral_seq, save_ret=False, save_path=None,
                                                 np_pw_dist=None, norm_sig_FADM=True, use_ln_to_vx_eig_convert=False,
                                                 df_ln_to_vx_eig_convert=None):
    logging.debug('[stratified_graph_spectra_and_transformations] starts.')
    timer_start = time.time()

    if np_vec_sig is not None and np_pw_dist is None:
        np_vec_sig = preprocessing.normalize(np_vec_sig)
        np_sig_pwd = np.sqrt(np.round((1.0 - np.matmul(np_vec_sig, np_vec_sig.T)), decimals=12) / 2.0)
    elif np_pw_dist is not None:
        np_sig_pwd = np_pw_dist

    np_sig_pwd = np.round(np_sig_pwd, decimals=12)

    l_analysis_rec = []
    for K, Srec in df_spectral_seq.iterrows():
        # >>> SAS
        l_eigs = Srec['eigs']
        l_eig_vals = [eig[0] for eig in l_eigs]
        np_vx_U = np.stack([eig[1] for eig in l_eigs])
        np_FEADMs = Srec['FEADMs']

        df_EADMs = Srec['df_EADMs']
        EADM_mask = df_EADMs.iloc[0]['EADM_mask']
        EADM_mask_row = [item[0] for item in EADM_mask]
        EADM_mask_col = [item[1] for item in EADM_mask]
        np_sig_pwd_FADM = np_sig_pwd[EADM_mask_row, EADM_mask_col]
        if norm_sig_FADM:
            np_sig_pwd_FADM = preprocessing.normalize(np_sig_pwd_FADM.reshape(1, -1))[0]

        sp_mask = sparse.coo_matrix(([1] * len(EADM_mask), (EADM_mask_row, EADM_mask_col)), shape=np_sig_pwd.shape)
        np_mask = sp_mask.toarray()
        np_mask_full = np_mask + np_mask.T
        np_deg = np.sum(np_mask_full, axis=0)
        np_measure = np_deg / np.sum(np_deg)
        np_measure_by_edge = np.asarray([np_measure[i] for i in EADM_mask_row])

        # >>> adj-diff
        # np_sig_pwd_FADM = np_sig_pwd_FADM * np_measure_by_edge
        np_sig_vs_eig_sas = np.matmul(np_sig_pwd_FADM, np_FEADMs.T)
        np_sig_vs_eig_sas = np_sig_vs_eig_sas / np.linalg.norm(np_FEADMs)

        # >>> Line graph space
        l_ln_eigs = Srec['ln_eigs']
        l_ln_eig_vals = [eig[0] for eig in l_ln_eigs]
        np_ln_U = np.stack([eig[1] for eig in l_ln_eigs])
        np_sig_pwd_ln_ft = graph_fourier(np_ln_U, np_sig_pwd_FADM)

        # >>> Vertex aggregation of sig_pwd_FADM
        np_Ik = Srec['sp_ln_inc'].toarray()
        np_sig_pwd_FADM_vx_agg = np.matmul(np_sig_pwd_FADM, np_Ik.T)
        np_sig_pwd_FADM_vx_agg = np.divide(np_sig_pwd_FADM_vx_agg, np.sum(np_Ik.astype(np.bool).astype(np.int32), axis=1))
        np_sig_pwd_FADM_vx_agg = np.nan_to_num(np_sig_pwd_FADM_vx_agg)
        if not np.isfinite(np_sig_pwd_FADM_vx_agg).all():
            raise Exception('[stratified_graph_spectra_and_transformations] np_sig_pwd_FADM_vx_agg is not finite!')
        np_sig_pwd_agg_vx_ft = graph_fourier(np_vx_U, np_sig_pwd_FADM_vx_agg)

        # >>> approximately solve linear system
        np_Ik_oriented = Srec['sp_ln_inc_oriented'].toarray()
        sol = np.linalg.lstsq(np_Ik_oriented.T, np_sig_pwd_FADM, rcond=None)
        np_apprx_ls_sig = np.round(sol[0], decimals=12)
        rmsd_apprx_ls_sig = np.sqrt(np.mean(np.square(np_Ik_oriented.T @ np_apprx_ls_sig - np_sig_pwd_FADM)))
        if not np.isfinite(np_apprx_ls_sig).all():
            raise Exception('[stratified_graph_spectra_and_transformations] np_apprx_ls_sig is not finite!')
        np_apprx_ls_ft = graph_fourier(np_vx_U, np_apprx_ls_sig)

        # >>> Taylor expansion at each node
        # np_sig_pwd_FADM_div_grad = np_sig_pwd_FADM_vx_agg
        # np_sig_pwd_FADM_taylor = np_sig_pwd_FADM_div_grad.reshape(-1, 1) - np_sig_pwd
        # np_sig_pwd_FADM_taylor = np_sig_pwd_FADM_taylor * np_mask
        # np_sig_pwd_FADM_taylor = np_sig_pwd_FADM_taylor + np.diag(np_sig_pwd_FADM_div_grad)
        # >>> a node does not approximate itself
        # np.fill_diagonal(np_sig_pwd_FADM_taylor, 0)
        # np_sig_pwd_FADM_taylor = np.sum(np_sig_pwd_FADM_taylor, axis=0)
        # np_sig_pwd_agg_vx_ft = graph_fourier(np_vx_U, np_sig_pwd_FADM_taylor)


        # >>> Convert line graph space back to vertex graph space
        if use_ln_to_vx_eig_convert:
            th_ln_to_vx_model = LN_to_VX(num_ln_eigs=np_ln_U.shape[0], num_ln_edges=np_ln_U.shape[1],
                                         num_vx_eigs=len(l_eigs), num_vx_vertices=len(l_eigs[0][1]))
            th_ln_to_vx_model.load_state_dict(df_ln_to_vx_eig_convert.loc[K]['ln_to_vx_model'])
            # th_sig_pwd_rec_vx_ft = th_ln_to_vx_model(th.from_numpy(np_sig_pwd_FADM).type(th.float32))
            # np_sig_pwd_rec_vx_ft = th_sig_pwd_rec_vx_ft.detach().numpy()
            np_sig_pwd_ln_ft = np.abs(np_sig_pwd_ln_ft)
            np_w_ln_U = np_ln_U * np_sig_pwd_ln_ft.reshape(-1, 1)
            th_rec_w_vx_U = th_ln_to_vx_model(th.from_numpy(np_w_ln_U).type(th.float32))
            np_sig_pwd_rec_vx_ft = np.einsum('ij..., ij...->i', th_rec_w_vx_U.detach().numpy(), np_vx_U)
        else:
            np_sig_pwd_rec_vx_ft = None

        # >>> FEADM based space
        np_FEADM_basis = Srec['FEADM_Basis']
        np_FEADM_Singulars = Srec['FEADM_Singulars']
        np_FEADM_Singulars = np.round(np_FEADM_Singulars, decimals=12)
        num_eff_singulars = np.count_nonzero(np_FEADM_Singulars)
        np_eff_FEADM_Singulars = np_FEADM_Singulars[:num_eff_singulars]
        np_eff_sig_pwd_FEADM_Embed = np.matmul(np_sig_pwd_FADM, np_FEADM_basis[:num_eff_singulars, :].T)

        l_analysis_rec.append((K, np_sig_pwd_FADM,
                               l_eig_vals, np_sig_vs_eig_sas,
                               l_ln_eig_vals, np_sig_pwd_ln_ft,
                               np_eff_FEADM_Singulars, np_eff_sig_pwd_FEADM_Embed,
                               np_sig_pwd_agg_vx_ft, np_sig_pwd_rec_vx_ft,
                               np_apprx_ls_ft, rmsd_apprx_ls_sig))
        logging.debug('[stratified_graph_spectra_and_transformations] all done with K=%s in %s secs.'
                      % (K, time.time() - timer_start))

    df_analysis = pd.DataFrame(l_analysis_rec, columns=['K', 'sig_pwd_FADM',
                                                        'eig_vals', 'sig_vs_eig_sas',
                                                        'ln_eig_vals', 'sig_pwd_ln_ft',
                                                        'eff_FEADM_Singulars', 'eff_sig_pwd_FEADM_Embed',
                                                        'sig_pwd_agg_vx_ft', 'sig_pwd_rec_vx_ft',
                                                        'apprx_ls_ft', 'rmsd_apprx_ls_sig'])
    df_analysis = df_analysis.set_index('K')
    if save_ret:
        pd.to_pickle(df_analysis, save_path + 'sas_and_trans.pickle')
    logging.debug('[stratified_graph_spectra_and_transformations] all done with %s recs in %s secs.'
                  % (len(df_analysis), time.time() - timer_start))
    return df_analysis


def draw_sas_and_trans_rets(df_sas_ana, save_path=None, save_img=True, show_img=False):
    logging.debug('[draw_sas_and_trans_rets] starts.')
    plt.set_loglevel('error')
    l_K = list(df_sas_ana.index)
    img_name = 'sig_vs_eig_sas'
    vmax = np.max([np.max(np.abs(item)) for item in df_sas_ana['sig_vs_eig_sas'].to_list()])
    vmin = np.min([np.min(np.abs(item)) for item in df_sas_ana['sig_vs_eig_sas'].to_list()])
    fig_width = len(df_sas_ana) * 3
    fig, axes = plt.subplots(ncols=1, nrows=len(df_sas_ana), figsize=(10, fig_width))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in l_K:
        l_eig_vals = df_sas_ana.loc[K]['eig_vals']
        np_sig_vs_eig_sas_1 = df_sas_ana.loc[K]['sig_vs_eig_sas']
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        axes[idx].stem(np_sig_vs_eig_sas_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt)
        axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
        axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.2, 0.2)])
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()

    if df_sas_ana.loc[1]['sig_pwd_rec_vx_ft'] is not None:
        img_name = 'sig_pwd_rec_vx_ft'
        vmax = np.max([np.max(np.abs(item)) for item in df_sas_ana['sig_pwd_rec_vx_ft'].to_list()])
        vmin = np.min([np.min(np.abs(item)) for item in df_sas_ana['sig_pwd_rec_vx_ft'].to_list()])
        fig_width = len(df_sas_ana) * 3
        fig, axes = plt.subplots(ncols=1, nrows=len(df_sas_ana), figsize=(10, fig_width))
        fig.suptitle(img_name, fontsize=15, fontweight='semibold')
        idx = 0
        for K in l_K:
            l_eig_vals = df_sas_ana.loc[K]['eig_vals']
            np_sig_pwd_rec_vx_ft = df_sas_ana.loc[K]['sig_pwd_rec_vx_ft']
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=15)
            sig_1_linefmt = 'tab:blue'
            sig_1_marker_fmt = 'o'
            axes[idx].stem(np_sig_pwd_rec_vx_ft, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt)
            axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
            axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
            axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.2, 0.2)])
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.94)
        if save_img:
            plt.savefig(save_path + img_name + '.PNG', format='PNG')
        if show_img:
            plt.show()

    img_name = 'sig_pwd_ln_ft'
    vmax = np.max([np.max(np.abs(item)) for item in df_sas_ana['sig_pwd_ln_ft'].to_list()])
    vmin = np.min([np.min(np.abs(item)) for item in df_sas_ana['sig_pwd_ln_ft'].to_list()])
    fig_width = len(df_sas_ana) * 3
    fig, axes = plt.subplots(ncols=1, nrows=len(df_sas_ana), figsize=(10, fig_width))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in l_K:
        l_ln_eig_vals = df_sas_ana.loc[K]['ln_eig_vals']
        np_sig_pwd_ln_ft_1 = np.abs(df_sas_ana.loc[K]['sig_pwd_ln_ft'])
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        axes[idx].stem(np_sig_pwd_ln_ft_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt)
        axes[idx].set_xticks([i for i in range(len(l_ln_eig_vals))])
        axes[idx].set_xticklabels(np.round(l_ln_eig_vals, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.2, 0.2)])
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()

    img_name = 'sig_pwd_agg_vx_ft'
    vmax = np.max([np.max(np.abs(item)) for item in df_sas_ana['sig_pwd_agg_vx_ft'].to_list()])
    vmin = np.min([np.min(np.abs(item)) for item in df_sas_ana['sig_pwd_agg_vx_ft'].to_list()])
    fig_width = len(df_sas_ana) * 3
    fig, axes = plt.subplots(ncols=1, nrows=len(df_sas_ana), figsize=(10, fig_width))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in l_K:
        l_eig_vals = df_sas_ana.loc[K]['eig_vals']
        np_sig_pwd_agg_vx_ft_1 = np.abs(df_sas_ana.loc[K]['sig_pwd_agg_vx_ft'])
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        axes[idx].stem(np_sig_pwd_agg_vx_ft_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt)
        axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
        axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.2, 0.2)])
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()

    img_name = 'eff_sig_pwd_FEADM_embed'
    fig_width = len(df_sas_ana) * 3
    vmax = np.max([np.max(np.abs(item)) for item in df_sas_ana['eff_sig_pwd_FEADM_Embed'].to_list()])
    vmin = np.min([np.min(np.abs(item)) for item in df_sas_ana['eff_sig_pwd_FEADM_Embed'].to_list()])
    fig, axes = plt.subplots(ncols=1, nrows=len(df_sas_ana), figsize=(10, fig_width))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in l_K:
        l_FEADM_singulars = df_sas_ana.loc[K]['eff_FEADM_Singulars']
        num_eff_FEADM_singulars = np.count_nonzero(l_FEADM_singulars)
        l_FEADM_singulars = l_FEADM_singulars[:num_eff_FEADM_singulars]
        np_sig_pwd_FEADM_embed_1 = np.abs(df_sas_ana.loc[K]['eff_sig_pwd_FEADM_Embed'])
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        axes[idx].stem(np_sig_pwd_FEADM_embed_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt)
        axes[idx].set_xticks([i for i in range(len(l_FEADM_singulars))])
        axes[idx].set_xticklabels(np.round(l_FEADM_singulars, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.2, 0.2)])
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()
    plt.clf()
    plt.close()
    logging.debug('[draw_sas_and_trans_rets] All done.')


def compare_two_sigs_with_sas_and_trans(df_spectral_seq,
                                        np_vec_sig_1, np_vec_sig_2,
                                        sig_1_label, sig_2_label,
                                        np_pw_dist_1=None, np_pw_dist_2=None,
                                        use_ln_to_vx_eig_convert=False,
                                        df_ln_to_vx_eig_convert=None,
                                        norm_sig_FADM=True,
                                        save_ret=True, save_path=None,
                                        save_img=True, show_img=False):
    logging.debug('[compare_two_sigs_with_sas_and_trans] starts.')
    df_analysis_1 = stratified_graph_spectra_and_transformations(np_vec_sig_1, df_spectral_seq,
                                                                 save_ret=save_ret, save_path=save_path,
                                                                 np_pw_dist=np_pw_dist_1, norm_sig_FADM=norm_sig_FADM)
    df_analysis_2 = stratified_graph_spectra_and_transformations(np_vec_sig_2, df_spectral_seq,
                                                                 save_ret=save_ret, save_path=save_path,
                                                                 np_pw_dist=np_pw_dist_2, norm_sig_FADM=norm_sig_FADM)

    l_K = list(df_analysis_1.index)
    img_name = 'sig_vs_eig_sas'
    ana_1_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_1['sig_vs_eig_sas'].to_list()])
    ana_1_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_1['sig_vs_eig_sas'].to_list()])
    ana_2_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_2['sig_vs_eig_sas'].to_list()])
    ana_2_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_2['sig_vs_eig_sas'].to_list()])
    vmax = np.max([ana_1_vmax, ana_2_vmax])
    vmin = np.min([ana_1_vmin, ana_2_vmin])
    fig_width = len(df_analysis_1) * 3
    fig, axes = plt.subplots(ncols=1, nrows=len(df_analysis_1), figsize=(10, fig_width))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in l_K:
        l_eig_vals = df_analysis_1.loc[K]['eig_vals']
        np_sig_vs_eig_sas_1 = df_analysis_1.loc[K]['sig_vs_eig_sas']
        np_sig_vs_eig_sas_2 = df_analysis_2.loc[K]['sig_vs_eig_sas']

        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        sig_2_linefmt = 'tab:orange'
        sig_2_marker_fmt = 'x'
        axes[idx].stem(np_sig_vs_eig_sas_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt, label=sig_1_label)
        axes[idx].stem(np_sig_vs_eig_sas_2, linefmt=sig_2_linefmt, markerfmt=sig_2_marker_fmt, label=sig_2_label)
        axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
        axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.1, 0.1)])
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()

    if df_analysis_1.loc[1]['sig_pwd_rec_vx_ft'] is not None and df_analysis_2.loc[1]['sig_pwd_rec_vx_ft'] is not None:
        img_name = 'sig_pwd_rec_vx_ft'
        ana_1_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_1['sig_pwd_rec_vx_ft'].to_list()])
        ana_1_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_1['sig_pwd_rec_vx_ft'].to_list()])
        ana_2_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_2['sig_pwd_rec_vx_ft'].to_list()])
        ana_2_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_2['sig_pwd_rec_vx_ft'].to_list()])
        vmax = np.max([ana_1_vmax, ana_2_vmax])
        vmin = np.min([ana_1_vmin, ana_2_vmin])
        fig_width = len(df_analysis_1) * 3
        fig, axes = plt.subplots(ncols=1, nrows=len(df_analysis_1), figsize=(10, fig_width))
        fig.suptitle(img_name, fontsize=15, fontweight='semibold')
        idx = 0
        for K in l_K:
            l_eig_vals = df_analysis_1.loc[K]['eig_vals']
            np_sig_pwd_rec_vx_ft_1 = df_analysis_1.loc[K]['sig_pwd_rec_vx_ft']
            np_sig_pwd_rec_vx_ft_2 = df_analysis_2.loc[K]['sig_pwd_rec_vx_ft']
            axes[idx].grid(True)
            axes[idx].set_title('K = %s' % K, fontsize=15)
            sig_1_linefmt = 'tab:blue'
            sig_1_marker_fmt = 'o'
            sig_2_linefmt = 'tab:orange'
            sig_2_marker_fmt = 'x'
            axes[idx].stem(np_sig_pwd_rec_vx_ft_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt, label=sig_1_label)
            axes[idx].stem(np_sig_pwd_rec_vx_ft_2, linefmt=sig_2_linefmt, markerfmt=sig_2_marker_fmt, label=sig_2_label)
            axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
            axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
            axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.2, 0.2)])
            axes[idx].legend()
            idx += 1
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.94)
        if save_img:
            plt.savefig(save_path + img_name + '.PNG', format='PNG')
        if show_img:
            plt.show()

    img_name = 'sig_pwd_ln_ft'
    ana_1_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_1['sig_pwd_ln_ft'].to_list()])
    ana_1_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_1['sig_pwd_ln_ft'].to_list()])
    ana_2_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_2['sig_pwd_ln_ft'].to_list()])
    ana_2_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_2['sig_pwd_ln_ft'].to_list()])
    vmax = np.max([ana_1_vmax, ana_2_vmax])
    vmin = np.min([ana_1_vmin, ana_2_vmin])
    fig_width = len(df_analysis_1) * 3
    fig, axes = plt.subplots(ncols=1, nrows=len(df_analysis_1), figsize=(10, fig_width))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in l_K:
        l_ln_eig_vals = df_analysis_1.loc[K]['ln_eig_vals']
        np_sig_pwd_ln_ft_1 = np.abs(df_analysis_1.loc[K]['sig_pwd_ln_ft'])
        np_sig_pwd_ln_ft_2 = np.abs(df_analysis_2.loc[K]['sig_pwd_ln_ft'])

        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        sig_2_linefmt = 'tab:orange'
        sig_2_marker_fmt = 'x'
        axes[idx].stem(np_sig_pwd_ln_ft_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt, label=sig_1_label)
        axes[idx].stem(np_sig_pwd_ln_ft_2, linefmt=sig_2_linefmt, markerfmt=sig_2_marker_fmt, label=sig_2_label)
        axes[idx].set_xticks([i for i in range(len(l_ln_eig_vals))])
        axes[idx].set_xticklabels(np.round(l_ln_eig_vals, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.1, 0.1)])
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()

    img_name = 'np_sig_pwd_agg_vx_ft'
    ana_1_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_1['sig_pwd_agg_vx_ft'].to_list()])
    ana_1_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_1['sig_pwd_agg_vx_ft'].to_list()])
    ana_2_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_2['sig_pwd_agg_vx_ft'].to_list()])
    ana_2_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_2['sig_pwd_agg_vx_ft'].to_list()])
    vmax = np.max([ana_1_vmax, ana_2_vmax])
    vmin = np.min([ana_1_vmin, ana_2_vmin])
    fig_width = len(df_analysis_1) * 3
    fig, axes = plt.subplots(ncols=1, nrows=len(df_analysis_1), figsize=(10, fig_width))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in l_K:
        l_eig_vals = df_analysis_1.loc[K]['eig_vals']
        np_sig_pwd_agg_vx_ft_1 = np.abs(df_analysis_1.loc[K]['sig_pwd_agg_vx_ft'])
        np_sig_pwd_agg_vx_ft_2 = np.abs(df_analysis_2.loc[K]['sig_pwd_agg_vx_ft'])
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        sig_2_linefmt = 'tab:orange'
        sig_2_marker_fmt = 'x'
        axes[idx].stem(np_sig_pwd_agg_vx_ft_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt, label=sig_1_label)
        axes[idx].stem(np_sig_pwd_agg_vx_ft_2, linefmt=sig_2_linefmt, markerfmt=sig_2_marker_fmt, label=sig_2_label)
        axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
        axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.2, 0.2)])
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()

    img_name = 'eff_sig_pwd_FEADM_embed'
    fig_width = len(df_analysis_1) * 3
    ana_1_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_1['eff_sig_pwd_FEADM_Embed'].to_list()])
    ana_1_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_1['eff_sig_pwd_FEADM_Embed'].to_list()])
    ana_2_vmax = np.max([np.max(np.abs(item)) for item in df_analysis_2['eff_sig_pwd_FEADM_Embed'].to_list()])
    ana_2_vmin = np.min([np.min(np.abs(item)) for item in df_analysis_2['eff_sig_pwd_FEADM_Embed'].to_list()])
    vmax = np.max([ana_1_vmax, ana_2_vmax])
    vmin = np.min([ana_1_vmin, ana_2_vmin])
    fig, axes = plt.subplots(ncols=1, nrows=len(df_analysis_1), figsize=(10, fig_width))
    fig.suptitle(img_name, fontsize=15, fontweight='semibold')
    idx = 0
    for K in l_K:
        l_FEADM_singulars = df_analysis_1.loc[K]['eff_FEADM_Singulars']
        num_eff_FEADM_singulars = np.count_nonzero(l_FEADM_singulars)
        l_FEADM_singulars = l_FEADM_singulars[:num_eff_FEADM_singulars]
        np_sig_pwd_FEADM_embed_1 = np.abs(df_analysis_1.loc[K]['eff_sig_pwd_FEADM_Embed'])
        np_sig_pwd_FEADM_embed_2 = np.abs(df_analysis_2.loc[K]['eff_sig_pwd_FEADM_Embed'])
        axes[idx].grid(True)
        axes[idx].set_title('K = %s' % K, fontsize=15)
        sig_1_linefmt = 'tab:blue'
        sig_1_marker_fmt = 'o'
        sig_2_linefmt = 'tab:orange'
        sig_2_marker_fmt = 'x'
        axes[idx].stem(np_sig_pwd_FEADM_embed_1, linefmt=sig_1_linefmt, markerfmt=sig_1_marker_fmt, label=sig_1_label)
        axes[idx].stem(np_sig_pwd_FEADM_embed_2, linefmt=sig_2_linefmt, markerfmt=sig_2_marker_fmt, label=sig_2_label)
        axes[idx].set_xticks([i for i in range(len(l_FEADM_singulars))])
        axes[idx].set_xticklabels(np.round(l_FEADM_singulars, decimals=3))
        axes[idx].set_yticks([i for i in np.arange(vmin, vmax + 0.2, 0.2)])
        axes[idx].legend()
        idx += 1
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.94)
    if save_img:
        plt.savefig(save_path + img_name + '.PNG', format='PNG')
    if show_img:
        plt.show()
    plt.close()
    logging.debug('[compare_two_sigs_with_sas_and_trans] All done.')


def compare_classic_gsp_with_sas_and_trans(df_graphs):
    logging.debug('[compare_classic_gsp_with_sas_and_trans] starts.')


################################################################################
def node_embed_spectrum(nx_graph, use_norm, draw_eigs=False):
    sp_L = graph_laplacian(nx_graph, use_norm=use_norm)
    l_eigs = graph_eigs(sp_L)
    if draw_eigs:
        draw_graph_eigs(l_eigs, nx_graph)
    return l_eigs


def eigenvec_pw_ele_diff_matrix(l_eigs, save_path, nx_graph, draw_diff_mat=False, draw_name=None, show_img=False,
                                save_img=True, img_path=None):
    # TODO
    # looking for more efficient methods to compute U.T @ U
    # when U is only one eigenvector, it's no problem.
    # but when U is a set of eigenvectors, need a method to do element-wise tensor multiplication,
    # in which each element is an eigenvector.
    logging.debug('[eigenvec_pw_ele_diff_matrix] starts.')
    timer_start = time.time()

    sp_u_A = sparse.triu(nx.linalg.adjacency_matrix(nx_graph), k=1).astype(np.bool).astype(np.int32)
    l_rec = []
    num_eig = len(l_eigs)
    for idx, eig in enumerate(l_eigs):
        eigenval = eig[0]
        eigenvec = eig[1]

        # for un-normalized Laplacian, the first eigenvector is constant, and thus sp_u_pw_ele_diff will be all zeros.
        # to make up for this, treat it differently as follows.
        if idx == 0 and np.allclose(eigenval, 0.0):
            sp_u_pw_ele_diff = sparse.coo_matrix(([], ([], [])), shape=(num_eig, num_eig))
        else:
            # >>> [[u(x) - u(y)]^2]^0.5 = |u(x) - u(y)|
            # >>> known: NORM[u]=1.0 => |u(x)| in [0, 1] => |u(x) - u(y)| in [0, 2] => |u(x) - u(y)|/2.0 in [0, 1]
            sp_u_pw_ele_prod = sparse.coo_matrix(
                np.triu(np.matmul(eigenvec.reshape(-1, 1), eigenvec.reshape(-1, 1).T), k=1))
            sp_u_pw_ele_prod = sp_u_pw_ele_prod.multiply(sp_u_A)
            sp_u_pw_ele_sqr = sparse.coo_matrix(
                np.triu(np.asarray([np.power(eigenvec, 2)] * num_eig) + np.power(eigenvec, 2).reshape(-1, 1), k=1))
            sp_u_pw_ele_sqr = sp_u_pw_ele_sqr.multiply(sp_u_A)
            # >>> guarantee every element is non-negative
            sp_u_pw_ele_diff_sqr = np.round(sp_u_pw_ele_sqr - 2 * sp_u_pw_ele_prod, decimals=12)
            sp_u_pw_ele_diff = np.sqrt(sp_u_pw_ele_diff_sqr) / 2.0
            # sp_u_pw_ele_diff = sp_u_pw_ele_diff.expm1() + sp_u_pw_ele_diff.astype(np.bool).astype(np.int32)

        # >>> as there can be many zeros or very small values, we make them non-trivial
        # >>> EXP[|u(x) - u(y)|/2.0] in [1, e]
        sp_u_pw_ele_diff.data = np.exp(sp_u_pw_ele_diff.data)
        sp_u_pw_ele_diff = (sp_u_A - sp_u_pw_ele_diff.astype(np.bool).astype(np.int32)) + sp_u_pw_ele_diff
        # sp_u_pw_ele_diff = sp_u_pw_ele_diff / sparse.linalg.norm(sp_u_pw_ele_diff)
        sp_u_pw_ele_diff = sparse.coo_matrix(sp_u_pw_ele_diff)
        if not np.isfinite(sp_u_pw_ele_diff.toarray()).all():
            raise Exception('[eigenvec_pw_ele_diff_matrix] invalid sp_u_pw_ele_diff_sqr: %s'
                            % sp_u_pw_ele_diff_sqr.toarray())
        l_rec.append((eigenval, eigenvec, sp_u_pw_ele_diff))
    df_rec = pd.DataFrame(l_rec, columns=['eigval', 'eigvec', 'eigvec_sp_pw_ele_diff_mat'])
    pd.to_pickle(df_rec, save_path)

    if draw_diff_mat:
        fig, axes = plt.subplots(ncols=1, nrows=len(df_rec), figsize=(5, 50))
        fig.suptitle(draw_name, fontsize=10)
        for idx, (eigenval, eigenvec, sp_u_pw_ele_diff) in enumerate(l_rec):
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eigenval, decimals=3), fontsize=10)
            axes[idx].set_xticks([i for i in range(len(eigenvec))])
            axes[idx].set_xticklabels([chr(i) for i in range(65, 65 + 13)])
            axes[idx].set_yticks([i for i in range(len(eigenvec))])
            axes[idx].set_yticklabels([chr(i) for i in range(65, 65 + 13)])
            # axes[idx].matshow(sp_u_pw_ele_diff.toarray())
            pos = axes[idx].imshow(sp_u_pw_ele_diff.toarray(), vmin=0, vmax=1.5, cmap='viridis')
            divider = make_axes_locatable(axes[idx])
            cax = divider.append_axes("right", size="3%", pad=0.1)
            fig.colorbar(pos, ax=axes[idx], cax=cax)
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.95)
        if save_img:
            plt.savefig(img_path, format='PNG')
        if show_img:
            plt.show()
        plt.clf()

    logging.debug('[eigenvec_pw_ele_diff_matrix] all done in %s secs.' % str(time.time() - timer_start))


def basis_for_eigenvec_pw_ele_diff_mat(df_eigvec_ele_pw_diff, nx_graph):
    logging.debug('[basis_for_eigenvec_pw_ele_diff_mat] starts.')
    timer_start = time.time()

    l_diff_mat_vec = []
    for _, diff_rec in df_eigvec_ele_pw_diff.iterrows():
        diff_mat = diff_rec['eigvec_sp_pw_ele_diff_mat']
        l_diff_mat_vec.append(diff_mat.data)

    np_diff_mat_vec = np.stack(l_diff_mat_vec)
    # >>> use col vec for each diff mat for the convenience of computing QR
    # >>> Q and R are all col vectors
    diff_mat_q, diff_mat_r = np.linalg.qr(np_diff_mat_vec.T)
    diff_mat_q = diff_mat_q.T
    diff_mat_r = diff_mat_r.T
    # !!!CAUTION!!!
    # the first basic component is a constant vector, which may not be useful for analysis.
    # we may consider to remove the first basic component.

    sample_diff_mat = df_eigvec_ele_pw_diff.iloc[0]['eigvec_sp_pw_ele_diff_mat']
    l_basis_dim = []
    l_node = list(nx_graph.nodes())
    num_edges = len(nx_graph.edges())
    for i in range(num_edges):
        col = sample_diff_mat.col[i]
        row = sample_diff_mat.row[i]
        col_char = l_node[col]
        row_char = l_node[row]
        l_basis_dim.append((row_char, col_char))

    return diff_mat_q, diff_mat_r, l_basis_dim


def comparison_based_graph_fourier(df_eigvec_ele_pw_diff, np_embed, nx_graph, no_1st_bc=True):
    logging.debug('[comparison_based_graph_fourier] starts.')
    timer_start = time.time()

    sp_gnu_A = sparse.coo_matrix(compute_gnu_A(nx_graph))
    # TODO
    # after the global normalization, the edge weights can become small easily,
    # which may further impact the fourier transformation.
    # we may need some better ways to offset this.
    # sp_gnu_A.data = np.exp(sp_gnu_A.data)
    num_edges = len(nx_graph.edges())

    np_embed = preprocessing.normalize(np_embed)
    # np_pw_cos = (1.0 + np.matmul(np_embed, np_embed.T)) / 2.0
    # sp_pw_dc = sparse.coo_matrix(np.triu(np_pw_cos, k=1))
    # sp_pw_dc = sp_pw_dc.multiply(sp_gnu_A.astype(np.bool).astype(np.int32))

    # >>> dc[x, y] in [0, 1]
    np_pw_dc = (1.0 - np.matmul(np_embed, np_embed.T)) / 2.0
    np_pw_dc[np.where(np_pw_dc < 0.0)] = 0.0
    sp_pw_dc = sparse.coo_matrix(np.triu(np_pw_dc, k=1))
    sp_pw_dc = sp_pw_dc.multiply(sp_gnu_A.astype(np.bool).astype(np.int32))

    pw_dc_vec = np.asarray(sp_pw_dc.data)

    diff_mat_q, diff_mat_r, l_basis_dim = basis_for_eigenvec_pw_ele_diff_mat(df_eigvec_ele_pw_diff, nx_graph)
    embed_pw_dc_vec_to_diff_mat_q = np.matmul(pw_dc_vec, diff_mat_q.T)
    embed_pw_dc_vec_to_diff_mat_q = preprocessing.normalize(embed_pw_dc_vec_to_diff_mat_q.reshape(1, -1))[0]
    diff_mat_r = preprocessing.normalize(diff_mat_r)
    if no_1st_bc:
        embed_pw_dc_vec_to_diff_mat_q = embed_pw_dc_vec_to_diff_mat_q[1:]
        diff_mat_r = diff_mat_r[:, 1:]
    np_eigvalmag = np.matmul(embed_pw_dc_vec_to_diff_mat_q, diff_mat_r.T)

    l_rec = []
    for idx, eig_rec in df_eigvec_ele_pw_diff.iterrows():
        eig_val = eig_rec['eigval']
        eig_vec = eig_rec['eigvec']
        eigvalmag = np_eigvalmag[idx]
        l_rec.append((eig_val, eig_vec, eigvalmag))

    # l_rec = []
    # for _, eig_rec in df_eigvec_ele_pw_diff.iterrows():
    #     eig_val = eig_rec['eigval']
    #     eig_vec = eig_rec['eigvec']
    #     sp_u_pw_ele_diff = eig_rec['eigvec_sp_pw_ele_diff_mat']
    #     if sp_u_pw_ele_diff.shape != sp_pw_dc.shape:
    #         raise Exception('[comparison_based_graph_fourier] invalid sp_pw_dc or sp_u_pw_ele_diff: %s, %s'
    #                         % (sp_pw_dc.shape, sp_u_pw_ele_diff.shape))
    #     # >>> SUM[w_xy * [cos[x, y] - d[u(x), u(y)]]],
    #     # >>> cos[x, y] in [0, 1], d[u(x), u(y)] in [0, 1], w_xy in [0, 1]
    #     # eig_val_mag = np.sum(np.abs(sp_pw_dc - sp_u_pw_ele_diff).multiply(sp_gnu_A))
    #     # >>> weighted inner product between dc and u_diff
    #     # sp_u_pw_ele_diff = sp_u_pw_ele_diff / sparse.linalg.norm(sp_u_pw_ele_diff)
    #     sp_pw_dc = sp_pw_dc / sparse.linalg.norm(sp_pw_dc)
    #     eig_val_mag = np.sum(sp_pw_dc.multiply(sp_u_pw_ele_diff).multiply(sp_gnu_A))
    #     l_rec.append((eig_val, eig_vec, eig_val_mag))
    df_rec = pd.DataFrame(l_rec, columns=['eigval', 'eigvec', 'eigvalmag'])
    logging.debug('[comparison_based_graph_fourier] all done in %s secs.' % str(time.time() - timer_start))
    return df_rec


def draw_comparison_based_graph_fourier_over_epoch(l_comp_based_fourier, img_name, save_ret=True, show_img=False,
                                                   save_path=None):
    df_init_comp_fourier = l_comp_based_fourier[-1]
    np_init_eigvalmags = np.asarray(df_init_comp_fourier['eigvalmag'].to_list())

    l_comp_based_fourier = l_comp_based_fourier[:-1]
    num_eig = len(l_comp_based_fourier[0])
    num_epoch = len(l_comp_based_fourier)
    l_epoch = [i for i in range(num_epoch)]
    l_eigvals = l_comp_based_fourier[0]['eigval'].to_list()

    l_eigvalmag = []
    for df_comp_fourier in l_comp_based_fourier:
        np_eigvalmags = np.asarray(df_comp_fourier['eigvalmag'].to_list())
        l_eigvalmag.append(np_eigvalmags)
    np_eigvalmags_over_epoch = np.stack(l_eigvalmag)
    np_final_eigvalmags = np_eigvalmags_over_epoch[num_epoch - 1]

    fig, axes = plt.subplots(ncols=1, nrows=num_eig + 2, figsize=(10, 40))
    fig.suptitle(img_name, fontsize=10)
    xtick_stride = math.ceil(num_epoch / 25)
    for idx in range(num_eig):
        np_eigvalmags_per_epoch = np_eigvalmags_over_epoch[:, idx]
        eigval = l_eigvals[idx]
        axes[idx].grid(True)
        axes[idx].set_title(r'$\lambda$ = %s' % np.round(eigval, decimals=3), fontsize=10)
        axes[idx].plot(l_epoch, np_eigvalmags_per_epoch)
        axes[idx].set_xticks(np.arange(0, num_epoch + xtick_stride, step=xtick_stride))
        axes[idx].set_yticks(np.arange(0.0, 1.0, step=0.2))

    # draw the init eigval mags
    idx = num_eig
    axes[idx].grid(True)
    axes[idx].set_title('Init Embed CBGF', fontsize=10)
    markerline, stemlines, baselineaxes = axes[idx].stem([i for i in range(len(l_eigvals))], np_init_eigvalmags)
    axes[idx].set_xticks([i for i in range(len(l_eigvals))])
    axes[idx].set_yticks(np.arange(0.0, 1.0, step=0.2))

    # draw the final eigval mags
    idx = num_eig + 1
    axes[idx].grid(True)
    axes[idx].set_title('Final Embed CBGF', fontsize=10)
    markerline, stemlines, baselineaxes = axes[idx].stem([i for i in range(len(l_eigvals))], np_final_eigvalmags)
    axes[idx].set_xticks([i for i in range(len(l_eigvals))])
    axes[idx].set_yticks(np.arange(0.0, 1.0, step=0.2))

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.96)
    if save_ret:
        plt.savefig(save_path, format='PNG')
    if show_img:
        plt.show()
    plt.clf()


################################################################################
#   NODE EMBEDDING GSP ANALYSIS ENDS
################################################################################


################################################################################
#   NODE EMBEDDING STARTS
################################################################################
def compute_diffusion_matrix_from_graph(nx_graph, J_prob):
    logging.debug('[compute_diffusion_matrix_from_graph] starts.')
    timer_start = time.time()

    A = nx.linalg.adjacency_matrix(nx_graph)
    A = preprocessing.normalize(A, axis=1, norm='l1')
    A = (1 - J_prob) * A
    # A = A.astype(np.float32)
    if J_prob == 0.0:
        logging.debug('[compute_diffusion_matrix_from_graph] all done with M in %s secs: %s'
                      % (time.time() - timer_start, A.shape))
        return A
    # T = sparse.diags([J_prob], shape=A.shape, dtype=np.float32)
    T = sparse.diags([J_prob], shape=A.shape)
    M = T + A
    logging.debug('[compute_diffusion_matrix_from_graph] all done with M in %s secs: %s'
                  % (time.time() - timer_start, M.shape))
    # return M.astype(np.float32)
    return M


def compute_gnu_A(nx_graph):
    '''
    Compute globally normalized upper-triangular adjacency excluding the diagonal
    '''
    logging.debug('[compute_gnu_A] starts.')
    timer_start = time.time()

    A = nx.linalg.adjacency_matrix(nx_graph)
    A = sparse.triu(A, k=1)
    A = np.divide(A, np.sum(A))
    logging.debug('[compute_gnu_A] all done with A in %s secs: %s'
                  % (time.time() - timer_start, A.shape))
    # return A.astype(np.float32)
    return A


def compute_ln_A(nx_graph):
    '''
    Compute locally normalized adjacency
    '''
    logging.debug('[compute_ln_A] starts.')
    timer_start = time.time()

    A = nx.linalg.adjacency_matrix(nx_graph)
    A = preprocessing.normalize(A, norm='l1')
    logging.debug('[compute_ln_A] all done with A in %s secs: %s'
                  % (time.time() - timer_start, A.shape))
    # return A.astype(np.float32)
    return A


def sparse_dense_element_mul(t_sp, t_de):
    t_de_typename = th.typename(t_de).split('.')[-1]
    sparse_tensortype = getattr(th.sparse, t_de_typename)

    i = t_sp._indices()
    v = t_sp._values()
    dv = t_de[i[0, :], i[1, :]]
    return sparse_tensortype(i, v * dv, t_sp.size())


def graph_tv_bv_lv_loss(t_gnu_sp_A, t_node_embed, t_ln_sp_A=None, use_tv_loss=True, use_bv_loss=True, use_lv_loss=True,
                        use_lg_loss=True, use_gs_loss=True):
    '''
    cos_sim = (1.0 + <vec_i, vec_j>) / 2.0, for |vec_i|=1 and |vec_j|=1, in range [0, 1]
    cos_dist = (1.0 - cos_sim) / 2.0, in range [0, 1]
    dc = cos_dist
    tv_loss = mean(globally_normalized_edge_weight * cos_dist(i, j)), for all (i, j) being an edge.
    bv_loss = mean(globally_normalized_edge_weight * cos_sim(i, j)), for all i, j not adjacent.
    lv_loss = mean(locally_normalized_weight * cos_dist(i, j)), for all i and j in N(i, 1)
    lg_loss = mean(E[(cos_dist - E[cos_dist])^2])
    gs_loss = mean(E[cos_dis^3] / (E[cos_dis^2])^(2/3)), i.e. 3rd moment
    {}_[row, col] => Matrix
    Intuition: adjacent nodes should be similar, non-adjacent nodes should be dissimilar.
    '''
    t_node_embed_norm = th.nn.functional.normalize(t_node_embed)
    t_pw_cos = th.matmul(t_node_embed_norm, th.transpose(t_node_embed_norm, 0, 1))

    # tv
    if use_tv_loss:
        t_cos_dev = sparse_dense_element_mul(t_gnu_sp_A, t_pw_cos)
        t_cos_dev = th.div(th.sub(th.sparse.sum(t_gnu_sp_A), th.sparse.sum(t_cos_dev)), 2.0)
    else:
        t_cos_dev = None

    # bv
    if use_bv_loss:
        # cos dev beyond neighbors => maximize
        t_gnu_sp_A = t_gnu_sp_A.type(th.bool).type(th.int32)
        val_cnt = (t_gnu_sp_A.shape[0] ** 2 - 2 * len(t_gnu_sp_A._values()) - t_gnu_sp_A.shape[0]) / 2.0
        t_cos_dev_bynd_neig = th.div(th.add(val_cnt,
                                            th.sub(th.div(th.sub(th.sum(t_pw_cos), th.sum(th.diag(t_pw_cos))), 2.0),
                                                   th.sparse.sum(sparse_dense_element_mul(t_gnu_sp_A, t_pw_cos)))),
                                     2.0)
        if val_cnt > 0:
            t_cos_dev_bynd_neig = t_cos_dev_bynd_neig / val_cnt
        else:
            t_cos_dev_bynd_neig = None
    else:
        t_cos_dev_bynd_neig = None

    # lv
    if use_lv_loss or use_lg_loss or use_gs_loss:
        t_sp_edge_dc = sparse_dense_element_mul(t_ln_sp_A.type(th.bool).type(th.int32), (1.0 - t_pw_cos) / 2.0)
        # >>> t_sp_w_edge_dc = {w_ij * dc(v_i, v_j)}_[v_i, v_j], for all v_i, and for all v_j in N(v_i, 1)
        t_sp_w_edge_dc = sparse_dense_element_mul(t_ln_sp_A, t_sp_edge_dc.to_dense())

    if use_lv_loss:
        # >>> E[ {dc(v_i, v_j)}_[v_j] ], for all v_j in N(v_i, 1) = SUM[ {w_ij * dc(v_i, v_j)}_[v_j] ], for v_j in N(v_i, 1)
        # >>> lv_loss = E[ {E[ {dc(v_i, v_j)}_[v_j] ]}_[v_i] ], for all v_i = MEAN[ {E[ {dc(v_i, v_j)}_[v_j] ]}_[v_i] ]
        lv_loss = th.div(th.sparse.sum(t_sp_w_edge_dc), t_pw_cos.shape[0])
    else:
        lv_loss = None

    # lg
    if use_lg_loss or use_gs_loss:
        # >>> t_ds_exp_dc = {E[ {dc(v_i, v_j)}_[v_j] ]}_[v_i], for all v_i
        # NOTICE: dense matrix is used here for PyTorch does not support 'reshape' to sparse matrices,
        #         and row substraction is hard on matrices without 'reshape'.
        t_ds_exp_dc = th.sparse.sum(t_sp_w_edge_dc, dim=1).to_dense().reshape(-1, 1)

    if use_lg_loss:
        # >>> lg_loss = E[ {E[{(dc(v_i, v_j) - E[{dc(v_i, v_j)}_[v_j]])^2}_[v_j]]}_[v_i] ]
        # >>>         = MEAN[ {w_ij * (dc(v_i, v_j) - E[{dc(v_i, v_j)}_[v_j]])^2}_[v_i] ]
        lg_loss = th.sparse.sum(sparse_dense_element_mul(t_ln_sp_A,
                                                         th.square(t_sp_edge_dc.to_dense() - t_ds_exp_dc))) / \
                  t_ln_sp_A.shape[0]
    else:
        lg_loss = None

    # gs
    if use_gs_loss:
        # >>> E[dc(v_i, v_j)] = SUM_j[{w_ij * dc(v_i, v_j))_[v_j]], for all v_j in N(v_i, 1)
        # >>> t_ds_diff_dc_from_exp = {dc(v_i, v_j) - E[dc(v_i, v_j)]}_[v_i, v_j], for all v_j in N(v_i, 1)
        t_ds_diff_dc_from_exp = t_sp_edge_dc.to_dense() - t_ds_exp_dc
        # >>> t_k3 = k3 = {E[{(dc(v_i, vj) - E[dc(v_i, v_j)])^3_[v_j]]}_[v_i]
        # >>>           = {SUM_j[{w_ij * (dc(v_i, vj) - E[dc(v_i, v_j)])^3}_[v_j]]}_[v_i]
        t_k3 = th.sparse.sum(sparse_dense_element_mul(t_ln_sp_A, th.pow(t_ds_diff_dc_from_exp, 3)), dim=1).to_dense()
        # >>> t_k2_p1d5 = k2^1.5 = {E[{(dc(v_i, vj) - E[dc(v_i, v_j)])^2}_[v_j]]^1.5}_[v_i]
        # >>>                    = {SUM_j[{w_ij * t_ds_diff_dc_to_exp^2}_[v_j]]^1.5}_[v_i]
        t_k2_p1d5 = th.pow(
            th.sparse.sum(sparse_dense_element_mul(t_ln_sp_A, th.pow(t_ds_diff_dc_from_exp, 2)), dim=1).to_dense(), 1.5)
        # >>> t_u3 = u3_norm = {|k3 / k2^1.5|}_[v_i]
        # >>> CAUTION: k2^1.5 can be zero!!! WE IGNORE ZEROS.
        t_k2_p1d5_valid_pos = th.where(t_k2_p1d5 > 0)
        if len(t_k2_p1d5_valid_pos[0]) <= 0:
            gs_loss = None
        else:
            t_u3 = th.abs(th.div(t_k3[t_k2_p1d5_valid_pos], t_k2_p1d5[t_k2_p1d5_valid_pos]))
            gs_loss = th.div(th.sum(t_u3), len(t_k2_p1d5_valid_pos[0]))
        # t_u3 = th.abs(th.div(t_k3, t_k2_p1d5))
        # >>> gs_loss = MEAN[{|k3 / k2^1.5|}_[v_i]]
        # t_valid_cnt = th.sum((~th.isnan(t_u3)).type(th.int32))
        # if t_valid_cnt <= 0:
        #     gs_loss = None
        # else:
        #     gs_loss = th.div(th.nansum(t_u3), th.sum((~th.isnan(t_u3)).type(th.int32)))
    else:
        gs_loss = None

    return t_cos_dev, t_cos_dev_bynd_neig, lv_loss, lg_loss, gs_loss


def graph_dffsn_loss(t_node_embed, t_node_embed_t_1):
    '''
    cos_sim = (1.0 + <vec_i, vec_j>) / 2.0, for |vec_i|=1 and |vec_j|=1, in range [0, 1]
    cos_dist = (1.0 - cos_sim) / 2.0, in range [0, 1]
    dffsn_loss = mean(cos_dist(node_embed_t(i), node_embed_t_1(i))), for all nodes i.
    '''
    t_node_embed_t_norm = th.nn.functional.normalize(t_node_embed)
    t_node_embed_t_1_norm = th.nn.functional.normalize(t_node_embed_t_1)
    t_ret = th.einsum('ij..., ij...->i', t_node_embed_t_norm, t_node_embed_t_1_norm)
    t_ret = 1.0 - t_ret
    # low bound
    t_ret[t_ret < 0.0] = 0.0
    # up bound
    t_ret[t_ret > 2.0] = 2.0
    t_ret = th.divide(t_ret, 2.0)
    t_ret = th.mean(t_ret)
    return t_ret


def learn_node_embedding(nx_graph,
                         np_init_embed=None,
                         embed_dim=None,
                         dffsn_loss_w=1.0,
                         tv_loss_w=1.0,
                         bv_loss_w=1.0,
                         lv_loss_w=1.0,
                         lg_loss_w=1.0,
                         gs_loss_w=1.0,
                         dffsn_loss_th=None,
                         tv_loss_th=None,
                         bv_loss_th=None,
                         lv_loss_th=None,
                         lg_loss_th=None,
                         gs_loss_th=None,
                         use_neig_agg=False,
                         learn_dffsn_M=False,
                         J_prob=None,
                         max_epoch=500,
                         use_cuda=False,
                         save_int=False,
                         int_file=None):
    '''
    nx_graph needs to be connected
    '''
    if dffsn_loss_w > 0.0:
        use_dffsn_loss = True
    else:
        use_dffsn_loss = False
    if tv_loss_w > 0.0:
        use_tv_loss = True
    else:
        use_tv_loss = False
    if bv_loss_w > 0.0:
        use_bv_loss = True
    else:
        use_bv_loss = False
    if lv_loss_w > 0.0:
        use_lv_loss = True
    else:
        use_lv_loss = False
    if lg_loss_w > 0.0:
        use_lg_loss = True
    else:
        use_lg_loss = False
    if gs_loss_w > 0.0:
        use_gs_loss = True
    else:
        use_gs_loss = False

    logging.debug('[learn_node_embedding] Starts:\n nx_graph: %s\n np_init_embed: %s\n embed_dim=%s\n '
                  'use_dffsn_loss=%s\n use_tv_loss=%s\n use_bv_loss=%s\n use_lv_loss=%s\n use_lg_loss=%s\n use_gs_loss=%s\n '
                  'dfsn_loss_w=%s\n tv_loss_w=%s\n bv_loss_w=%s\n lv_loss_w=%s\n lg_loss_w=%s\n gs_loss_w=%s\n '
                  'dffsn_loss_th=%s\n tv_loss_th=%s\n bv_loss_th=%s\n lv_loss_th=%s\n lg_loss_th=%s\n gs_loss_th=%s\n '
                  'use_neig_agg=%s\n learn_dffsn_M=%s\n J_prob=%s\n max_epoch=%s\n use_cuda=%s\n' %
                  (nx.info(nx_graph), np_init_embed.shape if np_init_embed is not None else None, embed_dim,
                   use_dffsn_loss, use_tv_loss, use_bv_loss, use_lv_loss, use_lg_loss, use_gs_loss,
                   dffsn_loss_w, tv_loss_w, bv_loss_w, lv_loss_w, lg_loss_w, gs_loss_w,
                   dffsn_loss_th, tv_loss_th, bv_loss_th, lv_loss_th, lg_loss_th, gs_loss_th,
                   use_neig_agg, learn_dffsn_M, J_prob, max_epoch, use_cuda))
    timer_start = time.time()

    dffsn_loss_w = th.tensor(dffsn_loss_w)
    tv_loss_w = th.tensor(tv_loss_w)
    bv_loss_w = th.tensor(bv_loss_w)
    lv_loss_w = th.tensor(lv_loss_w)
    lg_loss_w = th.tensor(lg_loss_w)
    gs_loss_w = th.tensor(gs_loss_w)

    if use_dffsn_loss:
        if not learn_dffsn_M:
            sp_dffsn_M = compute_diffusion_matrix_from_graph(nx_graph, J_prob)
            sp_dffsn_M = sparse.coo_matrix(sp_dffsn_M)
            t_dffsn_M_indices = th.LongTensor(np.vstack((sp_dffsn_M.row, sp_dffsn_M.col)))
            t_dffsn_M_values = th.FloatTensor(sp_dffsn_M.data)
            t_dffsn_M = th.sparse.FloatTensor(t_dffsn_M_indices, t_dffsn_M_values, th.Size(sp_dffsn_M.shape))
        else:
            sp_dffsn_M = compute_diffusion_matrix_from_graph(nx_graph, J_prob)
            t_dffsn_M = th.from_numpy(sp_dffsn_M.toarray())
        t_dffsn_M = t_dffsn_M.type(th.float64)

    if use_tv_loss or use_bv_loss:
        sp_gnu_A = sparse.coo_matrix(compute_gnu_A(nx_graph))
        t_sp_gnu_A = th.sparse.FloatTensor(th.LongTensor(np.vstack((sp_gnu_A.row, sp_gnu_A.col))),
                                           th.FloatTensor(sp_gnu_A.data), th.Size(sp_gnu_A.shape))
        # t_sp_gnu_A = t_sp_gnu_A.type(th.float32)
    else:
        t_sp_gnu_A = None

    if use_lv_loss or use_lg_loss or use_gs_loss:
        sp_ln_A = sparse.coo_matrix(compute_ln_A(nx_graph))
        t_sp_ln_A = th.sparse.FloatTensor(th.LongTensor(np.vstack((sp_ln_A.row, sp_ln_A.col))),
                                          th.FloatTensor(sp_ln_A.data), th.Size(sp_ln_A.shape))
        # t_sp_ln_A = t_sp_ln_A.type(th.float32)
    else:
        t_sp_ln_A = None

    if np_init_embed is None:
        if embed_dim is None:
            raise Exception('[learn_node_embedding] np_init_embed and embed_dim should not be None at the same time.')
        np_init_embed = np.random.randn(nx.number_of_nodes(nx_graph), embed_dim)
    t_node_embed = th.from_numpy(np_init_embed)
    t_node_embed = th.nn.functional.normalize(t_node_embed)
    # t_node_embed = t_node_embed.type(th.float32)

    ##################################################
    #   !!! CAUTION STARTS !!!
    #   Sending tensors to CUDA must be done before
    #   setting 'requires_grad'! Otherwise, tensors
    #   that require grad will not be leaves anymore!
    #   And optimizers do not accept non-leaf tensors!
    #   In a word, the tensors sent to optimizers
    #   must be both leaves and requires_grad=True,
    #   as well as in a fixed device! In addition,
    #   sending tensors to CUDA must be done before
    #   sending them to optimizers, otherwise it will
    #   not learn at all!
    ##################################################
    if use_cuda:
        t_node_embed = t_node_embed.to('cuda')
        if use_dffsn_loss:
            t_dffsn_M = t_dffsn_M.to('cuda')
        if use_tv_loss or use_bv_loss:
            t_sp_gnu_A = t_sp_gnu_A.to('cuda')
        if use_lv_loss or use_lg_loss or use_gs_loss:
            t_sp_ln_A = t_sp_ln_A.to('cuda')
        const_zero = th.tensor(0.0).to('cuda')
        dffsn_loss_w = dffsn_loss_w.to('cuda')
        tv_loss_w = tv_loss_w.to('cuda')
        bv_loss_w = bv_loss_w.to('cuda')
        lv_loss_w = lv_loss_w.to('cuda')
        lg_loss_w = lg_loss_w.to('cuda')
        gs_loss_w = gs_loss_w.to('cuda')
    else:
        const_zero = th.tensor(0.0)

    if use_dffsn_loss:
        if not learn_dffsn_M:
            t_dffsn_M.requires_grad = False
        else:
            t_dffsn_M.requires_grad = True
    if use_tv_loss or use_bv_loss:
        t_sp_gnu_A.requires_grad = False
    if use_lv_loss or use_lg_loss or use_gs_loss:
        t_sp_ln_A.requires_grad = False
    t_node_embed.requires_grad = True
    dffsn_loss_w.requires_grad = False
    tv_loss_w.requires_grad = False
    bv_loss_w.requires_grad = False
    lv_loss_w.requires_grad = False
    lg_loss_w.requires_grad = False
    gs_loss_w.requires_grad = False

    # Potential choices: Adagrad > Adamax > AdamW > Adam
    if not learn_dffsn_M:
        optimizer = th.optim.Adagrad([t_node_embed])
    else:
        optimizer = th.optim.Adagrad([t_node_embed, t_dffsn_M])
    # optimizer = th.optim.SparseAdam([t_adj_embed])
    # SGD may have much higher GPU memory consumption and need more epochs to converge
    # optimizer = th.optim.SGD([t_adj_embed], lr=0.1)
    ##################################################
    #   !!! CAUTION ENDS !!!
    ##################################################
    logging.debug('[learn_node_embedding] All prepared in %s secs.' % str(time.time() - timer_start))

    timer_start = time.time()
    l_int_rec = []
    dffsn_loss_val = 0.0
    tv_loss_val = 0.0
    bv_loss_val = 0.0
    lv_loss_val = 0.0
    lg_loss_val = 0.0
    gs_loss_val = 0.0
    for i in range(max_epoch):
        optimizer.zero_grad()

        # diffusion loss
        if use_dffsn_loss:
            t_node_embed_t_1 = th.matmul(t_dffsn_M, t_node_embed)
            dffsn_loss = graph_dffsn_loss(t_node_embed, t_node_embed_t_1)
        else:
            dffsn_loss = const_zero

        # graph total variation and beyond variation losses
        if use_tv_loss or use_bv_loss or use_lv_loss or use_lg_loss or use_gs_loss:
            tv_loss, bv_loss, lv_loss, lg_loss, gs_loss = graph_tv_bv_lv_loss(t_sp_gnu_A, t_node_embed, t_sp_ln_A,
                                                                              use_tv_loss, use_bv_loss, use_lv_loss,
                                                                              use_lg_loss, use_gs_loss)
            if tv_loss is None:
                tv_loss = const_zero
            if bv_loss is None:
                bv_loss = const_zero
            if lv_loss is None:
                lv_loss = const_zero
            if lg_loss is None:
                lg_loss = const_zero
            if gs_loss is None:
                gs_loss = const_zero
        else:
            tv_loss = const_zero
            bv_loss = const_zero
            lv_loss = const_zero
            lg_loss = const_zero
            gs_loss = const_zero

        total_loss = dffsn_loss_w * dffsn_loss + tv_loss_w * tv_loss + bv_loss_w * bv_loss + lv_loss_w * lv_loss \
                     + lg_loss_w * lg_loss + gs_loss_w * gs_loss
        logging.debug(
            '[learn_node_embedding] epoch %s: dffsn_loss=%s, tv_loss=%s, bv_loss=%s, lv_loss=%s, lg_loss=%s, gs_loss=%s, '
            'w_dffsn_loss=%s, w_tv_loss=%s, w_bv_loss=%s, w_lv_loss=%s, w_lg_loss=%s, w_gs_loss=%s, '
            'total_loss=%s, elapse=%s'
            % (i, dffsn_loss, tv_loss, bv_loss, lv_loss, lg_loss, gs_loss,
               dffsn_loss_w * dffsn_loss, tv_loss_w * tv_loss,
               bv_loss_w * bv_loss, lv_loss_w * lv_loss, lg_loss_w * lg_loss, gs_loss_w * gs_loss,
               total_loss, time.time() - timer_start))

        if (dffsn_loss_th is not None and dffsn_loss <= dffsn_loss_th) \
                or (tv_loss_th is not None and tv_loss <= tv_loss_th) \
                or (bv_loss_th is not None and bv_loss <= bv_loss_th) \
                or (lv_loss_th is not None and lv_loss <= lv_loss_th) \
                or (lg_loss_th is not None and lg_loss <= lg_loss_th) \
                or (gs_loss_th is not None and gs_loss <= gs_loss_th):
            logging.debug('[learn_node_embedding] Return at epoch %s in %s secs.' % (i, time.time() - timer_start))
            t_node_embed = th.nn.functional.normalize(t_node_embed)
            return t_node_embed

        total_loss.backward()
        optimizer.step()

        ########################################
        #   TEST ONLY STARTS
        ########################################
        if th.isnan(t_node_embed).any():
            raise Exception('t_node_embed has nan!')
        ########################################
        #   TEST ONLY ENDS
        ########################################

        if save_int:
            l_int_rec.append((i,
                              t_node_embed.cpu().detach().numpy(),
                              dffsn_loss.cpu().detach().numpy().item(),
                              tv_loss.cpu().detach().numpy().item(),
                              bv_loss.cpu().detach().numpy().item(),
                              lv_loss.cpu().detach().numpy().item(),
                              lg_loss.cpu().detach().numpy().item(),
                              gs_loss.cpu().detach().numpy().item()))

        if use_dffsn_loss and use_neig_agg:
            with th.no_grad():
                t_node_embed = th.matmul(t_dffsn_M, t_node_embed)
                t_node_embed.requires_grad = True

        if i + 1 >= max_epoch:
            dffsn_loss_val = dffsn_loss.cpu().detach().numpy().item()
            tv_loss_val = tv_loss.cpu().detach().numpy().item()
            bv_loss_val = bv_loss.cpu().detach().numpy().item()
            lv_loss_val = lv_loss.cpu().detach().numpy().item()
            lg_loss_val = lg_loss.cpu().detach().numpy().item()
            gs_loss_val = gs_loss.cpu().detach().numpy().item()

    logging.debug('[learn_node_embedding] Return with full epoches in %s secs.' % str(time.time() - timer_start))
    if save_int:
        df_int = pd.DataFrame(l_int_rec,
                              columns=['epoch', 'node_embed_int', 'dffsn_loss', 'tv_loss', 'bv_loss', 'lv_loss',
                                       'lg_loss', 'gs_loss'])
        pd.to_pickle(df_int, int_file)
    t_node_embed = th.nn.functional.normalize(t_node_embed)
    return t_node_embed, dffsn_loss_val, tv_loss_val, bv_loss_val, lv_loss_val, lg_loss_val, gs_loss_val


def draw_learned_node_embed(nx_graph, np_node_embed_learned,
                            J_prob, dffsn_loss_w, tv_loss_w, bv_loss_w, lv_loss_w, lg_loss_w, gs_loss_w,
                            max_epoch, ari, nmi,
                            dffsn_loss_val, tv_loss_val, bv_loss_val, lv_loss_val, lg_loss_val, gs_loss_val,
                            l_pred_color, save_ret=False, save_path=None, show_img=True, show_title=True, epoch=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np_node_embed_learned[:, 0], np_node_embed_learned[:, 1], np_node_embed_learned[:, 2],
               s=50, c=l_pred_color, cmap='jet')
    origin_coord = [0.0, 0.0, 0.0]
    for i in range(len(np_node_embed_learned)):
        ax.plot([origin_coord[0], np_node_embed_learned[i][0]],
                [origin_coord[1], np_node_embed_learned[i][1]],
                [origin_coord[2], np_node_embed_learned[i][2]],
                c='tab:gray')
    if show_title:
        ax.set_title(
            'J_prob:%s dffsn_loss_w=%s tv_loss_w=%s bv_loss_w=%s lv_loss_w=%s lg_loss_w=%s gs_loss_w=%s\n epoch=%s ari=%s nmi=%s\n'
            'dffsn_loss=%s tv_loss=%s bv_loss=%s lv_loss=%s lg_loss=%s gs_loss=%s'
            % (np.round(J_prob, decimals=1), np.round(dffsn_loss_w, decimals=1), np.round(tv_loss_w, decimals=1),
               np.round(bv_loss_w, decimals=1), np.round(lv_loss_w, decimals=1), np.round(lg_loss_w, decimals=1),
               np.round(gs_loss_w, decimals=1),
               max_epoch, np.round(ari, decimals=2), np.round(nmi, decimals=2),
               np.round(dffsn_loss_val, decimals=4), np.round(tv_loss_val, decimals=4), np.round(bv_loss_val, decimals=4),
               np.round(lv_loss_val, decimals=4), np.round(lg_loss_val, decimals=4), np.round(gs_loss_val, decimals=4)))
    if epoch is not None:
        ax.set_title('Epoch %s' % epoch)
    for x, y, z, node in zip(np_node_embed_learned[:, 0],
                             np_node_embed_learned[:, 1],
                             np_node_embed_learned[:, 2],
                             nx_graph.nodes()):
        ax.text(x, y, z, node, size=15)
    if save_ret:
        plt.savefig(save_path, format='PNG')
    if show_img:
        plt.show()
    plt.clf()
    plt.close()


def configured_run(run_id, nx_graph, np_init_embed, embed_dim,
                   dffsn_loss_w_range, J_prob_range, tv_loss_w_range, bv_loss_w_range, lv_loss_w_range, lg_loss_w_range,
                   gs_loss_w_range,
                   max_epoch=1000, save_int=False, save_folder=None, show_img=True, show_init_img=False,
                   do_cluster=True, d_gt=None, k_cluster=None, cluster_alg='kmeans'):
    logging.debug('[configured_run] Run %s: start.' % str(run_id))
    timer_start = time.time()

    if dffsn_loss_w_range is None:
        dffsn_loss_w_range = [-0.0]
        J_prob_range = [-0.0]
    else:
        if J_prob_range is None:
            J_prob_range = np.arange(0.0, 1.0, 0.1)
    if tv_loss_w_range is None:
        tv_loss_w_range = [-0.0]
    if bv_loss_w_range is None:
        bv_loss_w_range = [-0.0]
    if lv_loss_w_range is None:
        lv_loss_w_range = [-0.0]
    if lg_loss_w_range is None:
        lg_loss_w_range = [-0.0]
    if gs_loss_w_range is None:
        gs_loss_w_range = [-0.0]

    if save_int:
        save_path = save_folder + 'ne_run_%s/' % str(run_id)
        os.mkdir(save_path)
        graph_img_name = 'ne_cluster_run@%s@init.png' % str(run_id)
        np.save(save_path + 'ne_run@' + str(run_id) + '@init_embed.npy', np_init_embed)
    else:
        save_path = None

    if show_init_img:
        ari, nmi, l_pred_color = node_clustering(d_gt, nx_graph, np_init_embed, k_cluster)
        draw_learned_node_embed(nx_graph, np_init_embed, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                max_epoch, np.round(ari, decimals=2), np.round(nmi, decimals=2),
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                l_pred_color,
                                save_ret=save_int, save_path=save_path + graph_img_name if save_int else None,
                                show_img=show_img)

    for dffsn_loss_w in dffsn_loss_w_range:
        for J_prob in J_prob_range:
            for tv_loss_w in tv_loss_w_range:
                for bv_loss_w in bv_loss_w_range:
                    for lv_loss_w in lv_loss_w_range:
                        for lg_loss_w in lg_loss_w_range:
                            for gs_loss_w in gs_loss_w_range:
                                if save_int:
                                    int_file = save_path + 'ne_run@%s@df%s_jp%s_tv%s_bv%s_lv%s_lg%s_gs%s_ep%s@ne_int.pickle' \
                                               % (run_id, np.round(dffsn_loss_w, decimals=1),
                                                  np.round(J_prob, decimals=1),
                                                  np.round(tv_loss_w, decimals=1), np.round(bv_loss_w, decimals=1),
                                                  np.round(lv_loss_w, decimals=1), np.round(lg_loss_w, decimals=1),
                                                  np.round(gs_loss_w, decimals=1),
                                                  max_epoch)
                                t_node_embed, dffsn_loss_val, tv_loss_val, bv_loss_val, lv_loss_val, lg_loss_val, gs_loss_val \
                                    = learn_node_embedding(nx_graph,
                                                           np_init_embed=np_init_embed,
                                                           embed_dim=embed_dim,
                                                           dffsn_loss_w=dffsn_loss_w,
                                                           tv_loss_w=tv_loss_w,
                                                           bv_loss_w=bv_loss_w,
                                                           lv_loss_w=lv_loss_w,
                                                           lg_loss_w=lg_loss_w,
                                                           gs_loss_w=gs_loss_w,
                                                           dffsn_loss_th=None,
                                                           tv_loss_th=None,
                                                           bv_loss_th=None,
                                                           lv_loss_th=None,
                                                           lg_loss_th=None,
                                                           gs_loss_th=None,
                                                           use_neig_agg=False,
                                                           learn_dffsn_M=False,
                                                           J_prob=J_prob,
                                                           max_epoch=max_epoch,
                                                           use_cuda=True,
                                                           save_int=save_int,
                                                           int_file=int_file if save_int else None)
                                np_node_embed_learned = t_node_embed.cpu().detach().numpy()
                                # np_pw_dc = np.round((1.0 - np.matmul(np_node_embed_learned, np_node_embed_learned.T)) / 2.0,
                                #                     decimals=2)

                                if do_cluster:
                                    ari, nmi, l_pred_color = node_clustering(d_gt, nx_graph, np_node_embed_learned,
                                                                             k_cluster, cluster_alg=cluster_alg)
                                else:
                                    ari = np.nan
                                    nmi = np.nan
                                    l_pred_color = None

                                logging.debug(
                                    '[configured_run] Run %s: Done with\ndffsn_loss_w=%s\nJ_prob=%s\ntv_loss_w=%s\n'
                                    'bv_loss_w=%s\nlv_loss_w=%s\nlg_loss_w=%s\ngs_loss_w=%s\nmax_epoch=%s\nari=%s\nnmi=%s\nin %s secs.'
                                    % (run_id, np.round(dffsn_loss_w, decimals=1), np.round(J_prob, decimals=1),
                                       np.round(tv_loss_w, decimals=1), np.round(bv_loss_w, decimals=1),
                                       np.round(lv_loss_w, decimals=1), np.round(lg_loss_w, decimals=1),
                                       np.round(gs_loss_w, decimals=1),
                                       max_epoch, np.round(ari, decimals=2), np.round(nmi, decimals=2),
                                       time.time() - timer_start))
                                # logging.debug('[configured_run] Run %s: np_pw_dc=%s' % (run_id, np_pw_dc))

                                if do_cluster:
                                    graph_img_name = 'ne_cluster_run@%s@df%s_jp%s_tv%s_bv%s_lv%s_lg%s_gs%s_ep%s.png' \
                                                     % (run_id, np.round(dffsn_loss_w, decimals=1),
                                                        np.round(J_prob, decimals=1),
                                                        np.round(tv_loss_w, decimals=1),
                                                        np.round(bv_loss_w, decimals=1),
                                                        np.round(lv_loss_w, decimals=1),
                                                        np.round(lg_loss_w, decimals=1),
                                                        np.round(gs_loss_w, decimals=1),
                                                        max_epoch)
                                    draw_learned_node_embed(nx_graph, np_node_embed_learned,
                                                            J_prob, dffsn_loss_w, tv_loss_w, bv_loss_w,
                                                            lv_loss_w, lg_loss_w, gs_loss_w,
                                                            max_epoch, ari, nmi,
                                                            dffsn_loss_val, tv_loss_val, bv_loss_val,
                                                            lv_loss_val, lg_loss_val, gs_loss_val,
                                                            l_pred_color,
                                                            save_ret=save_int,
                                                            save_path=save_path + graph_img_name if save_int else None,
                                                            show_img=show_img)

    logging.debug('[configured_run] Run %s: All done in %s secs.' % (run_id, time.time() - timer_start))


################################################################################
#   NODE EMBEDDING ENDS
################################################################################


################################################################################
#   NODE CLUSTERING STARTS
################################################################################
def node_clustering(d_gt, nx_graph, np_node_embed_learned, k_cluster, cluster_alg='kmeans'):
    if cluster_alg == 'kmeans':
        clustering = KMeans(n_clusters=k_cluster, random_state=0, n_jobs=-1).fit(np_node_embed_learned)
    elif cluster_alg == 'spectral':
        np_pw_dc = (1.0 + np.matmul(np_node_embed_learned, np_node_embed_learned.T)) / 2.0
        # np_pw_dc = (1.0 - np.matmul(np_node_embed_learned, np_node_embed_learned.T)) / 2.0
        clustering = SpectralClustering(n_clusters=k_cluster, assign_labels="discretize", random_state=0,
                                        affinity='precomputed', n_jobs=-1).fit(np_pw_dc)
    l_gt = [d_gt[node] for node in nx_graph.nodes()]
    nmi = metrics.adjusted_mutual_info_score(l_gt, clustering.labels_)
    ari = metrics.adjusted_rand_score(l_gt, clustering.labels_)
    l_cmap = np.arange(0.0, 1.0, 1.0 / k_cluster)[:k_cluster]
    l_pred_color = [l_cmap[x] for x in clustering.labels_]
    return ari, nmi, l_pred_color


################################################################################
#   NODE CLUSTERING ENDS
################################################################################


################################################################################
#   TEST ONLY STARTS
################################################################################
def stratified_graphs_sizes():
    logging.debug('[stratified_graphs_sizes] Starts.')
    timer_start = time.time()

    num_nodes_step = 1
    min_num_nodes = 5
    max_num_nodes = 1000 + num_nodes_step

    edge_density_step = 0.01
    min_edge_density = 0.01
    max_edge_density = 1.0 + edge_density_step

    l_rec = []
    rec_cnt = 0
    for num_nodes in np.arange(min_num_nodes, max_num_nodes, step=num_nodes_step, dtype=np.int32):
        for edge_density in np.arange(min_edge_density, max_edge_density, step=edge_density_step, dtype=np.float32):
            sp_A = sparse.rand(num_nodes, num_nodes, density=edge_density, dtype=np.float32)
            np_A = sp_A.toarray()
            np_A = np_A + np_A.T
            np.fill_diagonal(np_A, 0)
            nx_graph = nx.from_scipy_sparse_matrix(sp_A)
            if not nx.is_connected(nx_graph):
                continue
            diameter = nx.diameter(nx_graph)

            d_Bk_cnt_edges = dict()
            l_np_Ak = [np_A]
            if diameter >= 2:
                for k in range(2, diameter + 1):
                    np_Ak = np.matmul(np_A, l_np_Ak[-1])
                    np_Ak = np_Ak.astype(np.bool).astype(np.int32)
                    l_np_Ak.append(np_Ak)

            l_np_Bk = [l_np_Ak[0]]
            for i in range(1, len(l_np_Ak)):
                np_Bk = l_np_Ak[i] - np.cumsum(np.stack(l_np_Bk), axis=0)[-1]
                np.fill_diagonal(np_Bk, 0)
                np_Bk[np.where(np_Bk < 0)] = 0
                l_np_Bk.append(np_Bk)

            for i, np_Bk in enumerate(l_np_Bk):
                k = i + 1
                d_Bk_cnt_edges[k] = np.count_nonzero(np_Bk) / 2

            l_rec.append((num_nodes, np.round(edge_density, decimals=2), d_Bk_cnt_edges))
            rec_cnt += 1
            if rec_cnt % 5000 == 0 and rec_cnt >= 5000:
                logging.debug('[stratified_graphs_sizes] %s rec_cnt in %s secs.' % (rec_cnt, time.time() - timer_start))

    df_rec = pd.DataFrame(l_rec, columns=['num_nodes', 'edge_density', 'd_Bk_cnt_edges'])
    pd.to_pickle(df_rec, g_work_dir + 'LSG_sizes.pickle')
    logging.debug('[stratified_graphs_sizes] All done in %s secs.' % str(time.time() - timer_start))


################################################################################
#   TEST ONLY ENDS
################################################################################

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    cmd = sys.argv[1]

    if cmd == 'gen_graph':
        # gen_unweighted_symA_graph()
        # gen_weighted_symA_graph()
        # gen_unweighted_symA_K3_C1_graph()
        # gen_uw_symA_line_graph()
        # gen_complete_graph(20)
        # gen_star_graph(7)
        # gen_ring_graph(20)
        gen_turan_graph(6, 3)
        # gen_8_graph(21)
        # gen_grid_2d_graph(5, 5)
    # elif cmd == 'gen_rand_graphs':
    #     num_rand_graphs = 100
    #     num_nodes = 50
    #     edge_density = 0.05
    #     save_ret = True
    #     save_path = g_work_dir + 'experiments/gsp_vs_sas/rand_graphs/'
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     gen_rand_graphs(num_rand_graphs=num_rand_graphs, num_nodes=num_nodes, edge_density=edge_density,
    #                     graph_name_prefix=None, save_ret=save_ret, save_path=save_path)

    elif cmd == 'draw_graph':
        nx_graph = nx.read_gpickle(g_work_dir + 'star_graph.pickle')
        unweighted = True
        draw_graph(nx_graph, unweighted=unweighted)
        print()

    elif cmd == 'line_graph':
        plt.set_loglevel('error')
        nx_graph = nx.read_gpickle(g_work_dir + 'uw_symA.pickle')
        l_nodes = list(nx_graph.nodes())

        # >>> CONVERT TO LINE GRAPH <<<
        nx_line_graph = nx.line_graph(nx_graph)
        l_line_node_str = [node[0] + node[1] for node in nx_line_graph.nodes()]
        l_line_node_idx = [(l_nodes.index(ln_node[0]), l_nodes.index(ln_node[1])) for ln_node in nx_line_graph.nodes()]

        # >>> TRANSLATE ORIGINAL SIGNAL <<<
        np_sig = np.load('/home/mf3jh/workspace/data/papers/node_embed/classic_gsp/uw_symA_B_pulse_heat/np_sig.npy')
        np_sig_uxuy = np.matmul(np_sig.reshape(-1, 1), np_sig.reshape(-1, 1).T)
        np_sig_uxsqr_add_uysqr = np.asarray([np.power(np_sig, 2)] * len(np_sig)) + np.power(np_sig, 2).reshape(-1, 1)
        np_sig_ux_minus_uy_sqr = np.round(np_sig_uxsqr_add_uysqr - 2 * np_sig_uxuy, decimals=12)
        np_sig_adj_diff_mat = np.sqrt(np_sig_ux_minus_uy_sqr)
        np_ln_sig = np.asarray([np_sig_adj_diff_mat[ln_node_idx[0]][ln_node_idx[1]] for ln_node_idx in l_line_node_idx])

        np_filtered_sig = np.load(
            '/home/mf3jh/workspace/data/papers/node_embed/classic_gsp/uw_symA_B_pulse_heat/np_sig_filtered.npy')
        np_sig_filtered_uxuy = np.matmul(np_filtered_sig.reshape(-1, 1), np_filtered_sig.reshape(-1, 1).T)
        np_sig_filtered_uxsqr_add_uysqr = np.asarray([np.power(np_filtered_sig, 2)] * len(np_filtered_sig)) + np.power(
            np_filtered_sig, 2).reshape(-1, 1)
        np_sig_filtered_ux_minus_uy_sqr = np.round(np_sig_filtered_uxsqr_add_uysqr - 2 * np_sig_filtered_uxuy,
                                                   decimals=12)
        np_sig_filtered_adj_diff_mat = np.sqrt(np_sig_filtered_ux_minus_uy_sqr)
        np_ln_sig_filtered = np.asarray(
            [np_sig_filtered_adj_diff_mat[ln_node_idx[0]][ln_node_idx[1]] for ln_node_idx in l_line_node_idx])

        # >>> COMPUTE GRAPH LAPLACIAN <<<
        sp_L = graph_laplacian(nx_line_graph, use_norm=False)

        # >>> COMPUTE EIGENVALUES AND EIGENVECTORS FOR LAPLACIAN <<<
        l_eigs = graph_eigs(sp_L, use_sparse=False, top_M_eigs=None)
        np_eig_vals = np.asarray([item[0] for item in l_eigs])
        np_U = np.stack([item[1] for item in l_eigs])
        l_eig_vals = [np.round(eigval, decimals=3) for eigval in np_eig_vals]

        # >>> SIGNAL FOURIER TRANSFORMATION <<<
        np_ln_sig_hat = graph_fourier(np_U, np_ln_sig)
        np_ln_sig_filtered_hat = graph_fourier(np_U, np_ln_sig_filtered)

        # >>> DRAW <<<
        save_folder = g_work_dir + 'classic_gsp/uw_symA_B_pulse_heat/'
        # img_name = 'uw_symA pulse@B heat diffusion tau=%s sig & sig_filtered in line graph' % 0.5
        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 6))
        # fig.suptitle(img_name, fontsize=15,  fontweight='semibold')
        # draw signals
        idx = 0
        axes[idx].grid(True)
        axes[idx].set_title('ln_sig & ln_sig_filtered', fontsize=15)
        input_linefmt = 'tab:blue'
        input_marker_fmt = 'o'
        axes[idx].stem(np_ln_sig,
                       linefmt=input_linefmt,
                       markerfmt=input_marker_fmt,
                       label='pulse@B')
        filtered_linefmt = 'tab:orange'
        filtered_marker_fmt = 'o'
        axes[idx].stem(np_ln_sig_filtered,
                       linefmt=filtered_linefmt,
                       markerfmt=filtered_marker_fmt,
                       label='heat_filtered')
        axes[idx].set_xticks([i for i in range(len(l_line_node_str))])
        axes[idx].set_xticklabels(l_line_node_str)
        axes[idx].legend()
        idx += 1
        # draw transformation
        axes[idx].grid(True)
        axes[idx].set_title('ln_sig_hat & ln_sig_filtered_hat', fontsize=15)
        input_linefmt = 'tab:blue'
        input_marker_fmt = 'o'
        axes[idx].stem(np.abs(np_ln_sig_hat),
                       linefmt=input_linefmt,
                       markerfmt=input_marker_fmt,
                       label='pulse@B')
        filtered_linefmt = 'tab:orange'
        filtered_marker_fmt = 'o'
        axes[idx].stem(np.abs(np_ln_sig_filtered_hat),
                       linefmt=filtered_linefmt,
                       markerfmt=filtered_marker_fmt,
                       label='heat_filtered')
        axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
        axes[idx].set_xticklabels(l_eig_vals)
        axes[idx].legend()
        idx += 1

        plt.tight_layout(pad=1.0)
        # plt.subplots_adjust(top=0.92)

        plt.savefig(save_folder + 'line_graph_sig.PNG', format='PNG')
        plt.show()

        print()

    elif cmd == 'draw_eigs':
        plt.set_loglevel('error')
        save_img = False
        show_img = True
        graph_name = 'star_graph'
        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')
        l_eigs = graph_eigs(nx.linalg.laplacian_matrix(nx_graph))
        l_nodes = list(nx_graph.nodes())

        # >>> draw eigenvalues and eigenvectors
        eig_img_name = graph_name + '#eigs.PNG'
        # max_eig_val = np.max([eig[1] for eig in l_eigs])
        # min_eig_val = np.min([eig[1] for eig in l_eigs])
        img_height = math.ceil(len(l_eigs) / 10) * 24
        fig, axes = plt.subplots(ncols=1, nrows=len(l_eigs), figsize=(15, img_height))
        # fig.suptitle('Eigenvalues & Eigenvectors', fontsize=20, fontweight='semibold')
        for idx, (eig_val, eig_vec) in enumerate(l_eigs):
            axes[idx].grid(True)
            axes[idx].set_title(r'$\lambda$ = %s' % np.round(eig_val, decimals=3), fontsize=10)
            axes[idx].errorbar([i for i in range(len(l_nodes))], np.round(eig_vec, decimals=12),
                               fmt='o-', c='tab:blue', capsize=1, capthick=1)
            # axes[idx].set_xticks([i for i in range(len(l_nodes))])
            # axes[idx].set_yticks(np.arange(min_eig_val, max_eig_val, step=0.1))
        plt.tight_layout(pad=1.0)
        if save_img:
            plt.savefig(g_work_dir + eig_img_name, format='PNG')
        if show_img:
            plt.show()
        plt.clf()
        plt.close()

    elif cmd == 'gnn_test':
        # >>> GEN WEIGHTED KARATE CLUB GRAPH <<<
        # gen_weighted_karate_club_graph()

        # >>> LOAD WEIGHTED KARATE CLUB GRAPH <<<
        nx_karate = nx.read_gpickle('weighted_karate_club.pickle')

        # >>> DRAW KARATE CLUB GRAPH <<<
        # draw_weighted_karate_club_graph(nx_karate)

        # >>> COMPUTE GRAPH LAPLACIAN <<<
        sp_L = graph_laplacian(nx_karate, use_norm=False)

        # >>> COMPUTE EIGENVALUES AND EIGENVECTORS FOR LAPLACIAN <<<
        l_eigs = graph_eigs(sp_L, use_sparse=False, top_M_eigs=None)
        np_eig_vals = np.asarray([item[0] for item in l_eigs])
        np_U = np.stack([item[1] for item in l_eigs]).T

        # >>> GEN RANDOM SIGNALS ON NODES <<<
        feature_dim = 3
        np_sig = np.random.randn(nx.number_of_nodes(nx_karate), feature_dim)
        np_sig = preprocessing.normalize(np_sig)

        # >>> COMPUTE FOURIER TRANSFORMED SIGNALS <<<
        np_sig_hat = graph_fourier(np_U, np_sig)

        # >>> COMPUTE DIFFUSION MATRIX <<<
        tau = 0.0
        sp_M = compute_diffusion_matrix_from_graph(nx_karate, tau)

    elif cmd == 'classic_gsp':
        plt.set_loglevel('error')

        graph_name = 'uw_symA'

        # >>> LOAD IN GRAPH <<<
        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')
        l_nodes = list(nx_graph.nodes())

        # >>> CONSTRUCT SIGNAL <<<
        # set a pulse at B
        np_sig = np.asarray([0] * nx.number_of_nodes(nx_graph))
        np_sig[1] = 1.0

        # >>> COMPUTE GRAPH LAPLACIAN <<<
        sp_L = graph_laplacian(nx_graph, use_norm=False)

        # >>> COMPUTE EIGENVALUES AND EIGENVECTORS FOR LAPLACIAN <<<
        l_eigs = graph_eigs(sp_L, use_sparse=False, top_M_eigs=None)
        np_eig_vals = np.asarray([item[0] for item in l_eigs])
        np_U = np.stack([item[1] for item in l_eigs])

        # >>> SIGNAL FOURIER TRANSFORMATION <<<
        np_sig_hat = graph_fourier(np_U, np_sig)

        # >>> FILTERING <<<
        # heat diffusion
        tau = 0.5
        np_filter = np.exp(- tau * np_eig_vals)
        np_sig_filtered_hat = np_sig_hat * np_filter

        # >>> SIGNAL INVERSE FOURIER TRANSFORMATION <<<
        np_sig_filtered = graph_inverse_fourier(np_U, np_sig_filtered_hat)

        # >>> SAVE RESULTS <<<
        save_folder = g_work_dir + 'experiments/uw_symA_pulse_B_vs_heat_filtered/'
        np.save(save_folder + 'np_sig', np_sig)
        np.save(save_folder + 'np_sig_filtered', np_sig_filtered)

        # >>> DRAW <<<
        img_name = 'uw_symA pulse@B heat diffusion tau=%s' % 0.5
        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 6))
        # fig.suptitle(img_name, fontsize=15,  fontweight='semibold')
        # draw signals
        idx = 0
        axes[idx].grid(True)
        axes[idx].set_title('Input & Filtered Signals', fontsize=15)
        input_linefmt = 'tab:blue'
        input_marker_fmt = 'o'
        axes[idx].stem(np_sig,
                       linefmt=input_linefmt,
                       markerfmt=input_marker_fmt,
                       label='pulse@B')
        filtered_linefmt = 'tab:orange'
        filtered_marker_fmt = 'o'
        axes[idx].stem(np_sig_filtered,
                       linefmt=filtered_linefmt,
                       markerfmt=filtered_marker_fmt,
                       label='heat_filtered')
        axes[idx].set_xticks([i for i in range(len(l_nodes))])
        axes[idx].set_xticklabels(l_nodes)
        axes[idx].legend()
        idx += 1

        # draw transformed signals
        l_eig_vals = [np.round(eigval, decimals=3) for eigval in np_eig_vals]
        axes[idx].grid(True)
        axes[idx].set_title('Input & Filtered Transformed Signals', fontsize=15)
        input_linefmt = 'tab:blue'
        input_marker_fmt = 'o'
        axes[idx].stem(np.abs(np_sig_hat),
                       linefmt=input_linefmt,
                       markerfmt=input_marker_fmt,
                       label='pulse@B')
        filtered_linefmt = 'tab:orange'
        filtered_marker_fmt = 'o'
        axes[idx].stem(np.abs(np_sig_filtered_hat),
                       linefmt=filtered_linefmt,
                       markerfmt=filtered_marker_fmt,
                       label='heat_filtered')
        axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
        axes[idx].set_xticklabels(l_eig_vals)
        axes[idx].legend()
        plt.tight_layout(pad=1.0)
        # plt.subplots_adjust(top=0.92)

        # plt.savefig(save_folder + 'sig_and_trans.PNG', format='PNG')
        plt.show()

    elif cmd == 'node_embed':
        plt.set_loglevel('error')
        # >>> DATA SETTINGS <<<
        graph_name = 'uw_symA'
        embed_dim = 3
        rand_run_cnt = 1
        k_cluster = 4
        d_gt = {'A': 0, 'B': 1, 'E': 1, 'F': 1, 'K': 1, 'C': 2, 'G': 2, 'H': 2, 'L': 2, 'D': 3, 'I': 3, 'J': 3, 'M': 3}
        use_manual_ne = True
        manual_ne_type = 'single_pulse'
        # manual_ne_type = 'ideal'

        # >>> LOAD IN GRAPH <<<
        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')

        # >>> LEARN NODE EMBEDDING <<<
        l_work_trial = []
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        work_trial_name = 'ne_run_' + graph_name + '_' + now
        save_folder = g_work_dir + work_trial_name + '/'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        for run in range(rand_run_cnt):
            if use_manual_ne:
                np_init_embed = manual_node_embed(ne_type=manual_ne_type)
            else:
                np_init_embed = np.random.randn(nx.number_of_nodes(nx_graph), embed_dim)
            np_init_embed = preprocessing.normalize(np_init_embed)

            run_id = graph_name + '_' + now + '_' + str(run)
            l_work_trial.append(save_folder + run_id)
            dffsn_loss_w_range = None
            J_prob_range = None
            tv_loss_w_range = [1.0]
            bv_loss_w_range = None
            lv_loss_w_range = None
            lg_loss_w_range = None
            gs_loss_w_range = None
            max_epoch = 100
            save_int = True
            show_img = False
            show_init_img = True
            do_cluster = True
            d_gt = d_gt
            k_cluster = k_cluster
            cluster_alg = 'spectral'

            configured_run(run_id, nx_graph, np_init_embed, embed_dim,
                           dffsn_loss_w_range, J_prob_range, tv_loss_w_range,
                           bv_loss_w_range, lv_loss_w_range, lg_loss_w_range, gs_loss_w_range,
                           max_epoch=max_epoch, save_int=save_int, save_folder=save_folder, show_img=show_img,
                           show_init_img=show_init_img, do_cluster=do_cluster, d_gt=d_gt, k_cluster=k_cluster,
                           cluster_alg=cluster_alg)

        # with open(g_work_dir + 'work_trial_' + now + '.txt', 'w+') as out_fd:
        #     out_str = '\n'.join(l_work_trial)
        #     out_fd.write(out_str)
        #     out_fd.close()
        logging.debug('All done.')

    elif cmd == 'node_embed_specific':
        '''
        TEST ONLY!!!
        '''
        plt.set_loglevel('error')
        # >>> DATA SETTINGS <<<
        nx_graph = nx.read_gpickle('uw_symA.pickle')
        embed_dim = 3
        rand_run_cnt = 1
        k_cluster = 4
        d_gt = {'A': 0, 'B': 1, 'E': 1, 'F': 1, 'K': 1, 'C': 2, 'G': 2, 'H': 2, 'L': 2, 'D': 3, 'I': 3, 'J': 3, 'M': 3}

        # >>> LEARN NODE EMBEDDING <<<
        l_work_trial = []
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for run in range(rand_run_cnt):
            # np_init_embed = np.load('/home/mf3jh/workspace/data/papers/node_embed/ne_run_uw_symA_20210318123648_2/ne_run@uw_symA_20210318123648_2@init_embed.npy')
            np_init_embed = np.asarray([[0.2, 0.2, 0.2],  # A
                                        [1.0, 0.0, 1.0],  # B
                                        [0.0, 1.0, 1.0],  # C
                                        [1.0, 1.0, 0.0],  # D
                                        [1.9659, 0.0, 1.2588],  # E
                                        [1.2588, 0.0, 1.9659],  # F
                                        [0.0, 1.9659, 1.2588],  # G
                                        [0.0, 1.2588, 1.9659],  # H
                                        [1.9659, 1.2588, 0.0],  # I
                                        [1.2588, 1.9659, 0.0],  # J
                                        [2.2247, 0.0, 2.2247],  # K
                                        [0.0, 2.2247, 2.2247],  # L
                                        [2.2247, 2.2247, 0.0],  # M
                                        ])
            np_init_embed = preprocessing.normalize(np_init_embed)
            np_pairwise_dc = (1.0 - np.matmul(np_init_embed, np_init_embed.T)) / 2.0
            logging.debug('np_pairwise_dc = %s' % np.round(np_pairwise_dc, decimals=2))

            run_id = 'uw_symA_' + now + '_' + str(run)
            l_work_trial.append(run_id)
            dffsn_loss_w_range = None
            J_prob_range = None
            tv_loss_w_range = [1.0]
            bv_loss_w_range = None
            lv_loss_w_range = None
            lg_loss_w_range = None
            gs_loss_w_range = None
            max_epoch = 1000
            save_int = True
            save_folder = g_work_dir
            show_img = False
            show_init_img = True
            do_cluster = True
            d_gt = d_gt
            k_cluster = k_cluster
            cluster_alg = 'spectral'

            configured_run(run_id, nx_graph, np_init_embed, embed_dim,
                           dffsn_loss_w_range, J_prob_range, tv_loss_w_range,
                           bv_loss_w_range, lv_loss_w_range, lg_loss_w_range, gs_loss_w_range,
                           max_epoch=max_epoch, save_int=save_int, save_folder=save_folder, show_img=show_img,
                           show_init_img=show_init_img,
                           do_cluster=do_cluster, d_gt=d_gt, k_cluster=k_cluster, cluster_alg=cluster_alg)

        with open(g_work_dir + 'work_trial_' + now + '.txt', 'w+') as out_fd:
            out_str = '\n'.join(l_work_trial)
            out_fd.write(out_str)
            out_fd.close()

    elif cmd == 'spectral':
        '''
        Temporarily obsoleted
        '''
        plt.set_loglevel('error')
        graph_name = 'uw_symA'
        work_trial_name = 'work_trial_20210318123648'
        # l_metrics = ['dffsn_loss', 'tv_loss', 'bv_loss', 'lv_loss', 'lg_loss', 'gs_loss']
        l_metrics = ['tv_loss']

        # >>> LOAD IN GRAPH <<<
        nx_graph = nx.read_gpickle(graph_name + '.pickle')

        # >>> COMPUTE EIGEN STUFF <<<
        l_eigs = node_embed_spectrum(nx_graph, use_norm=False, draw_eigs=True)
        eigvec_pw_ele_diff_save_path = g_work_dir + graph_name + '_eigvec_pw_ele_diff_mats.pickle'
        eigenvec_pw_ele_diff_matrix(l_eigs, eigvec_pw_ele_diff_save_path, nx_graph,
                                    draw_diff_mat=True,
                                    draw_name=graph_name + '_eigvec_pw_ele_diff_mat',
                                    show_img=False,
                                    save_img=True,
                                    img_path=g_work_dir + graph_name + '_eigvec_pw_ele_diff_mat.png')

        # >>> COMPUTE COMPARISON BASED GRAPH FOURIER <<<
        df_eigvec_ele_pw_diff = pd.read_pickle(eigvec_pw_ele_diff_save_path)
        l_trial_folder = []
        with open(g_work_dir + work_trial_name + '.txt', 'r') as in_fd:
            for ln in in_fd:
                l_trial_folder.append(ln.strip())
            in_fd.close()
        for trial_folder in l_trial_folder:
            for (dirpath, dirname, filenames) in walk(g_work_dir + 'ne_run_' + trial_folder):
                for filename in filenames:
                    if filename[-14:] != 'init_embed.npy' or filename[:6] != 'ne_run':
                        continue
                    np_init_embed = np.load(dirpath + '/' + filename)
                    df_init_comp_fourier = comparison_based_graph_fourier(df_eigvec_ele_pw_diff, np_init_embed,
                                                                          nx_graph)

            for (dirpath, dirname, filenames) in walk(g_work_dir + 'ne_run_' + trial_folder):
                for filename in filenames:
                    if filename[-7:] != '.pickle' or filename[:6] != 'ne_run':
                        continue
                    run_name = filename.split('@')[1].strip()
                    param_str = filename.split('@')[2].strip()[:-7]
                    l_comp_based_fourier = []
                    df_embed_rec = pd.read_pickle(dirpath + '/' + filename)
                    for _, embed_rec in df_embed_rec.iterrows():
                        epoch = embed_rec['epoch']
                        np_embed = embed_rec['node_embed_int']
                        l_per_epoch_metrics = []
                        for metric_str in l_metrics:
                            l_per_epoch_metrics.append(embed_rec[metric_str])
                        df_comp_fourier = comparison_based_graph_fourier(df_eigvec_ele_pw_diff, np_embed, nx_graph)
                        l_comp_based_fourier.append(df_comp_fourier)
                    l_comp_based_fourier.append(df_init_comp_fourier)
                    save_path = dirpath + '/' + 'ne_cbf_run@' + run_name + '@' + param_str + '.png'
                    draw_comparison_based_graph_fourier_over_epoch(l_comp_based_fourier, param_str, save_ret=True,
                                                                   show_img=False, save_path=save_path)

    elif cmd == 'spectral_specific':
        '''
        TEST ONLY!!!
        '''
        plt.set_loglevel('error')
        graph_name = 'uw_symA'
        work_trial_name = 'work_trial_20210318123648'
        # l_metrics = ['dffsn_loss', 'tv_loss', 'bv_loss', 'lv_loss', 'lg_loss', 'gs_loss']
        l_metrics = ['tv_loss']

        # >>> LOAD IN GRAPH <<<
        nx_graph = nx.read_gpickle(graph_name + '.pickle')

        # >>> LOAD IN NODE EMBEDDING <<<
        np_embed = np.load(
            g_work_dir + 'ne_run_uw_symA_20210319113328_0/ne_run@uw_symA_20210319113328_0@init_embed.npy')

        # # >>> COMPUTE EIGEN STUFF <<<
        eigvec_pw_ele_diff_save_path = g_work_dir + graph_name + '_eigvec_pw_ele_diff_mats.pickle'
        l_eigs = node_embed_spectrum(nx_graph, use_norm=False, draw_eigs=True)
        eigenvec_pw_ele_diff_matrix(l_eigs, eigvec_pw_ele_diff_save_path, nx_graph,
                                    draw_diff_mat=True,
                                    draw_name=graph_name + '_eigvec_pw_ele_diff_mat',
                                    show_img=False,
                                    save_img=True,
                                    img_path=g_work_dir + graph_name + '_eigvec_pw_ele_diff_mat.png')

        # >>> COMPUTE COMPARISON BASED GRAPH FOURIER <<<
        df_eigvec_ele_pw_diff = pd.read_pickle(eigvec_pw_ele_diff_save_path)
        df_comp_fourier = comparison_based_graph_fourier(df_eigvec_ele_pw_diff, np_embed, nx_graph)
        np_eigvalmags = np.asarray(df_comp_fourier['eigvalmag'].to_list())
        param_str = 'manual_embed'
        save_path = g_work_dir + 'ne_run_uw_symA_20210319113328_0/' + 'ne_cbf_run@' + '20210319113328_0' + '@' + param_str + '.png'
        fig, axes = plt.subplots()
        # fig.suptitle(param_str, fontsize=10)
        axes.grid(True)
        axes.set_title('Manual Embed CBGF', fontsize=10)
        markerline, stemlines, baselineaxes = axes.stem([i for i in range(len(np_eigvalmags))], np_eigvalmags)
        axes.set_xticks([i for i in range(len(np_eigvalmags))])
        axes.set_yticks(np.arange(0.0, 1.0, step=0.2))
        plt.tight_layout(pad=1.0)
        # plt.subplots_adjust(top=0.96)
        plt.savefig(save_path, format='PNG')
        plt.clf()
        print()

    elif cmd == 'spectral_sequence':
        # >>> CONFIGURATION <<<
        plt.set_loglevel('error')
        graph_name = 'complete_graph'
        decomp = 'svd'
        val_weighted = False
        adj_only = True
        exp_EADMs = False
        max_k = None
        unweighted = True
        use_norm_L = False
        top_M_eigs = None
        rm_dc = False
        norm_FEADMs = True
        spec_seq_param_name = 'svd#adj#nfeadms'
        # >>> 'def': decomp='qr' max_k=None, unweighted=True, use_norm_L=False, top_M_eigs=None, rm_dc=False, exp_EADMs=True
        # >>> 'noexp': decomp='qr' max_k=None, unweighted=True, use_norm_L=False, top_M_eigs=None, rm_dc=False, exp_EADMs=False
        # >>> 'svd#noexp': decomp='svd' max_k=None, unweighted=True, use_norm_L=False, top_M_eigs=None, rm_dc=False, exp_EADMs=False

        # >>> 'svd#adj': decomp='svd', val_weighted=False, adj_only=True, max_k=None, unweighted=True, use_norm_L=False, top_M_eigs=None, rm_dc=False, exp_EADMs=True
        # >>> 'svd#vw': decomp='svd', val_weighted=True, adj_only=False, max_k=None, unweighted=True, use_norm_L=False, top_M_eigs=None, rm_dc=False, exp_EADMs=False
        # >>> 'svd#vw#exp': decomp='svd', val_weighted=True, adj_only=False, max_k=None, unweighted=True, use_norm_L=False, top_M_eigs=None, rm_dc=False, exp_EADMs=True
        # >>> 'svd#adj#exp#nfeadms': decomp='svd', val_weighted=False, adj_only=True, max_k=None, unweighted=True, use_norm_L=False, top_M_eigs=None, rm_dc=False, exp_EADMs=True, norm_FEADMs=True
        # >>> 'svd#adj#nfeadms': decomp='svd', val_weighted=False, adj_only=True, max_k=None, unweighted=True, use_norm_L=False, top_M_eigs=None, rm_dc=False, exp_EADMs=False, norm_FEADMs=True
        save_path = g_work_dir + graph_name + '_spectral_sequence@%s.pickle' % spec_seq_param_name

        # >>> LOAD IN GRAPH <<<
        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')

        # >>> COMPUTE SPECTRAL SEQUENCE <<<
        graph_spectra_sequence(nx_graph, save_path,
                               decomp=decomp,
                               adj_only=adj_only,
                               val_weighted=val_weighted,
                               exp_EADMs=exp_EADMs,
                               max_k=None,
                               unweighted=True,
                               use_norm_L=False,
                               top_M_eigs=None,
                               rm_dc=False,
                               norm_FEADMs=norm_FEADMs)

    elif cmd == 'draw_spectral_sequence':
        # >>> CONFIGURATION <<<
        plt.set_loglevel('error')
        graph_name = 'complete_graph'
        spec_seq_param_name = 'svd#adj#nfeadms'
        save_img = True
        show_img = False
        save_folder = g_work_dir + graph_name + '_spectral_sequence@%s/' % spec_seq_param_name
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # >>> LOAD IN GRAPH <<<
        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')

        # >>> LOAD IN SPECTRAL SEQUENCE <<<
        df_spectral_seq = pd.read_pickle(g_work_dir + graph_name + '_spectral_sequence@%s.pickle' % spec_seq_param_name)

        # >>> DRAW SPECTRAL SEQUENCE <<<
        draw_graph_spectra_sequence(df_spectral_seq, nx_graph, graph_name, save_folder, line_graph=False,
                                    save_img=save_img, show_img=show_img)

    elif cmd == 'convert_ln_eigs_to_vx_eigs':
        graph_name = 'uw_symA'
        spec_seq_param_name = 'svd#adj#nfeadms'
        loss_threshold = None
        max_epoch = 1000
        use_cuda = True
        save_ret = True
        save_path = g_work_dir + graph_name + '_spectral_sequence@' + spec_seq_param_name + '/'

        spec_seq_path = g_work_dir + graph_name + '_spectral_sequence@%s.pickle' % spec_seq_param_name
        df_spectral_seq = pd.read_pickle(spec_seq_path)

        convert_ln_eigs_to_vx_eigs(df_spectral_seq, loss_threshold=loss_threshold, max_epoch=max_epoch,
                                   use_cuda=use_cuda, save_ret=save_ret, save_path=save_path)

    elif cmd == 'analyze_signal_against_spectral_sequence':
        # >>> CONFIGURATION <<<
        plt.set_loglevel('error')
        graph_name = 'uw_symA'
        spec_seq_param_name = 'svd#adj#nfeadms'
        work_trial_name = '20210330234106'
        run_name = graph_name + '_' + work_trial_name
        # l_metrics = ['dffsn_loss', 'tv_loss', 'bv_loss', 'lv_loss', 'lg_loss', 'gs_loss']
        l_metrics = ['tv_loss']
        save_ret = False
        exp_pw_d = False
        norm_sig_FADM = False
        good_vs_bad = True
        use_manual_pw_dist = False
        pw_dist_type = 'uw_symA_pulse_B'

        # >>> LOAD IN GRAPH <<<
        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')

        # >>> LOAD IN SPECTRAL SEQUENCE <<<
        df_spectral_seq = pd.read_pickle(g_work_dir + graph_name + '_spectral_sequence@%s.pickle' % spec_seq_param_name)

        # >>> ANALYZE SPRECTRA OF NODE EMBEDDING <<<
        if good_vs_bad:
            l_trial_folder = [
                g_work_dir + 'experiments/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@good_vs_bad/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@bad_rets/']
        else:
            l_trial_folder = [g_work_dir + 'ne_run_' + graph_name + '_' + work_trial_name]

        # >>> MAKE UP MANUAL PAIRWISE DISTANCE MATRIX
        if use_manual_pw_dist:
            np_pw_dist = manual_pw_distance_mat(df_spectral_seq, nx_graph, pw_dist_type)
            df_analysis = analyze_vector_signal_against_spectral_sequence(None,
                                                                          df_spectral_seq,
                                                                          save_ret=save_ret,
                                                                          save_path=None,
                                                                          np_pw_dist=np_pw_dist,
                                                                          exp_pw_d=exp_pw_d,
                                                                          norm_sig_FADM=norm_sig_FADM)
            l_analysis_per_run = [(0, df_analysis, None, None, None, np_pw_dist)]
            save_path = g_work_dir + 'ne_run_' + run_name + '/' + 'ne_spec_seq_run@' + run_name + '@' \
                        + 'man_pw_d' + '#' + pw_dist_type + '.pickle'
            df_analysis_per_run = pd.DataFrame(l_analysis_per_run,
                                               columns=['epoch', 'df_analysis', 'metric_names', 'metric_vals', 'np_ne',
                                                        'np_pw_dist'])
            df_analysis_per_run = df_analysis_per_run.set_index('epoch')
            pd.to_pickle(df_analysis_per_run, save_path)
            sys.exit()

        for trial_folder in l_trial_folder:
            for (dirpath, dirname, filenames) in walk(trial_folder):
                for filename in filenames:
                    if filename[-7:] != '.pickle' or filename[:6] != 'ne_run':
                        continue
                    run_name = filename.split('@')[1].strip()
                    param_str = filename.split('@')[2].strip()[:-7]

                    init_embed_file_name = 'ne_run@' + run_name + '@init_embed.npy'
                    np_init_embed = np.load(dirpath + '/' + init_embed_file_name)
                    df_init_analysis = analyze_vector_signal_against_spectral_sequence(np_init_embed,
                                                                                       df_spectral_seq,
                                                                                       save_ret=save_ret,
                                                                                       save_path=None,
                                                                                       exp_pw_d=exp_pw_d)
                    l_analysis_per_run = [(0, df_init_analysis, None, None, np_init_embed, None)]

                    df_embed_rec = pd.read_pickle(dirpath + '/' + filename)
                    for _, embed_rec in df_embed_rec.iterrows():
                        epoch = embed_rec['epoch']
                        np_embed = embed_rec['node_embed_int']
                        l_per_epoch_metrics = []
                        for metric_str in l_metrics:
                            l_per_epoch_metrics.append(embed_rec[metric_str])
                        df_analysis = analyze_vector_signal_against_spectral_sequence(np_embed,
                                                                                      df_spectral_seq,
                                                                                      save_ret=save_ret,
                                                                                      save_path=None,
                                                                                      exp_pw_d=exp_pw_d)
                        l_analysis_per_run.append(
                            (epoch + 1, df_analysis, l_metrics, l_per_epoch_metrics, np_embed, None))
                    save_path = dirpath + '/' + 'ne_spec_seq_run@' + run_name + '@' + param_str + '.pickle'
                    # l_analysis_rec.append((l_analysis_per_run, run_name, param_str))
                    df_analysis_per_run = pd.DataFrame(l_analysis_per_run,
                                                       columns=['epoch', 'df_analysis', 'metric_names', 'metric_vals',
                                                                'np_ne', 'np_pw_dist'])
                    df_analysis_per_run = df_analysis_per_run.set_index('epoch')
                    pd.to_pickle(df_analysis_per_run, save_path)

    elif cmd == 'draw_signal_against_spectral_sequence_analysis':
        # >>> CONFIGURATION <<<
        plt.set_loglevel('error')
        graph_name = 'uw_symA'
        spec_seq_param_name = 'svd#adj#nfeadms'
        work_trial_name = '20210330234106'
        run_name = graph_name + '_' + work_trial_name.split('_')[-1].strip()
        init_only = False
        good_vs_bad = True

        if good_vs_bad:
            l_trial_folder = [
                g_work_dir + 'experiments/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@good_vs_bad/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@bad_rets/']
        else:
            l_trial_folder = [g_work_dir + 'ne_run_' + graph_name + '_' + work_trial_name]

        # >>> LOAD IN GRAPH <<<
        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')

        # >>> LOAD IN SPECTRAL SEQUENCE <<<
        df_spectral_seq = pd.read_pickle(g_work_dir + graph_name + '_spectral_sequence@%s.pickle' % spec_seq_param_name)

        # >>> DRAW SPECTRAL COMPONENTS OF NODE EMBEDDING <<<
        for trial_folder in l_trial_folder:
            for (dirpath, dirname, filenames) in walk(trial_folder):
                for filename in filenames:
                    if filename[-7:] != '.pickle' or filename[:15] != 'ne_spec_seq_run':
                        continue
                    run_name = filename.split('@')[1].strip()
                    param_str = filename.split('@')[2].strip()[:-7]

                    df_analysis_per_run = pd.read_pickle(dirpath + '/' + filename)
                    img_save_path_fmt = dirpath + '/' + 'ne_spec_seq_run@' + run_name + '@' + param_str + '@{0}' + '.png'
                    draw_vector_signal_analysis_against_spectral_sequence_over_epoch(df_analysis_per_run,
                                                                                     df_spectral_seq,
                                                                                     nx_graph,
                                                                                     param_str,
                                                                                     init_only=init_only,
                                                                                     save_ret=True,
                                                                                     show_img=False,
                                                                                     save_path_fmt=img_save_path_fmt)

    elif cmd == 'sas_and_trans':
        graph_name = 'uw_symA'
        spec_seq_param_name = 'svd#adj#nfeadms'
        use_ln_to_vx_eig_convert = True
        save_ret = False
        save_path = g_work_dir + 'experiments/uw_symA_pulse_B/'

        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')

        spec_seq_path = g_work_dir + graph_name + '_spectral_sequence@%s.pickle' % spec_seq_param_name
        df_spectral_seq = pd.read_pickle(spec_seq_path)

        if use_ln_to_vx_eig_convert:
            ln_to_vx_eig_convert_path = g_work_dir + graph_name + '_spectral_sequence@%s/df_ln_to_vx_eigs_convert.pickle' % spec_seq_param_name
            df_ln_to_vx_eig_convert = pd.read_pickle(ln_to_vx_eig_convert_path)
        else:
            df_ln_to_vx_eig_convert = None

        vec_sig_path = None
        if vec_sig_path is not None:
            np_vec_sig = np.load(vec_sig_path)
        else:
            np_vec_sig = None

        pw_dist_type = 'uw_symA_pulse_B'
        np_pw_dist = manual_pw_distance_mat(df_spectral_seq, nx_graph, pw_dist_type)

        df_analysis = stratified_graph_spectra_and_transformations(np_vec_sig,
                                                                   df_spectral_seq,
                                                                   save_ret=save_ret,
                                                                   save_path=save_path,
                                                                   np_pw_dist=np_pw_dist,
                                                                   norm_sig_FADM=False,
                                                                   use_ln_to_vx_eig_convert=use_ln_to_vx_eig_convert,
                                                                   df_ln_to_vx_eig_convert=df_ln_to_vx_eig_convert)
        draw_sas_and_trans_rets(df_analysis, save_path=save_path, save_img=True, show_img=False)

    elif cmd == 'compare_two_sigs':
        plt.set_loglevel('error')
        graph_name = 'uw_symA'
        spec_seq_param_name = 'svd#adj#nfeadms'
        use_ln_to_vx_eig_convert = True
        save_ret = True
        save_path = g_work_dir + 'experiments/uw_symA_pulse_B_vs_heat_filtered/'
        sig_1_label = 'pulse@B'
        sig_2_label = 'heat_filtered'

        spec_seq_path = g_work_dir + graph_name + '_spectral_sequence@%s.pickle' % spec_seq_param_name
        df_spectral_seq = pd.read_pickle(spec_seq_path)
        nx_graph = nx.read_gpickle(g_work_dir + graph_name + '.pickle')

        if use_ln_to_vx_eig_convert:
            ln_to_vx_eig_convert_path = g_work_dir + graph_name + '_spectral_sequence@%s/df_ln_to_vx_eigs_convert.pickle' % spec_seq_param_name
            df_ln_to_vx_eig_convert = pd.read_pickle(ln_to_vx_eig_convert_path)
        else:
            df_ln_to_vx_eig_convert = None

        pw_dist_type_1 = 'uw_symA_pulse_B'
        np_pw_dist_1 = manual_pw_distance_mat(df_spectral_seq, nx_graph, pw_dist_type_1)
        pw_dist_type_2 = 'uw_symA_pulse_B_heat'
        np_pw_dist_2 = manual_pw_distance_mat(df_spectral_seq, nx_graph, pw_dist_type_2)

        compare_two_sigs_with_sas_and_trans(df_spectral_seq,
                                            None, None,
                                            sig_1_label=sig_1_label, sig_2_label=sig_2_label,
                                            np_pw_dist_1=np_pw_dist_1, np_pw_dist_2=np_pw_dist_2,
                                            use_ln_to_vx_eig_convert=use_ln_to_vx_eig_convert,
                                            df_ln_to_vx_eig_convert=df_ln_to_vx_eig_convert,
                                            norm_sig_FADM=False,
                                            save_ret=save_ret, save_path=save_path,
                                            save_img=True, show_img=False)

    elif cmd == 'classic_gsp_on_sgs':
        plt.set_loglevel('error')
        graph_name = 'uw_symA'
        spec_seq_param_name = 'svd#adj#nfeadms'
        save_path = g_work_dir + 'experiments/uw_symA_pulse_B_vs_heat_filtered/'
        sig_1_label = 'pulse@B'
        sig_2_label = 'heat_filtered'

        spec_seq_path = g_work_dir + graph_name + '_spectral_sequence@%s.pickle' % spec_seq_param_name
        df_spectral_seq = pd.read_pickle(spec_seq_path)

        np_sig = np.load(g_work_dir + 'experiments/uw_symA_pulse_B_vs_heat_filtered/' + 'np_sig.npy')
        np_sig_filtered = np.load(g_work_dir + 'experiments/uw_symA_pulse_B_vs_heat_filtered/' + 'np_sig_filtered.npy')

        classic_gsp_on_stratified_graphs(df_spectral_seq,
                                         graph_sig_1=np_sig,
                                         graph_sig_2=np_sig_filtered,
                                         sig_1_label='pulse@B',
                                         sig_2_label='heat_filtered',
                                         save_path=save_path,
                                         save_img=True,
                                         show_img=True)

    elif cmd == 'compare_classic_gsp_with_sas_and_trans':
        plt.set_loglevel('error')
        job_id = '100_rand_graphs'
        # job_id = '100_rand_graphs_rand_pulse'
        # job_id = '100_sbm_graphs'
        # job_id = '100_sbm_graphs_rand_pulse'

        STAGE_rand_graphs = False
        STAGE_graph_sigs = False
        STAGE_graph_sig_pw_dist = False
        STAGE_spec_seqs = False
        STAGE_ln_to_vx_eigs = False
        STAGE_classic_gsp = False
        STAGE_sas_and_trans = True
        STAGE_ln_vx_only = False
        STAGE_classic_gsp_vs_ln_vx_only = False
        STAGE_draw_classic_gsp_vs_ln_vx_only = False
        STAGE_classic_gsp_vs_sas_and_trans = True
        STAGE_draw_classic_gsp_vs_sas_and_trans = False
        STAGE_draw_combined_gsp_vs_sas_rets = True
        STAGE_ln_to_vs_eigs_mse_stats = False
        STAGE_sgs_pairwise_comparisons = False
        STAGE_draw_sgs_pairwise_comparisons = False
        STAGE_spec_seq_sg_stats = False

        if STAGE_rand_graphs:
            rand_graph_type = 'rand'
            num_rand_graphs = 100
            num_nodes = 50
            edge_density = 0.05
            rand_graph_save_ret = True
            rand_graph_save_path = g_work_dir + 'experiments/gsp_vs_sas/rand_graphs/'
            if not os.path.exists(rand_graph_save_path):
                os.mkdir(rand_graph_save_path)
            gen_rand_graphs(rand_graph_type=rand_graph_type, num_rand_graphs=num_rand_graphs, num_nodes=num_nodes,
                            edge_density=edge_density, graph_name_prefix=None, save_ret=rand_graph_save_ret,
                            save_path=rand_graph_save_path, job_id=job_id)

        if STAGE_graph_sigs:
            max_sig_val = 1.0
            min_sig_val = -1.0
            sig_type = 'uniform'
            graph_sig_save_path = g_work_dir + 'experiments/gsp_vs_sas/graph_sigs/'
            if not os.path.exists(graph_sig_save_path):
                os.mkdir(graph_sig_save_path)

            graph_set_path = g_work_dir + 'experiments/gsp_vs_sas/rand_graphs/%s#rand_graphs.pickle' % str(job_id)
            df_graphs = pd.read_pickle(graph_set_path)

            l_graph_sig_rec = []
            for graph_name, graph_rec in df_graphs.iterrows():
                nx_graph = graph_rec['nx_graph']
                np_graph_sig = gen_rand_sigs(nx_graph, sig_type=sig_type, max_sig_val=max_sig_val,
                                             min_sig_val=min_sig_val)
                l_graph_sig_rec.append((graph_name, np_graph_sig))

            df_graph_sigs = pd.DataFrame(l_graph_sig_rec, columns=['graph_name', 'graph_sig'])
            df_graph_sigs = df_graph_sigs.set_index('graph_name')
            pd.to_pickle(df_graph_sigs, graph_sig_save_path + '%s#graph_sigs.pickle' % str(job_id))

        if STAGE_graph_sig_pw_dist:
            graph_sig_save_path = g_work_dir + 'experiments/gsp_vs_sas/graph_sigs/'
            df_graph_sigs = pd.read_pickle(graph_sig_save_path + '%s#graph_sigs.pickle' % str(job_id))

            l_sig_pw_dist_rec = []
            for graph_name, graph_sig_rec in df_graph_sigs.iterrows():
                np_sig = graph_sig_rec['graph_sig']
                np_pw_dist = compute_pw_dist_from_sig(np_sig)
                l_sig_pw_dist_rec.append((graph_name, np_pw_dist))
            df_sig_pw_dist = pd.DataFrame(l_sig_pw_dist_rec, columns=['graph_name', 'graph_sig_pw_dist'])
            df_sig_pw_dist = df_sig_pw_dist.set_index('graph_name')
            pd.to_pickle(df_sig_pw_dist, graph_sig_save_path + '%s#graph_sig_pw_dists.pickle' % str(job_id))

        if STAGE_spec_seqs:
            decomp = 'svd'
            val_weighted = False
            adj_only = True
            exp_EADMs = False
            max_k = None
            unweighted = True
            use_norm_L = False
            top_M_eigs = None
            rm_dc = False
            norm_FEADMs = True
            spec_seq_param_name = 'svd#adj#nfeadms'
            spec_seq_save_path = g_work_dir + 'experiments/gsp_vs_sas/spec_seqs/'
            if not os.path.exists(spec_seq_save_path):
                os.mkdir(spec_seq_save_path)

            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            for job_id in l_job_ids:
                graph_set_path = g_work_dir + 'experiments/gsp_vs_sas/rand_graphs/%s#rand_graphs.pickle' % str(job_id)
                df_graphs = pd.read_pickle(graph_set_path)

                l_spec_seq_rec = []
                for graph_name, graph_rec in df_graphs.iterrows():
                    nx_graph = graph_rec['nx_graph']
                    df_spec_seq = graph_spectra_sequence(nx_graph, None,
                                                         decomp=decomp,
                                                         adj_only=adj_only,
                                                         val_weighted=val_weighted,
                                                         exp_EADMs=exp_EADMs,
                                                         max_k=None,
                                                         unweighted=True,
                                                         use_norm_L=False,
                                                         top_M_eigs=None,
                                                         rm_dc=False,
                                                         norm_FEADMs=norm_FEADMs)
                    l_spec_seq_rec.append((graph_name, df_spec_seq))
                df_spec_seq = pd.DataFrame(l_spec_seq_rec, columns=['graph_name', 'df_spec_seq'])
                df_spec_seq = df_spec_seq.set_index('graph_name')
                pd.to_pickle(df_spec_seq, spec_seq_save_path + '%s#spec_seqs.pickle' % str(job_id))

        if STAGE_classic_gsp:
            classic_gsp_save_path = g_work_dir + 'experiments/gsp_vs_sas/classic_gsp/'
            if not os.path.exists(classic_gsp_save_path):
                os.mkdir(classic_gsp_save_path)
            spec_seq_param_name = 'svd#adj#nfeadms'
            spec_seq_save_path = g_work_dir + 'experiments/gsp_vs_sas/spec_seqs/'
            df_spec_seq = pd.read_pickle(spec_seq_save_path + '%s#spec_seqs.pickle' % str(job_id))

            graph_sig_save_path = g_work_dir + 'experiments/gsp_vs_sas/graph_sigs/'
            df_graph_sigs = pd.read_pickle(graph_sig_save_path + '%s#graph_sigs.pickle' % str(job_id))

            l_gsp_rec = []
            for graph_name, spec_seq_rec in df_spec_seq.iterrows():
                df_spec_seq_per_graph = spec_seq_rec['df_spec_seq']
                np_sig = df_graph_sigs.loc[graph_name]['graph_sig']

                df_gsp = classic_gsp_on_stratified_graphs(df_spec_seq_per_graph,
                                                          graph_sig_1=np_sig,
                                                          graph_sig_2=None,
                                                          sig_1_label=None,
                                                          sig_2_label=None,
                                                          save_ret=False,
                                                          save_path=None,
                                                          save_img=False,
                                                          show_img=False)
                l_gsp_rec.append((graph_name, df_gsp))
            df_classic_gsp = pd.DataFrame(l_gsp_rec, columns=['graph_name', 'df_gsp'])
            df_classic_gsp = df_classic_gsp.set_index('graph_name')
            pd.to_pickle(df_classic_gsp, classic_gsp_save_path + '%s#classic_gsp.pickle' % str(job_id))

        if STAGE_ln_to_vx_eigs:
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            loss_threshold = 0.000001
            max_epoch = 1000
            use_cuda = True
            num_trials = 50
            spec_seq_save_path = g_work_dir + 'experiments/gsp_vs_sas/spec_seqs/'

            for trial_id in range(num_trials):
                for job_id in l_job_ids:
                    df_spec_seq = pd.read_pickle(spec_seq_save_path + '%s#spec_seqs.pickle' % str(job_id))
                    l_ln_to_vx_eigs_rec = []
                    for graph_name, spec_seq_rec in df_spec_seq.iterrows():
                        df_spec_seq_per_graph = spec_seq_rec['df_spec_seq']
                        df_ln_to_vx = convert_ln_eigs_to_vx_eigs(df_spec_seq_per_graph,
                                                                 loss_threshold=loss_threshold,
                                                                 max_epoch=max_epoch,
                                                                 use_cuda=use_cuda,
                                                                 save_ret=False,
                                                                 save_path=None)
                        l_ln_to_vx_eigs_rec.append((graph_name, df_ln_to_vx))
                    df_ln_to_vx_eigs = pd.DataFrame(l_ln_to_vx_eigs_rec, columns=['graph_name', 'df_ln_to_vx_eigs'])
                    df_ln_to_vx_eigs = df_ln_to_vx_eigs.set_index('graph_name')
                    pd.to_pickle(df_ln_to_vx_eigs, spec_seq_save_path + '%s#%s#ln_to_vx_eigs.pickle' % (job_id, trial_id))
                    logging.debug('[STAGE_ln_to_vx_eigs] %s:%s done' % (job_id, trial_id))

        if STAGE_ln_to_vs_eigs_mse_stats:
            show_img = True
            save_img = True
            ln_to_vs_eigs_stats_save_path = g_work_dir + 'experiments/gsp_vs_sas/ln_to_vs_eigs_stats/'
            if not os.path.exists(ln_to_vs_eigs_stats_save_path):
                os.mkdir(ln_to_vs_eigs_stats_save_path)
            spec_seq_save_path = g_work_dir + 'experiments/gsp_vs_sas/spec_seqs/'
            df_ln_to_vx_eigs = pd.read_pickle(spec_seq_save_path + '%s#ln_to_vx_eigs.pickle' % str(job_id))

            d_mse_by_K = dict()
            for graph_name, df_ln_to_vs_eigs_rec in df_ln_to_vx_eigs.iterrows():
                df_ln_to_vx_eigs_per_graph = df_ln_to_vs_eigs_rec['df_ln_to_vx_eigs']
                for K, ln_to_vs_eigs_rec in df_ln_to_vx_eigs_per_graph.iterrows():
                    mse_loss = ln_to_vs_eigs_rec['mse_loss']
                    if K not in d_mse_by_K:
                        d_mse_by_K[K] = [mse_loss]
                    else:
                        d_mse_by_K[K].append(mse_loss)

            l_mse_mean = []
            l_mse_std = []
            for K in d_mse_by_K:
                l_mse_mean.append(np.mean(d_mse_by_K[K]))
                l_mse_std.append(np.std(d_mse_by_K[K]))

            if job_id == '100_rand_graphs':
                img_title = 'ERM-Rand'
            elif job_id == '100_rand_graphs_rand_pulse':
                img_title = 'ERM-Pulse'
            elif job_id == '100_sbm_graphs':
                img_title = 'SBM-Rand'
            elif job_id == '100_sbm_graphs_rand_pulse':
                img_title = 'SMB-Pulse'
            img_name = '%s#ln_to_vx_eigs_stats' % str(job_id)
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.grid(True)
            axes.set_title('%s' % img_title, fontsize=10, fontweight='semibold')
            axes.errorbar([i for i in range(len(l_mse_mean))], l_mse_mean, yerr=l_mse_std,
                               marker='o', fmt='o', mfc='tab:blue', capsize=2, capthick=1)
            axes.set_xticks([i for i in range(len(l_mse_mean))])
            axes.set_xticklabels(list(d_mse_by_K.keys()))
            # axes.set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(ln_to_vs_eigs_stats_save_path + img_name + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

        if STAGE_sas_and_trans:
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            use_ln_to_vx_eig_convert = False
            sas_and_trans_save_path = g_work_dir + 'experiments/gsp_vs_sas/sas_and_trans/'
            if not os.path.exists(sas_and_trans_save_path):
                os.mkdir(sas_and_trans_save_path)

            graph_sig_save_path = g_work_dir + 'experiments/gsp_vs_sas/graph_sigs/'
            spec_seq_save_path = g_work_dir + 'experiments/gsp_vs_sas/spec_seqs/'

            for job_id in l_job_ids:
                df_graph_sig_pw_dist = pd.read_pickle(graph_sig_save_path + '%s#graph_sig_pw_dists.pickle' % str(job_id))
                df_spec_seq = pd.read_pickle(spec_seq_save_path + '%s#spec_seqs.pickle' % str(job_id))
                # df_ln_to_vx_eigs = pd.read_pickle(spec_seq_save_path + '%s#ln_to_vx_eigs.pickle' % str(job_id))

                l_sas_and_trans_rec = []
                for graph_name, graph_sig_pw_dist_rec in df_graph_sig_pw_dist.iterrows():
                    np_sig_pw_dist = graph_sig_pw_dist_rec['graph_sig_pw_dist']
                    df_spec_seq_per_graph = df_spec_seq.loc[graph_name]['df_spec_seq']
                    # df_ln_to_vx_eigs_per_graph = df_ln_to_vx_eigs.loc[graph_name]['df_ln_to_vx_eigs']
                    df_analysis = stratified_graph_spectra_and_transformations(None,
                                                                               df_spec_seq_per_graph,
                                                                               save_ret=False,
                                                                               save_path=None,
                                                                               np_pw_dist=np_sig_pw_dist,
                                                                               norm_sig_FADM=False,
                                                                               use_ln_to_vx_eig_convert=use_ln_to_vx_eig_convert,
                                                                               df_ln_to_vx_eig_convert=None)
                    l_sas_and_trans_rec.append((graph_name, df_analysis))
                df_sas_and_trans = pd.DataFrame(l_sas_and_trans_rec, columns=['graph_name', 'df_sas_and_trans'])
                df_sas_and_trans = df_sas_and_trans.set_index('graph_name')
                pd.to_pickle(df_sas_and_trans, sas_and_trans_save_path + '%s#sas_and_trans.pickle' % str(job_id))

        if STAGE_ln_vx_only:
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            num_ln_vx_trials = 50
            ln_vx_only_save_path = g_work_dir + 'experiments/gsp_vs_sas/ln_vx_only/'
            if not os.path.exists(ln_vx_only_save_path):
                os.mkdir(ln_vx_only_save_path)

            graph_sig_save_path = g_work_dir + 'experiments/gsp_vs_sas/graph_sigs/'
            spec_seq_save_path = g_work_dir + 'experiments/gsp_vs_sas/spec_seqs/'

            for job_id in l_job_ids:
                df_graph_sig_pw_dist = pd.read_pickle(graph_sig_save_path + '%s#graph_sig_pw_dists.pickle' % str(job_id))
                df_spec_seq = pd.read_pickle(spec_seq_save_path + '%s#spec_seqs.pickle' % str(job_id))

                d_ln_vx_rec = dict()
                for ln_vx_id in range(num_ln_vx_trials):
                    df_ln_to_vx_eigs = pd.read_pickle(spec_seq_save_path + '%s#%s#ln_to_vx_eigs.pickle' % (job_id, ln_vx_id))
                    for graph_name, graph_sig_pw_dist_rec in df_graph_sig_pw_dist.iterrows():
                        df_ln_to_vx_eigs_per_graph = df_ln_to_vx_eigs.loc[graph_name]['df_ln_to_vx_eigs']
                        df_spec_seq_per_graph = df_spec_seq.loc[graph_name]['df_spec_seq']
                        np_sig_pw_dist = graph_sig_pw_dist_rec['graph_sig_pw_dist']
                        df_analysis = ln_vx(None, df_spec_seq_per_graph, df_ln_to_vx_eigs_per_graph, np_pw_dist=np_sig_pw_dist)
                        if graph_name not in d_ln_vx_rec:
                            d_ln_vx_rec[graph_name] = dict()
                        for K, ana_rec in df_analysis.iterrows():
                            ln_vx_trans = ana_rec['sig_pwd_rec_vx_ft']
                            if K not in d_ln_vx_rec[graph_name]:
                                d_ln_vx_rec[graph_name][K] = [ln_vx_trans]
                            else:
                                d_ln_vx_rec[graph_name][K].append(ln_vx_trans)

                l_ln_vx_per_job = []
                for graph_name in d_ln_vx_rec:
                    l_ln_vx_per_job.append((graph_name, d_ln_vx_rec[graph_name]))
                df_ln_vx_per_job = pd.DataFrame(l_ln_vx_per_job, columns=['graph_name', 'd_ln_vx_by_K'])
                df_ln_vx_per_job = df_ln_vx_per_job.set_index('graph_name')
                pd.to_pickle(df_ln_vx_per_job, ln_vx_only_save_path + '%s#ln_vx_only_by_job.pickle' % str(job_id))
                logging.debug('[STAGE_ln_vx_only] %s done.' % str(job_id))
            logging.debug('[STAGE_ln_vx_only] all done.')

        if STAGE_classic_gsp_vs_ln_vx_only:
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            num_ln_vx_trials = 50
            ln_vx_only_save_path = g_work_dir + 'experiments/gsp_vs_sas/ln_vx_only/'
            classic_gsp_save_path = g_work_dir + 'experiments/gsp_vs_sas/classic_gsp/'

            for job_id in l_job_ids:
                df_classic_gsp = pd.read_pickle(classic_gsp_save_path + '%s#classic_gsp.pickle' % str(job_id))
                df_ln_vx_per_job = pd.read_pickle(ln_vx_only_save_path + '%s#ln_vx_only_by_job.pickle' % str(job_id))

                l_gsp_vs_ln_vx = []
                for graph_name, classic_gsp_rec in df_classic_gsp.iterrows():
                    df_gsp_per_graph = classic_gsp_rec['df_gsp']
                    d_ln_vx_by_K = df_ln_vx_per_job.loc[graph_name]['d_ln_vx_by_K']
                    d_sim = dict()
                    for K, gsp_rec in df_gsp_per_graph.iterrows():
                        np_sig_ft_classic = gsp_rec['graph_sig_hat_1']
                        np_sig_ft_classic_hat = preprocessing.normalize(np.abs(np_sig_ft_classic).reshape(1, -1))[0]
                        l_ln_vx = d_ln_vx_by_K[K]
                        np_ln_vx = np.stack(l_ln_vx)
                        np_ln_vx_hat = preprocessing.normalize(np_ln_vx)
                        np_sim = np.matmul(np_ln_vx_hat, np_sig_ft_classic_hat)
                        np_sim[np_sim < -1.0] = -1.0
                        np_sim[np_sim > 1.0] = 1.0
                        if K not in d_sim:
                            d_sim[K] = np_sim
                        else:
                            raise Exception('[STAGE_classic_gsp_vs_ln_vx_only] repeated K')
                    l_gsp_vs_ln_vx.append((graph_name, d_sim))
                df_gsp_vs_ln_vx = pd.DataFrame(l_gsp_vs_ln_vx, columns=['graph_name', 'd_sim'])
                df_gsp_vs_ln_vx = df_gsp_vs_ln_vx.set_index('graph_name')
                pd.to_pickle(df_gsp_vs_ln_vx, ln_vx_only_save_path + '%s#gsp_vs_ln_vx_only.pickle' % str(job_id))
                logging.debug('[STAGE_classic_gsp_vs_ln_vx_only] %s done' % str(job_id))

        if STAGE_draw_classic_gsp_vs_ln_vx_only:
            save_img = True
            show_img = True
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            ln_vx_only_save_path = g_work_dir + 'experiments/gsp_vs_sas/ln_vx_only/'

            for job_id in l_job_ids:
                df_gsp_vs_ln_vx = pd.read_pickle(ln_vx_only_save_path + '%s#gsp_vs_ln_vx_only.pickle' % str(job_id))
                d_sim_mean_std_by_K = dict()
                for graph_name, gsp_vs_ln_vx_rec in df_gsp_vs_ln_vx.iterrows():
                    d_sim = gsp_vs_ln_vx_rec['d_sim']
                    for K in d_sim:
                        np_sim = d_sim[K]
                        sim_mean = np.mean(np_sim)
                        sim_std = np.std(np_sim)
                        if K not in d_sim_mean_std_by_K:
                            d_sim_mean_std_by_K[K] =[(sim_mean, sim_std)]
                        else:
                            d_sim_mean_std_by_K[K].append((sim_mean, sim_std))
                for K in d_sim_mean_std_by_K:
                    d_sim_mean_std_by_K[K] = sorted(d_sim_mean_std_by_K[K], key=lambda k: k[0], reverse=True)

                if job_id == '100_rand_graphs':
                    img_title = 'ERM-Rand'
                elif job_id == '100_rand_graphs_rand_pulse':
                    img_title = 'ERM-Pulse'
                elif job_id == '100_sbm_graphs':
                    img_title = 'SBM-Rand'
                elif job_id == '100_sbm_graphs_rand_pulse':
                    img_title = 'SMB-Pulse'
                l_K = list(d_sim_mean_std_by_K.keys())
                img_name = 'gsp_vs_ln_vx#' + str(job_id)
                fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(5, 15))
                fig.suptitle(img_title, fontsize=12, fontweight='semibold')
                idx = 0
                for K in l_K:
                    np_means = np.asarray([item[0] for item in d_sim_mean_std_by_K[K]])
                    np_stds = np.asarray([item[1] for item in d_sim_mean_std_by_K[K]])
                    axes[idx].grid(True)
                    axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                    axes[idx].plot([i for i in range(len(d_sim_mean_std_by_K[K]))], np_means, color='tab:blue')
                    axes[idx].fill_between([i for i in range(len(d_sim_mean_std_by_K[K]))],
                                           np_means - np_stds, np_means + np_stds, alpha=0.4, color='tab:blue')
                    if len(d_sim_mean_std_by_K[K]) >= 5:
                        axes[idx].set_xticks([i for i in range(0, len(d_sim_mean_std_by_K[K]) + math.ceil(len(d_sim_mean_std_by_K[K]) / 5),
                                                         math.ceil(len(d_sim_mean_std_by_K[K]) / 5))])
                        axes[idx].set_xticklabels([i for i in range(0, len(d_sim_mean_std_by_K[K]) + math.ceil(len(d_sim_mean_std_by_K[K]) / 5),
                                                                    math.ceil(len(d_sim_mean_std_by_K[K]) / 5))])
                    else:
                        axes[idx].set_xticks([i for i in range(len(d_sim_mean_std_by_K[K]))])
                        axes[idx].set_xticklabels([i for i in range(len(d_sim_mean_std_by_K[K]))])
                    idx += 1
                plt.tight_layout(pad=1.0)
                plt.subplots_adjust(top=0.95)
                if save_img:
                    plt.savefig(ln_vx_only_save_path + img_name + '.PNG', format='PNG')
                if show_img:
                    plt.show()
                plt.clf()
                plt.close()

        if STAGE_classic_gsp_vs_sas_and_trans:
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            no_in_agg = False
            norm_sig = True
            if norm_sig:
                out_file_suffix = '_norm'
            else:
                out_file_suffix = '_non_norm'
            if no_in_agg:
                out_file_suffix += '_no_in_agg'
            classic_gsp_vs_sas_and_trans_save_path = g_work_dir + 'experiments/gsp_vs_sas/classic_gsp_vs_sas_and_trans/'
            if not os.path.exists(classic_gsp_vs_sas_and_trans_save_path):
                os.mkdir(classic_gsp_vs_sas_and_trans_save_path)

            ln_vx_only_save_path = g_work_dir + 'experiments/gsp_vs_sas/ln_vx_only/'
            classic_gsp_save_path = g_work_dir + 'experiments/gsp_vs_sas/classic_gsp/'
            sas_and_trans_save_path = g_work_dir + 'experiments/gsp_vs_sas/sas_and_trans/'

            for job_id in l_job_ids:
                df_classic_gsp = pd.read_pickle(classic_gsp_save_path + '%s#classic_gsp.pickle' % str(job_id))
                df_sas_and_trans = pd.read_pickle(sas_and_trans_save_path + '%s#sas_and_trans.pickle' % str(job_id))
                df_ln_vx = pd.read_pickle(ln_vx_only_save_path + '%s#ln_vx_only_by_job.pickle' % str(job_id))

                l_classic_gsp_vs_sas_and_trans_rec = []
                for graph_name, classic_gsp_rec in df_classic_gsp.iterrows():
                    df_gsp_per_graph = classic_gsp_rec['df_gsp']
                    df_sas_and_trans_per_graph = df_sas_and_trans.loc[graph_name]['df_sas_and_trans']
                    d_ln_vx_by_K = df_ln_vx.loc[graph_name]['d_ln_vx_by_K']

                    l_sim_rec = []
                    for K, gsp_rec in df_gsp_per_graph.iterrows():
                        np_sig_ft_classic = gsp_rec['graph_sig_hat_1']
                        np_sig_ft_adj_diff = df_sas_and_trans_per_graph.loc[K]['sig_vs_eig_sas']
                        np_sig_ft_ln_agg = df_sas_and_trans_per_graph.loc[K]['sig_pwd_agg_vx_ft']
                        # np_sig_ft_ln_conv = df_sas_and_trans_per_graph.loc[K]['sig_pwd_rec_vx_ft']
                        np_sig_ft_apprx_ls = df_sas_and_trans_per_graph.loc[K]['apprx_ls_ft']
                        np_sig_ft_ln_conv = np.mean(d_ln_vx_by_K[K], axis=0)

                        l_eig_vals = df_sas_and_trans_per_graph.loc[K]['eig_vals']
                        l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                        num_eff_eig_vals = len(l_eff_eig_vals)
                        num_all_eig_vals = len(l_eig_vals)

                        if norm_sig:
                            np_sig_ft_classic_hat = preprocessing.normalize(np.abs(np_sig_ft_classic).reshape(1, -1))[0]
                            np_sig_ft_adj_diff_hat = preprocessing.normalize(np.abs(np_sig_ft_adj_diff).reshape(1, -1))[0]
                            np_sig_ft_ln_agg_hat = preprocessing.normalize(np.abs(np_sig_ft_ln_agg).reshape(1, -1))[0]
                            np_sig_ft_ln_conv_hat = preprocessing.normalize(np.abs(np_sig_ft_ln_conv).reshape(1, -1))[0]
                            np_sig_ft_apprx_ls = preprocessing.normalize(np.abs(np_sig_ft_apprx_ls).reshape(1, -1))[0]
                            if no_in_agg:
                                np_sig_ft_ens_hat = preprocessing.normalize(np.mean([np_sig_ft_adj_diff_hat,
                                                                                     np_sig_ft_ln_conv_hat,
                                                                                     np_sig_ft_apprx_ls], axis=0).reshape(1, -1))[0]
                            else:
                                np_sig_ft_ens_hat = preprocessing.normalize(np.mean([np_sig_ft_adj_diff_hat,
                                                                                     np_sig_ft_ln_agg_hat,
                                                                                     np_sig_ft_ln_conv_hat,
                                                                                     np_sig_ft_apprx_ls], axis=0).reshape(1, -1))[0]

                            sim_classic_vs_adj_diff = np.dot(np_sig_ft_classic_hat, np_sig_ft_adj_diff_hat)
                            if sim_classic_vs_adj_diff < -1.0:
                                sim_classic_vs_adj_diff = -1.0
                            elif sim_classic_vs_adj_diff > 1.0:
                                sim_classic_vs_adj_diff = 1.0

                            sim_classic_vs_ln_agg = np.dot(np_sig_ft_classic_hat, np_sig_ft_ln_agg_hat)
                            if sim_classic_vs_ln_agg < -1.0:
                                sim_classic_vs_ln_agg = -1.0
                            elif sim_classic_vs_ln_agg > 1.0:
                                sim_classic_vs_ln_agg = 1.0

                            sim_classic_vs_ln_conv = np.dot(np_sig_ft_classic_hat, np_sig_ft_ln_conv_hat)
                            if sim_classic_vs_ln_conv < -1.0:
                                sim_classic_vs_ln_conv = -1.0
                            elif sim_classic_vs_ln_conv > 1.0:
                                sim_classic_vs_ln_conv = 1.0

                            sim_classic_vs_apprx_ls = np.dot(np_sig_ft_classic_hat, np_sig_ft_apprx_ls)
                            if sim_classic_vs_apprx_ls < -1.0:
                                sim_classic_vs_apprx_ls = -1.0
                            elif sim_classic_vs_apprx_ls > 1.0:
                                sim_classic_vs_apprx_ls = 1.0

                            sim_classic_vs_ens = np.dot(np_sig_ft_classic_hat, np_sig_ft_ens_hat)
                            if sim_classic_vs_ens < -1.0:
                                sim_classic_vs_ens = -1.0
                            elif sim_classic_vs_ens > 1.0:
                                sim_classic_vs_ens = 1.0
                        else:
                            np_sig_ft_classic_hat = np.abs(np_sig_ft_classic)
                            np_sig_ft_adj_diff_hat = np.abs(np_sig_ft_adj_diff)
                            np_sig_ft_ln_agg_hat = np.abs(np_sig_ft_ln_agg)
                            np_sig_ft_ln_conv_hat = np.abs(np_sig_ft_ln_conv)
                            np_sig_ft_apprx_ls_hat = np.abs(np_sig_ft_apprx_ls)
                            if no_in_agg:
                                np_sig_ft_ens_hat = np.mean([np_sig_ft_adj_diff_hat, np_sig_ft_ln_conv_hat,
                                                             np_sig_ft_apprx_ls_hat], axis=0)
                            else:
                                np_sig_ft_ens_hat = np.mean([np_sig_ft_adj_diff_hat, np_sig_ft_ln_agg_hat,
                                                             np_sig_ft_ln_conv_hat, np_sig_ft_apprx_ls_hat], axis=0)
                            sim_classic_vs_adj_diff = np.dot(np_sig_ft_classic_hat, np_sig_ft_adj_diff_hat)
                            sim_classic_vs_ln_agg = np.dot(np_sig_ft_classic_hat, np_sig_ft_ln_agg_hat)
                            sim_classic_vs_ln_conv = np.dot(np_sig_ft_classic_hat, np_sig_ft_ln_conv_hat)
                            sim_classic_vs_apprx_ls = np.dot(np_sig_ft_classic_hat, np_sig_ft_apprx_ls_hat)
                            sim_classic_vs_ens = np.dot(np_sig_ft_classic_hat, np_sig_ft_ens_hat)

                        l_sim_rec.append((K, sim_classic_vs_adj_diff, sim_classic_vs_ln_agg, sim_classic_vs_ln_conv,
                                          sim_classic_vs_apprx_ls, sim_classic_vs_ens))
                    df_sim = pd.DataFrame(l_sim_rec, columns=['K', 'sim_classic_vs_adj_diff', 'sim_classic_vs_ln_agg',
                                                              'sim_classic_vs_ln_conv', 'sim_classic_vs_apprx_ls',
                                                              'sim_classic_vs_ens'])
                    df_sim = df_sim.set_index('K')
                    l_classic_gsp_vs_sas_and_trans_rec.append((graph_name, df_sim))
                df_classic_gsp_vs_sas_and_trans = pd.DataFrame(l_classic_gsp_vs_sas_and_trans_rec,
                                                               columns=['graph_name', 'df_sig_ft_sim'])
                df_classic_gsp_vs_sas_and_trans = df_classic_gsp_vs_sas_and_trans.set_index('graph_name')
                pd.to_pickle(df_classic_gsp_vs_sas_and_trans, classic_gsp_vs_sas_and_trans_save_path
                             + str(job_id) + '#classic_gsp_vs_sas_and_trans' + out_file_suffix + '.pickle')

        if STAGE_draw_classic_gsp_vs_sas_and_trans:
            save_img = True
            show_img = True
            no_in_agg = True
            norm_sig = False
            if norm_sig:
                out_file_suffix = '_norm'
            else:
                out_file_suffix = '_non_norm'
            if no_in_agg:
                out_file_suffix += '_no_in_agg'

            classic_gsp_vs_sas_and_trans_name = str(job_id) + '#classic_gsp_vs_sas_and_trans' + out_file_suffix + '.pickle'

            classic_gsp_vs_sas_and_trans_save_path = g_work_dir + 'experiments/gsp_vs_sas/classic_gsp_vs_sas_and_trans/'
            df_classic_gsp_vs_sas_and_trans = pd.read_pickle(classic_gsp_vs_sas_and_trans_save_path
                                                             + classic_gsp_vs_sas_and_trans_name % str(job_id))

            d_sim_classic_vs_adj_diff_by_K = dict()
            d_sim_classic_vs_ln_agg_by_K = dict()
            d_sim_classic_vs_ln_conv_by_K = dict()
            for graph_name, sim_rec in df_classic_gsp_vs_sas_and_trans.iterrows():
                df_sim_per_graph = sim_rec['df_sig_ft_sim']
                for K, sim_per_graph_rec in df_sim_per_graph.iterrows():
                    sim_classic_vs_adj_diff = sim_per_graph_rec['sim_classic_vs_adj_diff']
                    sim_classic_vs_ln_agg = sim_per_graph_rec['sim_classic_vs_ln_agg']
                    sim_classic_vs_ln_conv = sim_per_graph_rec['sim_classic_vs_ln_conv']

                    if K not in d_sim_classic_vs_adj_diff_by_K:
                        d_sim_classic_vs_adj_diff_by_K[K] = [sim_classic_vs_adj_diff]
                    else:
                        d_sim_classic_vs_adj_diff_by_K[K].append(sim_classic_vs_adj_diff)

                    if K not in d_sim_classic_vs_ln_agg_by_K:
                        d_sim_classic_vs_ln_agg_by_K[K] = [sim_classic_vs_ln_agg]
                    else:
                        d_sim_classic_vs_ln_agg_by_K[K].append(sim_classic_vs_ln_agg)

                    if K not in d_sim_classic_vs_ln_conv_by_K:
                        d_sim_classic_vs_ln_conv_by_K[K] = [sim_classic_vs_ln_conv]
                    else:
                        d_sim_classic_vs_ln_conv_by_K[K].append(sim_classic_vs_ln_conv)

            for K in d_sim_classic_vs_adj_diff_by_K:
                mean_sim_classic_vs_adj_diff_by_K = np.mean(d_sim_classic_vs_adj_diff_by_K[K])
                std_sim_classic_vs_adj_diff_by_K = np.std(d_sim_classic_vs_adj_diff_by_K[K])
                d_sim_classic_vs_adj_diff_by_K[K] = (mean_sim_classic_vs_adj_diff_by_K,
                                                     std_sim_classic_vs_adj_diff_by_K)

            for K in d_sim_classic_vs_ln_agg_by_K:
                mean_sim_classic_vs_ln_agg_by_K = np.mean(d_sim_classic_vs_ln_agg_by_K[K])
                std_sim_classic_vs_ln_agg_by_K = np.std(d_sim_classic_vs_ln_agg_by_K[K])
                d_sim_classic_vs_ln_agg_by_K[K] = (mean_sim_classic_vs_ln_agg_by_K,
                                                   std_sim_classic_vs_ln_agg_by_K)

            for K in d_sim_classic_vs_ln_conv_by_K:
                mean_sim_classic_vs_ln_conv_by_K = np.mean(d_sim_classic_vs_ln_conv_by_K[K])
                std_sim_classic_vs_ln_conv_by_K = np.std(d_sim_classic_vs_ln_conv_by_K[K])
                d_sim_classic_vs_ln_conv_by_K[K] = (mean_sim_classic_vs_ln_conv_by_K,
                                                    std_sim_classic_vs_ln_conv_by_K)

            l_K = d_sim_classic_vs_ln_conv_by_K.keys()
            img_name = '%s#ft_sim#classic_vs_sas_and_trans' % str(job_id)
            # vmax = np.max([np.max(np.abs(item)) for item in df_sas_ana['sig_pwd_ln_ft'].to_list()])
            # vmin = np.min([np.min(np.abs(item)) for item in df_sas_ana['sig_pwd_ln_ft'].to_list()])
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(5, 20))
            # fig.suptitle(img_name, fontsize=15, fontweight='semibold')
            idx = 0
            for K in l_K:
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1]]
                y_vmax = np.round(np.max([l_means[i] + l_stds[i] for i in range(3)]), decimals=1)
                y_vmin = np.round(np.min([l_means[i] - l_stds[i] for i in range(3)]), decimals=1)
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                y_vmax = y_vmax + y_stride if y_vmax + y_stride <= 1.0 else 1.0
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(3)], l_means, yerr=l_stds,
                                   marker='o', fmt='o', mfc='tab:blue', capsize=2, capthick=1)
                axes[idx].set_xticks([i for i in range(3)])
                axes[idx].set_xticklabels(['ADJ-DIFF', 'IN-AGG', 'LN-VX'])
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                idx += 1
            plt.tight_layout(pad=1.0)
            # plt.subplots_adjust(top=0.94)
            if save_img:
                plt.savefig(classic_gsp_vs_sas_and_trans_save_path + img_name + '.PNG', format='PNG')
            if show_img:
                plt.show()

        if STAGE_draw_combined_gsp_vs_sas_rets:
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']

            save_img = True
            show_img = True
            no_in_agg = False
            norm_sig = True
            if norm_sig:
                out_file_suffix = '_norm'
            else:
                out_file_suffix = '_non_norm'
            if no_in_agg:
                out_file_suffix += '_no_in_agg'

            classic_gsp_vs_sas_and_trans_save_path = g_work_dir + 'experiments/gsp_vs_sas/classic_gsp_vs_sas_and_trans/'

            d_ret_collection = dict()
            for job_id in l_job_ids:
                classic_gsp_vs_sas_and_trans_name = str(job_id) + '#classic_gsp_vs_sas_and_trans' + out_file_suffix + '.pickle'
                df_classic_gsp_vs_sas_and_trans = pd.read_pickle(classic_gsp_vs_sas_and_trans_save_path
                                                                 + classic_gsp_vs_sas_and_trans_name)

                d_sim_classic_vs_adj_diff_by_K = dict()
                d_sim_classic_vs_ln_agg_by_K = dict()
                d_sim_classic_vs_ln_conv_by_K = dict()
                d_sim_classic_vs_apprx_ls_by_K = dict()
                d_sim_classic_vs_ens_by_K = dict()
                for graph_name, sim_rec in df_classic_gsp_vs_sas_and_trans.iterrows():
                    df_sim_per_graph = sim_rec['df_sig_ft_sim']
                    for K, sim_per_graph_rec in df_sim_per_graph.iterrows():
                        sim_classic_vs_adj_diff = sim_per_graph_rec['sim_classic_vs_adj_diff']
                        sim_classic_vs_ln_agg = sim_per_graph_rec['sim_classic_vs_ln_agg']
                        sim_classic_vs_ln_conv = sim_per_graph_rec['sim_classic_vs_ln_conv']
                        sim_classic_vs_apprx_ls = sim_per_graph_rec['sim_classic_vs_apprx_ls']
                        sim_classic_vs_ens = sim_per_graph_rec['sim_classic_vs_ens']

                        if K not in d_sim_classic_vs_adj_diff_by_K:
                            d_sim_classic_vs_adj_diff_by_K[K] = [sim_classic_vs_adj_diff]
                        else:
                            d_sim_classic_vs_adj_diff_by_K[K].append(sim_classic_vs_adj_diff)

                        if K not in d_sim_classic_vs_ln_agg_by_K:
                            d_sim_classic_vs_ln_agg_by_K[K] = [sim_classic_vs_ln_agg]
                        else:
                            d_sim_classic_vs_ln_agg_by_K[K].append(sim_classic_vs_ln_agg)

                        if K not in d_sim_classic_vs_ln_conv_by_K:
                            d_sim_classic_vs_ln_conv_by_K[K] = [sim_classic_vs_ln_conv]
                        else:
                            d_sim_classic_vs_ln_conv_by_K[K].append(sim_classic_vs_ln_conv)

                        if K not in d_sim_classic_vs_apprx_ls_by_K:
                            d_sim_classic_vs_apprx_ls_by_K[K] = [sim_classic_vs_apprx_ls]
                        else:
                            d_sim_classic_vs_apprx_ls_by_K[K].append(sim_classic_vs_apprx_ls)

                        if K not in d_sim_classic_vs_ens_by_K:
                            d_sim_classic_vs_ens_by_K[K] = [sim_classic_vs_ens]
                        else:
                            d_sim_classic_vs_ens_by_K[K].append(sim_classic_vs_ens)

                for K in d_sim_classic_vs_adj_diff_by_K:
                    mean_sim_classic_vs_adj_diff_by_K = np.mean(d_sim_classic_vs_adj_diff_by_K[K])
                    std_sim_classic_vs_adj_diff_by_K = np.std(d_sim_classic_vs_adj_diff_by_K[K])
                    d_sim_classic_vs_adj_diff_by_K[K] = (mean_sim_classic_vs_adj_diff_by_K,
                                                         std_sim_classic_vs_adj_diff_by_K)

                for K in d_sim_classic_vs_ln_agg_by_K:
                    mean_sim_classic_vs_ln_agg_by_K = np.mean(d_sim_classic_vs_ln_agg_by_K[K])
                    std_sim_classic_vs_ln_agg_by_K = np.std(d_sim_classic_vs_ln_agg_by_K[K])
                    d_sim_classic_vs_ln_agg_by_K[K] = (mean_sim_classic_vs_ln_agg_by_K,
                                                       std_sim_classic_vs_ln_agg_by_K)

                for K in d_sim_classic_vs_ln_conv_by_K:
                    mean_sim_classic_vs_ln_conv_by_K = np.mean(d_sim_classic_vs_ln_conv_by_K[K])
                    std_sim_classic_vs_ln_conv_by_K = np.std(d_sim_classic_vs_ln_conv_by_K[K])
                    d_sim_classic_vs_ln_conv_by_K[K] = (mean_sim_classic_vs_ln_conv_by_K,
                                                        std_sim_classic_vs_ln_conv_by_K)

                for K in d_sim_classic_vs_apprx_ls_by_K:
                    mean_sim_classic_vs_apprx_ls_by_K = np.mean(d_sim_classic_vs_apprx_ls_by_K[K])
                    std_sim_classic_vs_apprx_ls_by_K = np.std(d_sim_classic_vs_apprx_ls_by_K[K])
                    d_sim_classic_vs_apprx_ls_by_K[K] = (mean_sim_classic_vs_apprx_ls_by_K,
                                                        std_sim_classic_vs_apprx_ls_by_K)

                for K in d_sim_classic_vs_ens_by_K:
                    mean_sim_classic_vs_ens_by_K = np.mean(d_sim_classic_vs_ens_by_K[K])
                    std_sim_classic_vs_ens_by_K = np.std(d_sim_classic_vs_ens_by_K[K])
                    d_sim_classic_vs_ens_by_K[K] = (mean_sim_classic_vs_ens_by_K,
                                                    std_sim_classic_vs_ens_by_K)

                d_ret_collection[job_id] = [d_sim_classic_vs_adj_diff_by_K, d_sim_classic_vs_ln_agg_by_K,
                                            d_sim_classic_vs_ln_conv_by_K, d_sim_classic_vs_apprx_ls_by_K,
                                            d_sim_classic_vs_ens_by_K]

            l_K = d_ret_collection['100_rand_graphs'][0].keys()

            img_name = 'all#ft_sim#classic_vs_sas_and_trans' + out_file_suffix
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(12, 20))
            # fig.suptitle(img_name, fontsize=15, fontweight='semibold')
            idx = 0
            for K in l_K:
                d_sim_classic_vs_adj_diff_by_K = d_ret_collection['100_rand_graphs'][0]
                d_sim_classic_vs_ln_agg_by_K = d_ret_collection['100_rand_graphs'][1]
                d_sim_classic_vs_ln_conv_by_K = d_ret_collection['100_rand_graphs'][2]
                d_sim_classic_vs_apprx_ls_by_K = d_ret_collection['100_rand_graphs'][3]
                d_sim_classic_vs_ens_by_K = d_ret_collection['100_rand_graphs'][4]
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0],
                           d_sim_classic_vs_apprx_ls_by_K[K][0],
                           d_sim_classic_vs_ens_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1],
                          d_sim_classic_vs_apprx_ls_by_K[K][1],
                          d_sim_classic_vs_ens_by_K[K][1]]
                y_vmax_1 = np.round(np.max([l_means[i] + l_stds[i] for i in range(len(l_means))]), decimals=1)
                y_vmin_1 = np.round(np.min([l_means[i] - l_stds[i] for i in range(len(l_means))]), decimals=1)
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(len(l_means))], l_means, yerr=l_stds,
                                   marker='o', fmt='o', c='tab:blue', mfc='tab:blue', mec='tab:blue',
                                   capsize=2, capthick=1, label='ERM-Rand')

                d_sim_classic_vs_adj_diff_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][0]
                d_sim_classic_vs_ln_agg_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][1]
                d_sim_classic_vs_ln_conv_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][2]
                d_sim_classic_vs_apprx_ls_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][3]
                d_sim_classic_vs_ens_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][4]
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0],
                           d_sim_classic_vs_apprx_ls_by_K[K][0],
                           d_sim_classic_vs_ens_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1],
                          d_sim_classic_vs_apprx_ls_by_K[K][1],
                          d_sim_classic_vs_ens_by_K[K][1]]
                y_vmax_2 = np.round(np.max([l_means[i] + l_stds[i] for i in range(len(l_means))]), decimals=1)
                y_vmin_2 = np.round(np.min([l_means[i] - l_stds[i] for i in range(len(l_means))]), decimals=1)
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(1 * len(l_means), 2 * len(l_means))], l_means, yerr=l_stds,
                                   marker='o', fmt='o', c='tab:green', mfc='tab:green', mec='tab:green',
                                   capsize=2, capthick=1, label='ERM-Pulse')

                d_sim_classic_vs_adj_diff_by_K = d_ret_collection['100_sbm_graphs'][0]
                d_sim_classic_vs_ln_agg_by_K = d_ret_collection['100_sbm_graphs'][1]
                d_sim_classic_vs_ln_conv_by_K = d_ret_collection['100_sbm_graphs'][2]
                d_sim_classic_vs_apprx_ls_by_K = d_ret_collection['100_sbm_graphs'][3]
                d_sim_classic_vs_ens_by_K = d_ret_collection['100_sbm_graphs'][4]
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0],
                           d_sim_classic_vs_apprx_ls_by_K[K][0],
                           d_sim_classic_vs_ens_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1],
                          d_sim_classic_vs_apprx_ls_by_K[K][1],
                          d_sim_classic_vs_ens_by_K[K][1]]
                y_vmax_3 = np.round(np.max([l_means[i] + l_stds[i] for i in range(len(l_means))]), decimals=1)
                y_vmin_3 = np.round(np.min([l_means[i] - l_stds[i] for i in range(len(l_means))]), decimals=1)
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(2 * len(l_means), 3 * len(l_means))], l_means, yerr=l_stds,
                                   marker='o', fmt='o', c='tab:orange', mfc='tab:orange', mec='tab:orange',
                                   capsize=2, capthick=1, label='SBM-Rand')

                d_sim_classic_vs_adj_diff_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][0]
                d_sim_classic_vs_ln_agg_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][1]
                d_sim_classic_vs_ln_conv_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][2]
                d_sim_classic_vs_apprx_ls_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][3]
                d_sim_classic_vs_ens_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][4]
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0],
                           d_sim_classic_vs_apprx_ls_by_K[K][0],
                           d_sim_classic_vs_ens_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1],
                          d_sim_classic_vs_apprx_ls_by_K[K][1],
                          d_sim_classic_vs_ens_by_K[K][1]]
                y_vmax_4 = np.round(np.max([l_means[i] + l_stds[i] for i in range(len(l_means))]), decimals=1)
                y_vmin_4 = np.round(np.min([l_means[i] - l_stds[i] for i in range(len(l_means))]), decimals=1)
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(3 * len(l_means), 4 * len(l_means))], l_means, yerr=l_stds,
                                   marker='o', fmt='o', c='tab:red', mfc='tab:red', mec='tab:red',
                                   capsize=2, capthick=1, label='SBM-Pulse')

                y_vmin = np.min([y_vmin_1, y_vmin_2, y_vmin_3, y_vmin_4])
                y_vmax = np.max([y_vmax_1, y_vmax_2, y_vmax_3, y_vmax_4])
                y_step = np.round((y_vmax - y_vmin) / 8, decimals=2)
                y_vmax = y_vmax + y_step if y_vmax + y_step <= 1.0 else 1.0
                y_vmin = y_vmin if y_vmin >= 0.0 else 0.0
                axes[idx].set_xticks([i for i in range(4 * len(l_means))])
                axes[idx].set_xticklabels(
                    ['ADJ-DIFF', 'IN-AGG', 'LN-VX', 'LS-APPRX', 'ENS', 'ADJ-DIFF', 'IN-AGG', 'LN-VX', 'LS-APPRX', 'ENS',
                     'ADJ-DIFF', 'IN-AGG', 'LN-VX', 'LS-APPRX', 'ENS', 'ADJ-DIFF', 'IN-AGG', 'LN-VX', 'LS-APPRX', 'ENS'])
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_step)], decimals=2))
                axes[idx].legend()
                idx += 1
            plt.tight_layout(pad=1.0)
            # plt.subplots_adjust(top=0.94)
            if save_img:
                plt.savefig(classic_gsp_vs_sas_and_trans_save_path + img_name + '.PNG', format='PNG')
            if show_img:
                plt.show()

        if STAGE_sgs_pairwise_comparisons:
            '''
            ADJ-DIFF vs APPRX-LS, ADJ-DIFF vs IN_AGG, ADJ_DIFF vs LN-VX
            APPRX-LS vs IN-AGG, APPRX-LS vs LN-VX
            IN_AGG vs LN-VX
            '''
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']

            out_file_suffix = ''
            no_in_agg = False
            if no_in_agg:
                out_file_suffix += '_no_in_agg'

            sgs_pairwise_comparisons_save_path = g_work_dir + 'experiments/gsp_vs_sas/sgs_pairwise_comparisons/'
            if not os.path.exists(sgs_pairwise_comparisons_save_path):
                os.mkdir(sgs_pairwise_comparisons_save_path)

            sas_and_trans_save_path = g_work_dir + 'experiments/gsp_vs_sas/sas_and_trans/'
            ln_vx_only_save_path = g_work_dir + 'experiments/gsp_vs_sas/ln_vx_only/'

            for job_id in l_job_ids:
                df_sas_and_trans = pd.read_pickle(sas_and_trans_save_path + '%s#sas_and_trans.pickle' % str(job_id))
                df_ln_vx = pd.read_pickle(ln_vx_only_save_path + '%s#ln_vx_only_by_job.pickle' % str(job_id))
                l_sgs_pw_comp = []
                for graph_name, df_sas_and_trans_rec in df_sas_and_trans.iterrows():
                    df_sas_and_trans_per_graph = df_sas_and_trans_rec['df_sas_and_trans']
                    d_ln_vx_by_K = df_ln_vx.loc[graph_name]['d_ln_vx_by_K']
                    l_sim_rec = []
                    for K, sgs_rec in df_sas_and_trans_per_graph.iterrows():
                        np_sig_ft_adj_diff = sgs_rec['sig_vs_eig_sas']
                        np_sig_ft_ln_agg = sgs_rec['sig_pwd_agg_vx_ft']
                        # np_sig_ft_ln_conv = sgs_rec['sig_pwd_rec_vx_ft']
                        np_sig_ft_apprx_ls = sgs_rec['apprx_ls_ft']
                        np_sig_ft_ln_conv = np.mean(d_ln_vx_by_K[K], axis=0)

                        l_eig_vals = df_sas_and_trans_per_graph.loc[K]['eig_vals']
                        l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                        num_eff_eig_vals = len(l_eff_eig_vals)
                        num_all_eig_vals = len(l_eig_vals)

                        np_sig_ft_adj_diff_hat = preprocessing.normalize(np.abs(np_sig_ft_adj_diff).reshape(1, -1))[0]
                        np_sig_ft_ln_agg_hat = preprocessing.normalize(np.abs(np_sig_ft_ln_agg).reshape(1, -1))[0]
                        np_sig_ft_ln_conv_hat = preprocessing.normalize(np.abs(np_sig_ft_ln_conv).reshape(1, -1))[0]
                        np_sig_ft_apprx_ls = preprocessing.normalize(np.abs(np_sig_ft_apprx_ls).reshape(1, -1))[0]
                        if no_in_agg:
                            np_sig_ft_ens_hat = preprocessing.normalize(np.mean([np_sig_ft_adj_diff_hat,
                                                                                 np_sig_ft_ln_conv_hat,
                                                                                 np_sig_ft_apprx_ls], axis=0).reshape(1, -1))[0]
                        else:
                            np_sig_ft_ens_hat = preprocessing.normalize(np.mean([np_sig_ft_adj_diff_hat,
                                                                                 np_sig_ft_ln_agg_hat,
                                                                                 np_sig_ft_ln_conv_hat,
                                                                                 np_sig_ft_apprx_ls], axis=0).reshape(1, -1))[0]

                        sim_adj_diff_vs_apprx_ls = np.dot(np_sig_ft_adj_diff_hat, np_sig_ft_apprx_ls)
                        # sim_adj_diff_vs_apprx_ls, _ = stats.spearmanr(np_sig_ft_adj_diff_hat, np_sig_ft_apprx_ls)
                        if sim_adj_diff_vs_apprx_ls < -1.0:
                            sim_adj_diff_vs_apprx_ls = -1.0
                        elif sim_adj_diff_vs_apprx_ls > 1.0:
                            sim_adj_diff_vs_apprx_ls = 1.0

                        sim_adj_diff_vs_in_agg = np.dot(np_sig_ft_adj_diff_hat, np_sig_ft_ln_agg_hat)
                        # sim_adj_diff_vs_in_agg, _ = stats.spearmanr(np_sig_ft_adj_diff_hat, np_sig_ft_ln_agg_hat)
                        if sim_adj_diff_vs_in_agg < -1.0:
                            sim_adj_diff_vs_in_agg = -1.0
                        elif sim_adj_diff_vs_in_agg > 1.0:
                            sim_adj_diff_vs_in_agg = 1.0

                        sim_adj_diff_vs_ln_vx = np.dot(np_sig_ft_adj_diff_hat, np_sig_ft_ln_conv_hat)
                        # sim_adj_diff_vs_ln_vx, _ = stats.spearmanr(np_sig_ft_adj_diff_hat, np_sig_ft_ln_conv_hat)
                        if sim_adj_diff_vs_ln_vx < -1.0:
                            sim_adj_diff_vs_ln_vx = -1.0
                        elif sim_adj_diff_vs_ln_vx > 1.0:
                            sim_adj_diff_vs_ln_vx = 1.0

                        sim_apprx_ls_vs_in_agg = np.dot(np_sig_ft_apprx_ls, np_sig_ft_ln_agg_hat)
                        # sim_apprx_ls_vs_in_agg, _ = stats.spearmanr(np_sig_ft_apprx_ls, np_sig_ft_ln_agg_hat)
                        if sim_apprx_ls_vs_in_agg < -1.0:
                            sim_apprx_ls_vs_in_agg = -1.0
                        elif sim_apprx_ls_vs_in_agg > 1.0:
                            sim_apprx_ls_vs_in_agg = 1.0

                        sim_apprx_lx_vs_ln_vx = np.dot(np_sig_ft_apprx_ls, np_sig_ft_ln_conv_hat)
                        # sim_apprx_lx_vs_ln_vx, _ = stats.spearmanr(np_sig_ft_apprx_ls, np_sig_ft_ln_conv_hat)
                        if sim_apprx_lx_vs_ln_vx < -1.0:
                            sim_apprx_lx_vs_ln_vx = -1.0
                        elif sim_apprx_lx_vs_ln_vx > 1.0:
                            sim_apprx_lx_vs_ln_vx = 1.0

                        sim_in_agg_vs_ln_vx = np.dot(np_sig_ft_ln_agg_hat, np_sig_ft_ln_conv_hat)
                        # sim_in_agg_vs_ln_vx, _ = stats.spearmanr(np_sig_ft_ln_agg_hat, np_sig_ft_ln_conv_hat)
                        if sim_in_agg_vs_ln_vx < -1.0:
                            sim_in_agg_vs_ln_vx = -1.0
                        elif sim_in_agg_vs_ln_vx > 1.0:
                            sim_in_agg_vs_ln_vx = 1.0

                        l_sim_rec.append((K, sim_adj_diff_vs_apprx_ls, sim_adj_diff_vs_in_agg, sim_adj_diff_vs_ln_vx,
                                          sim_apprx_ls_vs_in_agg, sim_apprx_lx_vs_ln_vx, sim_in_agg_vs_ln_vx))
                    df_sim = pd.DataFrame(l_sim_rec, columns=['K', 'sim_adj_diff_vs_apprx_ls', 'sim_adj_diff_vs_in_agg',
                                                              'sim_adj_diff_vs_ln_vx', 'sim_apprx_ls_vs_in_agg',
                                                              'sim_apprx_lx_vs_ln_vx', 'sim_in_agg_vs_ln_vx'])
                    df_sim = df_sim.set_index('K')
                    l_sgs_pw_comp.append((graph_name, df_sim))
                df_sgs_pw_comp = pd.DataFrame(l_sgs_pw_comp, columns=['graph_name', 'df_sig_ft_sim'])
                df_sgs_pw_comp = df_sgs_pw_comp.set_index('graph_name')
                pd.to_pickle(df_sgs_pw_comp, sgs_pairwise_comparisons_save_path + str(job_id)
                             + '#sgs_pw_comp' + out_file_suffix + '.pickle')
                logging.debug('[STAGE_sgs_pairwise_comparisons] done for %s' % job_id)

        if STAGE_draw_sgs_pairwise_comparisons:
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            save_img = True
            show_img = True
            sgs_pw_comp_suffix = ''
            no_in_agg = False
            if no_in_agg:
                sgs_pw_comp_suffix += '_no_in_agg'
            sgs_pairwise_comparisons_save_path = g_work_dir + 'experiments/gsp_vs_sas/sgs_pairwise_comparisons/'

            for job_id in l_job_ids:
                df_sgs_pw_comp = pd.read_pickle(sgs_pairwise_comparisons_save_path + str(job_id) + '#sgs_pw_comp'
                                                + sgs_pw_comp_suffix + '.pickle')
                d_sgs_pw_sim = dict()
                for graph_name, df_sim_rec in df_sgs_pw_comp.iterrows():
                    df_sim = df_sim_rec['df_sig_ft_sim']
                    for K, sim_rec in df_sim.iterrows():
                        sim_adj_diff_vs_apprx_ls = sim_rec['sim_adj_diff_vs_apprx_ls']
                        sim_adj_diff_vs_in_agg = sim_rec['sim_adj_diff_vs_in_agg']
                        sim_adj_diff_vs_ln_vx = sim_rec['sim_adj_diff_vs_ln_vx']
                        sim_apprx_ls_vs_in_agg = sim_rec['sim_apprx_ls_vs_in_agg']
                        sim_apprx_lx_vs_ln_vx = sim_rec['sim_apprx_lx_vs_ln_vx']
                        sim_in_agg_vs_ln_vx = sim_rec['sim_in_agg_vs_ln_vx']
                        np_sim = np.asarray([sim_adj_diff_vs_apprx_ls, sim_adj_diff_vs_in_agg, sim_adj_diff_vs_ln_vx,
                                             sim_apprx_ls_vs_in_agg, sim_apprx_lx_vs_ln_vx, sim_in_agg_vs_ln_vx])
                        if K not in d_sgs_pw_sim:
                            d_sgs_pw_sim[K] = [np_sim]
                        else:
                            d_sgs_pw_sim[K].append(np_sim)

                if job_id == '100_rand_graphs':
                    img_title = 'ERM-Rand'
                elif job_id == '100_rand_graphs_rand_pulse':
                    img_title = 'ERM-Pulse'
                elif job_id == '100_sbm_graphs':
                    img_title = 'SBM-Rand'
                elif job_id == '100_sbm_graphs_rand_pulse':
                    img_title = 'SMB-Pulse'
                l_K = list(d_sgs_pw_sim.keys())
                l_xlabels = ['AD-AP', 'AD-IN', 'AD-LN', 'AP-IN', 'AP-LN', 'IN-LN']
                img_name = 'sgs_pw_comp#' + str(job_id) + sgs_pw_comp_suffix
                fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(4, 15))
                fig.suptitle(img_title, fontsize=12, fontweight='semibold')
                idx = 0
                for K in l_K:
                    sim_means = np.mean(d_sgs_pw_sim[K], axis=0)
                    sim_stds = np.std(d_sgs_pw_sim[K], axis=0)
                    axes[idx].grid(True)
                    axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                    axes[idx].errorbar([i for i in range(len(l_xlabels))], sim_means, yerr=sim_stds,
                                       marker='o', fmt='o', c='tab:blue', mfc='tab:blue', mec='tab:blue',
                                       capsize=2, capthick=1)
                    axes[idx].set_xticks([i for i in range(len(l_xlabels))])
                    axes[idx].set_xticklabels(l_xlabels)
                    axes[idx].set_yticks(np.round([i for i in np.arange(0, 1.2, 0.2)], decimals=2))
                    idx += 1
                plt.tight_layout(pad=1.0)
                plt.subplots_adjust(top=0.95)
                if save_img:
                    plt.savefig(sgs_pairwise_comparisons_save_path + img_name + '.PNG', format='PNG')
                if show_img:
                    plt.show()
                plt.clf()
                plt.close()

        if STAGE_spec_seq_sg_stats:
            save_img = True
            show_img = True
            sg_stats_save_path = g_work_dir + 'experiments/gsp_vs_sas/sg_stats/'
            if not os.path.exists(sg_stats_save_path):
                os.mkdir(sg_stats_save_path)
            graph_sig_save_path = g_work_dir + 'experiments/gsp_vs_sas/graph_sigs/'
            spec_seq_save_path = g_work_dir + 'experiments/gsp_vs_sas/spec_seqs/'
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']
            for job_id in l_job_ids:
                if job_id == '100_rand_graphs':
                    img_title = 'ERM-Rand'
                elif job_id == '100_rand_graphs_rand_pulse':
                    img_title = 'ERM-Pulse'
                elif job_id == '100_sbm_graphs':
                    img_title = 'SBM-Rand'
                elif job_id == '100_sbm_graphs_rand_pulse':
                    img_title = 'SMB-Pulse'
                d_num_comp_by_K = dict()
                d_num_singleton_by_K = dict()
                df_graph_sig_pw_dist = pd.read_pickle(graph_sig_save_path + '%s#graph_sig_pw_dists.pickle' % str(job_id))
                df_spec_seq = pd.read_pickle(spec_seq_save_path + '%s#spec_seqs.pickle' % str(job_id))
                for graph_name, graph_sig_pw_dist_rec in df_graph_sig_pw_dist.iterrows():
                    df_spec_seq_per_graph = df_spec_seq.loc[graph_name]['df_spec_seq']
                    for K, spec_seq_rec in df_spec_seq_per_graph.iterrows():
                        connected_comp_sizes = spec_seq_rec['connected_comp_sizes']
                        if K not in d_num_comp_by_K:
                            d_num_comp_by_K[K] = [len(connected_comp_sizes)]
                        else:
                            d_num_comp_by_K[K].append(len(connected_comp_sizes))
                        if K not in d_num_singleton_by_K:
                            d_num_singleton_by_K[K] = [len([x for x in connected_comp_sizes if x == 1])]
                        else:
                            d_num_singleton_by_K[K].append(len([x for x in connected_comp_sizes if x == 1]))
                l_K = list(d_num_comp_by_K.keys())
                l_num_comp_means = []
                l_num_comp_stds = []
                l_num_sing_means = []
                l_num_sing_stds = []
                for K in l_K:
                    num_comp_mean = np.mean(d_num_comp_by_K[K])
                    l_num_comp_means.append(num_comp_mean)
                    num_comp_std = np.std(d_num_comp_by_K[K])
                    l_num_comp_stds.append(num_comp_std)
                    num_sing_mean = np.mean(d_num_singleton_by_K[K])
                    l_num_sing_means.append(num_sing_mean)
                    num_sing_std = np.std(d_num_singleton_by_K[K])
                    l_num_sing_stds.append(num_sing_std)

                img_name = 'sg_stats#num_comp#' + str(job_id)
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
                axes.grid(True)
                axes.set_title('%s: SG Connected' % img_title, fontsize=10, fontweight='semibold')
                axes.errorbar([i for i in range(len(l_K))], l_num_comp_means, yerr=l_num_comp_stds,
                                   marker='o', fmt='o', c='tab:blue', mfc='tab:blue', mec='tab:blue',
                                   capsize=2, capthick=1)
                axes.set_xticks([i for i in range(len(l_K))])
                axes.set_xticklabels(l_K)
                plt.tight_layout(pad=1.0)
                if save_img:
                    plt.savefig(sg_stats_save_path + img_name + '.PNG', format='PNG')
                if show_img:
                    plt.show()
                plt.clf()
                plt.close()

                img_name = 'sg_stats#num_sing#' + str(job_id)
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
                axes.grid(True)
                axes.set_title('%s: SG Singleton' % img_title, fontsize=10, fontweight='semibold')
                axes.errorbar([i for i in range(len(l_K))], l_num_sing_means, yerr=l_num_sing_stds,
                              marker='o', fmt='o', c='tab:blue', mfc='tab:blue', mec='tab:blue',
                              capsize=2, capthick=1)
                axes.set_xticks([i for i in range(len(l_K))])
                axes.set_xticklabels(l_K)
                plt.tight_layout(pad=1.0)
                if save_img:
                    plt.savefig(sg_stats_save_path + img_name + '.PNG', format='PNG')
                if show_img:
                    plt.show()
                plt.clf()
                plt.close()

    elif cmd == 'compare_good_and_bad_node_embeds':
        plt.set_loglevel('error')
        job_id = 'uw_symA'
        # draw_data_suffix = '_tv_bv_pulse_BCD'
        draw_data_suffix = ''
        good_vs_bad_draw_data_save_name = 'good_vs_bad_draw_data' + draw_data_suffix
        good_ne_folder_file = 'good_ne_folders' + draw_data_suffix + '.txt'
        bad_ne_folder_file = 'bad_ne_folders' + draw_data_suffix + '.txt'
        all_draw_data_save_name = 'all_draw_data'
        # all_ne_folder_file = 'bad_ne_folders.txt'
        # all_ne_folder_file = 'good_ne_folders.txt'
        # all_ne_folder_file = 'all_ne_folders.txt'
        # all_ne_folder_file = 'all_good_and_bad_ne_folders.txt'
        all_ne_folder_file = 'man_sig_folder.txt'

        STAGE_learn_node_embeds = False
        STAGE_node_clustering = True
        STAGE_select_good_and_bad = False
        STAGE_select_good_and_bad_by_epoch = False
        STAGE_spec_seqs = False
        STAGE_ln_to_vx_eigs = False
        STAGE_ln_vx_only = False
        STAGE_sas_and_trans = False
        STAGE_good_vs_bad_draw_data = False
        STAGE_draw_good_vs_bad = False
        STAGE_draw_all = False
        STAGE_draw_ne_int = True
        STAGE_draw_good_vs_bad_final_only_or_init_only = False
        STAGE_draw_good_vs_bad_both_final_and_init = False
        STAGE_draw_all_both_final_and_init = False
        STAGE_find_similar_tv_runs = False
        STAGE_tv_sas_relation = False
        STAGE_tv_sas_relation_analysis = False
        STATE_sgs_init_final_relation = False
        STAGE_sim_tv_sim_sas_hypo = False
        STAGE_sgs_amp_distribution_comp = False
        STAGE_get_all_runs_folders = False
        STAGE_tv_ari_ami_vs_sas_amplitudes = False
        STAGE_tv_ari_ami_vs_sas_amplitudes_analysis = False

        if STAGE_learn_node_embeds:
            rand_run_cnt = 500
            graph_name = 'uw_symA'
            embed_dim = 3
            k_cluster = 4
            d_gt = {'A': 0, 'B': 1, 'E': 1, 'F': 1, 'K': 1, 'C': 2, 'G': 2, 'H': 2, 'L': 2, 'D': 3, 'I': 3, 'J': 3,
                    'M': 3}
            node_embed_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/learned_node_embeds/'
            if not os.path.exists(node_embed_save_path):
                os.mkdir(node_embed_save_path)

            graph_path = g_work_dir + graph_name + '.pickle'
            nx_graph = nx.read_gpickle(graph_path)

            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            work_trial_name = 'ne_run_' + graph_name + '_' + now

            for run in range(rand_run_cnt):
                np_init_ne = gen_rand_node_embed(nx_graph, embed_dim, node_embed_type='rand')
                np_init_ne = preprocessing.normalize(np_init_ne)

                run_id = graph_name + '_' + now + '_' + str(run)
                dffsn_loss_w_range = None
                J_prob_range = None
                tv_loss_w_range = [1.0]
                bv_loss_w_range = [1.0]
                lv_loss_w_range = None
                lg_loss_w_range = None
                gs_loss_w_range = None
                max_epoch = 1000
                save_int = True
                show_img = False
                show_init_img = True
                do_cluster = True
                d_gt = d_gt
                k_cluster = k_cluster
                cluster_alg = 'spectral'

                configured_run(run_id, nx_graph, np_init_ne, embed_dim,
                               dffsn_loss_w_range, J_prob_range, tv_loss_w_range,
                               bv_loss_w_range, lv_loss_w_range, lg_loss_w_range, gs_loss_w_range,
                               max_epoch=max_epoch, save_int=save_int, save_folder=node_embed_save_path,
                               show_img=show_img,
                               show_init_img=show_init_img, do_cluster=do_cluster, d_gt=d_gt, k_cluster=k_cluster,
                               cluster_alg=cluster_alg)

        if STAGE_node_clustering:
            graph_name = 'uw_symA'
            # run_name = '20210427100644'
            # run_name = '20210407123413'
            # run_name = '20210609192829'
            run_name = '20210610160153'
            rand_run_cnt = 1
            rand_run_cnt_range = range(1)
            # rand_run_cnt_range = [308]
            k_cluster = 4
            cluster_alg = 'spectral'
            d_gt = {'A': 0, 'B': 1, 'E': 1, 'F': 1, 'K': 1, 'C': 2, 'G': 2, 'H': 2, 'L': 2, 'D': 3, 'I': 3, 'J': 3,
                    'M': 3}
            num_nc_jobs = 10

            node_embed_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/learned_node_embeds/'

            graph_path = g_work_dir + graph_name + '.pickle'
            nx_graph = nx.read_gpickle(graph_path)

            l_nc_task_folders = [node_embed_save_path + 'ne_run_uw_symA_' + run_name + '_' + str(i) + '/' for i in
                                 rand_run_cnt_range]
            num_nc_tasks = len(l_nc_task_folders)
            batch_size = math.ceil(num_nc_tasks / num_nc_jobs)
            l_nc_jobs = []
            for i in range(0, num_nc_tasks, batch_size):
                if i + batch_size < num_nc_tasks:
                    l_nc_jobs.append(l_nc_task_folders[i:i + batch_size])
                else:
                    l_nc_jobs.append(l_nc_task_folders[i:])

            def single_node_cluster_job(l_nc_tasks_per_job, job_id):
                logging.debug('[single_node_cluster_job] Proc %s: starts with %s nc tasks.'
                              % (job_id, len(l_nc_tasks_per_job)))
                timer_start = time.time()

                for task_id, task_folder in enumerate(l_nc_tasks_per_job):
                    for (dirpath, dirname, filenames) in walk(task_folder):
                        for filename in filenames:
                            if filename[-13:] != 'ne_int.pickle' or filename[:14] != 'ne_run@uw_symA':
                                continue
                            l_filename_fields = filename.split('@')
                            # np_ne_init = np.load(
                            #     dirpath + '/' + l_filename_fields[0] + '@' + l_filename_fields[1] + '@init_embed.npy')
                            df_ne_int = pd.read_pickle(dirpath + '/' + filename)

                            l_nc_perf = []
                            # np_ne_init = preprocessing.normalize(np_ne_init)
                            # ari, nmi, _ = node_clustering(d_gt, nx_graph, np_ne_init, k_cluster,
                            #                               cluster_alg=cluster_alg)
                            # l_nc_perf.append((ari, nmi))
                            for _, ne_int_rec in df_ne_int.iterrows():
                                epoch = ne_int_rec['epoch']
                                np_ne_int = ne_int_rec['node_embed_int']
                                np_ne_int = preprocessing.normalize(np_ne_int)
                                ari, nmi, l_pred_color = node_clustering(d_gt, nx_graph, np_ne_int, k_cluster,
                                                              cluster_alg=cluster_alg)
                                l_nc_perf.append((epoch, ari, nmi, l_pred_color))
                            df_nc_perf = pd.DataFrame(l_nc_perf, columns=['epoch', 'ari', 'nmi', 'l_pred_color'])
                            df_nc_perf = df_nc_perf.set_index('epoch')
                            pd.to_pickle(df_nc_perf, dirpath + '/' + l_filename_fields[0] + '@' + l_filename_fields[1]
                                         + '@nc_perf.pickle')
                            if task_id % 10 == 0 and task_id >= 10:
                                logging.debug('[single_node_cluster_job] Proc %s: done with %s nc tasks in %s secs.'
                                              % (job_id, task_id, time.time() - timer_start))
                logging.debug('[single_node_cluster_job] Proc %s: all done in %s secs.'
                              % (job_id, time.time() - timer_start))


            timer_start_all = time.time()
            l_proc = []
            for job_id, nc_job in enumerate(l_nc_jobs):
                p = multiprocessing.Process(target=single_node_cluster_job,
                                            args=(nc_job, job_id),
                                            name='Proc ' + str(job_id))
                p.start()
                l_proc.append(p)

            while len(l_proc) > 0:
                for p in l_proc:
                    if p.is_alive():
                        p.join(1)
                    else:
                        l_proc.remove(p)
                        logging.debug('[STAGE_node_clustering] %s is finished.' % p.name)
            logging.debug('[STAGE_node_clustering] All done in %s secs.' % str(time.time() - timer_start_all))

        if STAGE_select_good_and_bad:
            run_name = '20210426125446'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            if not os.path.exists(good_and_bad_runs_save_path):
                os.mkdir(good_and_bad_runs_save_path)
            good_threshold = 0.80
            bad_threshold = 0.30

            node_embed_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/learned_node_embeds/'

            l_good = []
            l_bad = []
            for (dirpath, dirname, filenames) in walk(node_embed_save_path):
                for filename in filenames:
                    if filename[-14:] != 'nc_perf.pickle' or filename[:29] != 'ne_run@uw_symA_' + run_name:
                        continue
                    l_filename_fields = filename.split('@')
                    df_nc_perf = pd.read_pickle(dirpath + '/' + filename)
                    last_nc_perf = df_nc_perf.iloc[len(df_nc_perf) - 1]
                    ari = last_nc_perf['ari']
                    nmi = last_nc_perf['nmi']
                    if ari >= good_threshold or nmi >= good_threshold:
                        l_good.append(dirpath + '/')
                        continue
                    if ari <= bad_threshold or nmi <= bad_threshold:
                        l_bad.append(dirpath + '/')
                        continue

            with open(good_and_bad_runs_save_path + good_ne_folder_file, 'w') as out_fd:
                out_str = '\n'.join(l_good)
                out_fd.write(out_str)
                out_fd.close()
            with open(good_and_bad_runs_save_path + bad_ne_folder_file, 'w') as out_fd:
                out_str = '\n'.join(l_bad)
                out_fd.write(out_str)
                out_fd.close()

        if STAGE_select_good_and_bad_by_epoch:
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'

            node_embed_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/learned_node_embeds/'
            l_good = [node_embed_save_path + 'ne_run_uw_symA_20210407123413_%s/' % i for i in range(500)]
            l_bad = [node_embed_save_path + 'ne_run_uw_symA_20210409084931_%s/' % i for i in range(500)]
            with open(good_and_bad_runs_save_path + 'good_ne_folders_by_epoch.txt', 'w') as out_fd:
                out_str = '\n'.join(l_good)
                out_fd.write(out_str)
                out_fd.close()
            with open(good_and_bad_runs_save_path + 'bad_ne_folders_by_epoch.txt', 'w') as out_fd:
                out_str = '\n'.join(l_bad)
                out_fd.write(out_str)
                out_fd.close()

        if STAGE_spec_seqs:
            decomp = 'svd'
            val_weighted = False
            adj_only = True
            exp_EADMs = False
            max_k = None
            unweighted = True
            use_norm_L = False
            top_M_eigs = None
            rm_dc = False
            norm_FEADMs = True
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            spec_seq_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/spec_seqs/'
            if not os.path.exists(spec_seq_save_path):
                os.mkdir(spec_seq_save_path)
            spec_seq_file_save_path = spec_seq_save_path + 'spec_seq@' + graph_name + '@' + spec_seq_param_name + '.pickle'

            graph_path = g_work_dir + graph_name + '.pickle'
            nx_graph = nx.read_gpickle(graph_path)

            graph_spectra_sequence(nx_graph,
                                   save_path=spec_seq_file_save_path,
                                   decomp=decomp,
                                   adj_only=adj_only,
                                   val_weighted=val_weighted,
                                   exp_EADMs=exp_EADMs,
                                   max_k=None,
                                   unweighted=True,
                                   use_norm_L=False,
                                   top_M_eigs=None,
                                   rm_dc=False,
                                   norm_FEADMs=norm_FEADMs)
            logging.debug('[STAGE_spec_seqs] All done.')

        if STAGE_ln_to_vx_eigs:
            loss_threshold = 0.000001
            max_epoch = 1000
            num_trials = 50
            use_cuda = True
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'

            spec_seq_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/spec_seqs/'
            df_spec_seq = pd.read_pickle(
                spec_seq_save_path + 'spec_seq@' + graph_name + '@' + spec_seq_param_name + '.pickle')

            for trial_id in range(num_trials):
                df_ln_to_vx = convert_ln_eigs_to_vx_eigs(df_spec_seq,
                                                         loss_threshold=loss_threshold,
                                                         max_epoch=max_epoch,
                                                         use_cuda=use_cuda,
                                                         save_ret=False,
                                                         save_path=None)
                pd.to_pickle(df_ln_to_vx,
                             spec_seq_save_path + 'ln_to_vx_eigs@' + graph_name + '@' + spec_seq_param_name
                             + '#%s.pickle' % str(trial_id))
            logging.debug('[STAGE_spec_seqs] All done.')

        if STAGE_ln_vx_only:
            all_runs = True
            num_ln_vx_trials = 20
            ln_vx_only_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/ln_vx_only/'
            if not os.path.exists(ln_vx_only_save_path):
                os.mkdir(ln_vx_only_save_path)

            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            spec_seq_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/spec_seqs/'
            df_spec_seq = pd.read_pickle(spec_seq_save_path + 'spec_seq@' + graph_name + '@' + spec_seq_param_name + '.pickle')

            l_df_ln_to_vx_eigs = []
            for trial_id in range(num_ln_vx_trials):
                df_ln_to_vx_eigs = pd.read_pickle(spec_seq_save_path + 'ln_to_vx_eigs@' + graph_name + '@'
                                                  + spec_seq_param_name + '#%s.pickle' % str(trial_id))
                l_df_ln_to_vx_eigs.append(df_ln_to_vx_eigs)

            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            l_selected_folders = []
            if all_runs:
                with open(good_and_bad_runs_save_path + all_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_selected_folders.append(ln.strip())
                    in_fd.close()
            else:
                with open(good_and_bad_runs_save_path + good_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_selected_folders.append(ln.strip())
                    in_fd.close()
                with open(good_and_bad_runs_save_path + bad_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_selected_folders.append(ln.strip())
                    in_fd.close()

            num_ln_vx_jobs = 10
            num_ln_vx_tasks = len(l_selected_folders)
            batch_size = math.ceil(num_ln_vx_tasks / num_ln_vx_jobs)
            l_ln_vx_jobs = []
            for i in range(0, num_ln_vx_tasks, batch_size):
                if i + batch_size < num_ln_vx_tasks:
                    l_ln_vx_jobs.append(l_selected_folders[i:i + batch_size])
                else:
                    l_ln_vx_jobs.append(l_selected_folders[i:])

            def single_ln_vx_job(l_ln_vx_tasks_per_job, job_id):
                logging.debug('[single_ln_vx_job] Proc %s: starts with %s ln_vx tasks.'
                              % (job_id, len(l_ln_vx_tasks_per_job)))
                timer_start= time.time()
                cnt = 0
                for task_id, task_folder in enumerate(l_ln_vx_tasks_per_job):
                    for (dirpath, dirname, filenames) in walk(task_folder):
                        for filename in filenames:
                            if filename[-13:] != 'ne_int.pickle' or filename[:14] != 'ne_run@uw_symA':
                                continue
                            l_filename_fields = filename.split('@')
                            if os.path.exists(dirpath + l_filename_fields[0] + '@' + l_filename_fields[1]
                                              + '@ln_vx_only.pickle'):
                                continue
                            df_ne_int = pd.read_pickle(dirpath + '/' + filename)
                            l_ln_vx_rec_by_epoch = []
                            for _, ne_int_rec in df_ne_int.iterrows():
                                epoch = ne_int_rec['epoch']
                                np_ne_int = ne_int_rec['node_embed_int']
                                np_ne_int = preprocessing.normalize(np_ne_int)
                                d_ln_vx_rec_by_K = dict()
                                for trial_id in range(num_ln_vx_trials):
                                    df_ln_to_vx_eigs = l_df_ln_to_vx_eigs[trial_id]
                                    df_analysis = ln_vx(np_ne_int, df_spec_seq, df_ln_to_vx_eigs)
                                    for K, ana_rec in df_analysis.iterrows():
                                        ln_vx_trans = ana_rec['sig_pwd_rec_vx_ft']
                                        if K not in d_ln_vx_rec_by_K:
                                            d_ln_vx_rec_by_K[K] = [ln_vx_trans]
                                        else:
                                            d_ln_vx_rec_by_K[K].append(ln_vx_trans)
                                l_ln_vx_rec_by_epoch.append((epoch, d_ln_vx_rec_by_K))
                            df_ln_vx_by_epoch = pd.DataFrame(l_ln_vx_rec_by_epoch, columns=['epoch', 'd_ln_vx_rec_by_K'])
                            df_ln_vx_by_epoch = df_ln_vx_by_epoch.set_index('epoch')
                            pd.to_pickle(df_ln_vx_by_epoch, dirpath + l_filename_fields[0] + '@' + l_filename_fields[1]
                                         + '@ln_vx_only.pickle')
                            cnt += 1
                            if cnt % 5 == 0 and cnt >= 5:
                                logging.debug('[single_ln_vx_job] %s ln_vx done in %s secs.'
                                              % (cnt, time.time() - timer_start))
                logging.debug('[single_ln_vx_job] Proc %s: all done in %s secs.'
                              % (job_id, time.time() - timer_start))

            timer_start_all = time.time()
            l_proc = []
            for job_id, ln_vx_job in enumerate(l_ln_vx_jobs):
                p = threading.Thread(target=single_ln_vx_job,
                                     args=(ln_vx_job, job_id),
                                     name='Proc ' + str(job_id))
                p.start()
                l_proc.append(p)

            while len(l_proc) > 0:
                for p in l_proc:
                    if p.is_alive():
                        p.join(1)
                    else:
                        l_proc.remove(p)
                        logging.debug('[single_ln_vx_job] %s is finished.' % p.name)
            logging.debug('[single_ln_vx_job] All done in %s secs.' % str(time.time() - timer_start_all))

        if STAGE_sas_and_trans:
            # !!!CAUTION!!!
            # PyTorch does not support loading a model to multiple processes
            # But multithreading is fine.
            all_runs = False
            num_sas_and_trans_jobs = 10
            use_ln_to_vx_eig_convert = False
            l_metrics = ['tv_loss']

            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            l_selected_folders = []
            if all_runs:
                with open(good_and_bad_runs_save_path + all_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_selected_folders.append(ln.strip())
                    in_fd.close()
            else:
                with open(good_and_bad_runs_save_path + good_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_selected_folders.append(ln.strip())
                    in_fd.close()
                with open(good_and_bad_runs_save_path + bad_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_selected_folders.append(ln.strip())
                    in_fd.close()

            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            spec_seq_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/spec_seqs/'
            df_spec_seq = pd.read_pickle(spec_seq_save_path + 'spec_seq@' + graph_name + '@' + spec_seq_param_name + '.pickle')
            # df_ln_to_vx_eigs = pd.read_pickle(spec_seq_save_path + 'ln_to_vx_eigs@' + graph_name + '@' + spec_seq_param_name + '.pickle')

            num_sas_and_trans_tasks = len(l_selected_folders)
            batch_size = math.ceil(num_sas_and_trans_tasks / num_sas_and_trans_jobs)
            l_sas_and_trans_jobs = []
            for i in range(0, num_sas_and_trans_tasks, batch_size):
                if i + batch_size < num_sas_and_trans_tasks:
                    l_sas_and_trans_jobs.append(l_selected_folders[i:i + batch_size])
                else:
                    l_sas_and_trans_jobs.append(l_selected_folders[i:])

            def single_sas_and_trans_job(l_sas_and_trans_tasks_per_job, job_id):
                logging.debug('[single_sas_and_trans_job] Proc %s: starts with %s sas_and_trans tasks.'
                              % (job_id, len(l_sas_and_trans_tasks_per_job)))
                timer_start = time.time()

                for task_id, task_folder in enumerate(l_sas_and_trans_tasks_per_job):
                    for (dirpath, dirname, filenames) in walk(task_folder):
                        for filename in filenames:
                            if filename[-13:] != 'ne_int.pickle' or filename[:14] != 'ne_run@uw_symA':
                                continue
                            l_filename_fields = filename.split('@')

                            l_analysis_per_run = []
                            df_ne_int = pd.read_pickle(dirpath + '/' + filename)
                            for _, ne_int_rec in df_ne_int.iterrows():
                                l_per_epoch_metrics = []
                                for metric_str in l_metrics:
                                    l_per_epoch_metrics.append(ne_int_rec[metric_str])
                                epoch = ne_int_rec['epoch']
                                np_ne_int = ne_int_rec['node_embed_int']
                                np_ne_int = preprocessing.normalize(np_ne_int)
                                df_analysis = stratified_graph_spectra_and_transformations(np_ne_int,
                                                                                           df_spec_seq,
                                                                                           save_ret=False,
                                                                                           save_path=None,
                                                                                           np_pw_dist=None,
                                                                                           norm_sig_FADM=False,
                                                                                           use_ln_to_vx_eig_convert=use_ln_to_vx_eig_convert,
                                                                                           df_ln_to_vx_eig_convert=None)
                                l_analysis_per_run.append(
                                    (epoch, df_analysis, l_metrics, l_per_epoch_metrics, np_ne_int))
                            df_analysis_per_run = pd.DataFrame(l_analysis_per_run, columns=['epoch',
                                                                                            'df_analysis',
                                                                                            'metric_names',
                                                                                            'metric_vals',
                                                                                            'np_ne'])
                            df_analysis_per_run = df_analysis_per_run.set_index('epoch')
                            pd.to_pickle(df_analysis_per_run, dirpath + l_filename_fields[0] + '@' + l_filename_fields[1]
                                         + '@sas_and_trans.pickle')
                            if task_id % 10 == 0 and task_id >= 10:
                                logging.debug('[single_sas_and_trans_job] Proc %s: done with %s sas_and_trans tasks in %s secs.'
                                              % (job_id, task_id, time.time() - timer_start))
                logging.debug('[single_sas_and_trans_job] Proc %s: all done in %s secs.'
                              % (job_id, time.time() - timer_start))

            timer_start_all = time.time()
            l_proc = []
            for job_id, sas_and_trans_job in enumerate(l_sas_and_trans_jobs):
                p = threading.Thread(target=single_sas_and_trans_job,
                                            args=(sas_and_trans_job, job_id),
                                            name='Proc ' + str(job_id))
                p.start()
                l_proc.append(p)

            while len(l_proc) > 0:
                for p in l_proc:
                    if p.is_alive():
                        p.join(1)
                    else:
                        l_proc.remove(p)
                        logging.debug('[STAGE_sas_and_trans] %s is finished.' % p.name)
            logging.debug('[STAGE_sas_and_trans] All done in %s secs.' % str(time.time() - timer_start_all))

        if STAGE_good_vs_bad_draw_data:
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            norm_sig = False
            no_in_agg = True
            sgs_chosen = False
            all_runs = True
            all_good_and_bad = False
            # good_or_bad_only = '_bad'
            good_or_bad_only = None
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'

            spec_seq_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/spec_seqs/'
            df_spec_seq = pd.read_pickle(spec_seq_save_path + 'spec_seq@' + graph_name + '@' + spec_seq_param_name + '.pickle')

            def collect_nc_perf_and_sas_and_trans_by_K(l_selected_folders):
                # >>> {K: [eig_val_0, eig_val_1, ...], ...}
                d_eig_vals = dict()
                got_eig_vals = False
                # >>> {tv_loss: {ep_0: [loss_per_run_0, ...], ep_1: [loss_per_run_0, ...]}, ...}
                d_metrics = dict()
                # >>> {K: {ep_0: [adj_diff_per_run_0, ...], ep_1: [adj_diff_per_run_0, ...]}, ...}
                d_adj_diff = dict()
                d_in_agg = dict()
                d_ln_vx = dict()
                d_apprx_ls = dict()
                d_ens = dict()
                d_sig_pw = dict()
                # >>> {ari: {ep_0: [ari_per_run_0, ...], ep_1: [ari_per_run_0, ...], ...}, nmi: {...}}
                d_nc_perf = dict()
                for selected_folder in l_selected_folders:
                    for (dirpath, dirname, filenames) in walk(selected_folder):
                        for filename in filenames:
                            if filename[-7:] != '.pickle' or filename[:14] != 'ne_run@uw_symA':
                                continue
                            if filename[-20:] == 'sas_and_trans.pickle':
                                l_filename_fields = filename.split('@')
                                df_sas_and_trans = pd.read_pickle(dirpath + '/' + filename)
                                df_ln_vx_only = pd.read_pickle(dirpath + '/' + l_filename_fields[0] + '@'
                                                                + l_filename_fields[1] + '@ln_vx_only.pickle')
                                for epoch, sas_and_trans_rec in df_sas_and_trans.iterrows():
                                    l_metric_names = sas_and_trans_rec['metric_names']
                                    l_metric_vals = sas_and_trans_rec['metric_vals']
                                    for metric_id, metric_name in enumerate(l_metric_names):
                                        if metric_name not in d_metrics:
                                            d_metrics[metric_name] = {ep: [] for ep in list(df_sas_and_trans.index)}
                                        d_metrics[metric_name][epoch].append(l_metric_vals[metric_id])

                                    df_analysis = sas_and_trans_rec['df_analysis']
                                    d_ln_vx_rec_by_K = df_ln_vx_only.loc[epoch]['d_ln_vx_rec_by_K']
                                    for K, ana_rec in df_analysis.iterrows():
                                        l_eig_vals = ana_rec['eig_vals']
                                        l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                                        num_eff_eig_vals = len(l_eff_eig_vals)
                                        num_all_eig_vals = len(l_eig_vals)
                                        sas_start_idx = num_all_eig_vals - num_eff_eig_vals
                                        np_sig_pw_FADM = ana_rec['sig_pwd_FADM']
                                        # np_sig_pw_FADM_norm = np.linalg.norm(np_sig_pw_FADM)
                                        if K not in d_sig_pw:
                                            d_sig_pw[K] = {ep: [] for ep in list(df_sas_and_trans.index)}
                                        d_sig_pw[K][epoch].append(np_sig_pw_FADM)

                                        if norm_sig:
                                            adj_diff = preprocessing.normalize(np.abs(ana_rec['sig_vs_eig_sas']).reshape(1, -1))[0]
                                            in_agg = preprocessing.normalize(np.abs(ana_rec['sig_pwd_agg_vx_ft']).reshape(1, -1))[0]
                                            ln_vx = np.mean(d_ln_vx_rec_by_K[K], axis=0)
                                            ln_vx = preprocessing.normalize(np.abs(ln_vx).reshape(1, -1))[0]
                                            apprx_ls = preprocessing.normalize(np.abs(ana_rec['apprx_ls_ft']).reshape(1, -1))[0]
                                            if no_in_agg:
                                                if int(K) <= 4:
                                                    # ens = preprocessing.normalize((0.4 * adj_diff + 0.4 * ln_vx + 0.2 * apprx_ls).reshape(1, -1))[0]
                                                    ens = 0.4 * adj_diff + 0.4 * ln_vx + 0.2 * apprx_ls
                                                else:
                                                    # ens = preprocessing.normalize(np.mean([adj_diff, ln_vx], axis=0).reshape(1, -1))[0]
                                                    ens = np.mean([adj_diff, ln_vx], axis=0)
                                            elif sgs_chosen:
                                                if int(K) <= 4:
                                                    # ens = preprocessing.normalize(np.mean([adj_diff, ln_vx, apprx_ls], axis=0).reshape(1, -1))[0]
                                                    ens = np.mean([adj_diff, ln_vx, apprx_ls], axis=0)
                                                else:
                                                    # ens = preprocessing.normalize(np.mean([in_agg, ln_vx], axis=0).reshape(1, -1))[0]
                                                    ens = np.mean([in_agg, ln_vx], axis=0)
                                            else:
                                                # ens = preprocessing.normalize(np.mean([adj_diff, in_agg, ln_vx, apprx_ls], axis=0).reshape(1, -1))[0]
                                                ens = np.mean([adj_diff, in_agg, ln_vx, apprx_ls], axis=0)
                                        else:
                                            adj_diff = np.abs(ana_rec['sig_vs_eig_sas'])
                                            in_agg = np.abs(ana_rec['sig_pwd_agg_vx_ft'])
                                            # ln_vx = np.abs(ana_rec['sig_pwd_rec_vx_ft'])
                                            ln_vx = np.mean(d_ln_vx_rec_by_K[K], axis=0)
                                            apprx_ls = np.abs(ana_rec['apprx_ls_ft'])

                                            # adj_diff_norm = preprocessing.normalize(adj_diff.reshape(1, -1))[0]
                                            # in_agg_norm = preprocessing.normalize(in_agg.reshape(1, -1))[0]
                                            # ln_vx_norm = preprocessing.normalize(ln_vx.reshape(1, -1))[0]
                                            # apprx_ls_norm = preprocessing.normalize(apprx_ls.reshape(1, -1))[0]
                                            if no_in_agg:
                                                if int(K) <= 4:
                                                    # ens = (np.linalg.norm(adj_diff) + np.linalg.norm(ln_vx_norm) + np.linalg.norm(apprx_ls_norm)) \
                                                    #       * (0.4 * adj_diff_norm + 0.4 * ln_vx_norm + 0.2 * apprx_ls_norm)
                                                    ens = 0.4 * adj_diff + 0.4 * ln_vx + 0.2 * apprx_ls
                                                else:
                                                    # ens = (np.linalg.norm(adj_diff) * np.linalg.norm(ln_vx_norm)) \
                                                    #       * np.mean([adj_diff, ln_vx], axis=0)
                                                    ens = np.mean([adj_diff, ln_vx], axis=0)
                                            elif sgs_chosen:
                                                if int(K) <= 4:
                                                    ens = np.mean([adj_diff, ln_vx, apprx_ls], axis=0)
                                                else:
                                                    ens = np.mean([in_agg, ln_vx], axis=0)
                                            else:
                                                ens = np.mean([adj_diff, in_agg, ln_vx, apprx_ls], axis=0)

                                        if not got_eig_vals:
                                            if K not in d_eig_vals:
                                                d_eig_vals[K] = l_eig_vals

                                        if K not in d_adj_diff:
                                            d_adj_diff[K] = {ep: [] for ep in list(df_sas_and_trans.index)}
                                        d_adj_diff[K][epoch].append(adj_diff)

                                        if K not in d_in_agg:
                                            d_in_agg[K] = {ep: [] for ep in list(df_sas_and_trans.index)}
                                        d_in_agg[K][epoch].append(in_agg)

                                        if K not in d_ln_vx:
                                            d_ln_vx[K] = {ep: [] for ep in list(df_sas_and_trans.index)}
                                        d_ln_vx[K][epoch].append(ln_vx)

                                        if K not in d_apprx_ls:
                                            d_apprx_ls[K] = {ep: [] for ep in list(df_sas_and_trans.index)}
                                        d_apprx_ls[K][epoch].append(apprx_ls)

                                        if K not in d_ens:
                                            d_ens[K] = {ep: [] for ep in list(df_sas_and_trans.index)}
                                        d_ens[K][epoch].append(ens)
                                    got_eig_vals = True
                            elif filename[-14:] == 'nc_perf.pickle':
                                df_nc_perf = pd.read_pickle(dirpath + '/' + filename)
                                for epoch, nc_perf_rec in df_nc_perf.iterrows():
                                    if 'ari' not in d_nc_perf:
                                        d_nc_perf['ari'] = {ep: [] for ep in list(df_nc_perf.index)}
                                    d_nc_perf['ari'][epoch].append(nc_perf_rec['ari'])
                                    if 'nmi' not in d_nc_perf:
                                        d_nc_perf['nmi'] = {ep: [] for ep in list(df_nc_perf.index)}
                                    d_nc_perf['nmi'][epoch].append(nc_perf_rec['nmi'])
                            else:
                                continue
                return d_metrics, d_eig_vals, d_adj_diff, d_in_agg, d_ln_vx, d_apprx_ls, d_ens, d_nc_perf, d_sig_pw

            if all_runs:
                l_all_folders = []
                with open(good_and_bad_runs_save_path + all_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_all_folders.append(ln.strip())
                    in_fd.close()
                d_metrics_all, d_eig_vals_all, d_adj_diff_all, d_in_agg_all, d_ln_vx_all, d_apprx_ls_all, d_ens_all, d_nc_perf_all, d_sig_pw_all \
                    = collect_nc_perf_and_sas_and_trans_by_K(l_all_folders)
                l_all_draw_data = []
                l_all_draw_data.append(('all', d_metrics_all, d_eig_vals_all, d_adj_diff_all, d_in_agg_all,
                                        d_ln_vx_all, d_apprx_ls_all, d_ens_all, d_nc_perf_all, d_sig_pw_all))
                df_all_draw_data = pd.DataFrame(l_all_draw_data, columns=['cat',
                                                                          'd_metrics',
                                                                          'd_eig_vals',
                                                                          'd_adj_diff',
                                                                          'd_in_agg',
                                                                          'd_ln_vx',
                                                                          'd_apprx_ls',
                                                                          'd_ens',
                                                                          'd_nc_perf',
                                                                          'd_sig_pw'])
                df_all_draw_data = df_all_draw_data.set_index('cat')
                if norm_sig:
                    if no_in_agg:
                        draw_data_save_name = all_draw_data_save_name + '_norm_no_in_agg'
                    elif sgs_chosen:
                        draw_data_save_name = all_draw_data_save_name + '_norm_sgs_chosen'
                    else:
                        draw_data_save_name = all_draw_data_save_name + '_norm'
                else:
                    if no_in_agg:
                        draw_data_save_name = all_draw_data_save_name + '_non_norm_no_in_agg'
                    elif sgs_chosen:
                        draw_data_save_name = all_draw_data_save_name + '_non_norm_sgs_chosen'
                    else:
                        draw_data_save_name = all_draw_data_save_name + '_non_norm'
                if all_good_and_bad:
                    draw_data_save_name += '_all_good_bad'
                if good_or_bad_only is not None:
                    draw_data_save_name += good_or_bad_only
                pd.to_pickle(df_all_draw_data, good_and_bad_runs_save_path + draw_data_save_name +
                             '@' + graph_name + '@' + spec_seq_param_name + '.pickle')

            else:
                l_good_folders = []
                l_bad_folders = []
                with open(good_and_bad_runs_save_path + good_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_good_folders.append(ln.strip())
                    in_fd.close()
                with open(good_and_bad_runs_save_path + bad_ne_folder_file, 'r') as in_fd:
                    for ln in in_fd:
                        l_bad_folders.append(ln.strip())
                    in_fd.close()

                l_good_vs_bad_draw_data = []
                d_metrics_good, d_eig_vals_good, d_adj_diff_good, d_in_agg_good, d_ln_vx_good, d_apprx_ls_good, d_ens_good, d_nc_perf_good, d_sig_pw_good \
                    = collect_nc_perf_and_sas_and_trans_by_K(l_good_folders)
                l_good_vs_bad_draw_data.append(('good', d_metrics_good, d_eig_vals_good, d_adj_diff_good, d_in_agg_good,
                                                d_ln_vx_good, d_apprx_ls_good, d_ens_good, d_nc_perf_good, d_sig_pw_good))

                d_metrics_bad, d_eig_vals_bad, d_adj_diff_bad, d_in_agg_bad, d_ln_vx_bad, d_apprx_ls_bad, d_ens_bad, d_nc_perf_bad, d_sig_pw_bad \
                    = collect_nc_perf_and_sas_and_trans_by_K(l_bad_folders)
                l_good_vs_bad_draw_data.append(('bad', d_metrics_bad, d_eig_vals_bad, d_adj_diff_bad, d_in_agg_bad,
                                                d_ln_vx_bad, d_apprx_ls_bad, d_ens_bad, d_nc_perf_bad, d_sig_pw_bad))
                df_good_vs_bad_draw_data = pd.DataFrame(l_good_vs_bad_draw_data, columns=['cat',
                                                                                          'd_metrics',
                                                                                          'd_eig_vals',
                                                                                          'd_adj_diff',
                                                                                          'd_in_agg',
                                                                                          'd_ln_vx',
                                                                                          'd_apprx_ls',
                                                                                          'd_ens',
                                                                                          'd_nc_perf',
                                                                                          'd_sig_pw'])
                df_good_vs_bad_draw_data = df_good_vs_bad_draw_data.set_index('cat')
                if norm_sig:
                    if no_in_agg:
                        draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_no_in_agg'
                    elif sgs_chosen:
                        draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_sgs_chosen'
                    else:
                        draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm'
                else:
                    if no_in_agg:
                        draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_no_in_agg'
                    elif sgs_chosen:
                        draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_sgs_chosen'
                    else:
                        draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm'
                pd.to_pickle(df_good_vs_bad_draw_data, good_and_bad_runs_save_path + draw_data_save_name +
                             '@' + graph_name + '@' + spec_seq_param_name + '.pickle')

        if STAGE_draw_good_vs_bad:
            save_img = True
            show_img = False
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            norm_sig = True
            if norm_sig:
                img_name_prefix = 'good_vs_bad_norm'
            else:
                img_name_prefix = 'good_vs_bad_non_norm'
            sgs_chosen = False
            no_in_agg = True
            all_runs = False
            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_sgs_chosen'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_sgs_chosen'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_good_vs_bad_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name + '@'
                                                      + graph_name + '@' + spec_seq_param_name + '.pickle')

            d_metrics_good = df_good_vs_bad_draw_data.loc['good']['d_metrics']
            d_eig_vals_good = df_good_vs_bad_draw_data.loc['good']['d_eig_vals']
            d_adj_diff_good = df_good_vs_bad_draw_data.loc['good']['d_adj_diff']
            d_in_agg_good = df_good_vs_bad_draw_data.loc['good']['d_in_agg']
            d_ln_vx_good = df_good_vs_bad_draw_data.loc['good']['d_ln_vx']
            d_apprx_ls_good = df_good_vs_bad_draw_data.loc['good']['d_apprx_ls']
            d_ens_good = df_good_vs_bad_draw_data.loc['good']['d_ens']
            d_nc_perf_good = df_good_vs_bad_draw_data.loc['good']['d_nc_perf']

            d_metrics_bad = df_good_vs_bad_draw_data.loc['bad']['d_metrics']
            d_eig_vals_bad = df_good_vs_bad_draw_data.loc['bad']['d_eig_vals']
            d_adj_diff_bad = df_good_vs_bad_draw_data.loc['bad']['d_adj_diff']
            d_in_agg_bad = df_good_vs_bad_draw_data.loc['bad']['d_in_agg']
            d_ln_vx_bad = df_good_vs_bad_draw_data.loc['bad']['d_ln_vx']
            d_apprx_ls_bad = df_good_vs_bad_draw_data.loc['bad']['d_apprx_ls']
            d_ens_bad = df_good_vs_bad_draw_data.loc['bad']['d_ens']
            d_nc_perf_bad = df_good_vs_bad_draw_data.loc['bad']['d_nc_perf']

            l_K = sorted(d_eig_vals_good.keys())
            l_epoch = sorted(list(d_metrics_good['tv_loss'].keys()))
            sas_xtick_stride = math.ceil(len(l_epoch) / 35)
            l_sas_xticks = [i for i in range(0, len(l_epoch), sas_xtick_stride)]
            l_sas_xticklabels = [l_epoch[i] for i in l_sas_xticks]

            if no_in_agg:
                img_name_suffix = '_no_in_agg'
            elif sgs_chosen:
                img_name_suffix = '_sgs_chosen'
            else:
                img_name_suffix = ''

            for sas_method in ['ADJ-DIFF', 'IN-AGG', 'LN-VX', 'APPRX-LS', 'ENS']:
                for K in l_K:
                    l_eig_vals = d_eig_vals_good[K]
                    l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                    num_all_eig_vals = len(l_eig_vals)
                    num_eff_eig_vals = len(l_eff_eig_vals)

                    img_name = img_name_prefix + '@' + graph_name + '@' + spec_seq_param_name + '#K%s#%s' % (K, sas_method)
                    fig_height = (num_all_eig_vals + 4) * 2.5
                    fig, axes = plt.subplots(ncols=1, nrows=num_all_eig_vals + 4, figsize=(15, fig_height))
                    # fig.suptitle('Globally Weighted Total Variation', fontsize=15, fontweight='semibold')

                    # >>> Globally Weighted Total Variation
                    idx = 0
                    l_tv_mean_over_ep_good = [np.mean(d_metrics_good['tv_loss'][ep]) for ep in l_epoch]
                    l_tv_std_over_ep_good = [np.std(d_metrics_good['tv_loss'][ep]) for ep in l_epoch]
                    l_tv_mean_over_ep_bad = [np.mean(d_metrics_bad['tv_loss'][ep]) for ep in l_epoch]
                    l_tv_std_over_ep_bad = [np.std(d_metrics_bad['tv_loss'][ep]) for ep in l_epoch]

                    good_plot_color = 'tab:blue'
                    bad_plot_color = 'tab:orange'
                    linewidth = 2
                    axes[idx].grid(True)
                    axes[idx].set_title(r'Total Variation $\tau$', fontsize=10, fontweight='semibold')
                    axes[idx].plot(l_epoch, l_tv_mean_over_ep_good, linewidth=linewidth, color=good_plot_color, label='tv_good')
                    axes[idx].plot(l_epoch, l_tv_mean_over_ep_bad, linewidth=linewidth, color=bad_plot_color, label='tv_bad')
                    axes[idx].set_xticks(l_sas_xticks)
                    axes[idx].set_xticklabels(l_sas_xticklabels)
                    axes[idx].legend()
                    idx += 1

                    # >>> Gradients of Globally Weighted Total Variation
                    axes[idx].grid(True)
                    axes[idx].set_title(r'$\partial \tau / \partial t$', fontsize=10, fontweight='semibold')
                    axes[idx].plot(l_epoch, np.gradient(l_tv_mean_over_ep_good), linewidth=linewidth, color=good_plot_color, label='tv_good')
                    axes[idx].plot(l_epoch, np.gradient(l_tv_mean_over_ep_bad), linewidth=linewidth, color=bad_plot_color, label='tv_bad')
                    axes[idx].set_xticks(l_sas_xticks)
                    axes[idx].set_xticklabels(l_sas_xticklabels)
                    axes[idx].legend()
                    idx += 1

                    # >>> ARI & NMI
                    l_ari_mean_over_ep_good = [np.mean(d_nc_perf_good['ari'][ep]) for ep in l_epoch]
                    l_ari_std_over_ep_good = [np.std(d_nc_perf_good['ari'][ep]) for ep in l_epoch]
                    l_nmi_mean_over_ep_good = [np.mean(d_nc_perf_good['nmi'][ep]) for ep in l_epoch]
                    l_nmi_std_over_ep_good = [np.std(d_nc_perf_good['nmi'][ep]) for ep in l_epoch]
                    l_ari_mean_over_ep_bad = [np.mean(d_nc_perf_bad['ari'][ep]) for ep in l_epoch]
                    l_ari_std_over_ep_bad = [np.std(d_nc_perf_bad['ari'][ep]) for ep in l_epoch]
                    l_nmi_mean_over_ep_bad = [np.mean(d_nc_perf_bad['nmi'][ep]) for ep in l_epoch]
                    l_nmi_std_over_ep_bad = [np.std(d_nc_perf_bad['nmi'][ep]) for ep in l_epoch]

                    good_ari_plot_color = 'tab:blue'
                    good_nmi_plot_color = 'tab:cyan'
                    bad_ari_plot_color = 'tab:red'
                    bad_nmi_plot_color = 'tab:pink'
                    axes[idx].grid(True)
                    axes[idx].set_title('ARI & AMI', fontsize=10, fontweight='semibold')
                    axes[idx].plot(l_epoch, l_ari_mean_over_ep_good, linewidth=linewidth, color=good_ari_plot_color, label='ARI_good')
                    axes[idx].plot(l_epoch, l_nmi_mean_over_ep_good, linewidth=linewidth, color=good_nmi_plot_color, label='AMI_good')
                    axes[idx].plot(l_epoch, l_ari_mean_over_ep_bad, linewidth=linewidth, color=bad_ari_plot_color, label='ARI_bad')
                    axes[idx].plot(l_epoch, l_nmi_mean_over_ep_bad, linewidth=linewidth, color=bad_nmi_plot_color, label='AMI_bad')
                    axes[idx].set_xticks(l_sas_xticks)
                    axes[idx].set_xticklabels(l_sas_xticklabels)
                    axes[idx].legend()
                    idx += 1

                    # >>> Gradients of ARI & NMI
                    good_ari_plot_color = 'tab:blue'
                    good_nmi_plot_color = 'tab:cyan'
                    bad_ari_plot_color = 'tab:red'
                    bad_nmi_plot_color = 'tab:pink'
                    axes[idx].grid(True)
                    axes[idx].set_title(r'$\partial ARI / \partial t$ & $\partial AMI / \partial t$', fontsize=10, fontweight='semibold')
                    axes[idx].plot(l_epoch, np.gradient(l_ari_mean_over_ep_good), linewidth=linewidth, color=good_ari_plot_color, label='ARI_good')
                    axes[idx].plot(l_epoch, np.gradient(l_nmi_mean_over_ep_good), linewidth=linewidth, color=good_nmi_plot_color, label='AMI_good')
                    axes[idx].plot(l_epoch, np.gradient(l_ari_mean_over_ep_bad), linewidth=linewidth, color=bad_ari_plot_color, label='ARI_bad')
                    axes[idx].plot(l_epoch, np.gradient(l_nmi_mean_over_ep_bad), linewidth=linewidth, color=bad_nmi_plot_color, label='AMI_bad')
                    axes[idx].set_xticks(l_sas_xticks)
                    axes[idx].set_xticklabels(l_sas_xticklabels)
                    axes[idx].legend()
                    idx += 1

                    # >>> ADJ-DIFF, IN-AGG, LN-VX, APPRX-LS & ENS
                    l_adj_diff_mean_gradient_good = []
                    l_in_agg_mean_gradient_good = []
                    l_ln_vx_mean_gradient_good = []
                    l_apprx_ls_mean_gradient_good = []
                    l_ens_mean_gradient_good = []
                    
                    l_adj_diff_mean_gradient_bad = []
                    l_in_agg_mean_gradient_bad = []
                    l_ln_vx_mean_gradient_bad = []
                    l_apprx_ls_mean_gradient_bad = []
                    l_ens_mean_gradient_bad = []
                    for eig_idx in range(num_all_eig_vals):
                        eig_val = l_eig_vals[eig_idx]
                        l_adj_diff_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_adj_diff_good[K][ep]]) for ep in l_epoch]
                        l_adj_diff_std_over_ep_good = [np.std([item[eig_idx] for item in d_adj_diff_good[K][ep]]) for ep in l_epoch]
                        l_in_agg_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_in_agg_good[K][ep]]) for ep in l_epoch]
                        l_in_agg_std_over_ep_good = [np.std([item[eig_idx] for item in d_in_agg_good[K][ep]]) for ep in l_epoch]
                        l_ln_vx_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_ln_vx_good[K][ep]]) for ep in l_epoch]
                        l_ln_vx_std_over_ep_good = [np.std([item[eig_idx] for item in d_ln_vx_good[K][ep]]) for ep in l_epoch]
                        l_apprx_ls_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_apprx_ls_good[K][ep]]) for ep in l_epoch]
                        l_apprx_ls_std_over_ep_good = [np.std([item[eig_idx] for item in d_apprx_ls_good[K][ep]]) for ep in l_epoch]
                        l_ens_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_ens_good[K][ep]]) for ep in l_epoch]
                        l_ens_std_over_ep_good = [np.std([item[eig_idx] for item in d_ens_good[K][ep]]) for ep in l_epoch]

                        l_adj_diff_mean_over_ep_bad = [np.mean([item[eig_idx] for item in d_adj_diff_bad[K][ep]]) for ep in l_epoch]
                        l_adj_diff_std_over_ep_bad = [np.std([item[eig_idx] for item in d_adj_diff_bad[K][ep]]) for ep in l_epoch]
                        l_in_agg_mean_over_ep_bad = [np.mean([item[eig_idx] for item in d_in_agg_bad[K][ep]]) for ep in l_epoch]
                        l_in_agg_std_over_ep_bad = [np.std([item[eig_idx] for item in d_in_agg_bad[K][ep]]) for ep in l_epoch]
                        l_ln_vx_mean_over_ep_bad = [np.mean([item[eig_idx] for item in d_ln_vx_bad[K][ep]]) for ep in l_epoch]
                        l_ln_vx_std_over_ep_bad = [np.std([item[eig_idx] for item in d_ln_vx_bad[K][ep]]) for ep in l_epoch]
                        l_apprx_ls_mean_over_ep_bad = [np.mean([item[eig_idx] for item in d_apprx_ls_bad[K][ep]]) for ep in l_epoch]
                        l_apprx_ls_std_over_ep_bad = [np.std([item[eig_idx] for item in d_apprx_ls_bad[K][ep]]) for ep in l_epoch]
                        l_ens_mean_over_ep_bad = [np.mean([item[eig_idx] for item in d_ens_bad[K][ep]]) for ep in l_epoch]
                        l_ens_std_over_ep_bad = [np.std([item[eig_idx] for item in d_ens_bad[K][ep]]) for ep in l_epoch]

                        l_adj_diff_mean_gradient_good.append(np.gradient(l_adj_diff_mean_over_ep_good))
                        l_in_agg_mean_gradient_good.append(np.gradient(l_in_agg_mean_over_ep_good))
                        l_ln_vx_mean_gradient_good.append(np.gradient(l_ln_vx_mean_over_ep_good))
                        l_apprx_ls_mean_gradient_good.append(np.gradient(l_apprx_ls_mean_over_ep_good))
                        l_ens_mean_gradient_good.append(np.gradient(l_ens_mean_over_ep_good))

                        l_adj_diff_mean_gradient_bad.append(np.gradient(l_adj_diff_mean_over_ep_bad))
                        l_in_agg_mean_gradient_bad.append(np.gradient(l_in_agg_mean_over_ep_bad))
                        l_ln_vx_mean_gradient_bad.append(np.gradient(l_ln_vx_mean_over_ep_bad))
                        l_apprx_ls_mean_gradient_bad.append(np.gradient(l_apprx_ls_mean_over_ep_bad))
                        l_ens_mean_gradient_bad.append(np.gradient(l_ens_mean_over_ep_bad))

                        good_plot_color = 'tab:blue'
                        bad_plot_color = 'tab:orange'
                        good_fmt = '-o'
                        bad_fmt = '-o'
                        axes[idx].grid(True)

                        if sas_method == 'ADJ-DIFF':
                            y_vmax_good = np.max(l_adj_diff_mean_over_ep_good)
                            y_vmin_good = np.min(l_adj_diff_mean_over_ep_good)
                            y_vmax_bad = np.max(l_adj_diff_mean_over_ep_bad)
                            y_vmin_bad = np.min(l_adj_diff_mean_over_ep_bad)
                            y_vmax = np.round(np.max([y_vmax_good, y_vmax_bad]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good, y_vmin_bad]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            good_vs_bad_max_mean_diff = np.max(np.abs(np.asarray(l_adj_diff_mean_over_ep_good) - np.asarray(l_adj_diff_mean_over_ep_bad)))
                            good_vs_bad_min_mean_diff = np.min(np.abs(np.asarray(l_adj_diff_mean_over_ep_good) - np.asarray(l_adj_diff_mean_over_ep_bad)))
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s: Max Mean Diff=%s, Min Mean Diff=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K,
                                                   np.round(good_vs_bad_max_mean_diff, decimals=4),
                                                   np.round(good_vs_bad_min_mean_diff, decimals=4)),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_adj_diff_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1, label='ADJ-DIFF_good')
                            axes[idx].errorbar(l_sas_xticks, [l_adj_diff_mean_over_ep_bad[ep] for ep in l_sas_xticklabels],
                                               fmt=bad_fmt, c=bad_plot_color, capsize=2, capthick=1, label='ADJ-DIFF_bad')
                        elif sas_method == 'IN-AGG':
                            y_vmax_good = np.max(l_in_agg_mean_over_ep_good)
                            y_vmin_good = np.min(l_in_agg_mean_over_ep_good)
                            y_vmax_bad = np.max(l_in_agg_mean_over_ep_bad)
                            y_vmin_bad = np.min(l_in_agg_mean_over_ep_bad)
                            y_vmax = np.round(np.max([y_vmax_good, y_vmax_bad]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good, y_vmin_bad]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            good_vs_bad_max_mean_diff = np.max(np.abs(np.asarray(l_in_agg_mean_over_ep_good) - np.asarray(l_in_agg_mean_over_ep_bad)))
                            good_vs_bad_min_mean_diff = np.min(np.abs(np.asarray(l_in_agg_mean_over_ep_good) - np.asarray(l_in_agg_mean_over_ep_bad)))
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s: Max Mean Diff=%s, Min Mean Diff=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K,
                                                   np.round(good_vs_bad_max_mean_diff, decimals=4),
                                                   np.round(good_vs_bad_min_mean_diff, decimals=4)),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_in_agg_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1, label='IN-AGG_good')
                            axes[idx].errorbar(l_sas_xticks, [l_in_agg_mean_over_ep_bad[ep] for ep in l_sas_xticklabels],
                                               fmt=bad_fmt, c=bad_plot_color, capsize=2, capthick=1, label='IN-AGG_bad')
                        elif sas_method == 'LN-VX':
                            y_vmax_good = np.max(l_ln_vx_mean_over_ep_good)
                            y_vmin_good = np.min(l_ln_vx_mean_over_ep_good)
                            y_vmax_bad = np.max(l_ln_vx_mean_over_ep_bad)
                            y_vmin_bad = np.min(l_ln_vx_mean_over_ep_bad)
                            y_vmax = np.round(np.max([y_vmax_good, y_vmax_bad]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good, y_vmin_bad]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            good_vs_bad_max_mean_diff = np.max(np.abs(np.asarray(l_ln_vx_mean_over_ep_good) - np.asarray(l_ln_vx_mean_over_ep_bad)))
                            good_vs_bad_min_mean_diff = np.min(np.abs(np.asarray(l_ln_vx_mean_over_ep_good) - np.asarray(l_ln_vx_mean_over_ep_bad)))
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s: Max Mean Diff=%s, Min Mean Diff=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K,
                                                   np.round(good_vs_bad_max_mean_diff, decimals=4),
                                                   np.round(good_vs_bad_min_mean_diff, decimals=4)),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_ln_vx_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1, label='ln_vx_good')
                            axes[idx].errorbar(l_sas_xticks, [l_ln_vx_mean_over_ep_bad[ep] for ep in l_sas_xticklabels],
                                               fmt=bad_fmt, c=bad_plot_color, capsize=2, capthick=1, label='ln_vx_bad')
                        elif sas_method == 'APPRX-LS':
                            y_vmax_good = np.max(l_apprx_ls_mean_over_ep_good)
                            y_vmin_good = np.min(l_apprx_ls_mean_over_ep_good)
                            y_vmax_bad = np.max(l_apprx_ls_mean_over_ep_bad)
                            y_vmin_bad = np.min(l_apprx_ls_mean_over_ep_bad)
                            y_vmax = np.round(np.max([y_vmax_good, y_vmax_bad]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good, y_vmin_bad]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            good_vs_bad_max_mean_diff = np.max(np.abs(np.asarray(l_apprx_ls_mean_over_ep_good) - np.asarray(l_apprx_ls_mean_over_ep_bad)))
                            good_vs_bad_min_mean_diff = np.min(np.abs(np.asarray(l_apprx_ls_mean_over_ep_good) - np.asarray(l_apprx_ls_mean_over_ep_bad)))
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s: Max Mean Diff=%s, Min Mean Diff=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K,
                                                   np.round(good_vs_bad_max_mean_diff, decimals=4),
                                                   np.round(good_vs_bad_min_mean_diff, decimals=4)),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_apprx_ls_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1, label='ln_vx_good')
                            axes[idx].errorbar(l_sas_xticks, [l_apprx_ls_mean_over_ep_bad[ep] for ep in l_sas_xticklabels],
                                               fmt=bad_fmt, c=bad_plot_color, capsize=2, capthick=1, label='ln_vx_bad')
                        elif sas_method == 'ENS':
                            y_vmax_good = np.max(l_ens_mean_over_ep_good)
                            y_vmin_good = np.min(l_ens_mean_over_ep_good)
                            y_vmax_bad = np.max(l_ens_mean_over_ep_bad)
                            y_vmin_bad = np.min(l_ens_mean_over_ep_bad)
                            y_vmax = np.round(np.max([y_vmax_good, y_vmax_bad]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good, y_vmin_bad]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            good_vs_bad_max_mean_diff = np.max(np.abs(np.asarray(l_ens_mean_over_ep_good) - np.asarray(l_ens_mean_over_ep_bad)))
                            good_vs_bad_min_mean_diff = np.min(np.abs(np.asarray(l_ens_mean_over_ep_good) - np.asarray(l_ens_mean_over_ep_bad)))
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s: Max Mean Diff=%s, Min Mean Diff=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K,
                                                   np.round(good_vs_bad_max_mean_diff, decimals=4),
                                                   np.round(good_vs_bad_min_mean_diff, decimals=4)),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_ens_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1, label='ENS_good')
                            axes[idx].errorbar(l_sas_xticks, [l_ens_mean_over_ep_bad[ep] for ep in l_sas_xticklabels],
                                               fmt=bad_fmt, c=bad_plot_color, capsize=2, capthick=1, label='ENS_bad')

                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)
                        axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                        axes[idx].legend()
                        idx += 1

                    plt.tight_layout(pad=1.0)
                    # plt.subplots_adjust(top=0.94)
                    if save_img:
                        plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                    if show_img:
                        plt.show()
                    plt.clf()
                    plt.close()

                    img_name = img_name_prefix + '_gradient' + '@' + graph_name + '@' + spec_seq_param_name + '#K%s#%s' % (K, sas_method)
                    fig_height = 2 * 2.5
                    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(15, fig_height))
                    if sas_method == 'ADJ-DIFF':
                        idx = 0
                        linewidth = 2
                        axes[idx].grid(True)
                        axes[idx].set_title('ADJ-DIFF Good @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_adj_diff_mean_gradient_good)))
                        for eig_idx, adj_diff_mean_gradient_good in enumerate(l_adj_diff_mean_gradient_good):
                            axes[idx].plot(adj_diff_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)

                        idx += 1
                        axes[idx].grid(True)
                        axes[idx].set_title('ADJ-DIFF Bad @ K=%s' % K, fontsize=10, fontweight='semibold')
                        for eig_idx, adj_diff_mean_gradient_bad in enumerate(l_adj_diff_mean_gradient_bad):
                            axes[idx].plot(adj_diff_mean_gradient_bad, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()
                    elif sas_method == 'IN-AGG':
                        idx = 0
                        linewidth = 2
                        axes[idx].grid(True)
                        axes[idx].set_title('IN-AGG Good @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_in_agg_mean_gradient_good)))
                        for eig_idx, in_agg_mean_gradient_good in enumerate(l_in_agg_mean_gradient_good):
                            axes[idx].plot(in_agg_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)

                        idx += 1
                        axes[idx].grid(True)
                        axes[idx].set_title('IN-AGG Bad @ K=%s' % K, fontsize=10, fontweight='semibold')
                        for eig_idx, in_agg_mean_gradient_bad in enumerate(l_in_agg_mean_gradient_bad):
                            axes[idx].plot(in_agg_mean_gradient_bad, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()
                    elif sas_method == 'LN-VX':
                        idx = 0
                        linewidth = 2
                        axes[idx].grid(True)
                        axes[idx].set_title('LN-VX Good @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_ln_vx_mean_gradient_good)))
                        for eig_idx, ln_vx_mean_gradient_good in enumerate(l_ln_vx_mean_gradient_good):
                            axes[idx].plot(ln_vx_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)

                        idx += 1
                        axes[idx].grid(True)
                        axes[idx].set_title('LN-VX Bad @ K=%s' % K, fontsize=10, fontweight='semibold')
                        for eig_idx, ln_vx_mean_gradient_bad in enumerate(l_ln_vx_mean_gradient_bad):
                            axes[idx].plot(ln_vx_mean_gradient_bad, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()
                    elif sas_method == 'APPRX-LS':
                        idx = 0
                        linewidth = 2
                        axes[idx].grid(True)
                        axes[idx].set_title('APPRX-LS Good @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_apprx_ls_mean_gradient_good)))
                        for eig_idx, apprx_ls_mean_gradient_good in enumerate(l_apprx_ls_mean_gradient_good):
                            axes[idx].plot(apprx_ls_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)

                        idx += 1
                        axes[idx].grid(True)
                        axes[idx].set_title('APPRX-LS Bad @ K=%s' % K, fontsize=10, fontweight='semibold')
                        for eig_idx, apprx_ls_mean_gradient_bad in enumerate(l_apprx_ls_mean_gradient_bad):
                            axes[idx].plot(apprx_ls_mean_gradient_bad, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()
                    elif sas_method == 'ENS':
                        idx = 0
                        linewidth = 2
                        axes[idx].grid(True)
                        axes[idx].set_title('ENS Good @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_ens_mean_gradient_good)))
                        for eig_idx, ens_mean_gradient_good in enumerate(l_ens_mean_gradient_good):
                            axes[idx].plot(ens_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)

                        idx += 1
                        axes[idx].grid(True)
                        axes[idx].set_title('ENS Bad @ K=%s' % K, fontsize=10, fontweight='semibold')
                        for eig_idx, ens_mean_gradient_bad in enumerate(l_ens_mean_gradient_bad):
                            axes[idx].plot(ens_mean_gradient_bad, c=l_colors[eig_idx], linewidth=linewidth)
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()

        if STAGE_draw_all:
            save_img = True
            show_img = False
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            sgs_chosen = False
            norm_sig = True
            no_in_agg = True
            multi_loss = False
            all_good_and_bad = False
            good_or_bad_only = 'good'
            if norm_sig:
                img_name_prefix = 'all_norm'
            else:
                img_name_prefix = 'all_non_norm'
            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = all_draw_data_save_name + '_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = all_draw_data_save_name + '_norm_sgs_chosen'
                else:
                    draw_data_save_name = all_draw_data_save_name + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = all_draw_data_save_name + '_non_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = all_draw_data_save_name + '_non_norm_sgs_chosen'
                else:
                    draw_data_save_name = all_draw_data_save_name + '_non_norm'
            if all_good_and_bad:
                draw_data_save_name += '_all_good_bad'
            if good_or_bad_only is not None:
                draw_data_save_name += good_or_bad_only
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_all_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name + '@'
                                                      + graph_name + '@' + spec_seq_param_name + '.pickle')

            d_metrics_all = df_all_draw_data.loc['all']['d_metrics']
            d_eig_vals_all = df_all_draw_data.loc['all']['d_eig_vals']
            d_apprx_ls_all = df_all_draw_data.loc['all']['d_apprx_ls']
            d_adj_diff_all = df_all_draw_data.loc['all']['d_adj_diff']
            d_in_agg_all = df_all_draw_data.loc['all']['d_in_agg']
            d_ln_vx_all = df_all_draw_data.loc['all']['d_ln_vx']
            d_ens_all = df_all_draw_data.loc['all']['d_ens']
            d_nc_perf_all = df_all_draw_data.loc['all']['d_nc_perf']

            l_K = sorted(d_eig_vals_all.keys())
            l_epoch = sorted(list(d_metrics_all['tv_loss'].keys()))
            sas_xtick_stride = math.ceil(len(l_epoch) / 35)
            l_sas_xticks = [i for i in range(0, len(l_epoch), sas_xtick_stride)]
            l_sas_xticklabels = [l_epoch[i] for i in l_sas_xticks]

            if no_in_agg:
                img_name_suffix = '_no_in_agg'
            elif sgs_chosen:
                img_name_suffix = '_sgs_chosen'
            else:
                img_name_suffix = ''

            for sas_method in ['APPRX-LS', 'ADJ-DIFF', 'IN-AGG', 'LN-VX', 'ENS']:
                for K in l_K:
                    l_eig_vals = d_eig_vals_all[K]
                    l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                    num_all_eig_vals = len(l_eig_vals)
                    num_eff_eig_vals = len(l_eff_eig_vals)

                    img_name = img_name_prefix + '@' + graph_name + '@' + spec_seq_param_name + '#K%s#%s' % (K, sas_method)
                    if multi_loss:
                        add_row = 3
                    else:
                        add_row = 2
                    fig_height = (num_all_eig_vals + add_row) * 2.5
                    fig, axes = plt.subplots(ncols=1, nrows=num_all_eig_vals + add_row, figsize=(15, fig_height))

                    # >>> Globally Weighted Total Variation
                    idx = 0
                    l_tv_mean_over_ep_all = [np.mean(d_metrics_all['tv_loss'][ep]) for ep in l_epoch]
                    l_tv_std_over_ep_all = [np.std(d_metrics_all['tv_loss'][ep]) for ep in l_epoch]

                    good_plot_color = 'tab:blue'
                    linewidth = 2
                    axes[idx].grid(True)
                    axes[idx].set_title('Globally Weighted Total Variation', fontsize=10, fontweight='semibold')
                    # axes[idx].errorbar(l_epoch, l_tv_mean_over_ep_good, yerr=l_tv_std_over_ep_good,
                    #                    fmt='o-', c=good_plot_color, capsize=2, capthick=1, label='good')
                    # axes[idx].errorbar(l_epoch, l_tv_mean_over_ep_bad, yerr=l_tv_std_over_ep_bad,
                    #                    fmt='-x', c=bad_plot_color, capsize=2, capthick=1, label='good')
                    axes[idx].plot(l_epoch, l_tv_mean_over_ep_all, linewidth=linewidth, color=good_plot_color)
                    axes[idx].set_xticks(l_sas_xticks)
                    axes[idx].set_xticklabels(l_sas_xticklabels)
                    axes[idx].legend()
                    idx += 1

                    # >>> Multiple Loss
                    if multi_loss:
                        idx = 0
                        l_tv_mean_over_ep_all = [np.mean(d_metrics_all['tv_loss'][ep]) for ep in l_epoch]
                        l_tv_std_over_ep_all = [np.std(d_metrics_all['tv_loss'][ep]) for ep in l_epoch]
                        l_bv_mean_over_ep_all = [np.mean(d_metrics_all['bv_loss'][ep]) for ep in l_epoch]
                        l_bv_std_over_ep_all = [np.std(d_metrics_all['bv_loss'][ep]) for ep in l_epoch]

                        linewidth = 2
                        axes[idx].grid(True)
                        axes[idx].set_title('TV & Regularizer', fontsize=10, fontweight='semibold')
                        axes[idx].plot(l_epoch, l_tv_mean_over_ep_all, linewidth=linewidth, color='tab:blue', label='TV')
                        axes[idx].plot(l_epoch, l_bv_mean_over_ep_all, linewidth=linewidth, color='tab:orange', label='Regularizer')
                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)
                        axes[idx].legend()
                        idx += 1

                    # >>> ARI & NMI
                    l_ari_mean_over_ep_good = [np.mean(d_nc_perf_all['ari'][ep]) for ep in l_epoch]
                    l_ari_std_over_ep_good = [np.std(d_nc_perf_all['ari'][ep]) for ep in l_epoch]
                    l_nmi_mean_over_ep_good = [np.mean(d_nc_perf_all['nmi'][ep]) for ep in l_epoch]
                    l_nmi_std_over_ep_good = [np.std(d_nc_perf_all['nmi'][ep]) for ep in l_epoch]

                    good_ari_plot_color = 'tab:blue'
                    good_nmi_plot_color = 'tab:cyan'
                    axes[idx].grid(True)
                    axes[idx].set_title('ARI & AMI', fontsize=10, fontweight='semibold')
                    # axes[idx].errorbar(l_epoch, l_ari_mean_over_ep_good, yerr=l_ari_std_over_ep_good,
                    #                    fmt='-o', c=good_ari_plot_color, capsize=2, capthick=1, label='good')
                    # axes[idx].errorbar(l_epoch, l_nmi_mean_over_ep_good, yerr=l_nmi_std_over_ep_good,
                    #                    fmt='-o', c=good_nmi_plot_color, capsize=2, capthick=1, label='good')
                    # axes[idx].errorbar(l_epoch, l_ari_mean_over_ep_bad, yerr=l_ari_std_over_ep_bad,
                    #                    fmt='-x', c=bad_ari_plot_color, capsize=2, capthick=1, label='bad')
                    # axes[idx].errorbar(l_epoch, l_nmi_mean_over_ep_bad, yerr=l_nmi_std_over_ep_bad,
                    #                    fmt='-x', c=bad_nmi_plot_color, capsize=2, capthick=1, label='bad')
                    axes[idx].plot(l_epoch, l_ari_mean_over_ep_good, linewidth=linewidth, color=good_ari_plot_color, label='ARI')
                    axes[idx].plot(l_epoch, l_nmi_mean_over_ep_good, linewidth=linewidth, color=good_nmi_plot_color, label='AMI')
                    axes[idx].set_xticks(l_sas_xticks)
                    axes[idx].set_xticklabels(l_sas_xticklabels)
                    axes[idx].legend()
                    idx += 1

                    # >>> ADJ-DIFF, IN-AGG, LN-VX & ENS
                    l_apprx_ls_mean_gradient_good = []
                    l_adj_diff_mean_gradient_good = []
                    l_in_agg_mean_gradient_good = []
                    l_ln_vx_mean_gradient_good = []
                    l_ens_mean_gradient_good = []

                    for eig_idx in range(num_all_eig_vals):
                        eig_val = l_eig_vals[eig_idx]
                        l_apprx_ls_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_apprx_ls_all[K][ep]]) for ep in l_epoch]
                        l_apprx_ls_std_over_ep_good = [np.std([item[eig_idx] for item in d_apprx_ls_all[K][ep]]) for ep in l_epoch]
                        l_adj_diff_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_adj_diff_all[K][ep]]) for ep in l_epoch]
                        l_adj_diff_std_over_ep_good = [np.std([item[eig_idx] for item in d_adj_diff_all[K][ep]]) for ep in l_epoch]
                        l_in_agg_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_in_agg_all[K][ep]]) for ep in l_epoch]
                        l_in_agg_std_over_ep_good = [np.std([item[eig_idx] for item in d_in_agg_all[K][ep]]) for ep in l_epoch]
                        l_ln_vx_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_ln_vx_all[K][ep]]) for ep in l_epoch]
                        l_ln_vx_std_over_ep_good = [np.std([item[eig_idx] for item in d_ln_vx_all[K][ep]]) for ep in l_epoch]
                        l_ens_mean_over_ep_good = [np.mean([item[eig_idx] for item in d_ens_all[K][ep]]) for ep in l_epoch]
                        l_ens_std_over_ep_good = [np.std([item[eig_idx] for item in d_ens_all[K][ep]]) for ep in l_epoch]

                        l_apprx_ls_mean_gradient_good.append(np.gradient(l_apprx_ls_mean_over_ep_good))
                        l_adj_diff_mean_gradient_good.append(np.gradient(l_adj_diff_mean_over_ep_good))
                        l_in_agg_mean_gradient_good.append(np.gradient(l_in_agg_mean_over_ep_good))
                        l_ln_vx_mean_gradient_good.append(np.gradient(l_ln_vx_mean_over_ep_good))
                        l_ens_mean_gradient_good.append(np.gradient(l_ens_mean_over_ep_good))

                        good_plot_color = 'tab:blue'
                        good_fmt = '-o'
                        axes[idx].grid(True)

                        if sas_method == 'ADJ-DIFF':
                            y_vmax_good = np.max(l_adj_diff_mean_over_ep_good)
                            y_vmin_good = np.min(l_adj_diff_mean_over_ep_good)
                            y_vmax = np.round(np.max([y_vmax_good]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_adj_diff_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1)
                        elif sas_method == 'APPRX-LS':
                            y_vmax_good = np.max(l_apprx_ls_mean_over_ep_good)
                            y_vmin_good = np.min(l_apprx_ls_mean_over_ep_good)
                            y_vmax = np.round(np.max([y_vmax_good]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_apprx_ls_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1)
                        elif sas_method == 'IN-AGG':
                            y_vmax_good = np.max(l_in_agg_mean_over_ep_good)
                            y_vmin_good = np.min(l_in_agg_mean_over_ep_good)
                            y_vmax = np.round(np.max([y_vmax_good]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_in_agg_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1)
                        elif sas_method == 'LN-VX':
                            y_vmax_good = np.max(l_ln_vx_mean_over_ep_good)
                            y_vmin_good = np.min(l_ln_vx_mean_over_ep_good)
                            y_vmax = np.round(np.max([y_vmax_good]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_ln_vx_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1)
                        elif sas_method == 'ENS':
                            y_vmax_good = np.max(l_ens_mean_over_ep_good)
                            y_vmin_good = np.min(l_ens_mean_over_ep_good)
                            y_vmax = np.round(np.max([y_vmax_good]), decimals=2) + 0.1
                            y_vmin = np.round(np.min([y_vmin_good]), decimals=2) - 0.1
                            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
                            axes[idx].set_title('%s @ Eigenvalue=%s K=%s'
                                                % (sas_method, np.round(eig_val, decimals=3), K),
                                                fontsize=10, fontweight='semibold')
                            axes[idx].errorbar(l_sas_xticks, [l_ens_mean_over_ep_good[ep] for ep in l_sas_xticklabels],
                                               fmt=good_fmt, c=good_plot_color, capsize=2, capthick=1)

                        axes[idx].set_xticks(l_sas_xticks)
                        axes[idx].set_xticklabels(l_sas_xticklabels)
                        axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                        idx += 1

                    plt.tight_layout(pad=1.0)
                    # plt.subplots_adjust(top=0.94)
                    if save_img:
                        plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                    if show_img:
                        plt.show()
                    plt.clf()
                    plt.close()

                    img_name = img_name_prefix + '_gradient' + '@' + graph_name + '@' + spec_seq_param_name + '#K%s#%s' % (K, sas_method)
                    fig_height = 2 * 2.5
                    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(15, fig_height))
                    if sas_method == 'ADJ-DIFF':
                        linewidth = 2
                        axes.grid(True)
                        axes.set_title('ADJ-DIFF All @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_adj_diff_mean_gradient_good)))
                        for eig_idx, adj_diff_mean_gradient_good in enumerate(l_adj_diff_mean_gradient_good):
                            axes.plot(adj_diff_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()
                    elif sas_method == 'APPRX-LS':
                        linewidth = 2
                        axes.grid(True)
                        axes.set_title('APPRX-LS All @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_apprx_ls_mean_gradient_good)))
                        for eig_idx, apprx_ls_mean_gradient_good in enumerate(l_apprx_ls_mean_gradient_good):
                            axes.plot(apprx_ls_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()
                    elif sas_method == 'IN-AGG':
                        linewidth = 2
                        axes.grid(True)
                        axes.set_title('IN-AGG All @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_in_agg_mean_gradient_good)))
                        for eig_idx, in_agg_mean_gradient_good in enumerate(l_in_agg_mean_gradient_good):
                            axes.plot(in_agg_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()
                    elif sas_method == 'LN-VX':
                        linewidth = 2
                        axes.grid(True)
                        axes.set_title('LN-VX All @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_ln_vx_mean_gradient_good)))
                        for eig_idx, ln_vx_mean_gradient_good in enumerate(l_ln_vx_mean_gradient_good):
                            axes.plot(in_agg_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()
                    elif sas_method == 'ENS':
                        linewidth = 2
                        axes.grid(True)
                        axes.set_title('ENS All @ K=%s' % K, fontsize=10, fontweight='semibold')
                        l_colors = plt.cm.plasma(np.linspace(0.0, 0.8, len(l_ens_mean_gradient_good)))
                        for eig_idx, ens_mean_gradient_good in enumerate(l_ens_mean_gradient_good):
                            axes.plot(ens_mean_gradient_good, c=l_colors[eig_idx], linewidth=linewidth)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        axes.set_xticks(l_sas_xticks)
                        axes.set_xticklabels(l_sas_xticklabels)
                        plt.tight_layout(pad=1.0)
                        if save_img:
                            plt.savefig(good_and_bad_runs_save_path + img_name + img_name_suffix + '.PNG', format='PNG')
                        if show_img:
                            plt.show()
                        plt.clf()
                        plt.close()

        if STAGE_draw_ne_int:
            graph_name = 'uw_symA'
            ne_int_prefix_fmt = 'ne_run@uw_symA_%s_%s'
            # ne_int_param = '@df-0.0_jp-0.0_tv0.6_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500'
            # ne_int_param = '@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000'
            # ne_int_param = '@df-0.0_jp-0.0_tv0.2_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500'
            ne_int_param = '@df-0.0_jp-0.0_tv0.1_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500'
            ne_int_file_fmt = ne_int_prefix_fmt + ne_int_param + '@ne_int.pickle'
            nc_perf_file_fmt = ne_int_prefix_fmt + '@nc_perf.pickle'
            # run_name = '20210427100644'
            # run_name = '20210407123413'
            # run_name = '20210609192829'
            run_name = '20210610160153'
            run_cnt_range = range(1)
            # run_cnt_range = [148, 308]
            graph_path = g_work_dir + graph_name + '.pickle'
            nx_graph = nx.read_gpickle(graph_path)
            k_cluster = 4
            d_gt = {'A': 0, 'B': 1, 'E': 1, 'F': 1, 'K': 1, 'C': 2, 'G': 2, 'H': 2, 'L': 2, 'D': 3, 'I': 3, 'J': 3,
                    'M': 3}
            l_gt = [d_gt[node] for node in nx_graph.nodes()]
            l_cmap = np.arange(0.0, 1.0, 1.0 / k_cluster)[:k_cluster]
            l_pred_color = [l_cmap[x] for x in l_gt]
            for rc in run_cnt_range:
                ne_folder = 'ne_run_uw_symA_' + run_name + '_' + str(rc) + '/'
                node_embed_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/learned_node_embeds/'

                graph_path = g_work_dir + graph_name + '.pickle'
                nx_graph = nx.read_gpickle(graph_path)

                df_ne_int = pd.read_pickle(node_embed_save_path + ne_folder + ne_int_file_fmt % (run_name, str(rc)))
                df_nc_perf = pd.read_pickle(node_embed_save_path + ne_folder + nc_perf_file_fmt % (run_name, str(rc)))

                ne_int_img_folder = node_embed_save_path + ne_folder + 'ne_int_imgs/'
                if not os.path.exists(ne_int_img_folder):
                    os.mkdir(ne_int_img_folder)
                for _, ne_int_rec in df_ne_int.iterrows():
                    epoch = ne_int_rec['epoch']
                    np_ne_int = ne_int_rec['node_embed_int']
                    np_ne_int = preprocessing.normalize(np_ne_int)
                    l_pred_color = df_nc_perf.loc[epoch]['l_pred_color']
                    img_path = ne_int_img_folder + (ne_int_prefix_fmt % (run_name, str(rc))) \
                               + ('@ne_int_img_ep_%s.PNG' % str(epoch))
                    draw_learned_node_embed(nx_graph, np_ne_int,
                                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                            l_pred_color, save_ret=True, save_path=img_path, show_img=False
                                            , show_title=False, epoch=epoch)

        if STAGE_draw_good_vs_bad_final_only_or_init_only:
            final_or_init = 'init'
            save_img = True
            show_img = False
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            norm_sig = True
            if norm_sig:
                img_name_prefix = 'good_vs_bad' + draw_data_suffix
            else:
                img_name_prefix = 'good_vs_bad' + draw_data_suffix + '_non_norm'
            sgs_chosen = False
            no_in_agg = True
            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_sgs_chosen'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_sgs_chosen'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_good_vs_bad_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name +'@'
                                                      + graph_name + '@' + spec_seq_param_name + '.pickle')

            d_metrics_good = df_good_vs_bad_draw_data.loc['good']['d_metrics']
            d_eig_vals_good = df_good_vs_bad_draw_data.loc['good']['d_eig_vals']
            d_adj_diff_good = df_good_vs_bad_draw_data.loc['good']['d_adj_diff']
            d_in_agg_good = df_good_vs_bad_draw_data.loc['good']['d_in_agg']
            d_ln_vx_good = df_good_vs_bad_draw_data.loc['good']['d_ln_vx']
            d_apprx_ls_good = df_good_vs_bad_draw_data.loc['good']['d_apprx_ls']
            d_ens_good = df_good_vs_bad_draw_data.loc['good']['d_ens']
            d_nc_perf_good = df_good_vs_bad_draw_data.loc['good']['d_nc_perf']

            d_metrics_bad = df_good_vs_bad_draw_data.loc['bad']['d_metrics']
            d_eig_vals_bad = df_good_vs_bad_draw_data.loc['bad']['d_eig_vals']
            d_adj_diff_bad = df_good_vs_bad_draw_data.loc['bad']['d_adj_diff']
            d_in_agg_bad = df_good_vs_bad_draw_data.loc['bad']['d_in_agg']
            d_ln_vx_bad = df_good_vs_bad_draw_data.loc['bad']['d_ln_vx']
            d_apprx_ls_bad = df_good_vs_bad_draw_data.loc['bad']['d_apprx_ls']
            d_ens_bad = df_good_vs_bad_draw_data.loc['bad']['d_ens']
            d_nc_perf_bad = df_good_vs_bad_draw_data.loc['bad']['d_nc_perf']

            l_K = sorted(d_eig_vals_good.keys())
            if final_or_init == 'final':
                max_epoch_good = np.max(list(d_metrics_good['tv_loss'].keys()))
                max_epoch_bad = np.max(list(d_metrics_bad['tv_loss'].keys()))
            elif final_or_init == 'init':
                max_epoch_good = np.min(list(d_metrics_good['tv_loss'].keys()))
                max_epoch_bad = np.min(list(d_metrics_bad['tv_loss'].keys()))
            else:
                raise Exception('Invalid final_or_init!')

            if no_in_agg:
                img_suffix = '_no_in_agg'
            elif sgs_chosen:
                img_suffix = '_sgs_chosen'
            else:
                img_suffix = ''
            # >>> Draw metrics
            if final_or_init == 'final':
                img_name = img_name_prefix + '_final_only_metrics@' + graph_name + '@' + spec_seq_param_name
            else:
                img_name = img_name_prefix + '_init_only_metrics@' + graph_name + '@' + spec_seq_param_name
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(5, 3))

            idx = 0
            tv_mean_over_ep_good = np.mean(d_metrics_good['tv_loss'][max_epoch_good])
            tv_std_over_ep_good = np.std(d_metrics_good['tv_loss'][max_epoch_good])
            tv_mean_over_ep_bad = np.mean(d_metrics_bad['tv_loss'][max_epoch_bad])
            tv_std_over_ep_bad = np.std(d_metrics_bad['tv_loss'][max_epoch_bad])

            y_vmax = np.max([np.max(d_metrics_good['tv_loss'][max_epoch_good]), np.max(d_metrics_bad['tv_loss'][max_epoch_bad])])
            y_vmin = np.min([np.min(d_metrics_good['tv_loss'][max_epoch_good]), np.min(d_metrics_bad['tv_loss'][max_epoch_bad])])
            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

            good_plot_color = 'tab:blue'
            bad_plot_color = 'tab:orange'
            axes.grid(True)
            axes.set_title('Globally Weighted Total Variation', fontsize=10, fontweight='semibold')
            # axes[idx].errorbar([0], [tv_mean_over_ep_good], yerr=[tv_std_over_ep_good],
            #                    fmt='o', c=good_plot_color, capsize=2, capthick=1)
            # axes[idx].errorbar([1], [tv_mean_over_ep_bad], yerr=[tv_std_over_ep_bad],
            #                    fmt='o', c=bad_plot_color, capsize=2, capthick=1)
            sns.kdeplot(d_metrics_good['tv_loss'][max_epoch_good], ax=axes, color='tab:orange', cut=0, fill=True, label='TV_good')
            sns.kdeplot(d_metrics_bad['tv_loss'][max_epoch_bad], ax=axes, color='tab:blue', cut=0, fill=True, label='TV_bad')
            axes.set_xticks(np.round([i for i in np.arange(y_vmin, y_vmax + 0.05, 0.05)], decimals=2))
            axes.legend()
            # axes.set_xticklabels(['tv_good', 'tv_bad'])
            # if not np.allclose(y_stride, 0.0):
            #     axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
            idx += 1

            # ari_mean_over_ep_good = np.mean(d_nc_perf_good['ari'][max_epoch_good])
            # ari_std_over_ep_good = np.std(d_nc_perf_good['ari'][max_epoch_good])
            # nmi_mean_over_ep_good = np.mean(d_nc_perf_good['nmi'][max_epoch_good])
            # nmi_std_over_ep_good = np.std(d_nc_perf_good['nmi'][max_epoch_good])
            # ari_mean_over_ep_bad = np.mean(d_nc_perf_bad['ari'][max_epoch_bad])
            # ari_std_over_ep_bad = np.std(d_nc_perf_bad['ari'][max_epoch_bad])
            # nmi_mean_over_ep_bad = np.mean(d_nc_perf_bad['nmi'][max_epoch_bad])
            # nmi_std_over_ep_bad = np.std(d_nc_perf_bad['nmi'][max_epoch_bad])
            #
            # y_vmax = np.max([ari_mean_over_ep_good + ari_std_over_ep_good,
            #                  nmi_mean_over_ep_good + nmi_std_over_ep_good,
            #                  ari_mean_over_ep_bad + ari_std_over_ep_bad,
            #                  nmi_mean_over_ep_bad + nmi_std_over_ep_bad])
            # y_vmin = np.min([ari_mean_over_ep_good - ari_std_over_ep_good,
            #                  nmi_mean_over_ep_good - nmi_std_over_ep_good,
            #                  ari_mean_over_ep_bad - ari_std_over_ep_bad,
            #                  nmi_mean_over_ep_bad - nmi_std_over_ep_bad])
            # y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)
            #
            # good_plot_color = 'tab:blue'
            # bad_plot_color = 'tab:orange'
            # axes[idx].grid(True)
            # axes[idx].set_title('ARI & AMI', fontsize=10, fontweight='semibold')
            # axes[idx].errorbar([0, 2], [ari_mean_over_ep_good, nmi_mean_over_ep_good],
            #                    yerr=[ari_std_over_ep_good, nmi_std_over_ep_good],
            #                    fmt='o', c=good_plot_color, capsize=2, capthick=1)
            # axes[idx].errorbar([1, 3], [ari_mean_over_ep_bad, nmi_mean_over_ep_bad],
            #                    yerr=[ari_std_over_ep_bad, nmi_std_over_ep_bad],
            #                    fmt='o', c=bad_plot_color, capsize=2, capthick=1)
            # axes[idx].set_xticks([0, 1, 2, 3])
            # axes[idx].set_xticklabels(['ARI_good', 'ARI_bad', 'AMI_good', 'AMI_bad'])
            # if not np.allclose(y_stride, 0.0):
            #     axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw ADJ-DIFF
            if final_or_init == 'final':
                img_name = img_name_prefix + '_final_only_adj_diff@' + graph_name + '@' + spec_seq_param_name
            else:
                img_name = img_name_prefix + '_init_only_adj_diff@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_good[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                adj_diff_mean_over_ep_good = np.mean([item for item in d_adj_diff_good[K][max_epoch_good]], axis=0)
                adj_diff_std_over_ep_good = np.std([item for item in d_adj_diff_good[K][max_epoch_good]], axis=0)
                adj_diff_mean_over_ep_bad = np.mean([item for item in d_adj_diff_bad[K][max_epoch_bad]], axis=0)
                adj_diff_std_over_ep_bad = np.std([item for item in d_adj_diff_bad[K][max_epoch_bad]], axis=0)

                y_vmax_good = np.max(adj_diff_mean_over_ep_good + adj_diff_std_over_ep_good)
                y_vmax_bad = np.max(adj_diff_mean_over_ep_bad + adj_diff_std_over_ep_bad)
                y_vmin_good = np.min(adj_diff_mean_over_ep_good - adj_diff_std_over_ep_good)
                y_min_bad = np.min(adj_diff_mean_over_ep_bad - adj_diff_std_over_ep_bad)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                good_vs_bad_max_mean_diff = np.max(np.abs(adj_diff_mean_over_ep_good - adj_diff_mean_over_ep_bad))

                good_plot_color = 'tab:blue'
                bad_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('ADJ-DIFF @ K=%s: Max Mean Diff=%s' % (K, np.round(good_vs_bad_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], adj_diff_mean_over_ep_good,
                                   yerr=adj_diff_std_over_ep_good,
                                   fmt='-o', c=good_plot_color, capsize=2, capthick=1, label='good')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], adj_diff_mean_over_ep_bad,
                                   yerr=adj_diff_std_over_ep_bad,
                                   fmt='-o', c=bad_plot_color, capsize=2, capthick=1, label='bad')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw IN-AGG
            if final_or_init == 'final':
                img_name = img_name_prefix + '_final_only_in_agg@' + graph_name + '@' + spec_seq_param_name
            else:
                img_name = img_name_prefix + '_init_only_in_agg@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_good[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                in_agg_mean_over_ep_good = np.mean([item for item in d_in_agg_good[K][max_epoch_good]], axis=0)
                in_agg_std_over_ep_good = np.std([item for item in d_in_agg_good[K][max_epoch_good]], axis=0)
                in_agg_mean_over_ep_bad = np.mean([item for item in d_in_agg_bad[K][max_epoch_bad]], axis=0)
                in_agg_std_over_ep_bad = np.std([item for item in d_in_agg_bad[K][max_epoch_bad]], axis=0)

                y_vmax_good = np.max(in_agg_mean_over_ep_good + in_agg_std_over_ep_good)
                y_vmax_bad = np.max(in_agg_mean_over_ep_bad + in_agg_std_over_ep_bad)
                y_vmin_good = np.min(in_agg_mean_over_ep_good - in_agg_std_over_ep_good)
                y_min_bad = np.min(in_agg_mean_over_ep_bad - in_agg_std_over_ep_bad)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                good_vs_bad_max_mean_diff = np.max(np.abs(in_agg_mean_over_ep_good - in_agg_mean_over_ep_bad))

                good_plot_color = 'tab:blue'
                bad_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('IN-AGG @ K=%s: Max Mean Diff=%s' % (K, np.round(good_vs_bad_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], in_agg_mean_over_ep_good,
                                   yerr=in_agg_std_over_ep_good,
                                   fmt='-o', c=good_plot_color, capsize=2, capthick=1, label='good')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], in_agg_mean_over_ep_bad,
                                   yerr=in_agg_std_over_ep_bad,
                                   fmt='-o', c=bad_plot_color, capsize=2, capthick=1, label='bad')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw LN-VX
            if final_or_init == 'final':
                img_name = img_name_prefix + '_final_only_ln_vx@' + graph_name + '@' + spec_seq_param_name
            else:
                img_name = img_name_prefix + '_init_only_ln_vx@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_good[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                ln_vx_mean_over_ep_good = np.mean([item for item in d_ln_vx_good[K][max_epoch_good]], axis=0)
                ln_vx_std_over_ep_good = np.std([item for item in d_ln_vx_good[K][max_epoch_good]], axis=0)
                ln_vx_mean_over_ep_bad = np.mean([item for item in d_ln_vx_bad[K][max_epoch_bad]], axis=0)
                ln_vx_std_over_ep_bad = np.std([item for item in d_ln_vx_bad[K][max_epoch_bad]], axis=0)

                y_vmax_good = np.max(ln_vx_mean_over_ep_good + ln_vx_std_over_ep_good)
                y_vmax_bad = np.max(ln_vx_mean_over_ep_bad + ln_vx_std_over_ep_bad)
                y_vmin_good = np.min(ln_vx_mean_over_ep_good - ln_vx_std_over_ep_good)
                y_min_bad = np.min(ln_vx_mean_over_ep_bad - ln_vx_std_over_ep_bad)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                good_vs_bad_max_mean_diff = np.max(np.abs(ln_vx_mean_over_ep_good - ln_vx_mean_over_ep_bad))

                good_plot_color = 'tab:blue'
                bad_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('LN-VX @ K=%s: Max Mean Diff=%s' % (K, np.round(good_vs_bad_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ln_vx_mean_over_ep_good,
                                   yerr=ln_vx_std_over_ep_good,
                                   fmt='-o', c=good_plot_color, capsize=2, capthick=1, label='good')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ln_vx_mean_over_ep_bad,
                                   yerr=ln_vx_std_over_ep_bad,
                                   fmt='-o', c=bad_plot_color, capsize=2, capthick=1, label='bad')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw APPRX-LS
            if final_or_init == 'final':
                img_name = img_name_prefix + '_final_only_apprx_ls@' + graph_name + '@' + spec_seq_param_name
            else:
                img_name = img_name_prefix + '_init_only_apprx_ls@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_good[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                apprx_ls_mean_over_ep_good = np.mean([item for item in d_apprx_ls_good[K][max_epoch_good]], axis=0)
                apprx_ls_std_over_ep_good = np.std([item for item in d_apprx_ls_good[K][max_epoch_good]], axis=0)
                apprx_ls_mean_over_ep_bad = np.mean([item for item in d_apprx_ls_bad[K][max_epoch_bad]], axis=0)
                apprx_ls_std_over_ep_bad = np.std([item for item in d_apprx_ls_bad[K][max_epoch_bad]], axis=0)

                y_vmax_good = np.max(apprx_ls_mean_over_ep_good + apprx_ls_std_over_ep_good)
                y_vmax_bad = np.max(apprx_ls_mean_over_ep_bad + apprx_ls_std_over_ep_bad)
                y_vmin_good = np.min(apprx_ls_mean_over_ep_good - apprx_ls_std_over_ep_good)
                y_min_bad = np.min(apprx_ls_mean_over_ep_bad - apprx_ls_std_over_ep_bad)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                good_vs_bad_max_mean_diff = np.max(np.abs(apprx_ls_mean_over_ep_good - apprx_ls_mean_over_ep_bad))

                good_plot_color = 'tab:blue'
                bad_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('APPRX-LS @ K=%s: Max Mean Diff=%s' % (K, np.round(good_vs_bad_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], apprx_ls_mean_over_ep_good,
                                   yerr=apprx_ls_std_over_ep_good,
                                   fmt='-o', c=good_plot_color, capsize=2, capthick=1, label='good')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ln_vx_mean_over_ep_bad,
                                   yerr=apprx_ls_std_over_ep_bad,
                                   fmt='-o', c=bad_plot_color, capsize=2, capthick=1, label='bad')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw ENS
            if final_or_init == 'final':
                img_name = img_name_prefix + '_final_only_ens@' + graph_name + '@' + spec_seq_param_name
            else:
                img_name = img_name_prefix + '_init_only_ens@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_good[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                ens_mean_over_ep_good = np.mean([item for item in d_ens_good[K][max_epoch_good]], axis=0)
                ens_std_over_ep_good = np.std([item for item in d_ens_good[K][max_epoch_good]], axis=0)
                ens_mean_over_ep_bad = np.mean([item for item in d_ens_bad[K][max_epoch_good]], axis=0)
                ens_std_over_ep_bad = np.std([item for item in d_ens_bad[K][max_epoch_good]], axis=0)

                y_vmax_good = np.max(ens_mean_over_ep_good + ens_std_over_ep_good)
                y_vmax_bad = np.max(ens_mean_over_ep_bad + ens_std_over_ep_bad)
                y_vmin_good = np.min(ens_mean_over_ep_good - ens_std_over_ep_good)
                y_min_bad = np.min(ens_mean_over_ep_bad - ens_std_over_ep_bad)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                good_vs_bad_max_mean_diff = np.max(np.abs(ens_mean_over_ep_good - ens_mean_over_ep_bad))

                good_plot_color = 'tab:blue'
                bad_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('ENS @ K=%s: Max Mean Diff=%s' % (K, np.round(good_vs_bad_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ens_mean_over_ep_good,
                                   yerr=ens_std_over_ep_good,
                                   fmt='-o', c=good_plot_color, capsize=2, capthick=1, label='good')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ens_mean_over_ep_bad,
                                   yerr=ens_std_over_ep_bad,
                                   fmt='-o', c=bad_plot_color, capsize=2, capthick=1, label='bad')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

        if STAGE_draw_good_vs_bad_both_final_and_init:
            save_img = True
            show_img = True
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            all_runs = False
            if all_runs:
                img_name_prefix = 'all'
                good_vs_bad_draw_data_save_name = all_draw_data_save_name
            else:
                img_name_prefix = 'good_vs_bad'
            norm_sig = True
            no_in_agg = True
            sgs_chosen = False
            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_sgs_chosen'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_sgs_chosen'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_good_vs_bad_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name +'@'
                                                      + graph_name + '@' + spec_seq_param_name + '.pickle')

            d_metrics_good = df_good_vs_bad_draw_data.loc['good']['d_metrics']
            d_eig_vals_good = df_good_vs_bad_draw_data.loc['good']['d_eig_vals']
            d_adj_diff_good = df_good_vs_bad_draw_data.loc['good']['d_adj_diff']
            d_in_agg_good = df_good_vs_bad_draw_data.loc['good']['d_in_agg']
            d_ln_vx_good = df_good_vs_bad_draw_data.loc['good']['d_ln_vx']
            d_ens_good = df_good_vs_bad_draw_data.loc['good']['d_ens']
            d_nc_perf_good = df_good_vs_bad_draw_data.loc['good']['d_nc_perf']

            d_metrics_bad = df_good_vs_bad_draw_data.loc['bad']['d_metrics']
            d_eig_vals_bad = df_good_vs_bad_draw_data.loc['bad']['d_eig_vals']
            d_adj_diff_bad = df_good_vs_bad_draw_data.loc['bad']['d_adj_diff']
            d_in_agg_bad = df_good_vs_bad_draw_data.loc['bad']['d_in_agg']
            d_ln_vx_bad = df_good_vs_bad_draw_data.loc['bad']['d_ln_vx']
            d_ens_bad = df_good_vs_bad_draw_data.loc['bad']['d_ens']
            d_nc_perf_bad = df_good_vs_bad_draw_data.loc['bad']['d_nc_perf']

            l_K = sorted(d_eig_vals_good.keys())
            max_epoch_good = np.max(list(d_metrics_good['tv_loss'].keys()))
            max_epoch_bad = np.max(list(d_metrics_bad['tv_loss'].keys()))
            min_epoch_good = np.min(list(d_metrics_good['tv_loss'].keys()))
            min_epoch_bad = np.min(list(d_metrics_bad['tv_loss'].keys()))

            if no_in_agg:
                img_suffix = '_no_in_agg'
            else:
                img_suffix = ''
            # >>> Draw metrics
            img_name = img_name_prefix + '_final_init_metrics@' + graph_name + '@' + spec_seq_param_name
            fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(5, 4))

            idx = 0
            tv_mean_over_ep_good = np.mean(d_metrics_good['tv_loss'][max_epoch_good])
            tv_std_over_ep_good = np.std(d_metrics_good['tv_loss'][max_epoch_good])
            tv_mean_over_ep_bad = np.mean(d_metrics_bad['tv_loss'][max_epoch_bad])
            tv_std_over_ep_bad = np.std(d_metrics_bad['tv_loss'][max_epoch_bad])

            y_vmax = np.max([np.max(d_metrics_good['tv_loss'][max_epoch_good]), np.max(d_metrics_bad['tv_loss'][max_epoch_bad])])
            y_vmin = np.min([np.min(d_metrics_good['tv_loss'][max_epoch_good]), np.min(d_metrics_bad['tv_loss'][max_epoch_bad])])
            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

            good_plot_color = 'tab:blue'
            bad_plot_color = 'tab:orange'
            axes[idx].grid(True)
            # axes.set_title('Globally Weighted Total Variation', fontsize=10, fontweight='semibold')
            sns.kdeplot(d_metrics_good['tv_loss'][max_epoch_good], ax=axes[idx], color='tab:orange', cut=0, fill=True, label='TV_good_final')
            sns.kdeplot(d_metrics_bad['tv_loss'][max_epoch_bad], ax=axes[idx], color='tab:blue', cut=0, fill=True, label='TV_bad_final')
            axes[idx].set_xticks(np.round([i for i in np.arange(y_vmin, y_vmax + 0.05, 0.05)], decimals=2))
            axes[idx].legend()
            # axes.set_xticklabels(['tv_good', 'tv_bad'])
            # if not np.allclose(y_stride, 0.0):
            #     axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
            idx += 1

            tv_mean_over_ep_good = np.mean(d_metrics_good['tv_loss'][min_epoch_good])
            tv_std_over_ep_good = np.std(d_metrics_good['tv_loss'][min_epoch_good])
            tv_mean_over_ep_bad = np.mean(d_metrics_bad['tv_loss'][min_epoch_good])
            tv_std_over_ep_bad = np.std(d_metrics_bad['tv_loss'][min_epoch_good])

            y_vmax = np.max([np.max(d_metrics_good['tv_loss'][min_epoch_good]), np.max(d_metrics_bad['tv_loss'][min_epoch_good])])
            y_vmin = np.min([np.min(d_metrics_good['tv_loss'][min_epoch_good]), np.min(d_metrics_bad['tv_loss'][min_epoch_good])])
            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

            good_plot_color = 'tab:blue'
            bad_plot_color = 'tab:orange'
            axes[idx].grid(True)
            # axes.set_title('Globally Weighted Total Variation', fontsize=10, fontweight='semibold')
            sns.kdeplot(d_metrics_good['tv_loss'][min_epoch_good], ax=axes[idx], color='tab:orange', cut=0, fill=True, label='TV_good_init')
            sns.kdeplot(d_metrics_bad['tv_loss'][min_epoch_good], ax=axes[idx], color='tab:blue', cut=0, fill=True, label='TV_bad_init')
            axes[idx].set_xticks(np.round([i for i in np.arange(y_vmin, y_vmax + 0.1, 0.1)], decimals=2))
            axes[idx].legend()

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

        if STAGE_draw_all_both_final_and_init:
            save_img = True
            show_img = False
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            sgs_chosen = False
            norm_sig = False
            no_in_agg = True
            all_good_and_bad = False
            # good_or_bad_only = '_good'
            good_or_bad_only = None
            if norm_sig:
                img_name_prefix = 'all_norm'
            else:
                img_name_prefix = 'all_non_norm'
            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = all_draw_data_save_name + '_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = all_draw_data_save_name + '_norm_sgs_chosen'
                else:
                    draw_data_save_name = all_draw_data_save_name + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = all_draw_data_save_name + '_non_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = all_draw_data_save_name + '_non_norm_sgs_chosen'
                else:
                    draw_data_save_name = all_draw_data_save_name + '_non_norm'
            if all_good_and_bad:
                draw_data_save_name += '_all_good_bad'
            if good_or_bad_only is not None:
                draw_data_save_name += good_or_bad_only
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_all_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name + '@'
                                              + graph_name + '@' + spec_seq_param_name + '.pickle')

            d_metrics_all = df_all_draw_data.loc['all']['d_metrics']
            d_eig_vals_all = df_all_draw_data.loc['all']['d_eig_vals']
            d_apprx_ls_all = df_all_draw_data.loc['all']['d_apprx_ls']
            d_adj_diff_all = df_all_draw_data.loc['all']['d_adj_diff']
            d_in_agg_all = df_all_draw_data.loc['all']['d_in_agg']
            d_ln_vx_all = df_all_draw_data.loc['all']['d_ln_vx']
            d_ens_all = df_all_draw_data.loc['all']['d_ens']
            d_nc_perf_all = df_all_draw_data.loc['all']['d_nc_perf']

            l_K = sorted(d_eig_vals_all.keys())
            l_epoch = sorted(list(d_metrics_all['tv_loss'].keys()))
            sas_xtick_stride = math.ceil(len(l_epoch) / 35)
            l_sas_xticks = [i for i in range(0, len(l_epoch), sas_xtick_stride)]
            l_sas_xticklabels = [l_epoch[i] for i in l_sas_xticks]

            if no_in_agg:
                img_suffix = '_no_in_agg'
            elif sgs_chosen:
                img_suffix = '_sgs_chosen'
            else:
                img_suffix = ''

            max_epoch = np.max(list(d_metrics_all['tv_loss'].keys()))
            min_epoch = np.min(list(d_metrics_all['tv_loss'].keys()))

            img_name = img_name_prefix + '_init_final_metrics@' + graph_name + '@' + spec_seq_param_name
            
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(5, 3))
            tv_mean_over_ep_final = np.mean(d_metrics_all['tv_loss'][max_epoch])
            tv_std_over_ep_final = np.std(d_metrics_all['tv_loss'][max_epoch])
            tv_mean_over_ep_init = np.mean(d_metrics_all['tv_loss'][min_epoch])
            tv_std_over_ep_init = np.std(d_metrics_all['tv_loss'][min_epoch])

            y_vmax = np.max([np.max(d_metrics_all['tv_loss'][max_epoch]), np.max(d_metrics_all['tv_loss'][min_epoch])])
            y_vmin = np.min([np.min(d_metrics_all['tv_loss'][max_epoch]), np.min(d_metrics_all['tv_loss'][min_epoch])])
            y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

            final_plot_color = 'tab:blue'
            init_plot_color = 'tab:orange'
            axes.grid(True)
            axes.set_title('Globally Weighted Total Variation', fontsize=10, fontweight='semibold')
            sns.kdeplot(d_metrics_all['tv_loss'][max_epoch], ax=axes, color=final_plot_color, cut=0, fill=True, label='TV_final')
            sns.kdeplot(d_metrics_all['tv_loss'][min_epoch], ax=axes, color=init_plot_color, cut=0, fill=True, label='TV_init')
            axes.set_xticks(np.round([i for i in np.arange(y_vmin, y_vmax + 0.1, 0.1)], decimals=2))
            axes.legend()

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw ADJ-DIFF
            img_name = img_name_prefix + '_init_final_adj_diff@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_all[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                adj_diff_mean_over_ep_final = np.mean([item for item in d_adj_diff_all[K][max_epoch]], axis=0)
                adj_diff_std_over_ep_final = np.std([item for item in d_adj_diff_all[K][max_epoch]], axis=0)
                adj_diff_mean_over_ep_init = np.mean([item for item in d_adj_diff_all[K][min_epoch]], axis=0)
                adj_diff_std_over_ep_init = np.std([item for item in d_adj_diff_all[K][min_epoch]], axis=0)

                y_vmax_good = np.max(adj_diff_mean_over_ep_final + adj_diff_std_over_ep_final)
                y_vmax_bad = np.max(adj_diff_mean_over_ep_init + adj_diff_std_over_ep_init)
                y_vmin_good = np.min(adj_diff_mean_over_ep_final - adj_diff_std_over_ep_final)
                y_min_bad = np.min(adj_diff_mean_over_ep_init - adj_diff_std_over_ep_init)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                init_vs_final_max_mean_diff = np.max(np.abs(adj_diff_mean_over_ep_final - adj_diff_mean_over_ep_init))

                final_plot_color = 'tab:blue'
                init_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('ADJ-DIFF @ K=%s: Max Mean Diff=%s' % (K, np.round(init_vs_final_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], adj_diff_mean_over_ep_final,
                                   yerr=adj_diff_std_over_ep_final,
                                   fmt='-o', c=final_plot_color, capsize=2, capthick=1, label='final')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], adj_diff_mean_over_ep_init,
                                   yerr=adj_diff_std_over_ep_init,
                                   fmt='-o', c=init_plot_color, capsize=2, capthick=1, label='init')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw APPRX-LS
            img_name = img_name_prefix + '_init_final_apprx_ls@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_all[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                apprx_ls_mean_over_ep_final = np.mean([item for item in d_apprx_ls_all[K][max_epoch]], axis=0)
                apprx_ls_std_over_ep_final = np.std([item for item in d_apprx_ls_all[K][max_epoch]], axis=0)
                apprx_ls_mean_over_ep_init = np.mean([item for item in d_apprx_ls_all[K][min_epoch]], axis=0)
                apprx_ls_std_over_ep_init = np.std([item for item in d_apprx_ls_all[K][min_epoch]], axis=0)

                y_vmax_good = np.max(apprx_ls_mean_over_ep_final + apprx_ls_std_over_ep_final)
                y_vmax_bad = np.max(apprx_ls_mean_over_ep_init + apprx_ls_std_over_ep_init)
                y_vmin_good = np.min(apprx_ls_mean_over_ep_final - apprx_ls_std_over_ep_final)
                y_min_bad = np.min(apprx_ls_mean_over_ep_init - apprx_ls_std_over_ep_init)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                init_vs_final_max_mean_diff = np.max(np.abs(apprx_ls_mean_over_ep_final - apprx_ls_mean_over_ep_init))

                final_plot_color = 'tab:blue'
                init_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('APPRX-LS @ K=%s: Max Mean Diff=%s' % (K, np.round(init_vs_final_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], apprx_ls_mean_over_ep_final,
                                   yerr=apprx_ls_std_over_ep_final,
                                   fmt='-o', c=final_plot_color, capsize=2, capthick=1, label='final')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], apprx_ls_mean_over_ep_init,
                                   yerr=apprx_ls_std_over_ep_init,
                                   fmt='-o', c=init_plot_color, capsize=2, capthick=1, label='init')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw IN-AGG
            img_name = img_name_prefix + '_init_final_in_agg@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_all[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                in_agg_mean_over_ep_final = np.mean([item for item in d_in_agg_all[K][max_epoch]], axis=0)
                in_agg_std_over_ep_final = np.std([item for item in d_in_agg_all[K][max_epoch]], axis=0)
                in_agg_mean_over_ep_init = np.mean([item for item in d_in_agg_all[K][min_epoch]], axis=0)
                in_agg_std_over_ep_init = np.std([item for item in d_in_agg_all[K][min_epoch]], axis=0)

                y_vmax_good = np.max(in_agg_mean_over_ep_final + in_agg_std_over_ep_final)
                y_vmax_bad = np.max(in_agg_mean_over_ep_init + in_agg_std_over_ep_init)
                y_vmin_good = np.min(in_agg_mean_over_ep_final - in_agg_std_over_ep_final)
                y_min_bad = np.min(in_agg_mean_over_ep_init - in_agg_std_over_ep_init)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                init_vs_final_max_mean_diff = np.max(np.abs(in_agg_mean_over_ep_final - in_agg_mean_over_ep_init))

                final_plot_color = 'tab:blue'
                init_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('IN-AGG @ K=%s: Max Mean Diff=%s' % (K, np.round(init_vs_final_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], in_agg_mean_over_ep_final,
                                   yerr=in_agg_std_over_ep_final,
                                   fmt='-o', c=final_plot_color, capsize=2, capthick=1, label='final')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], in_agg_mean_over_ep_init,
                                   yerr=in_agg_std_over_ep_init,
                                   fmt='-o', c=init_plot_color, capsize=2, capthick=1, label='init')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw LN-VX
            img_name = img_name_prefix + '_init_final_ln_vx@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_all[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                ln_vx_mean_over_ep_final = np.mean([item for item in d_ln_vx_all[K][max_epoch]], axis=0)
                ln_vx_std_over_ep_final = np.std([item for item in d_ln_vx_all[K][max_epoch]], axis=0)
                ln_vx_mean_over_ep_init = np.mean([item for item in d_ln_vx_all[K][min_epoch]], axis=0)
                ln_vx_std_over_ep_init = np.std([item for item in d_ln_vx_all[K][min_epoch]], axis=0)

                y_vmax_good = np.max(ln_vx_mean_over_ep_final + ln_vx_std_over_ep_final)
                y_vmax_bad = np.max(ln_vx_mean_over_ep_init + ln_vx_std_over_ep_init)
                y_vmin_good = np.min(ln_vx_mean_over_ep_final - ln_vx_std_over_ep_final)
                y_min_bad = np.min(ln_vx_mean_over_ep_init - ln_vx_std_over_ep_init)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                init_vs_final_max_mean_diff = np.max(np.abs(ln_vx_mean_over_ep_final - ln_vx_mean_over_ep_init))

                final_plot_color = 'tab:blue'
                init_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('LN-VX @ K=%s: Max Mean Diff=%s' % (K, np.round(init_vs_final_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ln_vx_mean_over_ep_final,
                                   yerr=ln_vx_std_over_ep_final,
                                   fmt='-o', c=final_plot_color, capsize=2, capthick=1, label='final')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ln_vx_mean_over_ep_init,
                                   yerr=ln_vx_std_over_ep_init,
                                   fmt='-o', c=init_plot_color, capsize=2, capthick=1, label='init')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> Draw ENS
            img_name = img_name_prefix + '_init_final_ens@' + graph_name + '@' + spec_seq_param_name
            fig_height = len(l_K) * 2.5
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(10, fig_height))

            idx = 0
            for K in l_K:
                l_eig_vals = d_eig_vals_all[K]
                l_eff_eig_vals = [eig_val for eig_val in l_eig_vals if not np.allclose(eig_val, 0.0)]
                num_all_eig_vals = len(l_eig_vals)
                num_eff_eig_vals = len(l_eff_eig_vals)

                ens_mean_over_ep_final = np.mean([item for item in d_ens_all[K][max_epoch]], axis=0)
                ens_std_over_ep_final = np.std([item for item in d_ens_all[K][max_epoch]], axis=0)
                ens_mean_over_ep_init = np.mean([item for item in d_ens_all[K][min_epoch]], axis=0)
                ens_std_over_ep_init = np.std([item for item in d_ens_all[K][min_epoch]], axis=0)

                y_vmax_good = np.max(ens_mean_over_ep_final + ens_std_over_ep_final)
                y_vmax_bad = np.max(ens_mean_over_ep_init + ens_std_over_ep_init)
                y_vmin_good = np.min(ens_mean_over_ep_final - ens_std_over_ep_final)
                y_min_bad = np.min(ens_mean_over_ep_init - ens_std_over_ep_init)
                y_vmax = np.max([y_vmax_good, y_vmax_bad])
                y_vmin = np.min([y_vmin_good, y_min_bad])
                y_stride = np.round((y_vmax - y_vmin) / 8, decimals=2)

                init_vs_final_max_mean_diff = np.max(np.abs(ens_mean_over_ep_final - ens_mean_over_ep_init))

                final_plot_color = 'tab:blue'
                init_plot_color = 'tab:orange'
                axes[idx].grid(True)
                axes[idx].set_title('ENS @ K=%s: Max Mean Diff=%s' % (K, np.round(init_vs_final_max_mean_diff, decimals=4)), fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ens_mean_over_ep_final,
                                   yerr=ens_std_over_ep_final,
                                   fmt='-o', c=final_plot_color, capsize=2, capthick=1, label='final')
                axes[idx].errorbar([i for i in range(num_all_eig_vals)], ens_mean_over_ep_init,
                                   yerr=ens_std_over_ep_init,
                                   fmt='-o', c=init_plot_color, capsize=2, capthick=1, label='init')
                axes[idx].set_xticks([i for i in range(num_all_eig_vals)])
                axes[idx].set_xticklabels(np.round(l_eig_vals, decimals=3))
                axes[idx].set_yticks(np.round([i for i in np.arange(y_vmin, y_vmax, y_stride)], decimals=2))
                axes[idx].legend()
                idx += 1

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + img_name + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

        if STAGE_tv_sas_relation:
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            norm_sig = True
            init_or_final = 'final'
            if norm_sig:
                out_file_suffix = '_norm'
            else:
                out_file_suffix = '_non_norm'
            if init_or_final == 'init':
                out_file_suffix += '_init'
            elif init_or_final == 'final':
                out_file_suffix += '_final'
            no_in_agg = True
            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_no_in_agg'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_no_in_agg'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_good_vs_bad_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name + '@'
                                                      + graph_name + '@' + spec_seq_param_name + '.pickle')

            d_metrics_good = df_good_vs_bad_draw_data.loc['good']['d_metrics']
            d_eig_vals_good = df_good_vs_bad_draw_data.loc['good']['d_eig_vals']
            d_adj_diff_good = df_good_vs_bad_draw_data.loc['good']['d_adj_diff']
            d_in_agg_good = df_good_vs_bad_draw_data.loc['good']['d_in_agg']
            d_ln_vx_good = df_good_vs_bad_draw_data.loc['good']['d_ln_vx']
            d_apprx_ls_good = df_good_vs_bad_draw_data.loc['good']['d_apprx_ls']
            d_ens_good = df_good_vs_bad_draw_data.loc['good']['d_ens']
            d_nc_perf_good = df_good_vs_bad_draw_data.loc['good']['d_nc_perf']

            d_metrics_bad = df_good_vs_bad_draw_data.loc['bad']['d_metrics']
            d_eig_vals_bad = df_good_vs_bad_draw_data.loc['bad']['d_eig_vals']
            d_adj_diff_bad = df_good_vs_bad_draw_data.loc['bad']['d_adj_diff']
            d_in_agg_bad = df_good_vs_bad_draw_data.loc['bad']['d_in_agg']
            d_ln_vx_bad = df_good_vs_bad_draw_data.loc['bad']['d_ln_vx']
            d_apprx_ls_bad = df_good_vs_bad_draw_data.loc['bad']['d_apprx_ls']
            d_ens_bad = df_good_vs_bad_draw_data.loc['bad']['d_ens']
            d_nc_perf_bad = df_good_vs_bad_draw_data.loc['bad']['d_nc_perf']

            l_K = sorted(d_eig_vals_good.keys())
            max_epoch_good = np.max(list(d_metrics_good['tv_loss'].keys()))
            max_epoch_bad = np.max(list(d_metrics_bad['tv_loss'].keys()))
            min_epoch_good = np.min(list(d_metrics_good['tv_loss'].keys()))
            min_epoch_bad = np.min(list(d_metrics_bad['tv_loss'].keys()))
            if init_or_final == 'init':
                ep_idx_good = min_epoch_good
                ep_idx_bad = min_epoch_bad
            elif init_or_final == 'final':
                ep_idx_good = max_epoch_good
                ep_idx_bad = max_epoch_bad

            num_good = len(d_metrics_good['tv_loss'][ep_idx_good])
            num_bad = len(d_metrics_bad['tv_loss'][ep_idx_bad])

            timer_start = time.time()

            # >>> good vs bad
            l_tv_sas_rec = []
            for good_idx in range(num_good):
                tv_good = d_metrics_good['tv_loss'][ep_idx_good][good_idx]
                ari_good = d_nc_perf_good['ari'][ep_idx_good][good_idx]
                ami_good = d_nc_perf_good['nmi'][ep_idx_good][good_idx]
                d_adj_diff_final_good = {K: d_adj_diff_good[K][ep_idx_good][good_idx] for K in d_adj_diff_good}
                d_in_agg_final_good = {K: d_in_agg_good[K][ep_idx_good][good_idx] for K in d_in_agg_good}
                d_ln_vx_final_good = {K: d_ln_vx_good[K][ep_idx_good][good_idx] for K in d_ln_vx_good}
                d_apprx_ls_final_good = {K: d_apprx_ls_good[K][ep_idx_good][good_idx] for K in d_apprx_ls_good}
                d_ens_final_good = {K: d_ens_good[K][ep_idx_good][good_idx] for K in d_ens_good}

                for bad_idx in range(num_bad):
                    tv_bad = d_metrics_bad['tv_loss'][ep_idx_bad][bad_idx]
                    ari_bad = d_nc_perf_bad['ari'][ep_idx_bad][bad_idx]
                    ami_bad = d_nc_perf_bad['nmi'][ep_idx_bad][bad_idx]
                    d_adj_diff_final_bad = {K: d_adj_diff_bad[K][ep_idx_bad][bad_idx] for K in d_adj_diff_bad}
                    d_in_agg_final_bad = {K: d_in_agg_bad[K][ep_idx_bad][bad_idx] for K in d_in_agg_bad}
                    d_ln_vx_final_bad = {K: d_ln_vx_bad[K][ep_idx_bad][bad_idx] for K in d_ln_vx_bad}
                    d_apprx_ls_final_bad = {K: d_apprx_ls_bad[K][ep_idx_bad][bad_idx] for K in d_apprx_ls_bad}
                    d_ens_final_bad = {K: d_ens_bad[K][ep_idx_bad][bad_idx] for K in d_ens_bad}

                    tv_delta = np.abs(tv_good - tv_bad)
                    d_adj_diff_delta = {K: np.mean(np.abs(d_adj_diff_final_good[K] - d_adj_diff_final_bad[K])) for K in l_K}
                    d_in_agg_delta = {K: np.mean(np.abs(d_in_agg_final_good[K] - d_in_agg_final_bad[K])) for K in l_K}
                    d_ln_vx_delta = {K: np.mean(np.abs(d_ln_vx_final_good[K] - d_ln_vx_final_bad[K])) for K in l_K}
                    d_apprx_ls_delta = {K: np.mean(np.abs(d_apprx_ls_final_good[K] - d_apprx_ls_final_bad[K])) for K in l_K}
                    d_ens_delta = {K: np.mean(np.abs(d_ens_final_good[K] - d_ens_final_bad[K])) for K in l_K}
                    
                    d_adj_diff_delta_full = {K: np.abs(d_adj_diff_final_good[K] - d_adj_diff_final_bad[K]) for K in l_K}
                    d_in_agg_delta_full = {K: np.abs(d_in_agg_final_good[K] - d_in_agg_final_bad[K]) for K in l_K}
                    d_ln_vx_delta_full = {K: np.abs(d_ln_vx_final_good[K] - d_ln_vx_final_bad[K]) for K in l_K}
                    d_apprx_ls_delta_full = {K: np.abs(d_apprx_ls_final_good[K] - d_apprx_ls_final_bad[K]) for K in l_K}
                    d_ens_delta_full = {K: np.abs(d_ens_final_good[K] - d_ens_final_bad[K]) for K in l_K}

                    l_tv_sas_rec.append((good_idx, bad_idx, tv_delta,
                                         d_adj_diff_final_good, d_in_agg_final_good, d_ln_vx_final_good, d_ens_final_good,
                                         d_adj_diff_final_bad, d_in_agg_final_bad, d_ln_vx_final_bad, d_ens_final_bad,
                                         d_adj_diff_delta, d_in_agg_delta, d_ln_vx_delta, d_ens_delta,
                                         d_adj_diff_delta_full, d_in_agg_delta_full, d_ln_vx_delta_full,
                                         d_apprx_ls_delta_full, d_ens_delta_full))
                    if len(l_tv_sas_rec) % 500 == 0 and len(l_tv_sas_rec) >= 500:
                        logging.debug('%s good bad pairs done in %s secs.'
                                      % (len(l_tv_sas_rec), time.time() - timer_start))
            logging.debug('all %s good bad pairs done in %s secs.'
                          % (len(l_tv_sas_rec), time.time() - timer_start))
            df_tv_sas = pd.DataFrame(l_tv_sas_rec, columns=['good_idx', 'bad_idx', 'tv_delta',
                                                            'd_adj_diff_final_good', 'd_in_agg_final_good', 'd_ln_vx_final_good', 'd_ens_final_good',
                                                            'd_adj_diff_final_bad', 'd_in_agg_final_bad', 'd_ln_vx_final_bad', 'd_ens_final_bad',
                                                            'd_adj_diff_delta', 'd_in_agg_delta', 'd_ln_vx_delta', 'd_ens_delta',
                                                            'd_adj_diff_delta_full', 'd_in_agg_delta_full', 'd_ln_vx_delta_full',
                                                            'd_apprx_ls_delta_full', 'd_ens_delta_full'])
            pd.to_pickle(df_tv_sas, good_and_bad_runs_save_path + 'good_vs_bad_tv_sas_relations' + out_file_suffix + '.pickle')

            # >>> good vs good
            l_tv_sas_rec = []
            for good_idx_i in range(0, num_good - 1):
                tv_good_i = d_metrics_good['tv_loss'][ep_idx_good][good_idx_i]
                d_adj_diff_final_good_i = {K: d_adj_diff_good[K][ep_idx_good][good_idx_i] for K in d_adj_diff_good}
                d_in_agg_final_good_i = {K: d_in_agg_good[K][ep_idx_good][good_idx_i]for K in d_in_agg_good}
                d_ln_vx_final_good_i = {K: d_ln_vx_good[K][ep_idx_good][good_idx_i] for K in d_ln_vx_good}
                d_apprx_ls_final_good_i = {K: d_apprx_ls_good[K][ep_idx_good][good_idx_i] for K in d_apprx_ls_good}
                d_ens_final_good_i = {K: d_ens_good[K][ep_idx_good][good_idx_i] for K in d_ens_good}

                for good_idx_j in range(good_idx_i + 1, num_good):
                    tv_good_j = d_metrics_good['tv_loss'][ep_idx_good][good_idx_j]
                    d_adj_diff_final_good_j = {K: d_adj_diff_good[K][ep_idx_good][good_idx_j] for K in d_adj_diff_good}
                    d_in_agg_final_good_j = {K: d_in_agg_good[K][ep_idx_good][good_idx_j] for K in d_in_agg_good}
                    d_ln_vx_final_good_j = {K: d_ln_vx_good[K][ep_idx_good][good_idx_j] for K in d_ln_vx_good}
                    d_apprx_ls_final_good_j = {K: d_apprx_ls_good[K][ep_idx_good][good_idx_j] for K in d_apprx_ls_good}
                    d_ens_final_good_j = {K: d_ens_good[K][ep_idx_good][good_idx_j] for K in d_ens_good}

                    tv_delta = np.abs(tv_good_i - tv_good_j)
                    d_adj_diff_delta = {K: np.mean(np.abs(d_adj_diff_final_good_i[K] - d_adj_diff_final_good_j[K])) for K in l_K}
                    d_in_agg_delta = {K: np.mean(np.abs(d_in_agg_final_good_i[K] - d_in_agg_final_good_j[K])) for K in l_K}
                    d_ln_vx_delta = {K: np.mean(np.abs(d_ln_vx_final_good_i[K] - d_ln_vx_final_good_j[K])) for K in l_K}
                    d_ens_delta = {K: np.mean(np.abs(d_ens_final_good_i[K] - d_ens_final_good_j[K])) for K in l_K}

                    d_adj_diff_delta_full = {K: np.abs(d_adj_diff_final_good_i[K] - d_adj_diff_final_good_j[K]) for K in l_K}
                    d_in_agg_delta_full = {K: np.abs(d_in_agg_final_good_i[K] - d_in_agg_final_good_j[K]) for K in l_K}
                    d_ln_vx_delta_full = {K: np.abs(d_ln_vx_final_good_i[K] - d_ln_vx_final_good_j[K]) for K in l_K}
                    d_apprx_ls_delta_full = {K: np.abs(d_apprx_ls_final_good_i[K] - d_apprx_ls_final_good_j[K]) for K in l_K}
                    d_ens_delta_full = {K: np.abs(d_ens_final_good_i[K] - d_ens_final_good_j[K]) for K in l_K}

                    l_tv_sas_rec.append((good_idx_i, good_idx_j, tv_delta,
                                         d_adj_diff_delta, d_in_agg_delta, d_ln_vx_delta, d_ens_delta,
                                         d_adj_diff_delta_full, d_in_agg_delta_full, d_ln_vx_delta_full,
                                         d_apprx_ls_delta_full, d_ens_delta_full))
                    if len(l_tv_sas_rec) % 500 == 0 and len(l_tv_sas_rec) >= 500:
                        logging.debug('%s good bad pairs done in %s secs.'
                                      % (len(l_tv_sas_rec), time.time() - timer_start))
            logging.debug('all %s good bad pairs done in %s secs.'
                          % (len(l_tv_sas_rec), time.time() - timer_start))
            df_tv_sas = pd.DataFrame(l_tv_sas_rec, columns=['good_idx_i', 'good_idx_j', 'tv_delta',
                                                            'd_adj_diff_delta', 'd_in_agg_delta', 'd_ln_vx_delta', 'd_ens_delta',
                                                            'd_adj_diff_delta_full', 'd_in_agg_delta_full', 'd_ln_vx_delta_full',
                                                            'd_apprx_ls_delta_full', 'd_ens_delta_full'])
            pd.to_pickle(df_tv_sas, good_and_bad_runs_save_path + 'good_vs_good_tv_sas_relations' + out_file_suffix + '.pickle')

            # >>> bad vs bad
            l_tv_sas_rec = []
            for bad_idx_i in range(0, num_bad - 1):
                tv_bad_i = d_metrics_bad['tv_loss'][ep_idx_bad][bad_idx_i]
                d_adj_diff_final_bad_i = {K: d_adj_diff_bad[K][ep_idx_bad][bad_idx_i] for K in d_adj_diff_bad}
                d_in_agg_final_bad_i = {K: d_in_agg_bad[K][ep_idx_bad][bad_idx_i] for K in d_in_agg_bad}
                d_ln_vx_final_bad_i = {K: d_ln_vx_bad[K][ep_idx_bad][bad_idx_i] for K in d_ln_vx_bad}
                d_apprx_ls_final_bad_i = {K: d_apprx_ls_bad[K][ep_idx_bad][bad_idx_i] for K in d_apprx_ls_bad}
                d_ens_final_bad_i = {K: d_ens_bad[K][ep_idx_bad][bad_idx_i] for K in d_ens_bad}

                for bad_idx_j in range(bad_idx_i + 1, num_bad):
                    tv_bad_j = d_metrics_bad['tv_loss'][ep_idx_bad][bad_idx_j]
                    d_adj_diff_final_bad_j = {K: d_adj_diff_bad[K][ep_idx_bad][bad_idx_j] for K in d_adj_diff_bad}
                    d_in_agg_final_bad_j = {K: d_in_agg_bad[K][ep_idx_bad][bad_idx_j] for K in d_in_agg_bad}
                    d_ln_vx_final_bad_j = {K: d_ln_vx_bad[K][ep_idx_bad][bad_idx_j] for K in d_ln_vx_bad}
                    d_apprx_ls_final_bad_j = {K: d_apprx_ls_bad[K][ep_idx_bad][bad_idx_j] for K in d_apprx_ls_bad}
                    d_ens_final_bad_j = {K: d_ens_bad[K][ep_idx_bad][bad_idx_j] for K in d_ens_bad}

                    tv_delta = np.abs(tv_bad_i - tv_bad_j)
                    d_adj_diff_delta = {K: np.mean(np.abs(d_adj_diff_final_bad_i[K] - d_adj_diff_final_bad_j[K])) for K in l_K}
                    d_in_agg_delta = {K: np.mean(np.abs(d_in_agg_final_bad_i[K] - d_in_agg_final_bad_j[K])) for K in l_K}
                    d_ln_vx_delta = {K: np.mean(np.abs(d_ln_vx_final_bad_i[K] - d_ln_vx_final_bad_j[K])) for K in l_K}
                    d_ens_delta = {K: np.mean(np.abs(d_ens_final_bad_i[K] - d_ens_final_bad_j[K])) for K in l_K}

                    d_adj_diff_delta_full = {K: np.abs(d_adj_diff_final_bad_i[K] - d_adj_diff_final_bad_j[K]) for K in l_K}
                    d_in_agg_delta_full = {K: np.abs(d_in_agg_final_bad_i[K] - d_in_agg_final_bad_j[K]) for K in l_K}
                    d_ln_vx_delta_full = {K: np.abs(d_ln_vx_final_bad_i[K] - d_ln_vx_final_bad_j[K]) for K in l_K}
                    d_apprx_ls_delta_full = {K: np.abs(d_apprx_ls_final_bad_i[K] - d_apprx_ls_final_bad_j[K]) for K in l_K}
                    d_ens_delta_full = {K: np.abs(d_ens_final_bad_i[K] - d_ens_final_bad_j[K]) for K in l_K}

                    l_tv_sas_rec.append((tv_bad_i, bad_idx_j, tv_delta,
                                         d_adj_diff_delta, d_in_agg_delta, d_ln_vx_delta, d_ens_delta,
                                         d_adj_diff_delta_full, d_in_agg_delta_full, d_ln_vx_delta_full,
                                         d_apprx_ls_delta_full, d_ens_delta_full))
                    if len(l_tv_sas_rec) % 500 == 0 and len(l_tv_sas_rec) >= 500:
                        logging.debug('%s good bad pairs done in %s secs.'
                                      % (len(l_tv_sas_rec), time.time() - timer_start))
            logging.debug('all %s good bad pairs done in %s secs.'
                          % (len(l_tv_sas_rec), time.time() - timer_start))
            df_tv_sas = pd.DataFrame(l_tv_sas_rec, columns=['bad_idx_i', 'bad_idx_j', 'tv_delta',
                                                            'd_adj_diff_delta', 'd_in_agg_delta', 'd_ln_vx_delta', 'd_ens_delta',
                                                            'd_adj_diff_delta_full', 'd_in_agg_delta_full', 'd_ln_vx_delta_full',
                                                            'd_apprx_ls_delta_full', 'd_ens_delta_full'])
            pd.to_pickle(df_tv_sas, good_and_bad_runs_save_path + 'bad_vs_bad_tv_sas_relations' + out_file_suffix + '.pickle')

        if STAGE_tv_sas_relation_analysis:
            save_img = True
            show_img = False
            no_in_agg = True
            norm_sig = True
            if norm_sig:
                out_file_suffix = '_norm'
            else:
                out_file_suffix = '_non_norm'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_tv_sas_good_vs_bad = pd.read_pickle(good_and_bad_runs_save_path + 'good_vs_bad_tv_sas_relations' + out_file_suffix + '.pickle')
            df_tv_sas_good_vs_good = pd.read_pickle(good_and_bad_runs_save_path + 'good_vs_good_tv_sas_relations' + out_file_suffix + '.pickle')
            df_tv_sas_bad_vs_bad = pd.read_pickle(good_and_bad_runs_save_path + 'bad_vs_bad_tv_sas_relations' + out_file_suffix + '.pickle')

            df_tv_sas_good_vs_bad_sorted = df_tv_sas_good_vs_bad.sort_values(by=['tv_delta'])
            df_tv_sas_good_vs_good_sorted = df_tv_sas_good_vs_good.sort_values(by=['tv_delta'])
            df_tv_sas_bad_vs_bad_sorted = df_tv_sas_bad_vs_bad.sort_values(by=['tv_delta'])

            np_tv_delta_good_vs_bad = np.asarray(list(df_tv_sas_good_vs_bad_sorted['tv_delta']))
            np_tv_delta_good_vs_good = np.asarray(list(df_tv_sas_good_vs_good_sorted['tv_delta']))
            np_tv_delta_bad_vs_bad = np.asarray(list(df_tv_sas_bad_vs_bad_sorted['tv_delta']))

            l_K = list(df_tv_sas_good_vs_bad['d_ens_delta'][0].keys())

            l_good_vs_bad_mean_ratio_sas_vs_delta_adj_diff = []
            l_good_vs_bad_mean_ratio_sas_vs_delta_in_agg = []
            l_good_vs_bad_mean_ratio_sas_vs_delta_ln_vx = []
            l_good_vs_bad_mean_ratio_sas_vs_delta_ens = []

            l_good_vs_bad_sas_delta_adj_diff = []
            l_good_vs_bad_sas_delta_in_agg = []
            l_good_vs_bad_sas_delta_ln_vx = []
            l_good_vs_bad_sas_delta_ens = []

            l_good_vs_good_sas_delta_adj_diff = []
            l_good_vs_good_sas_delta_in_agg = []
            l_good_vs_good_sas_delta_ln_vx = []
            l_good_vs_good_sas_delta_ens = []

            l_bad_vs_bad_sas_delta_adj_diff = []
            l_bad_vs_bad_sas_delta_in_agg = []
            l_bad_vs_bad_sas_delta_ln_vx = []
            l_bad_vs_bad_sas_delta_ens = []

            l_good_vs_bad_correcoef_tv_adj_diff = []
            l_good_vs_good_correcoef_tv_adj_diff = []
            l_bad_vs_bad_correcoef_tv_adj_diff = []

            l_good_vs_bad_correcoef_tv_in_agg = []
            l_good_vs_good_correcoef_tv_in_agg = []
            l_bad_vs_bad_correcoef_tv_in_agg = []

            l_good_vs_bad_correcoef_tv_ln_vx = []
            l_good_vs_good_correcoef_tv_ln_vx = []
            l_bad_vs_bad_correcoef_tv_ln_vx = []

            l_good_vs_bad_correcoef_tv_ens = []
            l_good_vs_good_correcoef_tv_ens = []
            l_bad_vs_bad_correcoef_tv_ens = []

            if norm_sig:
                img_suffix = '_norm'
            else:
                img_suffix = '_non_norm'
            if no_in_agg:
                img_suffix += '_no_in_agg'
            else:
                img_suffix += ''
            # noinspection PyUnboundLocalVariable
            for K in l_K:
                mean_sas_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_bad_sorted['d_adj_diff_final_good'])]))
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_bad_sorted['d_adj_diff_delta'])]))
                l_good_vs_bad_mean_ratio_sas_vs_delta_adj_diff.append(mean_delta_element / mean_sas_element)
                l_good_vs_bad_sas_delta_adj_diff.append(mean_delta_element)
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_good_sorted['d_adj_diff_delta'])]))
                l_good_vs_good_sas_delta_adj_diff.append(mean_delta_element)
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_bad_vs_bad_sorted['d_adj_diff_delta'])]))
                l_bad_vs_bad_sas_delta_adj_diff.append(mean_delta_element)

                mean_sas_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_bad_sorted['d_in_agg_final_good'])]))
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_bad_sorted['d_in_agg_delta'])]))
                l_good_vs_bad_mean_ratio_sas_vs_delta_in_agg.append(mean_delta_element / mean_sas_element)
                l_good_vs_bad_sas_delta_in_agg.append(mean_delta_element)
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_good_sorted['d_in_agg_delta'])]))
                l_good_vs_good_sas_delta_in_agg.append(mean_delta_element)
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_bad_vs_bad_sorted['d_in_agg_delta'])]))
                l_bad_vs_bad_sas_delta_in_agg.append(mean_delta_element)

                mean_sas_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_bad_sorted['d_ln_vx_final_good'])]))
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_bad_sorted['d_ln_vx_delta'])]))
                l_good_vs_bad_mean_ratio_sas_vs_delta_ln_vx.append(mean_delta_element / mean_sas_element)
                l_good_vs_bad_sas_delta_ln_vx.append(mean_delta_element)
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_good_sorted['d_ln_vx_delta'])]))
                l_good_vs_good_sas_delta_ln_vx.append(mean_delta_element)
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_bad_vs_bad_sorted['d_ln_vx_delta'])]))
                l_bad_vs_bad_sas_delta_ln_vx.append(mean_delta_element)

                mean_sas_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_bad_sorted['d_ens_final_good'])]))
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_bad_sorted['d_ens_delta'])]))
                l_good_vs_bad_mean_ratio_sas_vs_delta_ens.append(mean_delta_element / mean_sas_element)
                l_good_vs_bad_sas_delta_ens.append(mean_delta_element)
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_good_vs_good_sorted['d_ens_delta'])]))
                l_good_vs_good_sas_delta_ens.append(mean_delta_element)
                mean_delta_element = np.mean(np.asarray([d_ens_delta[K] for d_ens_delta in list(df_tv_sas_bad_vs_bad_sorted['d_ens_delta'])]))
                l_bad_vs_bad_sas_delta_ens.append(mean_delta_element)

                # >>> good vs bad correcoef
                good_vs_bad_corrcoef_tv_adj_diff = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_good_vs_bad_sorted['d_adj_diff_delta']]),
                                                               np_tv_delta_good_vs_bad)
                l_good_vs_bad_correcoef_tv_adj_diff.append(good_vs_bad_corrcoef_tv_adj_diff[0][1])
                good_vs_bad_corrcoef_tv_in_agg = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_good_vs_bad_sorted['d_in_agg_delta']]),
                                                               np_tv_delta_good_vs_bad)
                l_good_vs_bad_correcoef_tv_in_agg.append(good_vs_bad_corrcoef_tv_in_agg[0][1])
                good_vs_bad_corrcoef_tv_ln_vx = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_good_vs_bad_sorted['d_ln_vx_delta']]),
                                                             np_tv_delta_good_vs_bad)
                l_good_vs_bad_correcoef_tv_ln_vx.append(good_vs_bad_corrcoef_tv_ln_vx[0][1])
                good_vs_bad_corrcoef_tv_ens = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_good_vs_bad_sorted['d_ens_delta']]),
                                                            np_tv_delta_good_vs_bad)
                l_good_vs_bad_correcoef_tv_ens.append(good_vs_bad_corrcoef_tv_ens[0][1])

                # >>> good vs good correcoef
                good_vs_good_corrcoef_tv_adj_diff = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_good_vs_good_sorted['d_adj_diff_delta']]),
                                                                np_tv_delta_good_vs_good)
                l_good_vs_good_correcoef_tv_adj_diff.append(good_vs_good_corrcoef_tv_adj_diff[0][1])
                good_vs_good_corrcoef_tv_in_agg = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_good_vs_good_sorted['d_in_agg_delta']]),
                                                                np_tv_delta_good_vs_good)
                l_good_vs_good_correcoef_tv_in_agg.append(good_vs_good_corrcoef_tv_in_agg[0][1])
                good_vs_good_corrcoef_tv_ln_vx = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_good_vs_good_sorted['d_ln_vx_delta']]),
                                                              np_tv_delta_good_vs_good)
                l_good_vs_good_correcoef_tv_ln_vx.append(good_vs_good_corrcoef_tv_ln_vx[0][1])
                good_vs_good_corrcoef_tv_ens = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_good_vs_good_sorted['d_ens_delta']]),
                                                             np_tv_delta_good_vs_good)
                l_good_vs_good_correcoef_tv_ens.append(good_vs_good_corrcoef_tv_ens[0][1])

                # >>> bad vs bad correcoef
                bad_vs_bad_corrcoef_tv_adj_diff = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_bad_vs_bad_sorted['d_adj_diff_delta']]),
                                                                np_tv_delta_bad_vs_bad)
                l_bad_vs_bad_correcoef_tv_adj_diff.append(bad_vs_bad_corrcoef_tv_adj_diff[0][1])
                bad_vs_bad_corrcoef_tv_in_agg = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_bad_vs_bad_sorted['d_in_agg_delta']]),
                                                              np_tv_delta_bad_vs_bad)
                l_bad_vs_bad_correcoef_tv_in_agg.append(bad_vs_bad_corrcoef_tv_in_agg[0][1])
                bad_vs_bad_corrcoef_tv_ln_vx = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_bad_vs_bad_sorted['d_ln_vx_delta']]),
                                                             np_tv_delta_bad_vs_bad)
                l_bad_vs_bad_correcoef_tv_ln_vx.append(bad_vs_bad_corrcoef_tv_ln_vx[0][1])
                bad_vs_bad_corrcoef_tv_ens = np.corrcoef(np.asarray([d_sas_delta[K] for d_sas_delta in df_tv_sas_bad_vs_bad_sorted['d_ens_delta']]),
                                                           np_tv_delta_bad_vs_bad)
                l_bad_vs_bad_correcoef_tv_ens.append(bad_vs_bad_corrcoef_tv_ens[0][1])

            # >>> draw mean ratios
            # fig, axes = plt.subplots(ncols=1, nrows=1)
            # axes.grid(True)
            # adj_diff_color = 'tab:blue'
            # in_agg_color = 'tab:orange'
            # ln_vx_color = 'tab:green'
            # ens_color = 'tab:red'
            # axes.errorbar(l_K, l_good_vs_bad_mean_ratio_sas_vs_delta_adj_diff,
            #                    fmt='-o', c=adj_diff_color, capsize=2, capthick=1, label='ADJ-DIFF')
            # axes.errorbar(l_K, l_good_vs_bad_mean_ratio_sas_vs_delta_in_agg,
            #                    fmt='-o', c=in_agg_color, capsize=2, capthick=1, label='IN-AGG')
            # axes.errorbar(l_K, l_good_vs_bad_mean_ratio_sas_vs_delta_ln_vx,
            #                    fmt='-o', c=ln_vx_color, capsize=2, capthick=1, label='LN-VX')
            # axes.errorbar(l_K, l_good_vs_bad_mean_ratio_sas_vs_delta_ens,
            #                    fmt='-o', c=ens_color, capsize=2, capthick=1, label='ENS')
            # axes.set_xticks(l_K)
            # axes.set_xticklabels(['K=%s' % K for K in l_K])
            # axes.legend()
            # plt.tight_layout(pad=1.0)
            # if save_img:
            #     plt.savefig(good_and_bad_runs_save_path + 'good_vs_bad_sas_vs_delta_mean_ratio' + img_suffix + '.PNG', format='PNG')
            # if show_img:
            #     plt.show()
            # plt.clf()
            # plt.close()

            # >>> draw good vs bad, good vs good, bad vs bad correcoefs for adj_diff
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.set_title('ADJ-DIFF', fontsize=10, fontweight='semibold')
            axes.grid(True)
            good_vs_bad_color = 'tab:blue'
            good_vs_good_color = 'tab:orange'
            bad_vs_bad_color = 'tab:green'
            axes.errorbar(l_K, l_good_vs_bad_correcoef_tv_adj_diff,
                          fmt='-o', c=good_vs_bad_color, capsize=2, capthick=1, label='good vs bad')
            axes.errorbar(l_K, l_good_vs_good_correcoef_tv_adj_diff,
                          fmt='-o', c=good_vs_good_color, capsize=2, capthick=1, label='good vs good')
            axes.errorbar(l_K, l_bad_vs_bad_correcoef_tv_adj_diff,
                          fmt='-o', c=bad_vs_bad_color, capsize=2, capthick=1, label='bad vs bad')
            axes.set_xticks(l_K)
            axes.set_xticklabels(['K=%s' % K for K in l_K])
            axes.legend()
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'correcoef_tv_adj_diff' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> draw good vs bad, good vs good, bad vs bad correcoefs for ens
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.set_title('ENS', fontsize=10, fontweight='semibold')
            axes.grid(True)
            good_vs_bad_color = 'tab:blue'
            good_vs_good_color = 'tab:orange'
            bad_vs_bad_color = 'tab:green'
            axes.errorbar(l_K, l_good_vs_bad_correcoef_tv_ens,
                          fmt='-o', c=good_vs_bad_color, capsize=2, capthick=1, label='good vs bad')
            axes.errorbar(l_K, l_good_vs_good_correcoef_tv_ens,
                          fmt='-o', c=good_vs_good_color, capsize=2, capthick=1, label='good vs good')
            axes.errorbar(l_K, l_bad_vs_bad_correcoef_tv_ens,
                          fmt='-o', c=bad_vs_bad_color, capsize=2, capthick=1, label='bad vs bad')
            axes.set_xticks(l_K)
            axes.set_xticklabels(['K=%s' % K for K in l_K])
            axes.legend()
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'correcoef_tv_ens' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> draw good vs bad, good vs good, bad vs bad correcoefs for in_agg
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.set_title('IN-AGG', fontsize=10, fontweight='semibold')
            axes.grid(True)
            good_vs_bad_color = 'tab:blue'
            good_vs_good_color = 'tab:orange'
            bad_vs_bad_color = 'tab:green'
            axes.errorbar(l_K, l_good_vs_bad_correcoef_tv_in_agg,
                          fmt='-o', c=good_vs_bad_color, capsize=2, capthick=1, label='good vs bad')
            axes.errorbar(l_K, l_good_vs_good_correcoef_tv_in_agg,
                          fmt='-o', c=good_vs_good_color, capsize=2, capthick=1, label='good vs good')
            axes.errorbar(l_K, l_bad_vs_bad_correcoef_tv_in_agg,
                          fmt='-o', c=bad_vs_bad_color, capsize=2, capthick=1, label='bad vs bad')
            axes.set_xticks(l_K)
            axes.set_xticklabels(['K=%s' % K for K in l_K])
            axes.legend()
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'correcoef_tv_in_agg' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> draw good vs bad, good vs good, bad vs bad correcoefs for ln_vx
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.set_title('LN-VX', fontsize=10, fontweight='semibold')
            axes.grid(True)
            good_vs_bad_color = 'tab:blue'
            good_vs_good_color = 'tab:orange'
            bad_vs_bad_color = 'tab:green'
            axes.errorbar(l_K, l_good_vs_bad_correcoef_tv_ln_vx,
                          fmt='-o', c=good_vs_bad_color, capsize=2, capthick=1, label='good vs bad')
            axes.errorbar(l_K, l_good_vs_good_correcoef_tv_ln_vx,
                          fmt='-o', c=good_vs_good_color, capsize=2, capthick=1, label='good vs good')
            axes.errorbar(l_K, l_bad_vs_bad_correcoef_tv_ln_vx,
                          fmt='-o', c=bad_vs_bad_color, capsize=2, capthick=1, label='bad vs bad')
            axes.set_xticks(l_K)
            axes.set_xticklabels(['K=%s' % K for K in l_K])
            axes.legend()
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'correcoef_tv_ln_vx' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> draw sas delta
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(5, 12))
            good_vs_bad_color = 'tab:blue'
            good_vs_good_color = 'tab:orange'
            idx = 0
            for K in l_K:
                axes[idx].grid(True)
                axes[idx].set_title('ADJ-DIFF @ K=%s' % K, fontsize=10, fontweight='semibold')
                sns.kdeplot([d_sas_delta[K] for d_sas_delta in list(df_tv_sas_good_vs_bad_sorted['d_adj_diff_delta'])],
                            ax=axes[idx], color='tab:orange', cut=0, fill=True, label='good vs bad')
                sns.kdeplot([d_sas_delta[K] for d_sas_delta in list(df_tv_sas_good_vs_good_sorted['d_adj_diff_delta'])],
                            ax=axes[idx], color='tab:blue', cut=0, fill=True, label='good vs good')
                axes[idx].legend()
                idx += 1
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'good_bad_vs_good_good_adj_diff_delta' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(5, 12))
            good_vs_bad_color = 'tab:blue'
            good_vs_good_color = 'tab:orange'
            idx = 0
            for K in l_K:
                axes[idx].grid(True)
                axes[idx].set_title('IN-AGG @ K=%s' % K, fontsize=10, fontweight='semibold')
                sns.kdeplot([d_sas_delta[K] for d_sas_delta in list(df_tv_sas_good_vs_bad_sorted['d_in_agg_delta'])],
                            ax=axes[idx], color='tab:orange', cut=0, fill=True, label='good vs bad')
                sns.kdeplot([d_sas_delta[K] for d_sas_delta in list(df_tv_sas_good_vs_good_sorted['d_in_agg_delta'])],
                            ax=axes[idx], color='tab:blue', cut=0, fill=True, label='good vs good')
                axes[idx].legend()
                idx += 1
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'good_bad_vs_good_good_in_agg_delta' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(5, 12))
            good_vs_bad_color = 'tab:blue'
            good_vs_good_color = 'tab:orange'
            idx = 0
            for K in l_K:
                axes[idx].grid(True)
                axes[idx].set_title('LN-VX @ K=%s' % K, fontsize=10, fontweight='semibold')
                sns.kdeplot([d_sas_delta[K] for d_sas_delta in list(df_tv_sas_good_vs_bad_sorted['d_ln_vx_delta'])],
                            ax=axes[idx], color='tab:orange', cut=0, fill=True, label='good vs bad')
                sns.kdeplot([d_sas_delta[K] for d_sas_delta in list(df_tv_sas_good_vs_good_sorted['d_ln_vx_delta'])],
                            ax=axes[idx], color='tab:blue', cut=0, fill=True, label='good vs good')
                axes[idx].legend()
                idx += 1
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'good_bad_vs_good_good_ln_vx_delta' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(5, 12))
            good_vs_bad_color = 'tab:blue'
            good_vs_good_color = 'tab:orange'
            idx = 0
            for K in l_K:
                axes[idx].grid(True)
                axes[idx].set_title('ENS @ K=%s' % K, fontsize=10, fontweight='semibold')
                sns.kdeplot([d_sas_delta[K] for d_sas_delta in list(df_tv_sas_good_vs_bad_sorted['d_ens_delta'])],
                            ax=axes[idx], color='tab:orange', cut=0, fill=True, label='good vs bad')
                sns.kdeplot([d_sas_delta[K] for d_sas_delta in list(df_tv_sas_good_vs_good_sorted['d_ens_delta'])],
                            ax=axes[idx], color='tab:blue', cut=0, fill=True, label='good vs good')
                axes[idx].legend()
                idx += 1
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'good_bad_vs_good_good_ens_delta' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

        if STAGE_sim_tv_sim_sas_hypo:
            save_img = True
            show_img = True
            no_in_agg = True
            norm_sig = False
            if norm_sig:
                out_file_suffix = '_norm'
            else:
                out_file_suffix = '_non_norm'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_tv_sas_good_vs_bad = pd.read_pickle(good_and_bad_runs_save_path + 'good_vs_bad_tv_sas_relations' + out_file_suffix +'.pickle')
            df_tv_sas_good_vs_good = pd.read_pickle(good_and_bad_runs_save_path + 'good_vs_good_tv_sas_relations' + out_file_suffix + '.pickle')
            df_tv_sas_bad_vs_bad = pd.read_pickle(good_and_bad_runs_save_path + 'bad_vs_bad_tv_sas_relations' + out_file_suffix + '.pickle')

            df_tv_sas_good_vs_bad_sorted = df_tv_sas_good_vs_bad.sort_values(by=['tv_delta'])
            df_tv_sas_good_vs_good_sorted = df_tv_sas_good_vs_good.sort_values(by=['tv_delta'])
            df_tv_sas_bad_vs_bad_sorted = df_tv_sas_bad_vs_bad.sort_values(by=['tv_delta'])

            l_K = list(df_tv_sas_good_vs_bad['d_ens_delta'][0].keys())
            num_good_vs_bad = len(df_tv_sas_good_vs_bad_sorted)
            l_sim_tv_threshold = np.round([t for t in np.arange(0.0, 0.1, 0.005)], decimals=2)
            l_sim_sas_threshold = np.round([t for t in np.arange(0.0, 0.1, 0.005)], decimals=2)

            if no_in_agg:
                img_suffix = '_no_in_agg'
            else:
                img_suffix = ''

            for sas_name in ['ens', 'adj_diff', 'in_agg', 'ln_vx']:
                d_sas_delta_name = 'd_' + sas_name + '_delta'
                l_sim_tv_sim_sas_rec = []
                for sim_tv in l_sim_tv_threshold:
                    l_sim_tv_sim_sas_per_tv = []
                    df_sim_tv_good_vs_bad = df_tv_sas_good_vs_bad_sorted.loc[df_tv_sas_good_vs_bad_sorted['tv_delta'] <= sim_tv]
                    if len(df_sim_tv_good_vs_bad) <= 0:
                        l_sim_tv_sim_sas_per_tv = [0.0] * len(l_sim_sas_threshold)
                        l_sim_tv_sim_sas_rec.append(l_sim_tv_sim_sas_per_tv)
                        continue
                    for sim_sas in l_sim_sas_threshold:
                        num_sim_tv_sim_sas = len([d_sas_delta for d_sas_delta in df_sim_tv_good_vs_bad[d_sas_delta_name]
                                                  if len([d_sas_delta[K] for K in l_K if d_sas_delta[K] <= sim_sas]) == len(l_K)])
                        if num_sim_tv_sim_sas > 0:
                            l_sim_tv_sim_sas_per_tv.append(num_sim_tv_sim_sas / num_good_vs_bad)
                        else:
                            l_sim_tv_sim_sas_per_tv.append(0.0)
                    l_sim_tv_sim_sas_rec.append(l_sim_tv_sim_sas_per_tv)

                if sas_name == 'ens':
                    img_title = 'ENS'
                elif sas_name == 'adj_diff':
                    img_title = 'ADJ-DIFF'
                elif sas_name == 'in_agg':
                    img_title = 'IN-AGG'
                elif sas_name == 'ln_vx':
                    img_title = 'LN-VX'
                img_name = 'good_vs_bad_sim_tv_sim_' + sas_name + '_percentages' + img_suffix + '.PNG'
                fig = plt.figure(figsize=(10, 10))
                ax = plt.axes(projection='3d')
                ax.grid(True)
                ax.set_title(img_title, fontsize=10, fontweight='semibold')
                TV, SAS = np.meshgrid(l_sim_tv_threshold, l_sim_sas_threshold)
                np_sim_tv_sim_sas_rec = np.asarray(l_sim_tv_sim_sas_rec)
                ax.plot_surface(TV, SAS, np_sim_tv_sim_sas_rec, rstride=1, cstride=1, cmap='coolwarm')
                ax.set_zticks(np.round([i for i in np.arange(0, 1.05, 0.05)], decimals=2))
                ax.set_xlabel(r'$\Delta \tau_t$')
                ax.set_ylabel(r'$\Delta \mathcal{M}_t$')
                ax.set_zlabel(r'$P$')
                ax.view_init(15, 255)
                if save_img:
                    plt.savefig(good_and_bad_runs_save_path + img_name, format='PNG')
                if show_img:
                    plt.show()
                plt.clf()
                plt.close()

        if STAGE_sgs_amp_distribution_comp:
            save_img = True
            show_img = True
            all_init_vs_final = True
            norm_sig = True
            if norm_sig:
                out_file_suffix = '_norm'
            else:
                out_file_suffix = '_non_norm'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            if all_init_vs_final:
                df_tv_sas_good_vs_bad = pd.read_pickle(good_and_bad_runs_save_path + 'init_vs_final_relation' + out_file_suffix +'.pickle')
                df_tv_sas_good_vs_good = pd.read_pickle(good_and_bad_runs_save_path + 'init_vs_init_relation' + out_file_suffix + '.pickle')
                df_tv_sas_bad_vs_bad = pd.read_pickle(good_and_bad_runs_save_path + 'final_vs_final_relation' + out_file_suffix + '.pickle')
            else:
                init_or_final = 'final'
                if init_or_final == 'init':
                    out_file_suffix += '_init'
                elif init_or_final == 'final':
                    out_file_suffix += '_final'
                df_tv_sas_good_vs_bad = pd.read_pickle(good_and_bad_runs_save_path + 'good_vs_bad_tv_sas_relations' + out_file_suffix +'.pickle')
                df_tv_sas_good_vs_good = pd.read_pickle(good_and_bad_runs_save_path + 'good_vs_good_tv_sas_relations' + out_file_suffix + '.pickle')
                df_tv_sas_bad_vs_bad = pd.read_pickle(good_and_bad_runs_save_path + 'bad_vs_bad_tv_sas_relations' + out_file_suffix + '.pickle')

            l_K = list(df_tv_sas_good_vs_bad['d_ens_delta_full'][0].keys())
            l_sgs_name = ['adj_diff', 'in_agg', 'ln_vx', 'apprx_ls', 'ens']
            d_sgs_img_legend = {'adj_diff': 'ADJ-DIFF', 'in_agg': 'IN-AGG', 'ln_vx': 'LN-VX', 'apprx_ls': 'APPRX-LS', 'ens': 'ENS'}
            # Expected
            d_good_bad_sgs_l1 = {sgs_name: {K: [] for K in l_K} for sgs_name in l_sgs_name}
            d_good_good_sgs_l1 = {sgs_name: {K: [] for K in l_K} for sgs_name in l_sgs_name}
            d_bad_bad_sgs_l1 = {sgs_name: {K: [] for K in l_K} for sgs_name in l_sgs_name}
            # L-infinity metric
            d_good_bad_sgs_lmax = {sgs_name: {K: [] for K in l_K} for sgs_name in l_sgs_name}
            d_good_good_sgs_lmax = {sgs_name: {K: [] for K in l_K} for sgs_name in l_sgs_name}
            d_bad_bad_sgs_lmax = {sgs_name: {K: [] for K in l_K} for sgs_name in l_sgs_name}

            for _, good_bad_rec in df_tv_sas_good_vs_bad.iterrows():
                d_adj_diff_delta_full = good_bad_rec['d_adj_diff_delta_full']
                d_in_agg_delta_full = good_bad_rec['d_in_agg_delta_full']
                d_ln_vx_delta_full = good_bad_rec['d_ln_vx_delta_full']
                d_apprx_ls_delta_full = good_bad_rec['d_apprx_ls_delta_full']
                d_ens_delta_full = good_bad_rec['d_ens_delta_full']
                for K in l_K:
                    # adj_diff
                    adj_diff_l1 = np.mean(d_adj_diff_delta_full[K])
                    d_good_bad_sgs_l1['adj_diff'][K].append(adj_diff_l1)
                    adj_diff_lmax = np.max(d_adj_diff_delta_full[K])
                    d_good_bad_sgs_lmax['adj_diff'][K].append(adj_diff_lmax)
                    # in_agg
                    in_agg_l1 = np.mean(d_in_agg_delta_full[K])
                    d_good_bad_sgs_l1['in_agg'][K].append(in_agg_l1)
                    in_agg_lmax = np.max(d_in_agg_delta_full[K])
                    d_good_bad_sgs_lmax['in_agg'][K].append(in_agg_lmax)
                    # ln_vx
                    ln_vx_l1 = np.mean(d_ln_vx_delta_full[K])
                    d_good_bad_sgs_l1['ln_vx'][K].append(ln_vx_l1)
                    ln_vx_lmax = np.max(d_ln_vx_delta_full[K])
                    d_good_bad_sgs_lmax['ln_vx'][K].append(ln_vx_lmax)
                    # apprx_ls
                    apprx_ls_l1 = np.mean(d_apprx_ls_delta_full[K])
                    d_good_bad_sgs_l1['apprx_ls'][K].append(apprx_ls_l1)
                    apprx_ls_lmax = np.max(d_apprx_ls_delta_full[K])
                    d_good_bad_sgs_lmax['apprx_ls'][K].append(apprx_ls_lmax)
                    # ens
                    ens_l1 = np.mean(d_ens_delta_full[K])
                    d_good_bad_sgs_l1['ens'][K].append(ens_l1)
                    ens_lmax = np.max(d_ens_delta_full[K])
                    d_good_bad_sgs_lmax['ens'][K].append(ens_lmax)

            for _, good_good_rec in df_tv_sas_good_vs_good.iterrows():
                d_adj_diff_delta_full = good_good_rec['d_adj_diff_delta_full']
                d_in_agg_delta_full = good_good_rec['d_in_agg_delta_full']
                d_ln_vx_delta_full = good_good_rec['d_ln_vx_delta_full']
                d_apprx_ls_delta_full = good_good_rec['d_apprx_ls_delta_full']
                d_ens_delta_full = good_good_rec['d_ens_delta_full']
                for K in l_K:
                    # adj_diff
                    adj_diff_l1 = np.mean(d_adj_diff_delta_full[K])
                    d_good_good_sgs_l1['adj_diff'][K].append(adj_diff_l1)
                    adj_diff_lmax = np.max(d_adj_diff_delta_full[K])
                    d_good_good_sgs_lmax['adj_diff'][K].append(adj_diff_lmax)
                    # in_agg
                    in_agg_l1 = np.mean(d_in_agg_delta_full[K])
                    d_good_good_sgs_l1['in_agg'][K].append(in_agg_l1)
                    in_agg_lmax = np.max(d_in_agg_delta_full[K])
                    d_good_good_sgs_lmax['in_agg'][K].append(in_agg_lmax)
                    # ln_vx
                    ln_vx_l1 = np.mean(d_ln_vx_delta_full[K])
                    d_good_good_sgs_l1['ln_vx'][K].append(ln_vx_l1)
                    ln_vx_lmax = np.max(d_ln_vx_delta_full[K])
                    d_good_good_sgs_lmax['ln_vx'][K].append(ln_vx_lmax)
                    # apprx_ls
                    apprx_ls_l1 = np.mean(d_apprx_ls_delta_full[K])
                    d_good_good_sgs_l1['apprx_ls'][K].append(apprx_ls_l1)
                    apprx_ls_lmax = np.max(d_apprx_ls_delta_full[K])
                    d_good_good_sgs_lmax['apprx_ls'][K].append(apprx_ls_lmax)
                    # ens
                    ens_l1 = np.mean(d_ens_delta_full[K])
                    d_good_good_sgs_l1['ens'][K].append(ens_l1)
                    ens_lmax = np.max(d_ens_delta_full[K])
                    d_good_good_sgs_lmax['ens'][K].append(ens_lmax)

            for _, bad_bad_rec in df_tv_sas_bad_vs_bad.iterrows():
                d_adj_diff_delta_full = bad_bad_rec['d_adj_diff_delta_full']
                d_in_agg_delta_full = bad_bad_rec['d_in_agg_delta_full']
                d_ln_vx_delta_full = bad_bad_rec['d_ln_vx_delta_full']
                d_apprx_ls_delta_full = bad_bad_rec['d_apprx_ls_delta_full']
                d_ens_delta_full = bad_bad_rec['d_ens_delta_full']
                for K in l_K:
                    # adj_diff
                    adj_diff_l1 = np.mean(d_adj_diff_delta_full[K])
                    d_bad_bad_sgs_l1['adj_diff'][K].append(adj_diff_l1)
                    adj_diff_lmax = np.max(d_adj_diff_delta_full[K])
                    d_bad_bad_sgs_lmax['adj_diff'][K].append(adj_diff_lmax)
                    # in_agg
                    in_agg_l1 = np.mean(d_in_agg_delta_full[K])
                    d_bad_bad_sgs_l1['in_agg'][K].append(in_agg_l1)
                    in_agg_lmax = np.max(d_in_agg_delta_full[K])
                    d_bad_bad_sgs_lmax['in_agg'][K].append(in_agg_lmax)
                    # ln_vx
                    ln_vx_l1 = np.mean(d_ln_vx_delta_full[K])
                    d_bad_bad_sgs_l1['ln_vx'][K].append(ln_vx_l1)
                    ln_vx_lmax = np.max(d_ln_vx_delta_full[K])
                    d_bad_bad_sgs_lmax['ln_vx'][K].append(ln_vx_lmax)
                    # apprx_ls
                    apprx_ls_l1 = np.mean(d_apprx_ls_delta_full[K])
                    d_bad_bad_sgs_l1['apprx_ls'][K].append(apprx_ls_l1)
                    apprx_ls_lmax = np.max(d_apprx_ls_delta_full[K])
                    d_bad_bad_sgs_lmax['apprx_ls'][K].append(apprx_ls_lmax)
                    # ens
                    ens_l1 = np.mean(d_ens_delta_full[K])
                    d_bad_bad_sgs_l1['ens'][K].append(ens_l1)
                    ens_lmax = np.max(d_ens_delta_full[K])
                    d_bad_bad_sgs_lmax['ens'][K].append(ens_lmax)

            plt.rcParams.update({
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{amssymb}'
            })
            if norm_sig:
                M_sym = '\mathcal{M}_1'
            else:
                M_sym = '\mathcal{M}'
            # good-bad vs good-good
            if all_init_vs_final:
                img_title_prefix = 'Init-Final vs Init-Init'
                img_save_surfix = '_if_vs_ii'
            else:
                img_title_prefix = 'Good-Bad vs Good-Good'
                img_save_surfix = '_gb_vs_gg'
            fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(5, 5))
            d_color = {'adj_diff': 'tab:blue', 'in_agg': 'tab:orange', 'ln_vx': 'tab:green', 'apprx_ls': 'tab:purple', 'ens': 'tab:red'}
            idx = 0
            axes[idx].set_title((r'$\mathbb{E}[\nabla %s]$ ' % M_sym) + img_title_prefix, fontsize=10, fontweight='semibold')
            axes[idx].grid(True)
            idx += 1
            axes[idx].set_title((r'$||\nabla %s||_{\infty}$ ' % M_sym) + img_title_prefix, fontsize=10, fontweight='semibold')
            axes[idx].grid(True)
            for sgs_name in l_sgs_name:
                l_gb_vs_gg_l1_wass_by_K = []
                l_gb_vs_gg_lmax_wass_by_K = []
                for K in l_K:
                    gb_vs_gg_l1_wass_by_K = stats.wasserstein_distance(np.asarray(d_good_bad_sgs_l1[sgs_name][K]),
                                                                    np.asarray(d_good_good_sgs_l1[sgs_name][K]))
                    l_gb_vs_gg_l1_wass_by_K.append(gb_vs_gg_l1_wass_by_K)
                    gb_vs_gg_lmax_wass_by_K = stats.wasserstein_distance(np.asarray(d_good_bad_sgs_lmax[sgs_name][K]),
                                                                       np.asarray(d_good_good_sgs_lmax[sgs_name][K]))
                    l_gb_vs_gg_lmax_wass_by_K.append(gb_vs_gg_lmax_wass_by_K)

                idx = 0
                axes[idx].errorbar(l_K, l_gb_vs_gg_l1_wass_by_K,
                              fmt='-o', c=d_color[sgs_name], capsize=2, capthick=1, label=d_sgs_img_legend[sgs_name])
                idx += 1
                axes[idx].errorbar(l_K, l_gb_vs_gg_lmax_wass_by_K,
                              fmt='-o', c=d_color[sgs_name], capsize=2, capthick=1, label=d_sgs_img_legend[sgs_name])
            idx = 0
            axes[idx].set_xticks(l_K)
            axes[idx].set_xticklabels(['K=%s' % K for K in l_K])
            axes[idx].legend()
            idx += 1
            axes[idx].set_xticks(l_K)
            axes[idx].set_xticklabels(['K=%s' % K for K in l_K])
            axes[idx].legend()

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'sgs_delta_distr_wass' + img_save_surfix + out_file_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # good-bad vs bad-bad
            if all_init_vs_final:
                img_title_prefix = 'Init-Final vs Final-Final'
                img_save_surfix = '_if_vs_ff'
            else:
                img_title_prefix = 'Good-Bad vs Bad-Bad'
                img_save_surfix = '_gb_vs_bb'
            fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(5, 5))
            d_color = {'adj_diff': 'tab:blue', 'in_agg': 'tab:orange', 'ln_vx': 'tab:green', 'apprx_ls': 'tab:purple', 'ens': 'tab:red'}
            idx = 0
            axes[idx].set_title((r'$\mathbb{E}[\nabla %s]$ ' % M_sym) + img_title_prefix, fontsize=10, fontweight='semibold')
            axes[idx].grid(True)
            idx += 1
            axes[idx].set_title((r'$||\nabla %s||_{\infty}$ ' % M_sym) + img_title_prefix, fontsize=10, fontweight='semibold')
            axes[idx].grid(True)
            for sgs_name in l_sgs_name:
                l_gb_vs_bb_l1_wass_by_K = []
                l_gb_vs_bb_lmax_wass_by_K = []
                for K in l_K:
                    gb_vs_bb_l1_wass_by_K = stats.wasserstein_distance(np.asarray(d_good_bad_sgs_l1[sgs_name][K]),
                                                                       np.asarray(d_bad_bad_sgs_l1[sgs_name][K]))
                    l_gb_vs_bb_l1_wass_by_K.append(gb_vs_bb_l1_wass_by_K)
                    gb_vs_bb_lmax_wass_by_K = stats.wasserstein_distance(np.asarray(d_good_bad_sgs_lmax[sgs_name][K]),
                                                                         np.asarray(d_bad_bad_sgs_lmax[sgs_name][K]))
                    l_gb_vs_bb_lmax_wass_by_K.append(gb_vs_bb_lmax_wass_by_K)

                idx = 0
                axes[idx].errorbar(l_K, l_gb_vs_bb_l1_wass_by_K,
                                   fmt='-o', c=d_color[sgs_name], capsize=2, capthick=1, label=d_sgs_img_legend[sgs_name])
                idx += 1
                axes[idx].errorbar(l_K, l_gb_vs_bb_lmax_wass_by_K,
                                   fmt='-o', c=d_color[sgs_name], capsize=2, capthick=1, label=d_sgs_img_legend[sgs_name])
            idx = 0
            axes[idx].set_xticks(l_K)
            axes[idx].set_xticklabels(['K=%s' % K for K in l_K])
            axes[idx].legend()
            idx += 1
            axes[idx].set_xticks(l_K)
            axes[idx].set_xticklabels(['K=%s' % K for K in l_K])
            axes[idx].legend()

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'sgs_delta_distr_wass' + img_save_surfix + out_file_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # good-good vs bad-bad
            if all_init_vs_final:
                img_title_prefix = 'Init-Init vs Final-Final'
                img_save_surfix = '_ii_vs_ff'
            else:
                img_title_prefix = 'Good-Good vs Bad-Bad'
                img_save_surfix = '_gg_vs_bb'
            fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(5, 5))
            d_color = {'adj_diff': 'tab:blue', 'in_agg': 'tab:orange', 'ln_vx': 'tab:green', 'apprx_ls': 'tab:purple', 'ens': 'tab:red'}
            idx = 0
            axes[idx].set_title((r'$\mathbb{E}[\nabla %s]$ ' % M_sym) + img_title_prefix, fontsize=10, fontweight='semibold')
            axes[idx].grid(True)
            idx += 1
            axes[idx].set_title((r'$||\nabla %s||_{\infty}$ ' % M_sym) + img_title_prefix, fontsize=10, fontweight='semibold')
            axes[idx].grid(True)
            for sgs_name in l_sgs_name:
                l_gg_vs_bb_l1_wass_by_K = []
                l_gg_vs_bb_lmax_wass_by_K = []
                for K in l_K:
                    gg_vs_bb_l1_wass_by_K = stats.wasserstein_distance(np.asarray(d_good_good_sgs_l1[sgs_name][K]),
                                                                       np.asarray(d_bad_bad_sgs_l1[sgs_name][K]))
                    l_gg_vs_bb_l1_wass_by_K.append(gg_vs_bb_l1_wass_by_K)
                    gg_vs_bb_lmax_wass_by_K = stats.wasserstein_distance(np.asarray(d_good_good_sgs_lmax[sgs_name][K]),
                                                                         np.asarray(d_bad_bad_sgs_lmax[sgs_name][K]))
                    l_gg_vs_bb_lmax_wass_by_K.append(gg_vs_bb_lmax_wass_by_K)

                idx = 0
                axes[idx].errorbar(l_K, l_gg_vs_bb_l1_wass_by_K,
                                   fmt='-o', c=d_color[sgs_name], capsize=2, capthick=1, label=d_sgs_img_legend[sgs_name])
                idx += 1
                axes[idx].errorbar(l_K, l_gg_vs_bb_lmax_wass_by_K,
                                   fmt='-o', c=d_color[sgs_name], capsize=2, capthick=1, label=d_sgs_img_legend[sgs_name])
            idx = 0
            axes[idx].set_xticks(l_K)
            axes[idx].set_xticklabels(['K=%s' % K for K in l_K])
            axes[idx].legend()
            idx += 1
            axes[idx].set_xticks(l_K)
            axes[idx].set_xticklabels(['K=%s' % K for K in l_K])
            axes[idx].legend()

            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'sgs_delta_distr_wass' + img_save_surfix + out_file_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

        if STATE_sgs_init_final_relation:
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            norm_sig = True
            if norm_sig:
                out_file_suffix = '_norm'
            else:
                out_file_suffix = '_non_norm'
            all_runs = True
            all_good_and_bad = True
            no_in_agg = True
            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = all_draw_data_save_name + '_norm_no_in_agg'
                else:
                    draw_data_save_name = all_draw_data_save_name + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = all_draw_data_save_name + '_non_norm_no_in_agg'
                else:
                    draw_data_save_name = all_draw_data_save_name + '_non_norm'
            if all_good_and_bad:
                draw_data_save_name += '_all_good_bad'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_all_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name + '@'
                                                      + graph_name + '@' + spec_seq_param_name + '.pickle')

            d_metrics_all = df_all_draw_data.loc['all']['d_metrics']
            d_eig_vals_all = df_all_draw_data.loc['all']['d_eig_vals']
            d_adj_diff_all = df_all_draw_data.loc['all']['d_adj_diff']
            d_in_agg_all = df_all_draw_data.loc['all']['d_in_agg']
            d_ln_vx_all = df_all_draw_data.loc['all']['d_ln_vx']
            d_apprx_ls_all = df_all_draw_data.loc['all']['d_apprx_ls']
            d_ens_all = df_all_draw_data.loc['all']['d_ens']
            d_nc_perf_all = df_all_draw_data.loc['all']['d_nc_perf']

            l_K = sorted(d_eig_vals_all.keys())
            max_epoch_all = np.max(list(d_metrics_all['tv_loss'].keys()))
            min_epoch_all = np.min(list(d_metrics_all['tv_loss'].keys()))

            num_init = len(d_metrics_all['tv_loss'][min_epoch_all])
            num_final = len(d_metrics_all['tv_loss'][max_epoch_all])

            timer_start = time.time()

            # >>> init vs final
            l_init_final_rec = []
            for init_idx in range(num_init):
                d_adj_diff_init = {K: d_adj_diff_all[K][min_epoch_all][init_idx] for K in d_adj_diff_all}
                d_in_agg_init = {K: d_in_agg_all[K][min_epoch_all][init_idx] for K in d_in_agg_all}
                d_ln_vx_init = {K: d_ln_vx_all[K][min_epoch_all][init_idx] for K in d_ln_vx_all}
                d_apprx_ls_init = {K: d_apprx_ls_all[K][min_epoch_all][init_idx] for K in d_apprx_ls_all}
                d_ens_init = {K: d_ens_all[K][min_epoch_all][init_idx] for K in d_ens_all}

                for final_idx in range(num_final):
                    d_adj_diff_final = {K: d_adj_diff_all[K][max_epoch_all][final_idx] for K in d_adj_diff_all}
                    d_in_agg_final = {K: d_in_agg_all[K][max_epoch_all][final_idx] for K in d_in_agg_all}
                    d_ln_vx_final = {K: d_ln_vx_all[K][max_epoch_all][final_idx] for K in d_ln_vx_all}
                    d_apprx_ls_final = {K: d_apprx_ls_all[K][max_epoch_all][final_idx] for K in d_apprx_ls_all}
                    d_ens_final = {K: d_ens_all[K][max_epoch_all][final_idx] for K in d_ens_all}

                    d_adj_diff_delta_full = {K: np.abs(d_adj_diff_init[K] - d_adj_diff_final[K]) for K in l_K}
                    d_in_agg_delta_full = {K: np.abs(d_in_agg_init[K] - d_in_agg_final[K]) for K in l_K}
                    d_ln_vx_delta_full = {K: np.abs(d_ln_vx_init[K] - d_ln_vx_final[K]) for K in l_K}
                    d_apprx_ls_delta_full = {K: np.abs(d_apprx_ls_init[K] - d_apprx_ls_final[K]) for K in l_K}
                    d_ens_delta_full = {K: np.abs(d_ens_init[K] - d_ens_final[K]) for K in l_K}

                    l_init_final_rec.append((init_idx, final_idx,
                                         d_adj_diff_delta_full, d_in_agg_delta_full, d_ln_vx_delta_full,
                                         d_apprx_ls_delta_full, d_ens_delta_full))
                    if len(l_init_final_rec) % 500 == 0 and len(l_init_final_rec) >= 500:
                        logging.debug('%s init final pairs done in %s secs.'
                                      % (len(l_init_final_rec), time.time() - timer_start))
            logging.debug('all %s init final pairs done in %s secs.'
                          % (len(l_init_final_rec), time.time() - timer_start))
            df_init_final = pd.DataFrame(l_init_final_rec, columns=['good_idx', 'bad_idx',
                                                            'd_adj_diff_delta_full', 'd_in_agg_delta_full', 'd_ln_vx_delta_full',
                                                            'd_apprx_ls_delta_full', 'd_ens_delta_full'])
            pd.to_pickle(df_init_final, good_and_bad_runs_save_path + 'init_vs_final_relation' + out_file_suffix + '.pickle')

            # >>> init vs init
            l_init_init_rec = []
            for init_idx_i in range(num_init - 1):
                d_adj_diff_init = {K: d_adj_diff_all[K][min_epoch_all][init_idx_i] for K in d_adj_diff_all}
                d_in_agg_init = {K: d_in_agg_all[K][min_epoch_all][init_idx_i] for K in d_in_agg_all}
                d_ln_vx_init = {K: d_ln_vx_all[K][min_epoch_all][init_idx_i] for K in d_ln_vx_all}
                d_apprx_ls_init = {K: d_apprx_ls_all[K][min_epoch_all][init_idx_i] for K in d_apprx_ls_all}
                d_ens_init = {K: d_ens_all[K][min_epoch_all][init_idx_i] for K in d_ens_all}

                for init_idx_j in range(init_idx_i + 1, num_init):
                    d_adj_diff_final = {K: d_adj_diff_all[K][min_epoch_all][init_idx_j] for K in d_adj_diff_all}
                    d_in_agg_final = {K: d_in_agg_all[K][min_epoch_all][init_idx_j] for K in d_in_agg_all}
                    d_ln_vx_final = {K: d_ln_vx_all[K][min_epoch_all][init_idx_j] for K in d_ln_vx_all}
                    d_apprx_ls_final = {K: d_apprx_ls_all[K][min_epoch_all][init_idx_j] for K in d_apprx_ls_all}
                    d_ens_final = {K: d_ens_all[K][min_epoch_all][init_idx_j] for K in d_ens_all}

                    d_adj_diff_delta_full = {K: np.abs(d_adj_diff_init[K] - d_adj_diff_final[K]) for K in l_K}
                    d_in_agg_delta_full = {K: np.abs(d_in_agg_init[K] - d_in_agg_final[K]) for K in l_K}
                    d_ln_vx_delta_full = {K: np.abs(d_ln_vx_init[K] - d_ln_vx_final[K]) for K in l_K}
                    d_apprx_ls_delta_full = {K: np.abs(d_apprx_ls_init[K] - d_apprx_ls_final[K]) for K in l_K}
                    d_ens_delta_full = {K: np.abs(d_ens_init[K] - d_ens_final[K]) for K in l_K}

                    l_init_init_rec.append((init_idx, final_idx,
                                             d_adj_diff_delta_full, d_in_agg_delta_full, d_ln_vx_delta_full,
                                             d_apprx_ls_delta_full, d_ens_delta_full))
                    if len(l_init_init_rec) % 500 == 0 and len(l_init_init_rec) >= 500:
                        logging.debug('%s init init pairs done in %s secs.'
                                      % (len(l_init_init_rec), time.time() - timer_start))
            logging.debug('all %s init init pairs done in %s secs.'
                          % (len(l_init_init_rec), time.time() - timer_start))
            df_init_init = pd.DataFrame(l_init_init_rec, columns=['init_idx_i', 'init_idx_j',
                                                                    'd_adj_diff_delta_full', 'd_in_agg_delta_full', 'd_ln_vx_delta_full',
                                                                    'd_apprx_ls_delta_full', 'd_ens_delta_full'])
            pd.to_pickle(df_init_init, good_and_bad_runs_save_path + 'init_vs_init_relation' + out_file_suffix + '.pickle')

            # >>> final vs final
            l_final_final_rec = []
            for final_idx_i in range(num_final - 1):
                d_adj_diff_init = {K: d_adj_diff_all[K][max_epoch_all][final_idx_i] for K in d_adj_diff_all}
                d_in_agg_init = {K: d_in_agg_all[K][max_epoch_all][final_idx_i] for K in d_in_agg_all}
                d_ln_vx_init = {K: d_ln_vx_all[K][max_epoch_all][final_idx_i] for K in d_ln_vx_all}
                d_apprx_ls_init = {K: d_apprx_ls_all[K][max_epoch_all][final_idx_i] for K in d_apprx_ls_all}
                d_ens_init = {K: d_ens_all[K][max_epoch_all][init_idx_i] for K in d_ens_all}

                for final_idx_j in range(final_idx_i + 1, num_final):
                    d_adj_diff_final = {K: d_adj_diff_all[K][max_epoch_all][final_idx_j] for K in d_adj_diff_all}
                    d_in_agg_final = {K: d_in_agg_all[K][max_epoch_all][final_idx_j] for K in d_in_agg_all}
                    d_ln_vx_final = {K: d_ln_vx_all[K][max_epoch_all][init_idx_j] for K in d_ln_vx_all}
                    d_apprx_ls_final = {K: d_apprx_ls_all[K][max_epoch_all][final_idx_j] for K in d_apprx_ls_all}
                    d_ens_final = {K: d_ens_all[K][max_epoch_all][final_idx_j] for K in d_ens_all}

                    d_adj_diff_delta_full = {K: np.abs(d_adj_diff_init[K] - d_adj_diff_final[K]) for K in l_K}
                    d_in_agg_delta_full = {K: np.abs(d_in_agg_init[K] - d_in_agg_final[K]) for K in l_K}
                    d_ln_vx_delta_full = {K: np.abs(d_ln_vx_init[K] - d_ln_vx_final[K]) for K in l_K}
                    d_apprx_ls_delta_full = {K: np.abs(d_apprx_ls_init[K] - d_apprx_ls_final[K]) for K in l_K}
                    d_ens_delta_full = {K: np.abs(d_ens_init[K] - d_ens_final[K]) for K in l_K}

                    l_final_final_rec.append((init_idx, final_idx,
                                            d_adj_diff_delta_full, d_in_agg_delta_full, d_ln_vx_delta_full,
                                            d_apprx_ls_delta_full, d_ens_delta_full))
                    if len(l_final_final_rec) % 500 == 0 and len(l_final_final_rec) >= 500:
                        logging.debug('%s final final pairs done in %s secs.'
                                      % (len(l_final_final_rec), time.time() - timer_start))
            logging.debug('all %s final final pairs done in %s secs.'
                          % (len(l_final_final_rec), time.time() - timer_start))
            df_final_final = pd.DataFrame(l_final_final_rec, columns=['final_idx_i', 'final_idx_j', 
                                                                   'd_adj_diff_delta_full', 'd_in_agg_delta_full', 'd_ln_vx_delta_full',
                                                                   'd_apprx_ls_delta_full', 'd_ens_delta_full'])
            pd.to_pickle(df_final_final, good_and_bad_runs_save_path + 'final_vs_final_relation' + out_file_suffix + '.pickle')

        if STAGE_get_all_runs_folders:
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            node_embed_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/learned_node_embeds/'
            l_run_folders = [node_embed_save_path + 'ne_run_uw_symA_20210407123413_%s/' % str(i) for i in range(1000)]
            with open(good_and_bad_runs_save_path + 'all_ne_folders.txt', 'w') as out_fd:
                out_str = '\n'.join(l_run_folders)
                out_fd.write(out_str)
                out_fd.close()

        if STAGE_tv_ari_ami_vs_sas_amplitudes:
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            norm_sig = False
            no_in_agg = True
            all_good_and_bad = True
            # good_or_bad = 'bad'
            good_or_bad = None
            init_or_final = 'final'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'

            if all_good_and_bad:
                draw_data_save_name_prefix = all_draw_data_save_name
                draw_data_col_name = 'all'
            else:
                draw_data_save_name_prefix = good_vs_bad_draw_data_save_name
                draw_data_col_name = good_or_bad

            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = draw_data_save_name_prefix + '_norm_no_in_agg'
                else:
                    draw_data_save_name = draw_data_save_name_prefix + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = draw_data_save_name_prefix + '_non_norm_no_in_agg'
                else:
                    draw_data_save_name = draw_data_save_name_prefix + '_non_norm'
            if all_good_and_bad:
                draw_data_save_name += '_all_good_bad'

            df_all_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name + '@'
                                                      + graph_name + '@' + spec_seq_param_name + '.pickle')
            d_metrics_all = df_all_draw_data.loc[draw_data_col_name]['d_metrics']
            d_eig_vals_all = df_all_draw_data.loc[draw_data_col_name]['d_eig_vals']
            d_adj_diff_all = df_all_draw_data.loc[draw_data_col_name]['d_adj_diff']
            d_in_agg_all = df_all_draw_data.loc[draw_data_col_name]['d_in_agg']
            d_ln_vx_all = df_all_draw_data.loc[draw_data_col_name]['d_ln_vx']
            d_apprx_ls_all = df_all_draw_data.loc[draw_data_col_name]['d_apprx_ls']
            d_ens_all = df_all_draw_data.loc[draw_data_col_name]['d_ens']
            d_nc_perf_all = df_all_draw_data.loc[draw_data_col_name]['d_nc_perf']
            d_sig_pw_all = df_all_draw_data.loc[draw_data_col_name]['d_sig_pw']

            l_K = sorted(d_eig_vals_all.keys())
            l_epoch = sorted(list(d_metrics_all['tv_loss'].keys()))
            max_epoch_all = np.max(list(d_metrics_all['tv_loss'].keys()))
            min_epoch_all = np.min(list(d_metrics_all['tv_loss'].keys()))
            num_all = len(d_metrics_all['tv_loss'][max_epoch_all])

            if no_in_agg:
                out_file_suffix = '_no_in_agg'
            else:
                out_file_suffix = ''
            if norm_sig:
                out_file_suffix += '_norm'
            else:
                out_file_suffix += '_non_norm'
            if all_good_and_bad:
                out_file_suffix += '_all_good_bad'
            else:
                out_file_suffix = out_file_suffix + '_' + good_or_bad
            if init_or_final == 'final':
                out_file_suffix += '_final'
            elif init_or_final == 'init':
                out_file_suffix += '_init'

            timer_start = time.time()
            l_nc_sas_rec = []
            for all_idx in range(num_all):
                tv_all = d_metrics_all['tv_loss'][max_epoch_all][all_idx]
                ari_all = d_nc_perf_all['ari'][max_epoch_all][all_idx]
                ami_all = d_nc_perf_all['nmi'][max_epoch_all][all_idx]
                if init_or_final == 'final':
                    d_adj_diff_final_all = {K: d_adj_diff_all[K][max_epoch_all][all_idx] for K in d_adj_diff_all}
                    d_in_agg_final_all = {K: d_in_agg_all[K][max_epoch_all][all_idx] for K in d_in_agg_all}
                    d_ln_vx_final_all = {K: d_ln_vx_all[K][max_epoch_all][all_idx] for K in d_ln_vx_all}
                    d_apprx_ls_final_all = {K: d_apprx_ls_all[K][max_epoch_all][all_idx] for K in d_apprx_ls_all}
                    d_ens_final_all = {K: d_ens_all[K][max_epoch_all][all_idx] for K in d_ens_all}
                    d_sig_pw_final_all = {K: d_sig_pw_all[K][max_epoch_all][all_idx] for K in d_sig_pw_all}
                elif init_or_final == 'init':
                    d_adj_diff_final_all = {K: d_adj_diff_all[K][min_epoch_all][all_idx] for K in d_adj_diff_all}
                    d_in_agg_final_all = {K: d_in_agg_all[K][min_epoch_all][all_idx] for K in d_in_agg_all}
                    d_ln_vx_final_all = {K: d_ln_vx_all[K][min_epoch_all][all_idx] for K in d_ln_vx_all}
                    d_apprx_ls_final_all = {K: d_apprx_ls_all[K][min_epoch_all][all_idx] for K in d_apprx_ls_all}
                    d_ens_final_all = {K: d_ens_all[K][min_epoch_all][all_idx] for K in d_ens_all}
                    d_sig_pw_final_all = {K: d_sig_pw_all[K][min_epoch_all][all_idx] for K in d_sig_pw_all}

                l_nc_sas_rec.append((all_idx, tv_all, ari_all, ami_all,
                                     d_adj_diff_final_all, d_in_agg_final_all, d_ln_vx_final_all, d_apprx_ls_final_all,
                                     d_ens_final_all, d_sig_pw_final_all))
                if len(l_nc_sas_rec) % 500 == 0 and len(l_nc_sas_rec) >= 500:
                    logging.debug('%s nc_sas_recs done in %s secs.'
                                  % (len(l_nc_sas_rec), time.time() - timer_start))
            logging.debug('all %s nc_sas_recs done in %s secs.'
                          % (len(l_nc_sas_rec), time.time() - timer_start))
            df_nc_sas = pd.DataFrame(l_nc_sas_rec, columns=['all_idx', 'tv', 'ari', 'ami',
                                                            'd_adj_diff_final_all', 'd_in_agg_final_all',
                                                            'd_ln_vx_final_all', 'd_apprx_ls_final_all',
                                                            'd_ens_final_all', 'd_sig_pw_final_all'])
            pd.to_pickle(df_nc_sas, good_and_bad_runs_save_path + 'all_nc_sas_amplitudes_relations' + out_file_suffix + '.pickle')

        if STAGE_tv_ari_ami_vs_sas_amplitudes_analysis:
            save_img = True
            show_img = False
            no_in_agg = True
            norm_sig = False
            all_good_and_bad = True
            # good_or_bad = 'bad'
            good_or_bad = None
            init_or_final = 'final'
            spearman_or_pearson = 'spearman'
            if spearman_or_pearson == 'spearman':
                correl_func = stats.spearmanr
            elif spearman_or_pearson == 'pearson':
                correl_func = np.corrcoef

            if no_in_agg:
                out_file_suffix = '_no_in_agg'
            else:
                out_file_suffix = ''
            if norm_sig:
                out_file_suffix += '_norm'
            else:
                out_file_suffix += '_non_norm'
            if all_good_and_bad:
                out_file_suffix += '_all_good_bad'
                img_prefix = 'Good & Bad '
            else:
                out_file_suffix = out_file_suffix + '_' + good_or_bad
                if good_or_bad == 'good':
                    img_prefix = 'Good Only '
                elif good_or_bad == 'bad':
                    img_prefix = 'Bad Only '
            if init_or_final == 'final':
                out_file_suffix += '_final'
            elif init_or_final == 'init':
                out_file_suffix += '_init'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_nc_sas = pd.read_pickle(good_and_bad_runs_save_path + 'all_nc_sas_amplitudes_relations' + out_file_suffix + '.pickle')

            np_ari = np.asarray(list(df_nc_sas['ari']))
            np_ami = np.asarray(list(df_nc_sas['ami']))
            np_tv = np.asarray(list(df_nc_sas['tv']))
            l_K = list(df_nc_sas['d_adj_diff_final_all'][0].keys())

            l_ari_sas_amp_corr_adj_diff = []
            l_ari_sas_amp_corr_in_agg = []
            l_ari_sas_amp_corr_ln_vx = []
            l_ari_sas_amp_corr_apprx_ls = []
            l_ari_sas_amp_corr_ens = []
            l_ari_sig_pw_corr = []

            l_ami_sas_amp_corr_adj_diff = []
            l_ami_sas_amp_corr_in_agg = []
            l_ami_sas_amp_corr_ln_vx = []
            l_ami_sas_amp_corr_apprx_ls = []
            l_ami_sas_amp_corr_ens = []
            l_ami_sig_pw_corr = []

            l_tv_sas_amp_corr_adj_diff = []
            l_tv_sas_amp_corr_in_agg = []
            l_tv_sas_amp_corr_ln_vx = []
            l_tv_sas_amp_corr_apprx_ls = []
            l_tv_sas_amp_corr_ens = []
            l_tv_sig_pw_corr = []
            for K in l_K:
                ari_sas_amp_corrcoef_adj_diff = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_adj_diff_final_all']]), np_ari)
                ami_sas_amp_corrcoef_adj_diff = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_adj_diff_final_all']]), np_ami)

                ari_sas_amp_corrcoef_in_agg = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_in_agg_final_all']]), np_ari)
                ami_sas_amp_corrcoef_in_agg = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_in_agg_final_all']]), np_ami)

                ari_sas_amp_corrcoef_ln_vx = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_ln_vx_final_all']]), np_ari)
                ami_sas_amp_corrcoef_ln_vx = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_ln_vx_final_all']]), np_ami)

                ari_sas_amp_corrcoef_apprx_ls = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_apprx_ls_final_all']]), np_ari)
                ami_sas_amp_corrcoef_apprx_ls = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_apprx_ls_final_all']]), np_ami)

                ari_sas_amp_corrcoef_ens = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_ens_final_all']]), np_ari)
                ami_sas_amp_corrcoef_ens = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_ens_final_all']]), np_ami)

                ari_sig_pw_corr = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_sig_pw_final_all']]), np_ari)
                ami_sig_pw_corr = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_sig_pw_final_all']]), np_ami)

                tv_sas_amp_corrcoef_adj_diff = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_adj_diff_final_all']]), np_tv)
                tv_sas_amp_corrcoef_in_agg = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_in_agg_final_all']]), np_tv)
                tv_sas_amp_corrcoef_ln_vx = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_ln_vx_final_all']]), np_tv)
                tv_sas_amp_corrcoef_apprx_ls = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_apprx_ls_final_all']]), np_tv)
                tv_sas_amp_corrcoef_ens = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_ens_final_all']]), np_tv)
                tv_sig_pw_corr = correl_func(np.asarray([np.linalg.norm(d_sas_final[K]) for d_sas_final in df_nc_sas['d_sig_pw_final_all']]), np_tv)
                
                if spearman_or_pearson == 'pearson':
                    l_ari_sas_amp_corr_adj_diff.append(ari_sas_amp_corrcoef_adj_diff[0][1])
                    l_ami_sas_amp_corr_adj_diff.append(ami_sas_amp_corrcoef_adj_diff[0][1])
                    l_ari_sas_amp_corr_in_agg.append(ari_sas_amp_corrcoef_in_agg[0][1])
                    l_ami_sas_amp_corr_in_agg.append(ami_sas_amp_corrcoef_in_agg[0][1])
                    l_ari_sas_amp_corr_ln_vx.append(ari_sas_amp_corrcoef_ln_vx[0][1])
                    l_ami_sas_amp_corr_ln_vx.append(ami_sas_amp_corrcoef_ln_vx[0][1])
                    l_ari_sas_amp_corr_apprx_ls.append(ari_sas_amp_corrcoef_apprx_ls[0][1])
                    l_ami_sas_amp_corr_apprx_ls.append(ami_sas_amp_corrcoef_apprx_ls[0][1])
                    l_ari_sas_amp_corr_ens.append(ari_sas_amp_corrcoef_ens[0][1])
                    l_ami_sas_amp_corr_ens.append(ami_sas_amp_corrcoef_ens[0][1])
                    l_ari_sig_pw_corr.append(ari_sig_pw_corr[0][1])
                    l_ami_sig_pw_corr.append(ami_sig_pw_corr[0][1])
                    l_tv_sas_amp_corr_adj_diff.append(tv_sas_amp_corrcoef_adj_diff[0][1])
                    l_tv_sas_amp_corr_in_agg.append(tv_sas_amp_corrcoef_in_agg[0][1])
                    l_tv_sas_amp_corr_ln_vx.append(tv_sas_amp_corrcoef_ln_vx[0][1])
                    l_tv_sas_amp_corr_apprx_ls.append(tv_sas_amp_corrcoef_apprx_ls[0][1])
                    l_tv_sas_amp_corr_ens.append(tv_sas_amp_corrcoef_ens[0][1])
                    l_tv_sig_pw_corr.append(tv_sig_pw_corr[0][1])
                elif spearman_or_pearson == 'spearman':
                    l_ari_sas_amp_corr_adj_diff.append(ari_sas_amp_corrcoef_adj_diff[0])
                    l_ami_sas_amp_corr_adj_diff.append(ami_sas_amp_corrcoef_adj_diff[0])
                    l_ari_sas_amp_corr_in_agg.append(ari_sas_amp_corrcoef_in_agg[0])
                    l_ami_sas_amp_corr_in_agg.append(ami_sas_amp_corrcoef_in_agg[0])
                    l_ari_sas_amp_corr_ln_vx.append(ari_sas_amp_corrcoef_ln_vx[0])
                    l_ami_sas_amp_corr_ln_vx.append(ami_sas_amp_corrcoef_ln_vx[0])
                    l_ari_sas_amp_corr_apprx_ls.append(ari_sas_amp_corrcoef_apprx_ls[0])
                    l_ami_sas_amp_corr_apprx_ls.append(ami_sas_amp_corrcoef_apprx_ls[0])
                    l_ari_sas_amp_corr_ens.append(ari_sas_amp_corrcoef_ens[0])
                    l_ami_sas_amp_corr_ens.append(ami_sas_amp_corrcoef_ens[0])
                    l_ari_sig_pw_corr.append(ari_sig_pw_corr[0])
                    l_ami_sig_pw_corr.append(ami_sig_pw_corr[0])
                    l_tv_sas_amp_corr_adj_diff.append(tv_sas_amp_corrcoef_adj_diff[0])
                    l_tv_sas_amp_corr_in_agg.append(tv_sas_amp_corrcoef_in_agg[0])
                    l_tv_sas_amp_corr_ln_vx.append(tv_sas_amp_corrcoef_ln_vx[0])
                    l_tv_sas_amp_corr_apprx_ls.append(tv_sas_amp_corrcoef_apprx_ls[0])
                    l_tv_sas_amp_corr_ens.append(tv_sas_amp_corrcoef_ens[0])
                    l_tv_sig_pw_corr.append(tv_sig_pw_corr[0])
                
            if norm_sig:
                img_suffix = '_norm'
            else:
                img_suffix = '_non_norm'
            if no_in_agg:
                img_suffix += '_no_in_agg'
            else:
                img_suffix += ''
            if all_good_and_bad:
                img_suffix += '_all_good_bad'
            else:
                if good_or_bad == 'good':
                    img_suffix += '_good_only'
                elif good_or_bad == 'bad':
                    img_suffix += '_bad_only'
            if init_or_final == 'final':
                img_suffix += '_final'
            elif init_or_final == 'init':
                img_suffix += '_init'

            # >>> draw correcoef between ari, ami and sig_pw
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.set_title(img_prefix + r'PPMCCs ARI, AMI vs $||\nabla s||_2$', fontsize=10, fontweight='semibold')
            axes.grid(True)
            ari_color = 'tab:blue'
            ami_color = 'tab:orange'
            axes.errorbar(l_K, l_ari_sig_pw_corr,
                          fmt='-o', c=ari_color, capsize=2, capthick=1, label=r'ARI vs $||s||_2$')
            axes.errorbar(l_K, l_ami_sig_pw_corr,
                          fmt='-o', c=ami_color, capsize=2, capthick=1, label=r'AMI vs $||s||_2$')
            axes.set_xticks(l_K)
            axes.set_xticklabels(['K=%s' % K for K in l_K])
            axes.legend()
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'correcoef_ari_ami_vs_sig_pw_norm' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> draw correcoef between ari and sgs
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.set_title(img_prefix + r'PPMCC ARI vs $||\mathcal{M}||_2$', fontsize=10, fontweight='semibold')
            axes.grid(True)
            adj_diff_color = 'tab:blue'
            in_agg_color = 'tab:orange'
            ln_vx_color = 'tab:green'
            apprx_ls_color = 'tab:purple'
            ens_color = 'tab:red'
            axes.errorbar(l_K, l_ari_sas_amp_corr_adj_diff,
                          fmt='-o', c=adj_diff_color, capsize=2, capthick=1, label='ADJ-DIFF')
            axes.errorbar(l_K, l_ari_sas_amp_corr_in_agg,
                          fmt='-o', c=in_agg_color, capsize=2, capthick=1, label='IN-AGG')
            axes.errorbar(l_K, l_ari_sas_amp_corr_ln_vx,
                          fmt='-o', c=ln_vx_color, capsize=2, capthick=1, label='LN-VX')
            axes.errorbar(l_K, l_ari_sas_amp_corr_apprx_ls,
                          fmt='-o', c=apprx_ls_color, capsize=2, capthick=1, label='APPRX-LS')
            axes.errorbar(l_K, l_ari_sas_amp_corr_ens,
                          fmt='-o', c=ens_color, capsize=2, capthick=1, label='ENS')
            axes.set_xticks(l_K)
            axes.set_xticklabels(['K=%s' % K for K in l_K])
            axes.legend()
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'correcoef_ari_vs_sgs_amp' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> draw correcoef between ami and sgs
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.set_title(img_prefix + r'PPMCC AMI vs $||\mathcal{M}||_2$', fontsize=10, fontweight='semibold')
            axes.grid(True)
            adj_diff_color = 'tab:blue'
            in_agg_color = 'tab:orange'
            ln_vx_color = 'tab:green'
            ens_color = 'tab:red'
            axes.errorbar(l_K, l_ami_sas_amp_corr_adj_diff,
                          fmt='-o', c=adj_diff_color, capsize=2, capthick=1, label='ADJ-DIFF')
            axes.errorbar(l_K, l_ami_sas_amp_corr_in_agg,
                          fmt='-o', c=in_agg_color, capsize=2, capthick=1, label='IN-AGG')
            axes.errorbar(l_K, l_ami_sas_amp_corr_ln_vx,
                          fmt='-o', c=ln_vx_color, capsize=2, capthick=1, label='LN-VX')
            axes.errorbar(l_K, l_ami_sas_amp_corr_apprx_ls,
                          fmt='-o', c=apprx_ls_color, capsize=2, capthick=1, label='APPRX-LS')
            axes.errorbar(l_K, l_ami_sas_amp_corr_ens,
                          fmt='-o', c=ens_color, capsize=2, capthick=1, label='ENS')
            axes.set_xticks(l_K)
            axes.set_xticklabels(['K=%s' % K for K in l_K])
            axes.legend()
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'correcoef_ami_vs_sgs_amp' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

            # >>> draw correcoef between tv and sgs
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 3))
            axes.set_title(img_prefix + r'PPMCC $\tau$ vs $||\mathcal{M}||_2$', fontsize=10, fontweight='semibold')
            axes.grid(True)
            adj_diff_color = 'tab:blue'
            in_agg_color = 'tab:orange'
            ln_vx_color = 'tab:green'
            apprx_ls_color = 'tab:purple'
            ens_color = 'tab:red'
            axes.errorbar(l_K, l_tv_sas_amp_corr_adj_diff,
                          fmt='-o', c=adj_diff_color, capsize=2, capthick=1, label='ADJ-DIFF')
            axes.errorbar(l_K, l_tv_sas_amp_corr_in_agg,
                          fmt='-o', c=in_agg_color, capsize=2, capthick=1, label='IN-AGG')
            axes.errorbar(l_K, l_tv_sas_amp_corr_ln_vx,
                          fmt='-o', c=ln_vx_color, capsize=2, capthick=1, label='LN-VX')
            axes.errorbar(l_K, l_tv_sas_amp_corr_apprx_ls,
                          fmt='-o', c=apprx_ls_color, capsize=2, capthick=1, label='APPRX-LS')
            axes.errorbar(l_K, l_tv_sas_amp_corr_ens,
                          fmt='-o', c=ens_color, capsize=2, capthick=1, label='ENS')
            axes.set_xticks(l_K)
            axes.set_xticklabels(['K=%s' % K for K in l_K])
            axes.legend()
            plt.tight_layout(pad=1.0)
            if save_img:
                plt.savefig(good_and_bad_runs_save_path + 'correcoef_tv_vs_sgs_amp' + img_suffix + '.PNG', format='PNG')
            if show_img:
                plt.show()
            plt.clf()
            plt.close()

    elif cmd == 'compare_pulse_and_smoothed_spectra':
        plt.set_loglevel('error')
        graph_name = 'uw_symA'
        ne_type = 'uw_symA_pulse_BCD'
        # ne_type = 'uw_symA_high_freq'

        STAGE_gen_man_sig = False
        STAGE_learn_node_embeds = False
        STAGE_sas_and_trans = True
        STAGE_ln_vx_only = True

        if STAGE_gen_man_sig:
            man_sig_folder = g_work_dir + 'experiments/uw_symA_spectra_by_cases/man_sigs/'
            if not os.path.exists(man_sig_folder):
                os.mkdir(man_sig_folder)
            graph_path = g_work_dir + graph_name + '.pickle'
            nx_graph = nx.read_gpickle(graph_path)
            np_init = manual_node_embed(ne_type)
            np_init = preprocessing.normalize(np_init)
            np.save(man_sig_folder + 'np_init@' + ne_type + '.npy', np_init)

        if STAGE_learn_node_embeds:
            node_embed_folder = g_work_dir + 'experiments/uw_symA_spectra_by_cases/node_embeds/'
            if not os.path.exists(node_embed_folder):
                os.mkdir(node_embed_folder)
            man_sig_folder = g_work_dir + 'experiments/uw_symA_spectra_by_cases/man_sigs/'
            np_init_ne = np.load(man_sig_folder + 'np_init@' + ne_type + '.npy')
            graph_path = g_work_dir + graph_name + '.pickle'
            nx_graph = nx.read_gpickle(graph_path)
            embed_dim = 3
            k_cluster = 4
            d_gt = {'A': 0, 'B': 1, 'E': 1, 'F': 1, 'K': 1, 'C': 2, 'G': 2, 'H': 2, 'L': 2, 'D': 3, 'I': 3, 'J': 3,
                    'M': 3}
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            run = 0
            run_id = graph_name + '_' + now + '_' + str(run)
            dffsn_loss_w_range = None
            J_prob_range = None
            tv_loss_w_range = np.arange(0.1, 1.1, 0.1)
            bv_loss_w_range = [1.0]
            lv_loss_w_range = None
            lg_loss_w_range = None
            gs_loss_w_range = None
            max_epoch = 3500
            save_int = True
            show_img = False
            show_init_img = True
            do_cluster = True
            d_gt = d_gt
            k_cluster = k_cluster
            cluster_alg = 'spectral'

            configured_run(run_id, nx_graph, np_init_ne, embed_dim,
                           dffsn_loss_w_range, J_prob_range, tv_loss_w_range,
                           bv_loss_w_range, lv_loss_w_range, lg_loss_w_range, gs_loss_w_range,
                           max_epoch=max_epoch, save_int=save_int, save_folder=node_embed_folder,
                           show_img=show_img,
                           show_init_img=show_init_img, do_cluster=do_cluster, d_gt=d_gt, k_cluster=k_cluster,
                           cluster_alg=cluster_alg)

        if STAGE_ln_vx_only:
            num_ln_vx_trials = 50
            ln_vx_only_save_path = g_work_dir + 'experiments/uw_symA_spectra_by_cases/ln_vx_only/'
            if not os.path.exists(ln_vx_only_save_path):
                os.mkdir(ln_vx_only_save_path)

            node_embed_folder = g_work_dir + 'experiments/uw_symA_spectra_by_cases/node_embeds/'
            # run_id = '20210427100644_0'
            # run_id = '20210609192829_0'
            run_id = '20210610160153_0'
            trg_ne_embed_folder = node_embed_folder + 'ne_run_uw_symA_' + run_id + '/'
            # ne_int_file_path = trg_ne_embed_folder + 'ne_run@uw_symA_20210427100644_0@df-0.0_jp-0.0_tv0.4_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500@ne_int.pickle'
            # ne_int_file_path = trg_ne_embed_folder + 'ne_run@uw_symA_20210609192829_0@df-0.0_jp-0.0_tv0.2_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500@ne_int.pickle'
            ne_int_file_path = trg_ne_embed_folder + 'ne_run@uw_symA_20210610160153_0@df-0.0_jp-0.0_tv0.1_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500@ne_int.pickle'

            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            spec_seq_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/spec_seqs/'
            df_spec_seq = pd.read_pickle(spec_seq_save_path + 'spec_seq@' + graph_name + '@' + spec_seq_param_name + '.pickle')

            l_df_ln_to_vx_eigs = []
            for trial_id in range(num_ln_vx_trials):
                df_ln_to_vx_eigs = pd.read_pickle(spec_seq_save_path + 'ln_to_vx_eigs@' + graph_name + '@'
                                                  + spec_seq_param_name + '#%s.pickle' % str(trial_id))
                l_df_ln_to_vx_eigs.append(df_ln_to_vx_eigs)

            timer_start = time.time()
            cnt = 0
            df_ne_int = pd.read_pickle(ne_int_file_path)
            l_ln_vx_rec_by_epoch = []
            for _, ne_int_rec in df_ne_int.iterrows():
                epoch = ne_int_rec['epoch']
                np_ne_int = ne_int_rec['node_embed_int']
                np_ne_int = preprocessing.normalize(np_ne_int)
                d_ln_vx_rec_by_K = dict()
                for trial_id in range(num_ln_vx_trials):
                    df_ln_to_vx_eigs = l_df_ln_to_vx_eigs[trial_id]
                    df_analysis = ln_vx(np_ne_int, df_spec_seq, df_ln_to_vx_eigs)
                    for K, ana_rec in df_analysis.iterrows():
                        ln_vx_trans = ana_rec['sig_pwd_rec_vx_ft']
                        if K not in d_ln_vx_rec_by_K:
                            d_ln_vx_rec_by_K[K] = [ln_vx_trans]
                        else:
                            d_ln_vx_rec_by_K[K].append(ln_vx_trans)
                l_ln_vx_rec_by_epoch.append((epoch, d_ln_vx_rec_by_K))
                cnt += 1
                if cnt % 1000 == 0 and cnt >= 1000:
                    logging.debug('[STAGE_ln_vx_only] %s epoch ln_vx done in %s secs.'
                                  % (cnt, time.time() - timer_start))
            df_ln_vx_by_epoch = pd.DataFrame(l_ln_vx_rec_by_epoch, columns=['epoch', 'd_ln_vx_rec_by_K'])
            df_ln_vx_by_epoch = df_ln_vx_by_epoch.set_index('epoch')
            pd.to_pickle(df_ln_vx_by_epoch, ln_vx_only_save_path + 'ne_run@uw_symA_' + run_id + '@ln_vx_only.pickle')

            logging.debug('[STAGE_ln_vx_only] all done in %s secs.' % str(time.time() - timer_start))

        if STAGE_sas_and_trans:
            sas_and_trans_folder = g_work_dir + 'experiments/uw_symA_spectra_by_cases/sas_and_trans/'
            if not os.path.exists(sas_and_trans_folder):
                os.mkdir(sas_and_trans_folder)
            node_embed_folder = g_work_dir + 'experiments/uw_symA_spectra_by_cases/node_embeds/'
            # run_id = '20210427100644_0'
            # run_id = '20210609192829_0'
            run_id = '20210610160153_0'
            trg_ne_embed_folder = node_embed_folder + 'ne_run_uw_symA_' + run_id + '/'
            # ne_int_file_path = trg_ne_embed_folder + 'ne_run@uw_symA_20210427100644_0@df-0.0_jp-0.0_tv0.4_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500@ne_int.pickle'
            # ne_int_file_path = trg_ne_embed_folder + 'ne_run@uw_symA_20210609192829_0@df-0.0_jp-0.0_tv0.2_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500@ne_int.pickle'
            ne_int_file_path = trg_ne_embed_folder + 'ne_run@uw_symA_20210610160153_0@df-0.0_jp-0.0_tv0.1_bv1.0_lv-0.0_lg-0.0_gs-0.0_ep3500@ne_int.pickle'

            use_ln_to_vx_eig_convert = False
            l_metrics = ['tv_loss']
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            spec_seq_save_path = g_work_dir + 'experiments/uw_symA_spectra_by_cases/spec_seqs/'
            df_spec_seq = pd.read_pickle(spec_seq_save_path + 'spec_seq@' + graph_name + '@' + spec_seq_param_name + '.pickle')
            df_ln_to_vx_eigs = pd.read_pickle(spec_seq_save_path + 'ln_to_vx_eigs@' + graph_name + '@' + spec_seq_param_name + '.pickle')

            l_analysis_per_run = []
            df_ne_int = pd.read_pickle(ne_int_file_path)
            for _, ne_int_rec in df_ne_int.iterrows():
                l_per_epoch_metrics = []
                for metric_str in l_metrics:
                    l_per_epoch_metrics.append(ne_int_rec[metric_str])
                epoch = ne_int_rec['epoch']
                np_ne_int = ne_int_rec['node_embed_int']
                np_ne_int = preprocessing.normalize(np_ne_int)
                df_analysis = stratified_graph_spectra_and_transformations(np_ne_int,
                                                                           df_spec_seq,
                                                                           save_ret=False,
                                                                           save_path=None,
                                                                           np_pw_dist=None,
                                                                           norm_sig_FADM=False,
                                                                           use_ln_to_vx_eig_convert=use_ln_to_vx_eig_convert,
                                                                           df_ln_to_vx_eig_convert=None)
                l_analysis_per_run.append(
                    (epoch, df_analysis, l_metrics, l_per_epoch_metrics, np_ne_int))
            df_analysis_per_run = pd.DataFrame(l_analysis_per_run, columns=['epoch',
                                                                            'df_analysis',
                                                                            'metric_names',
                                                                            'metric_vals',
                                                                            'np_ne'])
            df_analysis_per_run = df_analysis_per_run.set_index('epoch')
            pd.to_pickle(df_analysis_per_run, sas_and_trans_folder + 'ne_run@uw_symA_' + run_id + '@sas_and_trans.pickle')

    elif cmd == 'spec_seq_good_vs_bad':
        plt.set_loglevel('error')
        good_folder = g_work_dir + 'experiments/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@good_vs_bad/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@good_rets/'
        bad_folder = g_work_dir + 'experiments/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@good_vs_bad/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@bad_rets/'
        save_folder = g_work_dir + 'experiments/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@good_vs_bad/'
        draw_spec_seq_good_vs_bad(good_folder, bad_folder, save_img=True, show_img=False, save_folder=save_folder)

    # elif cmd == 'spec_seq_epoch1000_vs_epoch200':
    #     good_folder = g_work_dir + 'experiments/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@good_rets/'
    #     bad_folder = g_work_dir + 'experiments/uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0_ep1000@bad_rets/'
    #     img_title = 'uw_symA@df-0.0_jp-0.0_tv1.0_bv-0.0_lv-0.0_lg-0.0_gs-0.0 epoch_1000 vs epoch_200'
    #     draw_spec_seq_good_vs_bad(good_folder, bad_folder, img_title)

    elif cmd == 'pure_test':
        STAGE_combine_gsp_vs_sas_rets = False
        STAGE_find_similar_tv_runs = False
        STATE_distributions_of_ari_nmi = True
        STATE_real_valued_graph_sig = False

        if STAGE_combine_gsp_vs_sas_rets:
            plt.set_loglevel('error')
            l_job_ids = ['100_rand_graphs', '100_rand_graphs_rand_pulse', '100_sbm_graphs', '100_sbm_graphs_rand_pulse']

            save_img = True
            show_img = True
            classic_gsp_vs_sas_and_trans_save_path = g_work_dir + 'experiments/gsp_vs_sas/classic_gsp_vs_sas_and_trans/'

            d_ret_collection = dict()
            for job_id in l_job_ids:
                df_classic_gsp_vs_sas_and_trans = pd.read_pickle(classic_gsp_vs_sas_and_trans_save_path
                                                                 + '%s#classic_gsp_vs_sas_and_trans.pickle' % str(
                    job_id))

                d_sim_classic_vs_adj_diff_by_K = dict()
                d_sim_classic_vs_ln_agg_by_K = dict()
                d_sim_classic_vs_ln_conv_by_K = dict()
                for graph_name, sim_rec in df_classic_gsp_vs_sas_and_trans.iterrows():
                    df_sim_per_graph = sim_rec['df_sig_ft_sim']
                    for K, sim_per_graph_rec in df_sim_per_graph.iterrows():
                        sim_classic_vs_adj_diff = sim_per_graph_rec['sim_classic_vs_adj_diff']
                        sim_classic_vs_ln_agg = sim_per_graph_rec['sim_classic_vs_ln_agg']
                        sim_classic_vs_ln_conv = sim_per_graph_rec['sim_classic_vs_ln_conv']

                        if K not in d_sim_classic_vs_adj_diff_by_K:
                            d_sim_classic_vs_adj_diff_by_K[K] = [sim_classic_vs_adj_diff]
                        else:
                            d_sim_classic_vs_adj_diff_by_K[K].append(sim_classic_vs_adj_diff)

                        if K not in d_sim_classic_vs_ln_agg_by_K:
                            d_sim_classic_vs_ln_agg_by_K[K] = [sim_classic_vs_ln_agg]
                        else:
                            d_sim_classic_vs_ln_agg_by_K[K].append(sim_classic_vs_ln_agg)

                        if K not in d_sim_classic_vs_ln_conv_by_K:
                            d_sim_classic_vs_ln_conv_by_K[K] = [sim_classic_vs_ln_conv]
                        else:
                            d_sim_classic_vs_ln_conv_by_K[K].append(sim_classic_vs_ln_conv)

                for K in d_sim_classic_vs_adj_diff_by_K:
                    mean_sim_classic_vs_adj_diff_by_K = np.mean(d_sim_classic_vs_adj_diff_by_K[K])
                    std_sim_classic_vs_adj_diff_by_K = np.std(d_sim_classic_vs_adj_diff_by_K[K])
                    d_sim_classic_vs_adj_diff_by_K[K] = (mean_sim_classic_vs_adj_diff_by_K,
                                                         std_sim_classic_vs_adj_diff_by_K)

                for K in d_sim_classic_vs_ln_agg_by_K:
                    mean_sim_classic_vs_ln_agg_by_K = np.mean(d_sim_classic_vs_ln_agg_by_K[K])
                    std_sim_classic_vs_ln_agg_by_K = np.std(d_sim_classic_vs_ln_agg_by_K[K])
                    d_sim_classic_vs_ln_agg_by_K[K] = (mean_sim_classic_vs_ln_agg_by_K,
                                                       std_sim_classic_vs_ln_agg_by_K)

                for K in d_sim_classic_vs_ln_conv_by_K:
                    mean_sim_classic_vs_ln_conv_by_K = np.mean(d_sim_classic_vs_ln_conv_by_K[K])
                    std_sim_classic_vs_ln_conv_by_K = np.std(d_sim_classic_vs_ln_conv_by_K[K])
                    d_sim_classic_vs_ln_conv_by_K[K] = (mean_sim_classic_vs_ln_conv_by_K,
                                                        std_sim_classic_vs_ln_conv_by_K)

                d_ret_collection[job_id] = [d_sim_classic_vs_adj_diff_by_K, d_sim_classic_vs_ln_agg_by_K,
                                            d_sim_classic_vs_ln_conv_by_K]

            l_K = d_ret_collection['100_rand_graphs'][0].keys()
            img_name = 'all#ft_sim#classic_vs_sas_and_trans'
            # vmax = np.max([np.max(np.abs(item)) for item in df_sas_ana['sig_pwd_ln_ft'].to_list()])
            # vmin = np.min([np.min(np.abs(item)) for item in df_sas_ana['sig_pwd_ln_ft'].to_list()])
            fig, axes = plt.subplots(ncols=1, nrows=len(l_K), figsize=(8, 20))
            # fig.suptitle(img_name, fontsize=15, fontweight='semibold')
            idx = 0
            for K in l_K:
                d_sim_classic_vs_adj_diff_by_K = d_ret_collection['100_rand_graphs'][0]
                d_sim_classic_vs_ln_agg_by_K = d_ret_collection['100_rand_graphs'][1]
                d_sim_classic_vs_ln_conv_by_K = d_ret_collection['100_rand_graphs'][2]
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1]]
                y_vmax_1 = np.round(np.max([l_means[i] + l_stds[i] for i in range(3)]), decimals=1)
                y_vmin_1 = np.round(np.min([l_means[i] - l_stds[i] for i in range(3)]), decimals=1)
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(3)], l_means, yerr=l_stds,
                                   marker='o', fmt='o', c='tab:blue', mfc='tab:blue', mec='tab:blue',
                                   capsize=2, capthick=1, label='ERM-Rand')

                d_sim_classic_vs_adj_diff_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][0]
                d_sim_classic_vs_ln_agg_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][1]
                d_sim_classic_vs_ln_conv_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][2]
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1]]
                y_vmax_2 = np.round(np.max([l_means[i] + l_stds[i] for i in range(3)]), decimals=1)
                y_vmin_2 = np.round(np.min([l_means[i] - l_stds[i] for i in range(3)]), decimals=1)
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(3, 6)], l_means, yerr=l_stds,
                                   marker='o', fmt='o', c='tab:green', mfc='tab:green', mec='tab:green',
                                   capsize=2, capthick=1, label='ERM-Pulse')

                d_sim_classic_vs_adj_diff_by_K = d_ret_collection['100_sbm_graphs'][0]
                d_sim_classic_vs_ln_agg_by_K = d_ret_collection['100_sbm_graphs'][1]
                d_sim_classic_vs_ln_conv_by_K = d_ret_collection['100_sbm_graphs'][2]
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1]]
                y_vmax_3 = np.round(np.max([l_means[i] + l_stds[i] for i in range(3)]), decimals=1)
                y_vmin_3 = np.round(np.min([l_means[i] - l_stds[i] for i in range(3)]), decimals=1)
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(6, 9)], l_means, yerr=l_stds,
                                   marker='o', fmt='o', c='tab:orange', mfc='tab:orange', mec='tab:orange',
                                   capsize=2, capthick=1, label='SBM-Rand')

                d_sim_classic_vs_adj_diff_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][0]
                d_sim_classic_vs_ln_agg_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][1]
                d_sim_classic_vs_ln_conv_by_K = d_ret_collection['100_rand_graphs_rand_pulse'][2]
                l_means = [d_sim_classic_vs_adj_diff_by_K[K][0],
                           d_sim_classic_vs_ln_agg_by_K[K][0],
                           d_sim_classic_vs_ln_conv_by_K[K][0]]
                l_stds = [d_sim_classic_vs_adj_diff_by_K[K][1],
                          d_sim_classic_vs_ln_agg_by_K[K][1],
                          d_sim_classic_vs_ln_conv_by_K[K][1]]
                y_vmax_4 = np.round(np.max([l_means[i] + l_stds[i] for i in range(3)]), decimals=1)
                y_vmin_4 = np.round(np.min([l_means[i] - l_stds[i] for i in range(3)]), decimals=1)
                axes[idx].grid(True)
                axes[idx].set_title('K = %s' % K, fontsize=10, fontweight='semibold')
                axes[idx].errorbar([i for i in range(9, 12)], l_means, yerr=l_stds,
                                   marker='o', fmt='o', c='tab:red', mfc='tab:red', mec='tab:red',
                                   capsize=2, capthick=1, label='SBM-Pulse')

                y_vmin = np.min([y_vmin_1, y_vmin_2, y_vmin_3, y_vmin_4])
                y_vmax = np.max([y_vmax_1, y_vmax_2, y_vmax_3, y_vmax_4])
                y_step = np.round((y_vmax - y_vmin) / 8, decimals=2)
                axes[idx].set_xticks([i for i in range(12)])
                axes[idx].set_xticklabels(
                    ['ADJ-DIFF', 'IN-AGG', 'LN-VX', 'ADJ-DIFF', 'IN-AGG', 'LN-VX', 'ADJ-DIFF', 'IN-AGG', 'LN-VX',
                     'ADJ-DIFF', 'IN-AGG', 'LN-VX'])
                axes[idx].set_yticks([i for i in np.arange(y_vmin, y_vmax + y_step, y_step)])
                axes[idx].legend()
                idx += 1
            plt.tight_layout(pad=1.0)
            # plt.subplots_adjust(top=0.94)
            if save_img:
                plt.savefig(classic_gsp_vs_sas_and_trans_save_path + img_name + '.PNG', format='PNG')
            if show_img:
                plt.show()

        if STAGE_find_similar_tv_runs:
            tv_loss_delta = 0.001
            sas_and_trans_file_folder = g_work_dir + 'experiments/uw_symA_node_embeds/learned_node_embeds/'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            good_ne_folder_file = 'good_ne_folders.txt'
            bad_ne_folder_file = 'bad_ne_folders.txt'
            sas_and_trans_file_fmt = '{0}@sas_and_trans.pickle'
            l_good_folders = []
            l_bad_folders = []
            with open(good_and_bad_runs_save_path + good_ne_folder_file, 'r') as in_fd:
                for ln in in_fd:
                    l_good_folders.append(ln.strip())
                in_fd.close()
            with open(good_and_bad_runs_save_path + bad_ne_folder_file, 'r') as in_fd:
                for ln in in_fd:
                    l_bad_folders.append(ln.strip())
                in_fd.close()

            timer_start = time.time()
            l_good_vs_bad_tv_loss_delta = []
            for good_idx, good_folder in enumerate(l_good_folders):
                l_name_fields_good = good_folder.split('/')[-2].split('_')
                sas_and_trans_file_prefix = l_name_fields_good[0] + '_' + l_name_fields_good[1] + '@' \
                                            + l_name_fields_good[2] + '_' + l_name_fields_good[3] + '_' \
                                            + l_name_fields_good[4] + '_' + l_name_fields_good[5]
                sas_and_trans_file_name = sas_and_trans_file_fmt.format(sas_and_trans_file_prefix)
                df_sas_and_trans_good = pd.read_pickle(good_folder + sas_and_trans_file_name)
                max_epoch_good = np.max(list(df_sas_and_trans_good.index))
                l_metric_names_good = df_sas_and_trans_good.loc[max_epoch_good]['metric_names']
                l_metric_vals_good = df_sas_and_trans_good.loc[max_epoch_good]['metric_vals']
                tv_loss_final_good = l_metric_vals_good[l_metric_names_good.index('tv_loss')]

                for bad_idx, bad_folder in enumerate(l_bad_folders):
                    l_name_fields_bad = bad_folder.split('/')[-2].split('_')
                    sas_and_trans_file_prefix = l_name_fields_bad[0] + '_' + l_name_fields_bad[1] + '@' \
                                                + l_name_fields_bad[2] + '_' + l_name_fields_bad[3] + '_' \
                                                + l_name_fields_bad[4] + '_' + l_name_fields_bad[5]
                    sas_and_trans_file_name = sas_and_trans_file_fmt.format(sas_and_trans_file_prefix)
                    df_sas_and_trans_bad = pd.read_pickle(bad_folder + sas_and_trans_file_name)
                    max_epoch_bad = np.max(list(df_sas_and_trans_bad.index))
                    l_metric_names_bad = df_sas_and_trans_bad.loc[max_epoch_bad]['metric_names']
                    l_metric_vals_bad = df_sas_and_trans_bad.loc[max_epoch_bad]['metric_vals']
                    tv_loss_final_bad = l_metric_vals_bad[l_metric_names_bad.index('tv_loss')]

                    l_good_vs_bad_tv_loss_delta.append((good_idx, bad_idx, np.abs(tv_loss_final_good - tv_loss_final_bad)))
                    if len(l_good_vs_bad_tv_loss_delta) % 500 == 0 and len(l_good_vs_bad_tv_loss_delta) >= 500:
                        logging.debug('%s comparisons done in %s secs.'
                                      % (len(l_good_vs_bad_tv_loss_delta), time.time() - timer_start))
            logging.debug('%s comparisons done in %s secs.'
                          % (len(l_good_vs_bad_tv_loss_delta), time.time() - timer_start))
            l_good_vs_bad_tv_loss_delta = sorted(l_good_vs_bad_tv_loss_delta, key=lambda k: k[2])
            l_good_vs_bad_tv_loss_delta_rec = [(l_good_folders[item[0]], l_bad_folders[item[1]], item[2]) for item in l_good_vs_bad_tv_loss_delta]
            df_good_vs_bad_tv_loss_delta = pd.DataFrame(l_good_vs_bad_tv_loss_delta_rec, columns=['good_folder',
                                                                                                  'bad_folder',
                                                                                                  'tv_loss_delta'])
            pd.to_pickle(df_good_vs_bad_tv_loss_delta, good_and_bad_runs_save_path + 'good_vs_bad_tv_loss_delta.pickle')
            logging.debug('all done.')

        if STATE_distributions_of_ari_nmi:
            good_vs_bad_draw_data_save_name = 'good_vs_bad_draw_data_by_epoch'
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            norm_sig = True
            no_in_agg = False
            sgs_chosen = False
            if norm_sig:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm_sgs_chosen'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_norm'
            else:
                if no_in_agg:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_no_in_agg'
                elif sgs_chosen:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm_sgs_chosen'
                else:
                    draw_data_save_name = good_vs_bad_draw_data_save_name + '_non_norm'
            good_and_bad_runs_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/good_and_bad_runs/'
            df_good_vs_bad_draw_data = pd.read_pickle(good_and_bad_runs_save_path + draw_data_save_name + '@'
                                                      + graph_name + '@' + spec_seq_param_name + '.pickle')
            d_nc_perf = df_good_vs_bad_draw_data.loc['good']['d_nc_perf']
            max_epoch = np.max(list(d_nc_perf['ari'].keys()))
            l_ari_final = d_nc_perf['ari'][max_epoch]
            l_nmi_final = d_nc_perf['nmi'][max_epoch]
            y_vmin_final = np.min([np.min(l_ari_final), np.min(l_nmi_final)])
            y_vmax_final = np.min([np.max(l_ari_final), np.max(l_nmi_final)])

            fig, axs = plt.subplots(ncols=1, nrows=2)
            idx = 0
            axs[idx].grid(True)
            sns.kdeplot(l_ari_final, ax=axs[idx], color='tab:orange', cut=0, fill=True, label='ARI_final')
            sns.kdeplot(l_nmi_final, ax=axs[idx], color='tab:blue', cut=0, fill=True, label='AMI_final')
            axs[idx].set_xticks(np.round([i for i in np.arange(y_vmin_final, y_vmax_final + 0.1, 0.1)], decimals=2))
            axs[idx].legend()

            idx += 1
            axs[idx].grid(True)
            max_epoch = np.max(list(d_nc_perf['ari'].keys()))
            l_ari_init = d_nc_perf['ari'][0]
            l_nmi_init = d_nc_perf['nmi'][0]
            y_vmin_init = np.min([np.min(l_ari_init), np.min(l_nmi_init)])
            y_vmax_init = np.min([np.max(l_ari_init), np.max(l_nmi_init)])

            # fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
            sns.kdeplot(l_ari_init, ax=axs[idx], color='tab:orange', cut=0, fill=True, label='ARI_init')
            sns.kdeplot(l_nmi_init, ax=axs[idx], color='tab:blue', cut=0, fill=True, label='AMI_init')
            axs[idx].set_xticks(np.round([i for i in np.arange(y_vmin_init, y_vmax_init + 0.1, 0.1)], decimals=2))
            axs[idx].legend()
            plt.show()
            plt.clf()
            plt.close()
            print()

        if STATE_real_valued_graph_sig:
            use_ln_to_vx_eig_convert = True
            graph_name = 'uw_symA'
            spec_seq_param_name = 'svd#adj#nfeadms'
            spec_seq_save_path = g_work_dir + 'experiments/uw_symA_node_embeds/spec_seqs/'
            df_spec_seq = pd.read_pickle(spec_seq_save_path + 'spec_seq@' + graph_name + '@' + spec_seq_param_name + '.pickle')
            df_ln_to_vx_eigs = pd.read_pickle(spec_seq_save_path + 'ln_to_vx_eigs@' + graph_name + '@' + spec_seq_param_name + '.pickle')
            graph_path = g_work_dir + graph_name + '.pickle'
            nx_graph = nx.read_gpickle(graph_path)
            np_pw_dist = manual_pw_distance_mat(df_spec_seq, nx_graph, 'uw_symA_ideal_cluster')

            df_analysis = stratified_graph_spectra_and_transformations(None,
                                                                       df_spec_seq,
                                                                       save_ret=False,
                                                                       save_path=None,
                                                                       np_pw_dist=np_pw_dist,
                                                                       norm_sig_FADM=False,
                                                                       use_ln_to_vx_eig_convert=use_ln_to_vx_eig_convert,
                                                                       df_ln_to_vx_eig_convert=df_ln_to_vx_eigs)

        #     'K', 'sig_pwd_FADM',
        # 'eig_vals', 'sig_vs_eig_sas',
        # 'ln_eig_vals', 'sig_pwd_ln_ft',
        # 'eff_FEADM_Singulars', 'eff_sig_pwd_FEADM_Embed',
        # 'sig_pwd_agg_vx_ft', 'sig_pwd_rec_vx_ft'


            adj_diff_color = 'tab:blue'
            in_agg_color = 'tab:orange'
            ln_vx_color = 'tab:green'
            ens_color = 'tab:red'

            fig, axes = plt.subplots(ncols=1, nrows=len(df_analysis), figsize=(10, len(df_analysis) * 2.5))
            idx = 0
            # axes.set_title(img_prefix + 'PPMCC ARI vs SGS Amplitudes', fontsize=10, fontweight='semibold')
            for K, ana_rec in df_analysis.iterrows():
                axes[idx].grid(True)
                l_eig_vals = np.round(ana_rec['eig_vals'], decimals=3)
                adj_diff = ana_rec['sig_vs_eig_sas']
                in_agg = ana_rec['sig_pwd_agg_vx_ft']
                ln_vx = ana_rec['sig_pwd_rec_vx_ft']
                ens = np.mean([adj_diff, ln_vx], axis=0)
                axes[idx].errorbar([i for i in range(len(l_eig_vals))], adj_diff,
                              fmt='-o', c=adj_diff_color, capsize=2, capthick=1, label='ADJ-DIFF')
                axes[idx].set_xticks([i for i in range(len(l_eig_vals))])
                axes[idx].set_xticklabels(l_eig_vals)
                idx += 1
            plt.tight_layout(pad=1.0)
            plt.show()
            plt.clf()
            plt.close()
            print()