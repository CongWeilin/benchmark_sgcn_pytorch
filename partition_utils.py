"""Collections of partitioning functions."""

import time
import metis
import scipy.sparse as sp


def partition_graph(adj, idx_nodes, num_clusters):
    """partition a graph by METIS."""

    start_time = time.time()
    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]

    neighbor_intervals = []
    neighbors = []
    edge_cnt = 0
    neighbor_intervals.append(0)
    train_adj = adj[idx_nodes, :][:, idx_nodes]
    train_adj_lil = train_adj.tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        neighbors += rows
        edge_cnt += len(rows)
        neighbor_intervals.append(edge_cnt)
        train_ord_map[idx_nodes[i]] = i

    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes

    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]
    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)
        for nb_orig_idx in adj[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            if groups[nb_idx] == gp_idx:
                part_data.append(1)
                part_row.append(nd_orig_idx)
                part_col.append(nb_orig_idx)
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

    part_size = [len(part) for part in parts]
    print('Partitioning done. %f seconds.'%(time.time() - start_time))
    print('Max part size %d, min part size %d'%(max(part_size), min(part_size)))
    print('Before partition edges %d, after partition edges %d'%(train_adj.data.shape[0] , part_adj.data.shape[0]))

    return part_adj, parts
