from utils import *
import metis
from partition_utils import partition_graph


def CalculateThreshold(candidatesArray, sampleSize, sumSmall=0, nLarge=0):
    candidate = candidatesArray[candidatesArray > 0][0]
    smallArray = candidatesArray[candidatesArray < candidate]
    largeArray = candidatesArray[candidatesArray > candidate]
    equalArray = candidatesArray[candidatesArray == candidate]
    curSampleSize = (sum(smallArray) + sumSmall) / candidate + \
        len(largeArray) + nLarge + len(equalArray)
    if curSampleSize < sampleSize:
        if len(smallArray) == 0:
            return sumSmall/(sampleSize-nLarge-len(largeArray)-1)
        else:
            nLarge = nLarge + len(largeArray)+len(equalArray)
            return CalculateThreshold(smallArray, sampleSize, sumSmall, nLarge)
    else:
        if len(largeArray) == 0:
            return (sumSmall + sum(smallArray) + sum(equalArray))/(sampleSize-nLarge)
        else:
            sumSmall = sumSmall + sum(smallArray) + sum(equalArray)
            return CalculateThreshold(largeArray, sampleSize, sumSmall, nLarge)

class fastgcn_sampler:
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.lap_matrix_sq = self.lap_matrix.multiply(self.lap_matrix)

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = []
        pi = np.array(np.sum(self.lap_matrix_sq, axis=0))[0]
        p = pi / np.sum(pi)
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            s_num = np.min([np.sum(p > 0), samp_num_list[d]])
            after_nodes = np.random.choice(num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
            adj = U[:, after_nodes].multiply(1/p[after_nodes])
            adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
            sampled_nodes += [previous_nodes]
            previous_nodes = after_nodes
        sampled_nodes.reverse()
        adjs.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes

    def full_batch(self, batch_nodes, num_nodes, depth):
        adjs = [sparse_mx_to_torch_sparse_tensor(
            self.lap_matrix) for _ in range(depth)]
        input_nodes = np.arange(num_nodes)
        sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
        return adjs, input_nodes, sampled_nodes

class ladies_sampler:
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.lap_matrix_sq = self.lap_matrix.multiply(self.lap_matrix)

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            pi = np.array(np.sum(self.lap_matrix_sq[previous_nodes, :], axis=0))[0]
            p = pi / np.sum(pi)
            s_num = np.min([np.sum(p > 0), samp_num_list[d]])
            after_nodes = np.random.choice(num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
            adj = U[:, after_nodes].multiply(1/p[after_nodes])
            adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
            sampled_nodes += [previous_nodes]
            previous_nodes = after_nodes
        sampled_nodes.reverse()
        adjs.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes

    def full_batch(self, batch_nodes, num_nodes, depth):
        adjs = [sparse_mx_to_torch_sparse_tensor(
            self.lap_matrix) for _ in range(depth)]
        input_nodes = np.arange(num_nodes)
        sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
        return adjs, input_nodes, sampled_nodes


class cluster_sampler:
    def __init__(self, adj_matrix, train_nodes, num_clusters):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.lap_matrix = normalize_with_diag_enhance(
            adj_matrix, diag_lambda=1)
        self.train_nodes = train_nodes
        self.num_clusters = num_clusters
        part_adjs, self.parts = partition_graph(
            adj_matrix, train_nodes, num_clusters)
        self.part_adjs = normalize_with_diag_enhance(part_adjs, diag_lambda=1)

    def sample_subgraph(self, seed, size=1):
        np.random.seed(seed)
        select = np.random.choice(self.num_clusters, size, replace=False)
        select = [self.parts[i] for i in select]
        batch_nodes = np.concatenate(select)
        return batch_nodes

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        bsize = samp_num_list[0]
        batch_nodes = self.sample_subgraph(seed, bsize)
        if bsize == 1:
            sampled_nodes = []
            adj = self.part_adjs[batch_nodes, :][:, batch_nodes]
            adjs = []
            for d in range(depth):
                adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
                sampled_nodes.append(batch_nodes)
            adjs.reverse()
            sampled_nodes.reverse()
            return adjs, batch_nodes, batch_nodes, probs_nodes, sampled_nodes
        else:
            pass

    def full_batch(self, batch_nodes, num_nodes, depth):
        adjs = [sparse_mx_to_torch_sparse_tensor(
            self.lap_matrix) for _ in range(depth)]
        input_nodes = np.arange(num_nodes)
        sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
        return adjs, input_nodes, sampled_nodes


class graphsage_sampler:
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        sampled_nodes = []
        previous_nodes = batch_nodes
        adjs = []
        for d in range(depth):
            U = self.adj_matrix[previous_nodes, :]
            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                s_num = min(len(indices), samp_num_list[d])
                sampled_indices = np.random.choice(
                    indices, s_num, replace=False)
                after_nodes.append(sampled_indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(previous_nodes)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes

    def full_batch(self, batch_nodes, num_nodes, depth):
        adjs = [sparse_mx_to_torch_sparse_tensor(
            self.lap_matrix) for _ in range(depth)]
        input_nodes = np.arange(num_nodes)
        sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
        return adjs, input_nodes, sampled_nodes


class vrgcn_sampler:
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.train_nodes = train_nodes

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        sampled_nodes = []
        exact_input_nodes = []
        previous_nodes = batch_nodes
        adjs = []
        adjs_exact = []

        for d in range(depth):
            U = self.adj_matrix[previous_nodes, :]
            after_nodes = []
            after_nodes_exact = []
            for U_row in U:
                indices = U_row.indices
                s_num = min(len(indices), samp_num_list[d])
                sampled_indices = np.random.choice(
                    indices, s_num, replace=False)
                after_nodes.append(sampled_indices)
                after_nodes_exact.append(indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            after_nodes_exact = np.unique(np.concatenate(after_nodes_exact))
            after_nodes_exact = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, after_nodes_exact)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            adj_exact = U[:, after_nodes_exact]
            adj_exact = row_normalize(adj_exact)
            adjs_exact += [sparse_mx_to_torch_sparse_tensor(adj_exact)]
            sampled_nodes.append(previous_nodes)
            exact_input_nodes.append(after_nodes_exact)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        adjs_exact.reverse()
        exact_input_nodes.reverse()
        return adjs, adjs_exact, previous_nodes, batch_nodes, probs_nodes, sampled_nodes, exact_input_nodes

    def full_batch(self, batch_nodes, num_nodes, depth):
        adjs = [sparse_mx_to_torch_sparse_tensor(
            self.lap_matrix) for _ in range(depth)]
        input_nodes = np.arange(num_nodes)
        sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
        return adjs, input_nodes, sampled_nodes

# deprecated
def graphsaint_sampler(seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
    lap_matrix_coo = lap_matrix.tocoo()
    row, col = lap_matrix_coo.row, lap_matrix_coo.col

    A = sp.csr_matrix((np.ones_like(lap_matrix.data),
                       lap_matrix.indices, lap_matrix.indptr), shape=lap_matrix.shape)
    D_inv = 1.0/A.sum(axis=1)
    sample_prob = D_inv[row] + D_inv[col]
    sample_prob = len(batch_nodes) * sample_prob / sample_prob.sum()

    sampled, cnt = [], 0

    while cnt < len(batch_nodes):
        for e in range(len(sample_prob)):
            if np.random.rand() < sample_prob[e]:
                sampled.append(row[e])
                sampled.append(col[e])
                cnt += 1

    sampled = np.unique(np.array(sampled))

    adj = lap_matrix[sampled, :][:, sampled]

    adjs = [sparse_mx_to_torch_sparse_tensor(
        row_normalize(adj)) for d in range(depth)]
    sampled_nodes = [sampled for d in range(depth)]
    return adjs, sampled, sampled, probs_nodes, sampled_nodes

class subgraph_sampler:
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.train_nodes = train_nodes

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        adj = self.lap_matrix[batch_nodes, :][:, batch_nodes]
        adj = adj.multiply(1/probs_nodes)
        
        U = self.adj_matrix[batch_nodes, :]
        after_nodes_exact = []
        for U_row in U: 
            indices = U_row.indices
            after_nodes_exact.append(indices)
        after_nodes_exact = np.unique(np.concatenate(after_nodes_exact))
        after_nodes = np.concatenate(
                [batch_nodes, np.setdiff1d(after_nodes_exact, batch_nodes)])
        adj_exact = self.lap_matrix[batch_nodes, :][:, after_nodes]

        adjs = []
        adjs_exact = []
        sampled_nodes = []
        input_nodes_exact = []
        for d in range(depth):
            adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
            adjs_exact += [sparse_mx_to_torch_sparse_tensor(adj_exact)]
            sampled_nodes.append(batch_nodes)
            input_nodes_exact.append(after_nodes)
        adjs.reverse()
        adjs_exact.reverse()
        sampled_nodes.reverse()
        input_nodes_exact.reverse()

        return adjs, adjs_exact, batch_nodes, batch_nodes, probs_nodes, sampled_nodes, input_nodes_exact

    def full_batch(self, batch_nodes, num_nodes, depth):
        adjs = [sparse_mx_to_torch_sparse_tensor(
            self.lap_matrix) for _ in range(depth)]
        input_nodes = np.arange(num_nodes)
        sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
        return adjs, input_nodes, sampled_nodes
    
    def large_batch(self, batch_nodes, num_nodes, depth):
        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                after_nodes.append(indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(previous_nodes)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        return adjs, previous_nodes, sampled_nodes

class exact_sampler:
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        sampled_nodes = []
        previous_nodes = batch_nodes
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                after_nodes.append(indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(previous_nodes)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes

    def full_batch(self, batch_nodes, num_nodes, depth):
        adjs = [sparse_mx_to_torch_sparse_tensor(
            self.lap_matrix) for _ in range(depth)]
        input_nodes = np.arange(num_nodes)
        sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
        return adjs, input_nodes, sampled_nodes

    def large_batch(self, batch_nodes, num_nodes, depth):
        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                after_nodes.append(indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(previous_nodes)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        return adjs, previous_nodes, sampled_nodes

# def exact_sampler(seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
#     previous_nodes = batch_nodes
#     sampled_nodes = []
#     adjs = []
#     for d in range(depth):
#         U = lap_matrix[previous_nodes, :]
#         after_nodes = [previous_nodes]
#         for U_row in U:
#             indices = U_row.indices
#             after_nodes.append(indices)
#         after_nodes = np.unique(np.concatenate(after_nodes))
#         adj = U[:, after_nodes]
#         adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
#         sampled_nodes.append(previous_nodes)
#         previous_nodes = after_nodes
#     adjs.reverse()
#     sampled_nodes.reverse()
#     return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes


def full_batch_sampler(batch_nodes, num_nodes, lap_matrix, depth):
    adjs = [sparse_mx_to_torch_sparse_tensor(lap_matrix) for _ in range(depth)]
    input_nodes = np.arange(num_nodes)
    sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
    return adjs, input_nodes, sampled_nodes


def mini_batch_sampler(batch_nodes, num_nodes, lap_matrix, depth):
    previous_nodes = batch_nodes
    sampled_nodes = []
    adjs = []
    for d in range(depth):
        U = lap_matrix[previous_nodes, :]
        after_nodes = [previous_nodes]
        for U_row in U:
            indices = U_row.indices
            after_nodes.append(indices)
        after_nodes = np.unique(np.concatenate(after_nodes))
        adj = U[:, after_nodes]
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        sampled_nodes.append(previous_nodes)
        previous_nodes = after_nodes
    adjs.reverse()
    sampled_nodes.reverse()
    return adjs, previous_nodes, sampled_nodes
