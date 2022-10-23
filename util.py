from typing import List

import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import entropy

from model.gnn import GNN_node_Virtualnode, GNN_node
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from functools import partial

from sklearn.metrics import confusion_matrix


def accuracy_SBM(targets, pred_int):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """

    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight), pred
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(
            pred, true.float(), weight=weight[true]
        )
        return loss, torch.sigmoid(pred)


def edge_encoder_cls_zero(_):
    def zero(_):
        return 0

    return zero


def get_gnn(dim, dataset, gnn_type="gcn", virtual_node=False, depth=3, dropout=0.0):

    assert gnn_type in ["gcn", "gin"]
    if dataset in ["ogbg-molhiv", "ogbg-molpcba"]:
        if virtual_node:
            gnn = GNN_node_Virtualnode(
                depth,
                dim,
                AtomEncoder(dim),
                BondEncoder,
                gnn_type=gnn_type,
                drop_ratio=dropout,
            )
        else:
            gnn = GNN_node(
                depth,
                dim,
                AtomEncoder(dim),
                BondEncoder,
                gnn_type=gnn_type,
                drop_ratio=dropout,
            )

    elif dataset == "zinc":
        if virtual_node:
            gnn = GNN_node_Virtualnode(
                depth,
                dim,
                AtomEncoder(dim),
                partial(torch.nn.Embedding, 4),
                gnn_type=gnn_type,
                drop_ratio=dropout,
            )
        else:
            gnn = GNN_node(
                depth,
                dim,
                AtomEncoder(dim),
                partial(torch.nn.Embedding, 4),
                gnn_type=gnn_type,
                drop_ratio=dropout,
            )
    elif dataset == "pattern":
        if virtual_node:
            gnn = GNN_node_Virtualnode(
                depth,
                dim,
                torch.nn.Linear(3, dim),
                edge_encoder_cls_zero,
                gnn_type=gnn_type,
                drop_ratio=dropout,
            )
        else:
            gnn = GNN_node(
                depth,
                dim,
                torch.nn.Linear(3, dim),
                edge_encoder_cls_zero,
                gnn_type=gnn_type,
                drop_ratio=dropout,
            )
    else:
        raise NotImplementedError()
    return gnn


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1,
):
    """
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def attention_entropy(attention):

    attention = attention.cpu().numpy()
    return entropy(attention, axis=-1).squeeze(0)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["run"] = config["run"]
    hparams["train"] = config["train"]
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


class ASTNodeEncoder(torch.nn.Module):
    """
    Input:
        x: default node feature. the first and second column represents node type and node attributes.
        depth: The depth of the node in the AST.

    Output:
        emb_dim-dimensional vector

    """

    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return (
            self.type_encoder(x[:, 0])
            + self.attribute_encoder(x[:, 1])
            + self.depth_encoder(depth)
        )


def get_vocab_mapping(seq_list, num_vocab):
    """
    Input:
        seq_list: a list of sequences
        num_vocab: vocabulary size
    Output:
        vocab2idx:
            A dictionary that maps vocabulary into integer index.
            Additioanlly, we also index '__UNK__' and '__EOS__'
            '__UNK__' : out-of-vocabulary term
            '__EOS__' : end-of-sentence

        idx2vocab:
            A list that maps idx to actual vocabulary.

    """

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind="stable")[:num_vocab]

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # logger.info(topvocab)
    # logger.info([vocab_list[v] for v in topvocab[:10]])
    # logger.info([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx["__UNK__"] = num_vocab
    idx2vocab.append("__UNK__")

    vocab2idx["__EOS__"] = num_vocab + 1
    idx2vocab.append("__EOS__")

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert idx == vocab2idx[vocab]

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert vocab2idx["__EOS__"] == len(idx2vocab) - 1

    return vocab2idx, idx2vocab


def augment_edge(data):
    """
    Input:
        data: PyG data object
    Output:
        data (edges are augmented in the following ways):
            data.edge_index: Added next-token edge. The inverse edges were also added.
            data.edge_attr (torch.Long):
                data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    """

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = torch.cat(
        [
            torch.zeros(edge_index_ast_inverse.size(1), 1),
            torch.ones(edge_index_ast_inverse.size(1), 1),
        ],
        dim=1,
    )

    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(
        data.node_is_attributed.view(-1,) == 1
    )[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack(
        [attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]],
        dim=0,
    )
    edge_attr_nextoken = torch.cat(
        [
            torch.ones(edge_index_nextoken.size(1), 1),
            torch.zeros(edge_index_nextoken.size(1), 1),
        ],
        dim=1,
    )

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack(
        [edge_index_nextoken[1], edge_index_nextoken[0]], dim=0
    )
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat(
        [
            edge_index_ast,
            edge_index_ast_inverse,
            edge_index_nextoken,
            edge_index_nextoken_inverse,
        ],
        dim=1,
    )
    data.edge_attr = torch.cat(
        [
            edge_attr_ast,
            edge_attr_ast_inverse,
            edge_attr_nextoken,
            edge_attr_nextoken_inverse,
        ],
        dim=0,
    )

    return data


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    """
    Input:
        data: PyG graph object
        output: add y_arr to data
    """

    # PyG >= 1.5.0
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    """
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    """

    augmented_seq = seq[:max_seq_len] + ["__EOS__"] * max(0, max_seq_len - len(seq))
    return torch.tensor(
        [
            [
                vocab2idx[w] if w in vocab2idx else vocab2idx["__UNK__"]
                for w in augmented_seq
            ]
        ],
        dtype=torch.long,
    )


def decode_arr_to_seq(arr, idx2vocab):
    """
    Input: torch 1d array: y_arr
    Output: a sequence of words.
    """

    eos_idx_list = (
        arr == len(idx2vocab) - 1
    ).nonzero()  # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)]  # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))

