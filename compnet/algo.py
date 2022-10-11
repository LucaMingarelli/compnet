"""  Created on 10/10/2022::
------------- algo.py -------------

**Authors**: L. Mingarelli
"""

import numpy as np, pandas as pd
from compnet.__res.sample.sample0 import (sample0, sample_bilateral, sample_noncons1,
                                          sample_noncons2, sample_noncons3, sample_noncons4)
import numba
__SEP = '__<>__<>__'

df = sample0

class Compress:
    def __init__(self, edge_list):
        pass


# Nodes net flow
def _get_nodes_net_flow(f):
    return pd.concat([f.groupby('SOURCE').AMOUNT.sum(),
                      f.groupby('DESTINATION').AMOUNT.sum()],
                     axis=1).fillna(0).T.diff().iloc[-1,:].sort_index()

def compressed_market_size(g):
  return _get_nodes_net_flow(g).clip(lower=0).sum()



def charct_net(df):
    GMS = df.AMOUNT.sum()
    CMS = compressed_market_size(df)
    EMS = GMS - CMS
    return pd.Series({'GMS': GMS,  # Gross Market Size
                      'CMS': CMS,  # Compressed Market Size
                      'EMS': EMS   # Excess Market Size
                      })

def _flip_neg_amnts(df):
    f = df.copy(deep=True)
    f_flip = f[f.AMOUNT<0].iloc[:, [1,0,2]]
    f_flip.columns = df.columns
    f_flip['AMOUNT'] *= -1
    f[f.AMOUNT<0] = f_flip
    return f

def compressed_network_bilateral(df):
    """
    Returns bilaterally compressed network
    Args:
        df: pandas.DataFrame containing three columns SOURCE, DESTINATION, AMOUNT

    Returns:
        pandas.DataFrame containing edge list of bilaterally compressed network
    """
    rel_lab = df.SOURCE.astype(str) + __SEP + df.DESTINATION.astype(str)
    bil_rel = (df.SOURCE.astype(str).apply(list)+
              df.DESTINATION.astype(str).apply(list)
               ).apply(sorted).apply(lambda l: __SEP.join(l))

    rf = df.set_index(bil_rel)
    rf['AMOUNT'] *= (1-2*(rel_lab!=bil_rel).astype(int)).values

    rf = rf.sort_values(by=['SOURCE', 'AMOUNT']).reset_index().groupby('index').AMOUNT.sum().reset_index()
    rf = pd.concat([pd.DataFrame.from_records(rf['index'].str.split(__SEP).values,
                                              columns=['SOURCE', 'DESTINATION']),
                    rf],
                   axis=1).drop(columns='index')
    return _flip_neg_amnts(rf)

# TEST
assert (compressed_network_bilateral(df=sample_bilateral).AMOUNT == [5, 15]).all()
assert (_get_nodes_net_flow(compressed_network_bilateral(sample_bilateral)) == _get_nodes_net_flow(sample_bilateral)).all()
assert  (compressed_network_bilateral(df=sample_noncons2).AMOUNT == [10, 5, 20]).all()


def compressed_network_conservative(df):
    df = compressed_network_bilateral(df)
    ...


@numba.njit(fastmath=True)
def _noncons_compr_max_min(ordered_flows, max_links):
    EL = np.zeros(max_links)
    pairs = np.zeros((max_links, 2), dtype=np.uint32)
    i,j,n = 0,0,0
    while len(ordered_flows):
        v = min(ordered_flows[0], -ordered_flows[-1])
        err = ordered_flows[0] + ordered_flows[-1]
        EL[n] = v
        pairs[n, 0] = j
        pairs[n, 1] = i
        n += 1
        if err>0:
            ordered_flows = ordered_flows[:-1]
            ordered_flows[0] = err
            j += 1
        elif err<0:
            ordered_flows = ordered_flows[1:]
            ordered_flows[-1] = err
            i += 1
        else:
            ordered_flows = ordered_flows[1:-1]
            i += 1
            j += 1

    return EL, pairs


# For now assuming applied on fully connected subset
def compressed_network_non_conservative(df=sample_noncons4):
    """
    TODO: IN DOCS ADD https://github.com/sktime/sktime/issues/764
    Requirements of numba version and llvm
    Args:
        df:

    Returns:

    """
    nodes_flow = _get_nodes_net_flow(df)

    nodes = np.array(nodes_flow.index)
    flows = nodes_flow.values

    idx = flows[flows != 0].argsort()[::-1]

    ordered_flows = flows[flows != 0][idx]
    nodes = nodes[flows != 0][idx]
    nodes_r = nodes[::-1]

    from copy import copy

    EL, pairs = _noncons_compr_max_min(ordered_flows=copy(ordered_flows),
                                       max_links=len(nodes) # TODO - prove the following Theorem: for any compressed graph G=(N, E) one has |E|<=|N| (number of edges is at most the number of nodes)
                                       )

    fltr = EL!=0
    EL, pairs = EL[fltr], pairs[fltr, :]
    pairs = [*zip(nodes_r.reshape(1,-1)[:, pairs[:, 0]][0],
                    nodes.reshape(1,-1)[:, pairs[:, 1]][0])]

    fx = pd.DataFrame.from_records(pairs,columns=['SOURCE', 'DESTINATION'])
    fx['AMOUNT'] = EL
    return fx






# For now assuming applied on fully connected subset
def non_conservative_compression_ED(df=sample_noncons4):
    nodes_flow = _get_nodes_net_flow(df)

    flows = nodes_flow.values
    nodes = np.array(nodes_flow.index)[flows != 0]

    pos_flws = flows[flows > 0]
    neg_flws = -flows[flows < 0]
    pos_nds = nodes[flows > 0]
    neg_nds = nodes[flows < 0]

    # Total positive flow
    T_flow = pos_flws.sum()

    cmprsd_flws = neg_flws.reshape(-1,1) * pos_flws / T_flow
    cmprsd_edgs = neg_nds.reshape(-1, 1) + (__SEP + pos_nds)

    fx = pd.DataFrame.from_records(pd.Series(cmprsd_edgs.flatten()).str.split(__SEP),
                                   columns=['SOURCE', 'DESTINATION'])
    fx['AMOUNT'] = cmprsd_flws.flatten()
    return fx


charct_net(sample_noncons4)
charct_net(compressed_network_non_conservative(df=sample_noncons4))
charct_net(non_conservative_compression_ED(df=sample_noncons4))