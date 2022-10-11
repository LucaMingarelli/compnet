"""  Created on 10/10/2022::
------------- algo.py -------------

**Authors**: L. Mingarelli
"""

import numpy as np, pandas as pd
from compnet.__res.sample.sample0 import sample0, sample_bilateral, sample_noncons1, sample_noncons2,sample_noncons3
from numba import njit, jit
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


# For now assuming applied on fully connected subset
def compressed_network_non_conservative(df=sample_noncons4):
    nodes_flow = _get_nodes_net_flow(df)

    nodes = np.array(nodes_flow.index)
    flows = nodes_flow.values
    idx = flows[flows != 0].argsort()[::-1]
    ordered_flows = flows[flows != 0][idx]
    nodes = nodes[flows != 0][idx]


    @jit
    def __noncons_compr(nodes, ordered_flows):
        nodes_r = nodes[::-1]

        EL, pairs = [], []
        i,j = 0,0
        while len(ordered_flows):
            v = min(ordered_flows[0], -ordered_flows[-1])
            err = ordered_flows[0] + ordered_flows[-1]
            EL.append(v)
            pairs.append((nodes_r[j], nodes[i]))
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

    EL, pairs = __noncons_compr(nodes=nodes, ordered_flows=ordered_flows)

    fx = pd.DataFrame.from_records(pairs,columns=['SOURCE', 'DESTINATION'])
    fx['AMOUNT'] = EL
    return fx


