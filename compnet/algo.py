"""  Created on 10/10/2022::
------------- algo.py -------------

**Authors**: L. Mingarelli
"""

import numpy as np, pandas as pd
import numba
from tabulate import tabulate
__SEP = '__<>__<>__'

def _get_nodes_net_flow(df):
  return pd.concat([df.groupby('SOURCE').AMOUNT.sum(),
                    df.groupby('DESTINATION').AMOUNT.sum()],
                    axis=1).fillna(0).T.diff().iloc[-1,:].sort_index()
def compressed_market_size(g):
  return _get_nodes_net_flow(g).clip(lower=0).sum()
def market_desc(df):
    GMS = df.AMOUNT.sum()
    CMS = compressed_market_size(df)
    EMS = GMS - CMS
    return {'GMS':GMS, 'CMS':CMS, 'EMS':EMS}

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

def compression_factor(df1, df2, p=2, _max_comp_p=15):
    r"""Returns compression factor of df2 with respect to df1.

    The compression factor CF for two networks with N nodes and weighted adjacency matrix C_1 and C_2 is defined as

    .. math::
        CF_p = 1 - 2 / N(N-1)    (||L(C_2, N)||_p / ||L(C_1, N)||_p)

    where

    .. math::
        ||L(C, N)||_p = (2 / N(N-1) \sum_i\sum_{j=i+1} |C_ij|^p )^{1/p}

    Notice that in the limit we have TODO: NOT TRUE! The following applies only to bilateral (maybe to conservative as well)

    .. math::
        lim_{p\rightarrow\infty} CF_p = 1 - EMS_2 / EMS_1

    with EMS the excess market size.
    The compression ratio CR is related to CF as

    .. math::
        CF = 1 - CR

    Args:
        df1 (pd.DataFrame): Edge list of original network
        df2 (pd.DataFrame): Edge list of compressed network
        p: order of the norm (default is p=2). When p>_max_comp_p the limit p=∞ is automatically returned

    Returns:
        Compression factor
    """
    src_dst = ['SOURCE', 'DESTINATION']

    if p<=_max_comp_p:
        nds1, nds2 = ({k:v for v,k in enumerate(set(df1[src_dst].values.flatten()))},
                      {k:v for v,k in enumerate(set(df2[src_dst].values.flatten()))})
        N1, N2 = len(nds1), len(nds2)

        Lp1 = (2 / (N1*(N1-1)) * df1.replace(nds1).groupby('SOURCE').apply(lambda g: (g.AMOUNT.abs()[g.DESTINATION > g.SOURCE]**p).sum()).sum()) ** (1/p)
        Lp2 = (2 / (N2*(N2-1)) * df2.replace(nds2).groupby('SOURCE').apply(lambda g: (g.AMOUNT.abs()[g.DESTINATION > g.SOURCE]**p).sum()).sum()) ** (1/p)
        CR = 2 / (N2*(N2-1)) * (Lp2 / Lp1)
    else:  # If p>_max_comp_p (p>15 by default) returns the limit p=∞
        CR = market_desc(df2)['EMS'] / market_desc(df1)['EMS']

    CF = 1 - CR
    return CF




# class self: ...
# self = self()

class CompNet:
    __SEP = '__<>__<>__'

    def __init__(self, df):
        self._original_network = df
        self.net_flow = _get_nodes_net_flow(self._original_network)
        self.describe(print_props=False, ret=False)  # Builds GMS, CMS, EMS, and properties

    def _get_compressed_market_size(self, df=None):
        df = self._original_network if df is None else df
        return compressed_market_size(df)

    def describe(self, df=None, print_props=True, ret=False):
        df = self._original_network if df is None else df
        GMS, CMS, EMS = market_desc(df).values()
        props = pd.Series({'Gross size': GMS,  # Gross Market Size
                           'Compressed size': CMS,  # Compressed Market Size
                           'Excess size': EMS   # Excess Market Size
                           })
        if df is None:
            self.GMS, self.CMS, self.EMS = GMS, CMS, EMS
            self.properties = props
            if print_props and not ret:
                print(tabulate(props.reset_index().rename(columns={'index':'',0:'AMOUNT'}),
                               headers='keys', tablefmt='simple_outline', showindex=False))
        if ret:
            return props

    def _flip_neg_amnts(self, df):
        f = df.copy(deep=True)
        f_flip = f[f.AMOUNT < 0].iloc[:, [1, 0, 2]]
        f_flip.columns = df.columns
        f_flip['AMOUNT'] *= -1
        f[f.AMOUNT < 0] = f_flip
        return f

    def __bilateral_compression(self, df):
        """
        Returns bilaterally compressed network
        Args:
            df: pandas.DataFrame containing three columns SOURCE, DESTINATION, AMOUNT

        Returns:
            pandas.DataFrame containing edge list of bilaterally compressed network
        """
        rel_lab = df.SOURCE.astype(str) + self.__SEP + df.DESTINATION.astype(str)
        bil_rel = (df.SOURCE.astype(str).apply(list) +
                   df.DESTINATION.astype(str).apply(list)
                   ).apply(sorted).apply(lambda l: self.__SEP.join(l))

        rf = df.set_index(bil_rel)
        rf['AMOUNT'] *= (1 - 2 * (rel_lab != bil_rel).astype(int)).values

        rf = rf.sort_values(by=['SOURCE', 'AMOUNT']).reset_index().groupby('index').AMOUNT.sum().reset_index()
        rf = pd.concat([pd.DataFrame.from_records(rf['index'].str.split(self.__SEP).values,
                                                  columns=['SOURCE', 'DESTINATION']),
                        rf],
                       axis=1).drop(columns='index')
        return self._flip_neg_amnts(rf)

    def __non_conservative_compression_MAX(self, df):
        """
        TODO: IN DOCS ADD https://github.com/sktime/sktime/issues/764
        Requirements of numba version and llvm
        Args:
            df:

        Returns:

        """
        nodes_flow =self.net_flow if df is None else _get_nodes_net_flow(df)

        nodes = np.array(nodes_flow.index)
        flows = nodes_flow.values

        idx = flows[flows != 0].argsort()[::-1]

        ordered_flows = flows[flows != 0][idx]
        nodes = nodes[flows != 0][idx]
        nodes_r = nodes[::-1]

        EL, pairs = _noncons_compr_max_min(ordered_flows=ordered_flows,
                                           max_links=len(nodes)
                                           # TODO - prove the following Theorem: for any compressed graph G=(N, E) one has |E|<=|N| (number of edges is at most the number of nodes)
                                           )

        fltr = EL != 0
        EL, pairs = EL[fltr], pairs[fltr, :]
        pairs = [*zip(nodes_r.reshape(1, -1)[:, pairs[:, 0]][0],
                      nodes.reshape(1, -1)[:, pairs[:, 1]][0])]

        fx = pd.DataFrame.from_records(pairs, columns=['SOURCE', 'DESTINATION'])
        fx['AMOUNT'] = EL
        return fx

    def __non_conservative_compression_ED(self, df):
        nodes_flow = self.net_flow if df is None else _get_nodes_net_flow(df)

        flows = nodes_flow.values
        nodes = np.array(nodes_flow.index)[flows != 0]

        pos_flws = flows[flows > 0]
        neg_flws = -flows[flows < 0]
        pos_nds = nodes[flows > 0]
        neg_nds = nodes[flows < 0]

        # Total positive flow
        T_flow = pos_flws.sum()

        cmprsd_flws = neg_flws.reshape(-1, 1) * pos_flws / T_flow
        cmprsd_edgs = neg_nds.reshape(-1, 1) + (self.__SEP + pos_nds)

        fx = pd.DataFrame.from_records(pd.Series(cmprsd_edgs.flatten()).str.split(self.__SEP),
                                       columns=['SOURCE', 'DESTINATION'])
        fx['AMOUNT'] = cmprsd_flws.flatten()
        return fx

    def compress(self, type='bilateral', conn_sub=False, df=None,
                 compression_p=2, verbose=True, _max_comp_p=15):
        """
        Returns compressed network.
        Args:
            type: Type of compression. Either of ('NC-ED', 'NC-MAX', 'C', 'bilateral')
            df:

        Returns:
            Edge list (pandas.DataFrame) corresponding to compressed network.

        """
        df = self._original_network if df is None else df
        if type.lower() == 'nc-ed':
            compressed = self.__non_conservative_compression_ED(df=df)
        elif type.lower() == 'nc-max':
            compressed = self.__non_conservative_compression_MAX(df=df)
        elif type.lower() == 'c':
            ...
        elif type.lower() == 'bilateral':
            compressed = self.__bilateral_compression(df=df)
        else:
            raise Exception(f'Type {type} not recognised: please input either of NC-ED, NC-MAX, C, or bilateral.')

        if verbose:
            comp_rt = compression_factor(df1=df, df2=compressed, p=compression_p, _max_comp_p=_max_comp_p)
            print(f"Compression Factor  CF(p={compression_p})={comp_rt}")
        return compressed





# Nodes net flow
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

# For now assuming applied on fully connected subset
def compressed_network_non_conservative(df):
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
def non_conservative_compression_ED(df):
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

def compressed_network_conservative(df):
    df = compressed_network_bilateral(df)
    ...



