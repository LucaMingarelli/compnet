"""  Created on 23/10/2022::
------------- algo.py -------------

**Authors**: L. Mingarelli
"""

import numpy as np, pandas as pd
import numba, networkx as nx, warnings
from tabulate import tabulate
from tqdm import tqdm
from functools import lru_cache

__SEP = '__<>__<>__'

def _flip_neg_amnts(df):
    f = df.copy(deep=True)
    f_flip = f[f.AMOUNT<0].iloc[:, [1,0,2]]
    f_flip.columns = df.columns
    f_flip['AMOUNT'] *= -1
    f[f.AMOUNT<0] = f_flip
    return f

def _get_nodes_net_flow(df, grouper=None):
    def _get_group_nodes_net_flow(f):
        return pd.concat([f.groupby('SOURCE').AMOUNT.sum(),
                          f.groupby('TARGET').AMOUNT.sum()],
                          axis=1).fillna(0).T.diff().iloc[-1, :].sort_index()

    return df.groupby(grouper).apply(_get_group_nodes_net_flow) if grouper else _get_group_nodes_net_flow(df)

def _compressed_market_size(f, grouper=None):
  return _get_nodes_net_flow(f, grouper).clip(lower=0).sum()
def _market_desc(df, grouper=None):
    GMS = df.AMOUNT.abs().sum()
    CMS = _compressed_market_size(df, grouper)
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

def compression_efficiency(df, df_compressed):
    CE = 1 - _market_desc(df_compressed)['EMS'] / _market_desc(df)['EMS']
    return CE

def compression_factor(df1, df2, p=2):
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
        p: order of the norm (default is p=2). If p='ems_ratio' the ratio of EMS is provided. This corresponds in some cases to the limit p=∞.

    Returns:
        Compression factor
    """

    if str(p).lower()=='ems_ratio':  # In the bilateral compression case this corresponds to the limit p=∞
        CR = 1- compression_efficiency(df=df1, df_compressed=df2)
    else:
        N = len(set(df1[['SOURCE', 'TARGET']].values.flatten()))
        Lp1 = (df1.AMOUNT.abs()**p).sum() ** (1/p) # * (2 / (N*(N-1)))**(1/p)
        Lp2 = (df2.AMOUNT.abs()**p).sum() ** (1/p) # * (2 / (N*(N-1)))**(1/p)
        CR = 2 / (N*(N-1)) * (Lp2 / Lp1)
        # CR = Lp2 / Lp1

    CF = 1 - CR
    return CF



# class self: ...
# self = self()
# self.__SEP = '__<>__<>__'
# self.__GROUPER = 'GROUPER'

class Graph:
    __SEP = '__<>__<>__'

    def __init__(self, df, source='SOURCE', target='TARGET', amount='AMOUNT', grouper=None):

        self._labels = [source, target, amount]+([grouper] if grouper else [])
        self.__GROUPER = 'GROUPER' if grouper else None
        self._labels_map = {source: 'SOURCE', target: 'TARGET', amount: 'AMOUNT', grouper:self.__GROUPER}
        self._original_network = df[self._labels].rename(columns=self._labels_map)
        self.net_flow = _get_nodes_net_flow(self._original_network, grouper=self.__GROUPER)
        self.describe(print_props=False, ret=False)  # Builds GMS, CMS, EMS, and properties

    def describe(self, print_props=True, ret=False):
        df = self._original_network
        GMS, CMS, EMS = _market_desc(df, grouper=self.__GROUPER).values()
        props = pd.Series({'Gross size': GMS,       # Gross Market Size
                           'Compressed size': CMS,  # Compressed Market Size
                           'Excess size': EMS       # Excess Market Size
                           })
        self.GMS, self.CMS, self.EMS = GMS, CMS, EMS
        self.properties = props
        if print_props and not ret:
            print(tabulate(props.reset_index().rename(columns={'index':'',0:'AMOUNT'}),
                           headers='keys', tablefmt='simple_outline', showindex=False))
        if ret:
            return props

    def __bilateral_compression(self, df):
        """
        Returns bilaterally compressed network
        Args:
            df: pandas.DataFrame containing three columns SOURCE, TARGET, AMOUNT

        Returns:
            pandas.DataFrame containing edge list of bilaterally compressed network
        """
        rel_lab = df.SOURCE.astype(str) + self.__SEP + df.TARGET.astype(str)
        bil_rel = (df.SOURCE.astype(str).apply(lambda x: [x]) +
                   df.TARGET.astype(str).apply(lambda x: [x])
                   ).apply(sorted).apply(lambda l: self.__SEP.join(l))

        rf = df.set_index(bil_rel)
        rf['AMOUNT'] *= (1 - 2 * (rel_lab != bil_rel).astype(int)).values

        rf = rf.sort_values(by=['SOURCE', 'AMOUNT']).reset_index().groupby('index').AMOUNT.sum().reset_index()
        rf = pd.concat([pd.DataFrame.from_records(rf['index'].str.split(self.__SEP).values,
                                                  columns=['SOURCE', 'TARGET']),
                        rf],
                       axis=1).drop(columns='index')
        return _flip_neg_amnts(rf)

    def __conservative_compression(self, df):
        f = self.__bilateral_compression(_flip_neg_amnts(df))
        edgs = f.set_index(f.SOURCE + self.__SEP + f.TARGET)[['AMOUNT']].T
        @lru_cache()
        def loop2edg(tpl):
            return list(f'{x}{self.__SEP}{y}' for x, y in zip((tpl[-1],) + tpl[:-1], tpl))
        @lru_cache()
        def get_minedg(cycle):
            return edgs[loop2edg(cycle)].T.min().AMOUNT

        G = nx.DiGraph(list(f.iloc[:, :2].values))
        cycles_len_minedg = [(tuple(c), len(c) * get_minedg(tuple(c)))
                             for c in nx.simple_cycles(G)]
        while cycles_len_minedg:
            idx = np.argmax((c[1] for c in cycles_len_minedg))
            cycle = cycles_len_minedg[idx][0]
            cls = loop2edg(cycle)
            if pd.Series(cls).isin(edgs.columns).all():
                min_edg = edgs[cls].min(1).AMOUNT
                drop_col = edgs[cls].columns[(edgs[cls]==min_edg).values[0]][0]
                edgs[cls] -= min_edg
                edgs.drop(columns=[drop_col], inplace=True)
            cycles_len_minedg.pop(idx)
        edgs = edgs.T.reset_index()
        amnt = edgs.AMOUNT
        edgs = pd.DataFrame(edgs['index'].str.split(self.__SEP).to_list(),
                            columns=['SOURCE', 'TARGET'])
        edgs['AMOUNT'] = amnt
        return edgs

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

        fx = pd.DataFrame.from_records(pairs, columns=['SOURCE', 'TARGET'])
        fx['AMOUNT'] = EL
        return fx

    def _non_conservative_compression_ED(self, df):
        nodes_flow = self.net_flow if df is None else _get_nodes_net_flow(df)

        flows = nodes_flow.values
        nodes = np.array(nodes_flow.index)[flows != 0]
        flows = flows[flows != 0]

        pos_flws = flows[flows > 0]
        neg_flws = -flows[flows < 0]
        pos_nds = nodes[flows > 0]
        neg_nds = nodes[flows < 0]

        # Total positive flow
        T_flow = pos_flws.sum()

        cmprsd_flws = neg_flws.reshape(-1, 1) * pos_flws / T_flow
        cmprsd_edgs = neg_nds.reshape(-1, 1) + (self.__SEP + pos_nds)

        fx = pd.DataFrame.from_records(pd.Series(cmprsd_edgs.flatten()).str.split(self.__SEP),
                                       columns=['SOURCE', 'TARGET'])
        fx['AMOUNT'] = cmprsd_flws.flatten()
        return fx

    def _check_compression(self, df, df_compressed):
        GMS, CMS, EMS = _market_desc(df).values()
        GMS_comp, CMS_comp, EMS_comp = _market_desc(df_compressed).values()
        flows = _get_nodes_net_flow(df).sort_index()
        flows_comp = _get_nodes_net_flow(df_compressed).sort_index()
        assert EMS>EMS_comp or np.isclose(abs(EMS-EMS_comp), 0.0, atol=1e-6), f"Compression check failed on EMS. \n\n   Original EMS = {EMS} \n Compressed EMS = {EMS_comp}"
        assert np.isclose(pd.concat([flows, flows_comp], axis=1).fillna(0).diff(0).abs().max().max(), 0.0, atol=1e-6), f"Compression check failed on FLOWS. \n\n  Original flows = {flows.to_dict()} \nCompressed flows = {flows_comp.to_dict()}"
        assert np.isclose(CMS, CMS_comp, atol=1e-6), f"Compression check failed on CMS. \n\n   Original CMS = {CMS} \n Compressed CMS = {CMS_comp}"

    def compress(self, type='bilateral',
                 compression_p=2, verbose=True, _check_compr=True, progress=True):
        """
        Returns compressed network.
        Args:
            type: Type of compression. Either of ('NC-ED', 'NC-MAX', 'C', 'bilateral')
            df:

        Returns:
            Edge list (pandas.DataFrame) corresponding to compressed network.

        """
        df = self._original_network
        if type.lower() == 'nc-ed':
            compressor = self._non_conservative_compression_ED
        elif type.lower() == 'nc-max':
            compressor = self.__non_conservative_compression_MAX
        elif type.lower() == 'c':
            compressor = self.__conservative_compression
        elif type.lower() == 'bilateral':
            compressor = self.__bilateral_compression
        else:
            raise Exception(f'Type {type} not recognised: please input either of NC-ED, NC-MAX, C, or bilateral.')

        if self.__GROUPER:
            grpd_df = df.groupby(self.__GROUPER)
            if progress:
                tqdm.pandas()
                df_compressed = grpd_df.progress_apply(compressor).reset_index(drop=True)
            else:
                df_compressed = grpd_df.apply(compressor).reset_index(drop=True)
        else:
            df_compressed = compressor(df)

        if _check_compr:
            self._check_compression(df=df, df_compressed=df_compressed)
        if verbose:
            comp_rt = compression_factor(df1=df, df2=df_compressed, p=compression_p)
            comp_eff = compression_efficiency(df=df, df_compressed=df_compressed)
            print(f"Compression Efficiency CE = {comp_eff}")
            print(f"Compression Factor CF(p={compression_p}) = {comp_rt}")
        return df_compressed.rename(columns={v:k for k,v in self._labels_map.items()})


# Nodes net flow
def compressed_network_bilateral(df):
    """
    Returns bilaterally compressed network
    Args:
        df: pandas.DataFrame containing three columns SOURCE, TARGET, AMOUNT

    Returns:
        pandas.DataFrame containing edge list of bilaterally compressed network
    """
    rel_lab = df.SOURCE.astype(str) + __SEP + df.TARGET.astype(str)
    bil_rel = (df.SOURCE.astype(str).apply(list)+
              df.TARGET.astype(str).apply(list)
               ).apply(sorted).apply(lambda l: __SEP.join(l))

    rf = df.set_index(bil_rel)
    rf['AMOUNT'] *= (1-2*(rel_lab!=bil_rel).astype(int)).values

    rf = rf.sort_values(by=['SOURCE', 'AMOUNT']).reset_index().groupby('index').AMOUNT.sum().reset_index()
    rf = pd.concat([pd.DataFrame.from_records(rf['index'].str.split(__SEP).values,
                                              columns=['SOURCE', 'TARGET']),
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

    fx = pd.DataFrame.from_records(pairs,columns=['SOURCE', 'TARGET'])
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
                                   columns=['SOURCE', 'TARGET'])
    fx['AMOUNT'] = cmprsd_flws.flatten()
    return fx

def compressed_network_conservative(df):
    df = compressed_network_bilateral(df)
    ...



