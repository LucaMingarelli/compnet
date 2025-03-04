"""  Created on 23/10/2022::
------------- algo.py -------------

**Authors**: L. Mingarelli
"""

import numpy as np, pandas as pd
import numba, networkx as nx, warnings
from tabulate import tabulate
from tqdm import tqdm
from functools import lru_cache
from typing import Union
from collections.abc import Sequence

__SEP = '__<>__<>__'

def _flip_neg_amnts(df):
    f = df.copy(deep=True)
    f_flip = f[f.AMOUNT<0].iloc[:, [1,0,2]]
    f_flip.columns = df.columns
    f_flip['AMOUNT'] *= -1
    f[f.AMOUNT<0] = f_flip
    return f

def _get_all_nodes(df):
    return sorted(set(df['SOURCE']).union(df['TARGET']))

def _get_nodes_net_flow(df, grouper=None, adjust_labels=None):
    all_df_nodes = _get_all_nodes(df)
    def _get_group_nodes_net_flow(f):
        group_nodes_net_flow = pd.concat([f.groupby('SOURCE').AMOUNT.sum(),
                          f.groupby('TARGET').AMOUNT.sum()],
                          axis=1).fillna(0).T.diff().iloc[-1, :]
        if set(_get_all_nodes(f))!=set(all_df_nodes):
            return group_nodes_net_flow.reindex(all_df_nodes, fill_value=0).sort_index()
        else:
            return group_nodes_net_flow.sort_index()

    nodes_net_flow = df.groupby(grouper).apply(_get_group_nodes_net_flow) if grouper else _get_group_nodes_net_flow(df)
    _WARNING_MISSING_NODES = True # Re-enable warnings (this prevents printing warnings for each group)

    if grouper and adjust_labels:  # Adjust net_flow names
        original_grouper = [v for k,v in adjust_labels.items() if k.startswith('GROUPER')]
        nodes_net_flow = nodes_net_flow.reset_index().rename(columns=adjust_labels).set_index(original_grouper)
        nodes_net_flow.columns.name = adjust_labels['AMOUNT']

    return nodes_net_flow


def _get_nodes_gross_flow(df, grouper=None, adjust_labels=None):
    all_df_nodes = _get_all_nodes(df)
    def _get_group_nodes_gross_flow(f):
        group_nodes_gross_flow = pd.concat([f.groupby('SOURCE').AMOUNT.sum(),
                                           f.groupby('TARGET').AMOUNT.sum()],
                                           axis=1).fillna(0)
        group_nodes_gross_flow.index.name = 'ENTITY'
        group_nodes_gross_flow.columns = 'OUT', 'IN'
        group_nodes_gross_flow['GROSS_TOTAL'] = group_nodes_gross_flow.sum(1)
        if set(_get_all_nodes(f))!=set(all_df_nodes):
            return group_nodes_gross_flow.reindex(all_df_nodes, fill_value=0).sort_index()
        else:
            return group_nodes_gross_flow.sort_index()

    nodes_gross_flow = df.groupby(grouper).apply(_get_group_nodes_gross_flow) if grouper else _get_group_nodes_gross_flow(df)
    _WARNING_MISSING_NODES = True # Re-enable warnings (this prevents printing warnings for each group)

    if grouper and adjust_labels:  # Adjust net_flow names
        original_grouper = [v for k,v in adjust_labels.items() if k.startswith('GROUPER')]
        nodes_gross_flow = nodes_gross_flow.reset_index().rename(columns=adjust_labels).set_index(original_grouper)
        nodes_gross_flow.columns.name = adjust_labels['AMOUNT']
        nodes_gross_flow = {'IN': nodes_gross_flow.set_index('ENTITY', append=True)['IN'].unstack('ENTITY'),
                            'OUT': nodes_gross_flow.set_index('ENTITY', append=True)['OUT'].unstack('ENTITY'),
                            'GROSS_TOTAL': nodes_gross_flow.set_index('ENTITY', append=True)['GROSS_TOTAL'].unstack('ENTITY')}

    return nodes_gross_flow


def _compressed_market_size(f, grouper=None):
  return _get_nodes_net_flow(f, grouper).clip(lower=0).sum(1 if grouper else 0)

def _market_desc(df, grouper=None, grouper_rename=None):
    GMS = (df.groupby(grouper).apply(lambda g: g.AMOUNT.abs().sum())
           if grouper
           else df.AMOUNT.abs().sum())
    CMS = _compressed_market_size(df, grouper)
    EMS = GMS - CMS
    if isinstance(grouper, Sequence) and not isinstance(grouper, str):
        GMS.index.names = grouper_rename
        CMS.index.names = grouper_rename
        EMS.index.names = grouper_rename
    elif grouper:
        GMS.index.name = CMS.index.name = EMS.index.name = grouper_rename
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
        ||L(C, N)||_p = (1 / N(N-1) \sum_{i≠j} |C_ij|^p )^{1/p}

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
        CR = Lp2 / Lp1

    CF = 1 - CR
    return CF



# class self: ...
# self = self()
# self.__SEP = '__<>__<>__'
# self.__GROUPER = 'GROUPER'

class Graph:
    __SEP = '__<>__<>__'
    _MAX_DISPLAY_LENGTH = 20

    def __init__(self, df: pd.DataFrame,
                 source: str='SOURCE', target: str='TARGET', amount: str='AMOUNT',
                 grouper: Union[str, list]=None):
        """
        Initialises compnet.Group object.

        Args:
            df: An edge list containing at least a source, target, and amount columns.
            source: Name of the column corresponding to source nodes. Default is 'SOURCE'.
            target: Name of the column corresponding to target nodes. Default is 'TARGET'.
            amount: Name of the column corresponding to weights / amounts of corresponding source-target edge. Default is 'AMOUNT'.
            grouper: If an additional dimension exists (e.g. a date dimension), passing the corresponding column name will result in the creation of a graph for each category in the grouper column.
        """
        if isinstance(grouper, Sequence) and not isinstance(grouper, str):
            grouper = tuple(grouper)
            self._multi_grouper = True
        else:
            self._multi_grouper = False
        self.GMS = self.CMS = self.EMS = self.properties = None
        self._labels = [source, target, amount]+((list(grouper) if self._multi_grouper else [grouper]) if grouper else [])
        self.__GROUPER = ([f'GROUPER{n+1}' for n, grpr in enumerate(grouper)] if self._multi_grouper else 'GROUPER') if grouper else None
        self._labels_map = {**{source: 'SOURCE', target: 'TARGET', amount: 'AMOUNT'},
                            **({grpr: f'GROUPER{n+1}' for n, grpr in enumerate(grouper)}
                               if self._multi_grouper else
                               {grouper: self.__GROUPER or 'GROUPER'} if grouper else {})}
        self._labels_imap = {v:k for k,v in self._labels_map.items()}
        self.edge_list = df[self._labels].rename(columns=self._labels_map)

        if self.__GROUPER and any(set(_get_all_nodes(self.edge_list)) - set(_get_all_nodes(g)) for _, g in self.edge_list.groupby(self.__GROUPER)):
            warnings.warn(f"\n\nSome nodes (SOURCE `{source}` or TARGET `{target}`) are missing from some groups (GROUPER `{grouper}`).\n"
                          "These will be filled with zeros.\n")

        self.net_flow = _get_nodes_net_flow(self.edge_list, grouper=self.__GROUPER, adjust_labels=self._labels_imap)
        self.gross_flow = _get_nodes_gross_flow(df=self.edge_list, grouper=self.__GROUPER, adjust_labels=self._labels_imap)

        self.describe(print_props=False, ret=False)  # Builds GMS, CMS, EMS, and properties

    @property
    def SOURCE(self):
        return self.edge_list['SOURCE']

    @property
    def TARGET(self):
        return self.edge_list['TARGET']

    @property
    def AMOUNT(self):
        return self.edge_list['AMOUNT']

    @property
    def ENTITIES(self):
        if self.__GROUPER is None:
            entities = pd.DataFrame({'dealer_ratio': 1 - self.net_flow.abs() / self.gross_flow['GROSS_TOTAL']})
        else:
            entities = {'dealer_ratio': 1 - self.net_flow.abs() / self.gross_flow['GROSS_TOTAL']}

        entities['is_dealer'] = entities['dealer_ratio'] > 0
        entities['inflow'] = self.gross_flow['IN']
        entities['outflow'] = self.gross_flow['OUT']
        entities['gross_flow'] = self.gross_flow['GROSS_TOTAL']
        entities['net_flow'] = self.net_flow
        return entities

    def _grouper_rename(self):
        if self._multi_grouper:
            grouper_rename = [v for k,v in self._labels_imap.items() if k in self.__GROUPER]
        else:
            grouper_rename = self._labels_imap['GROUPER'] if self.__GROUPER else None
        return grouper_rename

    def describe(self, print_props: bool=True, ret: bool=False, recompute: bool=False):
        """
        Computes and prints / returns the graph's Gross, Compressed, and Excess market sizes.
        Args:
            print_props: If `True` (default) prints
            ret: If `True` returns
            recompute: If `True` forces re-computation. Otherwise, computes only at Graph's initialisation.

        Returns:
            If `ret==True`, pandas.Series if grouper is None, else pandas.DataFrame.
        """
        df = self.edge_list
        if (self.GMS is None
                or self.CMS is None
                or self.EMS is None
                or self.properties is None
                or recompute):
            GMS, CMS, EMS = _market_desc(df, grouper=self.__GROUPER, grouper_rename=self._grouper_rename()).values()

            props = (pd.DataFrame if self.__GROUPER else pd.Series)({
                'Gross size': GMS,  # Gross Market Size
                'Compressed size': CMS,  # Compressed Market Size
                'Excess size': EMS  # Excess Market Size
            })
            self.GMS, self.CMS, self.EMS = GMS, CMS, EMS
            self.properties = props
        if print_props and not ret:
            print(tabulate(self.properties.reset_index().rename(columns={'index':'',0:'AMOUNT'}),
                           headers='keys', tablefmt='simple_outline', showindex=False))
        if ret:
            return self.properties

    def _bilateral_compression(self, df: pd.DataFrame):
        """
        Returns bilaterally compressed network.
        Bilateral compression compresses exclusively multiple trades existing between two nodes.
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

    def _conservative_compression(self, df: pd.DataFrame):
        """
        Returns conservatively compressed network.
        Conservative compression only reduces or removes existing edges (trades)
        without however adding new ones.
        The resulting conservatively compressed graph is a sub-graph of the original graph.
        Moreover, the resulting conservatively compressed graph is always a directed
        acyclic graph (DAG).
        Args:
            df: pandas.DataFrame containing three columns SOURCE, TARGET, AMOUNT

        Returns:
            pandas.DataFrame containing edge list of conservatively compressed network

        """
        f = self._bilateral_compression(_flip_neg_amnts(df))
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

    def _non_conservative_compression_MAX(self, df: pd.DataFrame):
        """
        Returns non-conservatively compressed network.
        Non-conservative compression not only reduces or removes existing edges (trades)
        but can also introduce new ones.


        TODO: IN DOCS ADD https://github.com/sktime/sktime/issues/764
        Requirements of numba version and llvm
        Args:
            df:

        Returns:

        """
        nodes_flow = self.net_flow if df is None else _get_nodes_net_flow(df)

        nodes = np.array(nodes_flow.index)
        flows = nodes_flow.values

        idx = flows[flows != 0].argsort()[::-1]

        ordered_flows = flows[flows != 0][idx]
        nodes = nodes[flows != 0][idx]
        nodes_r = nodes[::-1]

        EL, pairs = _noncons_compr_max_min(ordered_flows=ordered_flows,
                                           max_links=len(nodes)
                                           # TODO - prove the following Theorem: for any FULLY compressed graph G=(N, E) one has |E|<=|N| (number of edges is at most the number of nodes)
                                           )

        fltr = EL != 0
        EL, pairs = EL[fltr], pairs[fltr, :]
        pairs = [*zip(nodes_r.reshape(1, -1)[:, pairs[:, 0]][0],
                      nodes.reshape(1, -1)[:, pairs[:, 1]][0])]

        fx = pd.DataFrame.from_records(pairs, columns=['SOURCE', 'TARGET'])
        fx['AMOUNT'] = EL
        return fx

    def _non_conservative_compression_ED(self, df: pd.DataFrame):
        """
        Returns the non-conservative equally-distributed compressed network.
        Args:
            df: pandas.DataFrame containing three columns SOURCE, TARGET, AMOUNT

        Returns:
            pandas.DataFrame containing edge list of non-conservative equally-distributed compressed network

        """

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

    def _check_compression(self, df: pd.DataFrame, df_compressed: pd.DataFrame, grouper: str=None):
        """
        Args:
            df:
            df_compressed:

        Returns:

        """
        GMS, CMS, EMS = _market_desc(df, grouper, grouper_rename=self._grouper_rename()).values()
        GMS_comp, CMS_comp, EMS_comp = _market_desc(df_compressed, grouper, grouper_rename=self._grouper_rename()).values()
        flows = _get_nodes_net_flow(df).sort_index()
        flows_comp = _get_nodes_net_flow(df_compressed, grouper).sort_index()
        assert EMS>EMS_comp or np.isclose(abs(EMS-EMS_comp), 0.0, atol=1e-6), f"Compression check failed on EMS. \n\n   Original EMS = {EMS} \n Compressed EMS = {EMS_comp}"
        assert np.isclose(pd.concat([flows, flows_comp], axis=1).fillna(0).diff(0).abs().max().max(), 0.0, atol=1e-6), f"Compression check failed on FLOWS. \n\n  Original flows = {flows.to_dict()} \nCompressed flows = {flows_comp.to_dict()}"
        assert np.isclose(CMS, CMS_comp, atol=1e-6), f"Compression check failed on CMS. \n\n   Original CMS = {CMS} \n Compressed CMS = {CMS_comp}"

    def compress(self,
                 type: str='bilateral',
                 compression_p: int=2,
                 verbose: bool=True,
                 progress: bool = True,
                 ret_edgelist: bool=False,
                 _check_compr: bool=True,
                 ):
        """
        Returns compressed network.
        Args:
            type: Type of compression. Either of ('NC-ED', 'NC-MAX', 'C', 'bilateral')
            compression_p: Compression order. Default is `p=1`.
            verbose: If `True` (default) prints out compression efficiency and compression factor.
            progress: Whether to display a progress bar. Default is True.
            ret_edgelist: If `False` (default) returns a compnet.Graph object. Otherwise only the compressed network's edge list.
            _check_compr: Whether to call Graph._check_compression. Default is True.

        Returns:
            Graph object or edge list (pandas.DataFrame) corresponding to compressed network.

        """
        df = self.edge_list
        if type.lower() == 'nc-ed':
            compressor = self._non_conservative_compression_ED
        elif type.lower() == 'nc-max':
            compressor = self._non_conservative_compression_MAX
        elif type.lower() == 'c':
            compressor = self._conservative_compression
        elif type.lower() == 'bilateral':
            compressor = self._bilateral_compression
        else:
            raise Exception(f'Type {type} not recognised: please input either of NC-ED, NC-MAX, C, or bilateral.')

        if self.__GROUPER:
            def clean_compressor(f):
                compressed_df = compressor(f.drop(columns=self.__GROUPER))
                compressed_df[self.__GROUPER] = f[self.__GROUPER].drop_duplicates().values[0]
                return compressed_df
            grpd_df = df.groupby(self.__GROUPER)
            if progress:
                tqdm.pandas()
                df_compressed = grpd_df.progress_apply(clean_compressor).reset_index(drop=True)
            else:
                df_compressed = grpd_df.apply(clean_compressor).reset_index(drop=True)
        else:
            df_compressed = compressor(df)

        if _check_compr:
            self._check_compression(df=df, df_compressed=df_compressed)
        if verbose:
            comp_rt = compression_factor(df1=df, df2=df_compressed, p=compression_p)
            comp_eff = compression_efficiency(df=df, df_compressed=df_compressed)
            print(f"Compression Efficiency CE = {comp_eff}")
            print(f"Compression Factor CF(p={compression_p}) = {comp_rt}")
        df_compressed = df_compressed.rename(columns={v: k for k, v in self._labels_map.items()})
        if ret_edgelist:
            return df_compressed
        else:
            kwargs = {v.lower() if isinstance(v, str) else v:k
                      for k,v in self._labels_map.items()
                      if v is not None and not v.lower().startswith('grouper')}
            kwargs = {**kwargs, **dict(grouper=self._grouper_rename())}
            return Graph(df_compressed, **kwargs)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.edge_list == other.edge_list).all().all()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        MAX_LEN = self._MAX_DISPLAY_LENGTH or 20
        is_long = len(self.edge_list) > MAX_LEN
        f = self.edge_list.head(MAX_LEN).rename(columns={v:k for k,v in self._labels_map.items()
                                                         if k is not None})[self._labels].astype(str)
        if is_long:
            f.loc[MAX_LEN, :] = ['⋮'] * f.shape[1]
        f.index = ['']*len(f)
        # return 'compnet.Graph object:\n' + f.to_string()
        return 'compnet.Graph object:\n' + tabulate(f, headers='keys', tablefmt='simple_outline', showindex=False)



# Nodes net flow
def compressed_network_bilateral(df: pd.DataFrame) -> pd.DataFrame:
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
def compressed_network_non_conservative(df: pd.DataFrame) -> pd.DataFrame:
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
def non_conservative_compression_ED(df: pd.DataFrame) -> pd.DataFrame:
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

def compressed_network_conservative(df: pd.DataFrame) -> pd.DataFrame:
    df = compressed_network_bilateral(df)
    ...



