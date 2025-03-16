"""  Created on 23/10/2022::
------------- algo.py -------------

**Authors**: L. Mingarelli
"""

import numpy as np, pandas as pd, scipy as sp
import numba, networkx as nx
from tabulate import tabulate
import duckdb
from tqdm import tqdm
from functools import lru_cache
from typing import Union
from collections.abc import Sequence

import warnings

def formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return f'{msg}\n'

warnings.formatwarning = formatwarning

__SEP = '__<>__<>__'

def _flip_neg_amnts(df):
    f = df#.copy(deep=True)
    f_flip = f[f.AMOUNT<0].iloc[:, [1,0,2]]
    f_flip.columns = df.columns
    f_flip['AMOUNT'] *= -1
    f[f.AMOUNT<0] = f_flip
    return f

def _get_all_nodes(df):
    return duckdb.sql("""SELECT DISTINCT SOURCE AS entity FROM df 
                         UNION 
                         SELECT DISTINCT TARGET AS entity FROM df 
                         ORDER BY entity
                      """).to_df().entity.to_list()

def _get_nodes_flows(df, grouper=None):
    grper_str = ','.join(grouper) + ',' if isinstance(grouper, list) else f'{grouper},' if grouper is not None else ''
    int_list = ','.join([str(n + 1) for n in
                         range(1 + len(grouper) if isinstance(grouper, list) else 2)]) if grouper is not None else '1'
    set_idx = (grouper if isinstance(grouper, list) else [grouper] if grouper is not None else []) + ['ENTITY']
    outflow = duckdb.sql(f"""SELECT {grper_str} SOURCE AS ENTITY, SUM(AMOUNT) AS 'OUT' 
                            FROM df 
                            GROUP BY {int_list} ORDER BY {int_list}
                         """).to_df().set_index(set_idx)
    inflow = duckdb.sql(f"""SELECT {grper_str} TARGET AS ENTITY, SUM(AMOUNT) AS 'IN' 
                             FROM df 
                             GROUP BY {int_list} ORDER BY {int_list}
                          """).to_df().set_index(set_idx)
    return inflow, outflow


def _get_nodes_net_flow(df, grouper=None, adjust_labels=None):
    inflow, outflow = _get_nodes_flows(df, grouper=grouper)
    nodes_net_flow = pd.concat([outflow, inflow], axis=1).fillna(0).T.diff().iloc[-1, :]

    if grouper and adjust_labels:  # Adjust net_flow names
        original_grouper = [v for k,v in adjust_labels.items() if k.startswith('GROUPER')]
        nodes_net_flow = nodes_net_flow.reset_index().rename(columns=adjust_labels).set_index(original_grouper)
        nodes_net_flow.columns.name = adjust_labels['AMOUNT']

    return nodes_net_flow


def _get_nodes_gross_flow(df, grouper=None, adjust_labels=None):
    inflow, outflow = _get_nodes_flows(df, grouper=grouper)
    nodes_gross_flow = pd.concat([inflow, outflow], axis=1).fillna(0).sort_index()
    nodes_gross_flow['GROSS_TOTAL'] = nodes_gross_flow[['IN', 'OUT']].sum(1)
    nodes_gross_flow = nodes_gross_flow[nodes_gross_flow.GROSS_TOTAL > 0]

    # if set(_get_all_nodes(f)) != set(all_df_nodes):
    #     group_nodes_gross_flow.reindex(all_df_nodes, fill_value=0).sort_index()
    # else:
    #     group_nodes_gross_flow.sort_index()


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

def _market_desc_OLD(df, grouper=None, grouper_rename=None):
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


def _market_desc(gross_flows, grouper=None, grouper_rename=None):
    GMS = gross_flows['GROSS_TOTAL'].sum(1)/2 if isinstance(gross_flows, dict) else gross_flows['GROSS_TOTAL'].sum()/2
    net_flows = gross_flows['IN'] - gross_flows['OUT']
    CMS = net_flows.clip(lower=0).sum(1 if grouper else 0)
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

def compression_efficiency(df, df_compressed, grouper=None):
    gross_flows = _get_nodes_gross_flow(df=df, grouper=grouper)
    gross_flows_comp = _get_nodes_gross_flow(df=df_compressed, grouper=grouper)
    GMS, CMS, EMS = _market_desc(gross_flows=gross_flows, grouper=grouper).values()
    GMS_comp, CMS_comp, EMS_comp = _market_desc(gross_flows=gross_flows_comp, grouper=grouper).values()
    CE = 1 - EMS_comp / EMS
    return CE


def compression_factor(df1, df2, p=2, grouper=None):
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
        CR = 1- compression_efficiency(df=df1, df_compressed=df2, grouper=grouper)
    else:
        # N = len(set(df1[['SOURCE', 'TARGET']].values.flatten()))
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
                 grouper: Union[str, list]=None,
                 ):
        """
        Initialises compnet.Group object.

        Args:
            df: An edge list containing at least a source, target, and amount columns.
            source: Name of the column corresponding to source nodes. Default is 'SOURCE'.
            target: Name of the column corresponding to target nodes. Default is 'TARGET'.
            amount: Name of the column corresponding to weights / amounts of corresponding source-target edge. Default is 'AMOUNT'.
            grouper: If an additional dimension exists (e.g. a date dimension), passing the corresponding column name (or list of column names) will result in the creation of a graph for each category in the grouper column.
        """
        from compnet import SUPPRESS_WARNINGS
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
        all_nodes = _get_all_nodes(self.edge_list)
        if self.__GROUPER and any(set(all_nodes) - set(_get_all_nodes(g)) for _, g in self.edge_list.groupby(self.__GROUPER)) and not SUPPRESS_WARNINGS:
            warnings.warn(f"\n\nSome nodes (SOURCE `{source}` or TARGET `{target}`) are missing from some groups (GROUPER `{grouper}`).\n"
                          "These will be filled with zeros.\n")

        self.gross_flow = _get_nodes_gross_flow(df=self.edge_list, grouper=self.__GROUPER, adjust_labels=self._labels_imap)
        self.net_flow = (self.gross_flow['IN'] - self.gross_flow['OUT'])
        # self.net_flow == _get_nodes_net_flow(df=self.edge_list, grouper=self.__GROUPER, adjust_labels=self._labels_imap)

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

    @property
    def DEALERS(self):
        if self.__GROUPER is None:
            dealers = self.ENTITIES.loc[self.ENTITIES['is_dealer'], ['dealer_ratio']]
            dealers.index.name = 'entity'
        else:
            dealers = (self.ENTITIES['dealer_ratio']
                           .melt(ignore_index=False)
                           .rename(columns={'amount': 'entity', 'value': 'dealer_ratio'})
                       )
            dealers = dealers[dealers.dealer_ratio>0]
        return dealers

    def dirichlet_energy(self, degree_type:str='net') -> pd.Series:
        """
        Computed the Dirichlet energy ½ ∑|L_{ij}|^2, with L the Laplacian matrix.
        This method automatically accounts for the grouper, if any was provided.
        Args:
            degree_type: Either of 'net' (default) or 'gross'.

        """
        DEGREE_OPERATOR = {'net': '-', 'gross': '+'}

        grouper = self.__GROUPER if isinstance(self.__GROUPER, list) else [self.__GROUPER] if self.__GROUPER is not None else None

        grper_str = ','.join(grouper) + ',' if isinstance(grouper, list) else f'{grouper},' if grouper is not None else ''
        int_list = ','.join([str(n + 1) for n in range(1+len(grouper) if isinstance(grouper, list) else 2)])  if grouper is not None else '1'
        degree_grper_str = ','.join([f'COALESCE(i.{g},o.{g}) AS {g}' for g in grouper]) + ',' if isinstance(grouper, list) else ''
        on_grper_str = 'i.ENTITY = o.ENTITY AND ' + ' AND '.join([f'i.{grpr} = o.{grpr}' for grpr in grouper]) if grouper is not None else 'i.ENTITY = o.ENTITY'

        edgelist = self.edge_list
        return duckdb.sql(f"""
                  WITH InDegree AS (SELECT {grper_str} TARGET AS ENTITY, SUM(AMOUNT) AS IN_DEGREE
                                    FROM edgelist
                                    GROUP BY {int_list}), 
                       OutDegree AS (SELECT {grper_str} SOURCE AS ENTITY, SUM(AMOUNT) AS OUT_DEGREE
                                     FROM edgelist
                                     GROUP BY {int_list}),
                       Degree AS (SELECT {degree_grper_str}
                                         COALESCE(i.ENTITY, o.ENTITY) AS ENTITY, 
                                         COALESCE(i.IN_DEGREE, 0) {DEGREE_OPERATOR[degree_type]} COALESCE(o.OUT_DEGREE, 0) AS DEGREE
                                  FROM InDegree i
                                  FULL OUTER JOIN OutDegree o ON {on_grper_str}),
                       LAPLACIAN AS (SELECT {grper_str}
                                            SOURCE AS ENTITY1, TARGET AS ENTITY2, 
                                            -AMOUNT AS LAPLACIAN_VALUE  -- Off-diagonal elements (-A)
                                     FROM edgelist e
                                     UNION ALL 
                                     SELECT {grper_str} 
                                            ENTITY AS ENTITY1, ENTITY AS ENTITY2, 
                                            DEGREE AS LAPLACIAN_VALUE -- Diagonal elements (D)
                                     FROM Degree)
                 SELECT {grper_str} SUM(LAPLACIAN_VALUE*LAPLACIAN_VALUE) / 2 AS DIRICHLET_ENERGY
                 FROM LAPLACIAN
                 {f'GROUP BY {grper_str[:-1]}' if grper_str else ''};
               """).to_df().rename(columns={v:k for k,v in self._labels_map.items()}).set_index([k for k,v in self._labels_map.items() if v.startswith('GROUPER')]).DIRICHLET_ENERGY

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
            GMS, CMS, EMS = _market_desc(gross_flows=self.gross_flow,
                                         grouper=self.__GROUPER,
                                         grouper_rename=self._grouper_rename()).values()

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

    def _bilateral_compression(self, df: pd.DataFrame, grouper=None):
        """
        Returns bilaterally compressed network.
        Bilateral compression compresses exclusively multiple trades existing between the same pair of nodes.
        Args:
            df: pandas.DataFrame containing three columns SOURCE, TARGET, AMOUNT
            grouper: If an additional dimension exists (e.g. a date dimension), passing the corresponding column name (or list of column names) will result in the creation of a graph for each category in the grouper column.

        Returns:
            pandas.DataFrame containing edge list of bilaterally compressed network
        """
        grper_str = ','.join(grouper) + ',' if isinstance(grouper, list) else f'{grouper},' if grouper is not None else ''
        int_list = ','.join([str(n + 1) for n in range(2 + len(grouper) if isinstance(grouper, list) else 3)]) if grouper is not None else '1,2'
        rf = duckdb.sql(f"""WITH net AS (SELECT {grper_str}
                                                CASE WHEN SOURCE < TARGET THEN SOURCE ELSE TARGET END AS node1,
                                                CASE WHEN SOURCE < TARGET THEN TARGET ELSE SOURCE END AS node2,
                                                SUM(CASE WHEN SOURCE < TARGET THEN AMOUNT ELSE -AMOUNT END) AS net_amount
                                         FROM df
                                         GROUP BY {int_list})
                             SELECT {grper_str}
                                    CASE WHEN net_amount > 0 THEN node1 ELSE node2 END AS SOURCE,
                                    CASE WHEN net_amount > 0 THEN node2 ELSE node1 END AS TARGET,
                                    ABS(net_amount) AS AMOUNT
                             FROM net
                             WHERE net_amount <> 0
                             ORDER BY {int_list};
                        """).to_df()

        return rf

    def _conservative_compression(self, df: pd.DataFrame, grouper=None):
        """
        Returns conservatively compressed network.
        Conservative compression only reduces or removes existing edges (trades)
        without however adding new ones.
        The resulting conservatively compressed graph is a sub-graph of the original graph.
        Moreover, the resulting conservatively compressed graph is always a directed
        acyclic graph (DAG).
        Args:
            df: pandas.DataFrame containing three columns SOURCE, TARGET, AMOUNT
            grouper: If an additional dimension exists (e.g. a date dimension), passing the corresponding column name (or list of column names) will result in the creation of a graph for each category in the grouper column.

        Returns:
            pandas.DataFrame containing edge list of conservatively compressed network

        """
        f = self._bilateral_compression(_flip_neg_amnts(df))
        edgs = f.set_index(f.SOURCE + self.__SEP + f.TARGET)[['AMOUNT']].T
        @lru_cache()
        def loop2edg(tpl):
            """Takes a tuple of nodes as input, defining the 'loop' or cycle,
               and returns the associated list of edges
            """
            return list(f'{x}{self.__SEP}{y}' for x, y in zip((tpl[-1],) + tpl[:-1], tpl))
        @lru_cache()
        def get_minedg(cycle):
            """Returns the smallest edge weight,
               from the list of weights associated with the edges of the cycle passed as input.
            """
            return edgs[loop2edg(cycle)].T.min().AMOUNT

        G = nx.DiGraph(list(f.iloc[:, :2].values))
        # For each cycle, find the associated smallest edge weight, multiplied by the length of the cycle ()
        cycles_len_minedg = [(tuple(c), len(c) * get_minedg(tuple(c)))
                             for c in nx.simple_cycles(G)]
        while cycles_len_minedg:
            idx = np.argmax((c[1] for c in cycles_len_minedg))
            cycle, min_edg_lencycle = cycles_len_minedg[idx]
            cls = loop2edg(cycle)
            if pd.Series(cls).isin(edgs.columns).all():
                min_edg = edgs[cls].min(1).AMOUNT # min_edg_lencycle / len(cycle) ## [WARNING: this second option may break the == search at the next lines with floats]
                drop_col = edgs[cls].columns[(edgs[cls]==min_edg).values[0]][0] # Edge to be removed
                edgs[cls] -= min_edg
                edgs.drop(columns=[drop_col], inplace=True)
            cycles_len_minedg.pop(idx)
        edgs = edgs.T.reset_index()
        amnt = edgs.AMOUNT
        edgs = pd.DataFrame(edgs['index'].str.split(self.__SEP).to_list(),
                            columns=['SOURCE', 'TARGET'])
        edgs['AMOUNT'] = amnt
        return edgs

    def _non_conservative_compression_MAX(self, df: pd.DataFrame, grouper=None):
        """
        Returns non-conservatively compressed network.
        Non-conservative compression not only reduces or removes existing edges (trades)
        but can also introduce new ones.


        TODO: IN DOCS ADD https://github.com/sktime/sktime/issues/764
        Requirements of numba version and llvm
        Args:
            df: pandas.DataFrame containing three columns SOURCE, TARGET, AMOUNT
            grouper: If an additional dimension exists (e.g. a date dimension), passing the corresponding column name (or list of column names) will result in the creation of a graph for each category in the grouper column.

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

    def _non_conservative_compression_ED(self, df: pd.DataFrame, grouper=None, fast=True):
        """
        Returns the non-conservative equally-distributed compressed network.
        Args:
            df: pandas.DataFrame containing three columns SOURCE, TARGET, AMOUNT
            grouper: If an additional dimension exists (e.g. a date dimension), passing the corresponding column name (or list of column names) will result in the creation of a graph for each category in the grouper column.
            fast: Use fast (duckdb based) or slow version of the algorithm. Slow version will eventually be deprecated.
        Returns:
            pandas.DataFrame containing edge list of non-conservative equally-distributed compressed network

        """

        nodes_flow = _get_nodes_net_flow(df, grouper=grouper)
        if fast:
            nodes_flow = nodes_flow.reset_index().rename(columns={'IN': 'net_flow'})

            if grouper is None:
                return duckdb.sql("""WITH positive AS (SELECT ENTITY, net_flow
                                                       FROM nodes_flow WHERE net_flow > 0),
                                          negative AS (SELECT ENTITY, net_flow
                                                       FROM nodes_flow WHERE net_flow < 0),
                                          total_positive AS (SELECT SUM(net_flow) AS total
                                                             FROM positive)
                                     SELECT negative.ENTITY AS SOURCE, positive.ENTITY AS TARGET,
                                           (-negative.net_flow * positive.net_flow) / total_positive.total AS AMOUNT
                                     FROM negative
                                     CROSS JOIN positive
                                     CROSS JOIN total_positive;
                                  """).to_df()
            else:
                if not isinstance(grouper, list):
                    grouper = [grouper]
                grper_str = ','.join(grouper) if isinstance(grouper, list) else f'{grouper},'
                int_list = ','.join([str(n + 1) for n in range(len(grouper) if isinstance(grouper, list) else 1)])
                tp_grper_str = grper_str.replace('GROUPER', 'tp.GROUPER')
                on_grper_str1 = ' AND '.join([f'negative.{grpr} = positive.{grpr}' for grpr in grouper])
                on_grper_str2 = ' AND '.join([f'negative.{grpr} = tp.{grpr}' for grpr in grouper])

                return duckdb.sql(f"""WITH positive AS (SELECT ENTITY, net_flow, {grper_str}
                                                        FROM nodes_flow WHERE net_flow > 0),
                                           negative AS (SELECT ENTITY, net_flow, {grper_str}
                                                        FROM nodes_flow WHERE net_flow < 0),
                                           total_positive AS (SELECT {grper_str}, SUM(net_flow) AS total
                                                              FROM positive 
                                                              GROUP BY {int_list})
                                      SELECT negative.ENTITY AS SOURCE, positive.ENTITY AS TARGET,
                                             (-negative.net_flow * positive.net_flow) / tp.total AS AMOUNT,
                                             {tp_grper_str}
                                      FROM negative
                                      JOIN positive ON {on_grper_str1}
                                      JOIN total_positive tp ON {on_grper_str2};
                                   """).to_df()
        # TODO: THIS PART BELOW CAN BE DROPPED EVENTUALLY - it's already not used anymore
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
            grouper: If an additional dimension exists (e.g. a date dimension), passing the corresponding column name (or list of column names) will result in the creation of a graph for each category in the grouper column.

        Returns:

        """
        gross_flows = _get_nodes_gross_flow(df=df, grouper=grouper)
        gross_flows_comp = _get_nodes_gross_flow(df=df_compressed, grouper=grouper)
        net_flows = gross_flows['IN'] - gross_flows['OUT']
        net_flows_comp = gross_flows_comp['IN'] - gross_flows_comp['OUT']
        GMS, CMS, EMS = _market_desc(gross_flows=gross_flows, grouper=grouper, grouper_rename=self._grouper_rename()).values()
        GMS_comp, CMS_comp, EMS_comp = _market_desc(gross_flows=gross_flows_comp, grouper=grouper, grouper_rename=self._grouper_rename()).values()

        assert EMS>EMS_comp or np.isclose(abs(EMS-EMS_comp), 0.0, atol=1e-6), f"Compression check failed on EMS. \n\n   Original EMS = {EMS} \n Compressed EMS = {EMS_comp}"
        assert np.isclose(pd.concat([net_flows, net_flows_comp], axis=1).fillna(0).diff(0).abs().max().max(), 0.0, atol=1e-6), f"Compression check failed on FLOWS. \n\n  Original flows = {flows.to_dict()} \nCompressed flows = {flows_comp.to_dict()}"
        assert np.isclose(CMS, CMS_comp, atol=1e-6), f"Compression check failed on CMS. \n\n   Original CMS = {CMS} \n Compressed CMS = {CMS_comp}"

    def compress(self,
                 type: str='bilateral',
                 compression_p: int=2,
                 verbose: bool=False,
                 progress: bool=True,
                 ret_edgelist: bool=False,
                 _check_compr: bool=True,
                 ):
        """
        Returns compressed network.
        Args:
            type: Type of compression. Either of ('NC-ED', 'NC-MAX', 'C', 'bilateral')
            compression_p: Compression order. Default is `p=1`.
            verbose: If `True` prints out compression efficiency and compression factor.
            progress: Whether to display a progress bar. Default is True.
            ret_edgelist: If `False` (default) returns a compnet.Graph object. Otherwise only the compressed network's edge list.
            _check_compr: Whether to call Graph._check_compression. Default is True.

        Returns:
            Graph object or edge list (pandas.DataFrame) corresponding to compressed network.

        """
        df = self.edge_list
        if type.lower() == 'nc-ed':
            compressor = self._non_conservative_compression_ED  # This has been optimised via duckdb. Will not be looped.
        elif type.lower() == 'nc-max':
            compressor = self._non_conservative_compression_MAX
        elif type.lower() == 'c':
            compressor = self._conservative_compression
        elif type.lower() == 'bilateral':
            compressor = self._bilateral_compression  # This has been optimised via duckdb. Will not be looped.
        else:
            raise Exception(f'Type {type} not recognised: please input either of NC-ED, NC-MAX, C, or bilateral.')

        if type.lower()=='bilateral':
            df_compressed = self._bilateral_compression(df, grouper=self.__GROUPER)
        elif type.lower()=='nc-ed':
            df_compressed = self._non_conservative_compression_ED(df, grouper=self.__GROUPER)
        else:
            if self.__GROUPER:
                def clean_compressor(f):
                    compressed_df = compressor(f.drop(columns=self.__GROUPER), grouper=None)
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

        if _check_compr and self.__GROUPER is None:
            self._check_compression(df=df, df_compressed=df_compressed, grouper=self.__GROUPER)
        if verbose and self.__GROUPER is None:
            comp_rt = compression_factor(df1=df, df2=df_compressed, p=compression_p, grouper=self.__GROUPER)
            comp_eff = compression_efficiency(df=df, df_compressed=df_compressed, grouper=self.__GROUPER)
            print(f"Compression Efficiency CE = {comp_eff}")
            print(f"Compression Factor CF(p={compression_p}) = {comp_rt}")
        elif verbose and self.__GROUPER is not None:
            raise ValueError('Verbose not yet available together with grouper.')
        df_compressed = df_compressed.rename(columns={v: k for k, v in self._labels_map.items()})
        if ret_edgelist:
            return df_compressed
        else:
            kwargs = {v.lower() if isinstance(v, str) else v:k
                      for k,v in self._labels_map.items()
                      if v is not None and not v.lower().startswith('grouper')}
            kwargs = {**kwargs, **dict(grouper=self._grouper_rename())}
            return Graph(df_compressed, **kwargs)

    def centrally_clear(self, ccp_name:str='CCP', net:bool=False):
        """
        Clears all positions centrally through a central clearing counterparty named `ccp_name`:
        if `ccp_name` does not exist already among the list of entities (in either source or target) it is introduced.
        Args:
            ccp_name: Name of the central clearing counterparty. Default is 'CCP'. If `ccp_name` does not exist already among the list of entities (in either source or target) it is introduced.
            net: If True returns the centrally cleared graph with bilateral compression.

        Returns:
            New Graph object, with all positions centrally cleared through `ccp_name`.
        """
        cleared_edge_list = pd.concat([self.edge_list.assign(SOURCE=ccp_name),
                                       self.edge_list.assign(TARGET=ccp_name)])
        if ccp_name in self.net_flow.index:
            cleared_edge_list = cleared_edge_list[(cleared_edge_list.SOURCE!=ccp_name)|(cleared_edge_list.TARGET!=ccp_name)]
        cleared_g = Graph(cleared_edge_list.rename(columns=self._labels_imap),
                          source=self._labels_imap['SOURCE'],
                          target=self._labels_imap['TARGET'],
                          amount=self._labels_imap['AMOUNT'],
                          grouper=self._grouper_rename())
        if net:
            return cleared_g.compress(type='bilateral')
        else:
            return cleared_g


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



# def adjacency_matrix(edgelist):
#     edgelist = edgelist[['SOURCE', 'TARGET', 'AMOUNT']]
#     nodes = _get_all_nodes(edgelist)
#     nodes_idx = {node: i for i, node in enumerate(nodes)}
#     row, col, weight = edgelist.replace(nodes_idx).values.T
#     # Create sparse adjacency matrix
#     adj_matrix = sp.sparse.coo_matrix((weight, (row, col)), shape=(len(nodes), len(nodes)))
#     # print(adj_matrix.toarray())
#     return adj_matrix
#
#
# def laplacian(edgelist):
#     net_flow = _get_nodes_net_flow(edgelist)
#     num_nodes = len(net_flow)
#     # D = np.eye(num_nodes) * self.net_flow.values
#     ids = np.arange(num_nodes)
#     D = sp.sparse.coo_matrix((net_flow.values, (ids, ids)),
#                              shape=(num_nodes, num_nodes))
#     return D - adjacency_matrix(edgelist)
#
#
# def dirichlet_energy(edgelist):
#     return (np.abs(laplacian(edgelist))**2).sum() / 2
