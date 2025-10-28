"""  Created on 11/10/2022::
------------- tests.py -------------

**Authors**: L. Mingarelli
"""
import numpy as np, pylab as plt
from tqdm import tqdm
import pandas as pd, networkx as nx
import compnet as cn

from compnet.tests.sample.sample0 import (sample0, sample_bilateral, sample_cycle, sample_entangled,
                                  sample_nested_cycle1, sample_nested_cycle2, sample_nested_cycle3, sample_nested_cycle4,
                                  sample_noncons1, sample_noncons1_compressed, sample_noncons2, sample_noncons2_compressed,
                                  sample_noncons2_compressed, sample_noncons3, sample_noncons3_compressed, sample_noncons4, 
                                  sample_noncons4_compressed)

def test_compression_factor():
    compressed = cn.Graph(sample_bilateral).compress(type='bilateral').edge_list
    ps = np.array(list(np.linspace(0.1, 15.01, 100)) + [16] )
    cfs = [cn.compression_factor(df1=sample_bilateral, df2=compressed, p=p) for p in ps]
    plt.axhline(cfs[-1], color='k')
    plt.plot(ps, cfs, color='red')
    plt.show()
    assert (np.array(cfs)>=cfs[-1]).all()

    ps = np.array(list(np.linspace(1, 20, 200))+[50])
    compressed1 = cn.Graph(sample_noncons4).compress(type='nc-ed').edge_list
    compressed2 = cn.Graph(sample_noncons4).compress(type='nc-max').edge_list
    cfs1 = [cn.compression_factor(sample_noncons4, compressed1, p=p)
            for p in ps]
    cfs2 = [cn.compression_factor(sample_noncons4, compressed2, p=p)
            for p in ps]

    plt.axhline(cfs1[-1], color='k')
    plt.axhline(cfs2[-1], color='k')
    plt.plot(ps, cfs1, color='blue', label='Non-conservative ED')
    plt.plot(ps, cfs2, color='red', label='Non-conservative MAX')
    plt.title('Compression factor')
    plt.xlabel('p')
    plt.legend()
    plt.xlim(1, 20)
    plt.show()




def test_compression_factor(df, plot=True):
    ps = np.array(list(np.linspace(1, 20, 191))+[50])
    graph = cn.Graph(df)
    compressed1 = graph.compress(type='nc-ed', verbose=False).edge_list
    compressed2 = graph.compress(type='nc-max', verbose=False).edge_list
    compressed3 = graph.compress(type='c', verbose=False).edge_list
    compressed4 = graph.compress(type='bilateral', verbose=False).edge_list
    cfs1 = [cn.compression_factor(df, compressed1, p=p)
            for p in ps]
    cfs2 = [cn.compression_factor(df, compressed2, p=p)
            for p in ps]
    cfs3 = [cn.compression_factor(df, compressed3, p=p)
            for p in ps]
    cfs4 = [cn.compression_factor(df, compressed4, p=p)
            for p in ps]
    cf_ems = cn.compression_factor(df, compressed4, p='ems_ratio')
    if plot:
        plt.axhline(cfs1[-1], color='k')
        plt.axhline(cfs2[-1], color='k')
        plt.axhline(cfs3[-1], color='k')
        # plt.axhline(cf_ems, color='orange', label='EMS compression factor')
        plt.plot(ps, cfs1, color='blue', label='Non-conservative ED')
        plt.plot(ps, cfs2, color='red', label='Non-conservative MAX')
        plt.plot(ps, cfs3, color='green', label='Conservative')
        plt.plot(ps, cfs4, color='purple', label='Bilateral')
        plt.title('Compression factor')
        plt.xlabel('p')
        plt.legend()
        plt.xlim(1, 20)
        plt.show()
    return np.array(cfs1), np.array(cfs2),

IMPROVED_COMPR = []
for _ in tqdm(range(1000)):
    # df = pd.DataFrame({'SOURCE':      ['A', 'A', 'A', 'B', 'B', 'C'],
    #                    'DESTINATION': ['B', 'C', 'D', 'C', 'D', 'D'],
    #                    # 'AMOUNT': np.random.randint(-100, 100, 6)}
    #                    # 'AMOUNT': np.random.randn(6) * 100+10}
    #                    # 'AMOUNT': np.random.power(0.5, 6) * 100 + 10}
    #                    'AMOUNT': (np.random.power(0.5, 6) * 100 + 10)*(np.random.randn(6) * 100+10)}
    #                   )
    df = pd.DataFrame(nx.erdos_renyi_graph(10, .25, directed=True).edges(),
                      columns=['SOURCE', 'TARGET']).astype(str)
    df['AMOUNT'] = (np.random.power(0.5, df.shape[0]) * 100 + 10) * (np.random.randn(df.shape[0]) * 100 + 10)
    cfs1, cfs2 = test_compression_factor(df, plot=False)
    IMPROVED_COMPR.append(cfs1[10]-cfs2[10])
    if (cfs1<cfs2).any():
        if (~np.isclose(cfs1[cfs1<cfs2], cfs2[cfs1<cfs2])).any():
            test_compression_factor(df, plot=True)
            raise Exception("You were wrong twat!")


plt.hist(IMPROVED_COMPR, bins=100)
plt.yscale('log')
plt.show()


########################
### FIND CLOSED CHAINS
########################
import networkx as nx
from compnet.tests.sample import (sample_cycle, sample_nested_cycle1, sample_nested_cycle2,
                                     sample_nested_cycle3, sample_nested_cycle4, sample_entangled)

# G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
# nx.find_cycle(G, orientation="original")
# list(nx.find_cycle(G, orientation="ignore"))

df = f = sample_entangled
G = nx.DiGraph(list(f.iloc[:,:2].values))
# G.edges
# list(nx.find_cycle(G, orientation="original"))
list(nx.simple_cycles(G))


