"""  Created on 11/10/2022::
------------- tests.py -------------

**Authors**: L. Mingarelli
"""

import numpy as np, pandas as pd, pylab as plt, networkx as nx
import compnet as cn
from compnet.tests.sample import (sample0, sample_bilateral,
                                     sample_noncons2, sample_noncons4)

class TestCompression:

    def test_describe(self):
        cn.Graph(sample_bilateral).describe()
        assert (cn.Graph(sample_bilateral).describe(ret=True) == [30, 15, 15]).all()

    def test_compress_bilateral(self):
        net = cn.Graph(sample_bilateral)
        bil_compr = net.compress(type='bilateral')

        assert (bil_compr.AMOUNT == [5, 15]).all()
        assert (cn.Graph(bil_compr).net_flow == cn.Graph(sample_bilateral).net_flow).all()

        assert (cn.Graph(sample_noncons2).compress(type='bilateral').AMOUNT == [10, 5, 20]).all()

    def test_compress_NC_ED(self):
        dsc = cn.Graph(sample_noncons4).describe(ret=True)
        ncedc = cn.Graph(sample_noncons4).compress(type='NC-ED')

        cmpr_dsc = cn.Graph(ncedc).describe(ret=True)
        # Check Null Excess
        assert cmpr_dsc['Excess size'] == 0
        # Check Conserved Compressed size
        assert cmpr_dsc['Compressed size'] == dsc['Compressed size'] == cmpr_dsc['Gross size']

    def test_compress_NC_MAX(self):
        dsc = cn.Graph(sample_noncons4).describe(ret=True)
        ncmaxc = cn.Graph(sample_noncons4).compress(type='NC-MAX')

        cmpr_dsc = cn.Graph(ncmaxc).describe(ret=True)
        # Check Null Excess
        assert cmpr_dsc['Excess size'] == 0
        # Check Conserved Compressed size
        assert cmpr_dsc['Compressed size'] == dsc['Compressed size'] == cmpr_dsc['Gross size']

    def test_compression_factor(self):
        import numpy as np, pylab as plt

        compressed = cn.Graph(sample_bilateral).compress(type='bilateral')
        ps = np.array(list(np.linspace(0.1, 15.01, 100)) + [16] )
        cfs = [cn.compression_factor(sample_bilateral, compressed, p=p) for p in ps]
        plt.axhline(cfs[-1], color='k')
        plt.plot(ps, cfs, color='red')
        plt.show()
        assert (np.array(cfs)>=cfs[-1]).all()

        ps = np.array(list(np.linspace(1, 20, 200))+[50])
        compressed1 = cn.Graph(sample_noncons4).compress(type='nc-ed')
        compressed2 = cn.Graph(sample_noncons4).compress(type='nc-max')
        cfs1 = [cn.compression_factor(sample_noncons4, compressed1, p=p)
                for p in ps]
        cfs2 = [cn.compression_factor(sample_noncons4, compressed2, p=p)
                for p in ps]

        plt.axhline(cfs1[-1], color='k')
        plt.axhline(cfs2[-1], color='k')
        plt.plot(ps, cfs1, color='blue', label='Non-conservative ED')
        plt.plot(ps, cfs2, color='red', label='Non-conservative MAX')
        plt.legend()
        plt.xlim(1, 20)
        plt.show()




def test_compression_factor(df, plot=True):
    ps = np.array(list(np.linspace(1, 20, 191))+[50])
    graph = cn.Graph(df)
    compressed1 = graph.compress(type='nc-ed', verbose=False)
    compressed2 = graph.compress(type='nc-max', verbose=False)
    compressed3 = graph.compress(type='c', verbose=False)
    compressed4 = graph.compress(type='bilateral', verbose=False)
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
        plt.legend()
        plt.xlim(1, 20)
        plt.show()
    return np.array(cfs1), np.array(cfs2),

IMPROVED_COMPR = []
for _ in range(500):
    # df = pd.DataFrame({'SOURCE':      ['A', 'A', 'A', 'B', 'B', 'C'],
    #                    'DESTINATION': ['B', 'C', 'D', 'C', 'D', 'D'],
    #                    # 'AMOUNT': np.random.randint(-100, 100, 6)}
    #                    # 'AMOUNT': np.random.randn(6) * 100+10}
    #                    # 'AMOUNT': np.random.power(0.5, 6) * 100 + 10}
    #                    'AMOUNT': (np.random.power(0.5, 6) * 100 + 10)*(np.random.randn(6) * 100+10)}
    #                   )
    df = pd.DataFrame(nx.erdos_renyi_graph(10, .25, directed=True).edges(),
                      columns=['SOURCE', 'DESTINATION']).astype(str)
    df['AMOUNT'] = (np.random.power(0.5, df.shape[0]) * 100 + 10) * (np.random.randn(df.shape[0]) * 100 + 10)
    cfs1, cfs2 = test_compression_factor(df, plot=True)
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


