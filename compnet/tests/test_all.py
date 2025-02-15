"""  Created on 23/07/2022::
------------- test_all.py -------------

**Authors**: L. Mingarelli
"""
import pandas as pd, numpy as np, pylab as plt, networkx as nx
import compnet as cn


from compnet.tests.sample.sample0 import (sample0, sample_bilateral, sample_cycle, sample_entangled,
                                  sample_nested_cycle1, sample_nested_cycle2, sample_nested_cycle3, sample_nested_cycle4,
                                  sample_noncons1, sample_noncons1_compressed, sample_noncons2, sample_noncons2_compressed,
                                  sample_noncons2_compressed, sample_noncons3, sample_noncons3_compressed, sample_noncons4, 
                                  sample_noncons4_compressed)


### Compare page 64 here: https://www.esrb.europa.eu/pub/pdf/wp/esrbwp44.en.pdf
sample_derrico = pd.DataFrame([['Node A','Node B', 5],
     ['Node B','Node C', 10],
     ['Node C','Node A', 20],
     ],columns=['SOURCE', 'TARGET' ,'AMOUNT'])

class Test_DErrico:
    def test_conservative_compression(self):
        c_comp = cn.Graph(df=sample_derrico).compress(type='c')
        ncmax_comp = cn.Graph(sample_derrico).compress(type='nc-max')
        nced__comp = cn.Graph(sample_derrico).compress(type='nc-ed')


class TestCompression:

    def test_describe(self):
        cn.Graph(sample_bilateral).describe()
        assert (cn.Graph(sample_bilateral).describe(ret=True) == [30, 15, 15]).all()

    def test_with_grouper(self):
        # One grouper
        el = pd.DataFrame([['A', 'B', 15, '2025-02-10'],
                           ['B', 'C', 15, '2025-02-10'],
                           ['B', 'A', 5, '2025-02-10'],
                           ['A', 'B', 20, '2025-02-11'],
                           ['B', 'C', 15, '2025-02-11'],
                           ['B', 'A', 6, '2025-02-11'],
                           ['A', 'B', 25, '2025-02-12'],
                           ['B', 'C', 15, '2025-02-12'],
                           ['B', 'A', 7, '2025-02-12'],
                           ],
                          columns=['lender', 'borrower', 'amount', 'date'])
        g = cn.Graph(el, source='lender', target='borrower', amount='amount', grouper='date')

        c_comp = g.compress(type='c')
        ncmax_comp = g.compress(type='nc-max')
        nced__comp = g.compress(type='nc-ed')

        # Multiple groupers
        el = pd.DataFrame([['A', 'B', 10, '2025-02-10', 'ISIN_A'],
                                ['B', 'C', 5, '2025-02-10', 'ISIN_A'],
                                ['B', 'A', 3,  '2025-02-10', 'ISIN_A'],

                                ['A', 'B', 5, '2025-02-10', 'ISIN_B'],
                                ['B', 'C', 10, '2025-02-10', 'ISIN_B'],
                                ['B', 'A', 2,  '2025-02-10', 'ISIN_B'],

                                ['A', 'B', 12, '2025-02-11', 'ISIN_A'],
                                ['B', 'C', 5, '2025-02-11', 'ISIN_A'],
                                ['B', 'A', 4, '2025-02-11', 'ISIN_A'],

                                ['A', 'B', 8, '2025-02-11', 'ISIN_B'],
                                ['B', 'C', 14, '2025-02-11', 'ISIN_B'],
                                ['B', 'A', 5, '2025-02-11', 'ISIN_B'],
                                ],
                          columns=['lender', 'borrower', 'amount', 'date', 'collateral'])
        g = cn.Graph(df=el, source='lender', target='borrower', amount='amount', grouper=['date', 'collateral'])

        g.describe()

        c_comp = g.compress(type='c')
        ncmax_comp = g.compress(type='nc-max')
        nced__comp = g.compress(type='nc-ed')

    def test_compress_bilateral(self):
        net = cn.Graph(sample_bilateral)
        bil_compr = net.compress(type='bilateral')

        assert (bil_compr.AMOUNT == [5, 15]).all()
        assert (bil_compr.net_flow == cn.Graph(sample_bilateral).net_flow).all()

        assert (cn.Graph(sample_noncons2).compress(type='bilateral').AMOUNT == [10, 5, 20]).all()

    def test_compress_NC_ED(self):
        dsc = cn.Graph(sample_noncons4).describe(ret=True)
        ncedc = cn.Graph(sample_noncons4).compress(type='NC-ED')

        cmpr_dsc = ncedc.describe(ret=True)
        # Check Null Excess
        assert cmpr_dsc['Excess size'] == 0
        # Check Conserved Compressed size
        assert cmpr_dsc['Compressed size'] == dsc['Compressed size'] == cmpr_dsc['Gross size']

    def test_compress_NC_MAX(self):
        dsc = cn.Graph(sample_noncons4).describe(ret=True)
        ncmaxc = cn.Graph(sample_noncons4).compress(type='NC-MAX')

        cmpr_dsc = ncmaxc.describe(ret=True)
        # Check Null Excess
        assert cmpr_dsc['Excess size'] == 0
        # Check Conserved Compressed size
        assert cmpr_dsc['Compressed size'] == dsc['Compressed size'] == cmpr_dsc['Gross size']

    def test_compression_factor(self):

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







