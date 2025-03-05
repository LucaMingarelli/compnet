"""  Created on 23/07/2022::
------------- test_all.py -------------

**Authors**: L. Mingarelli
"""
import pandas as pd, numpy as np, pylab as plt, networkx as nx
import compnet as cn


from compnet.tests.sample.sample0 import (sample0, sample_bilateral, sample_cycle, sample_entangled, sample_purebilateral,
                                          sample_nested_cycle1, sample_nested_cycle2, sample_nested_cycle3, sample_nested_cycle4,
                                          sample_noncons1, sample_noncons1_compressed, sample_noncons2, sample_noncons2_compressed,
                                          sample_noncons2_compressed, sample_noncons3, sample_noncons3_compressed, sample_noncons4,
                                          sample_noncons4_compressed,
                                          sample_onegrouper, sample_twogrouper, sample_warning)


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
        g = cn.Graph(sample_bilateral)
        g.describe()
        assert (cn.Graph(sample_bilateral).describe(ret=True) == [30, 15, 15]).all()
        assert g.gross_flow['IN'].to_dict() == {'A': 5, 'B': 10, 'C': 15}
        assert g.gross_flow['OUT'].to_dict() == {'A': 10.0, 'B': 20.0, 'C': 0.0}
        assert g.gross_flow['GROSS_TOTAL'].to_dict()=={'A': 15.0, 'B': 30.0, 'C': 15.0}
        assert g.net_flow.to_dict()== {'A': -5.0, 'B': -10.0, 'C': 15.0}

    def test_with_grouper(self):
        # One grouper
        g = cn.Graph(sample_onegrouper, source='lender', target='borrower', amount='amount', grouper='date')

        assert g.gross_flow['IN'].to_dict() == {'A': {'2025-02-10': 5, '2025-02-11': 6, '2025-02-12': 7},
                                                'B': {'2025-02-10': 15, '2025-02-11': 20, '2025-02-12': 25},
                                                'C': {'2025-02-10': 15, '2025-02-11': 15, '2025-02-12': 15}}

        assert g.net_flow.to_dict() == {'A': {'2025-02-10': -10.0, '2025-02-11': -14.0, '2025-02-12': -18.0},
                                        'B': {'2025-02-10': -5.0, '2025-02-11': -1.0, '2025-02-12': 3.0},
                                        'C': {'2025-02-10': 15.0, '2025-02-11': 15.0, '2025-02-12': 15.0}}

        c_comp = g.compress(type='c')
        ncmax_comp = g.compress(type='nc-max')
        nced__comp = g.compress(type='nc-ed')

        # Multiple groupers
        g = cn.Graph(df=sample_twogrouper, source='lender', target='borrower', amount='amount', grouper=['date', 'collateral'])

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
        cfs1 = [cn.compression_factor(df1=sample_noncons4, df2=compressed1, p=p)
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


    def test_ENTITIES(self):
        assert cn.Graph(sample_derrico).ENTITIES.is_dealer.all()
        assert not cn.Graph(sample_purebilateral).ENTITIES.is_dealer.any()

        assert cn.Graph(df=sample_onegrouper,
                 source='lender',
                 target='borrower',
                 amount='amount',
                 grouper='date').ENTITIES['is_dealer'].A.all()

        assert cn.Graph(df=sample_twogrouper,
                        source='lender',
                        target='borrower',
                        amount='amount',
                        grouper=['date', 'collateral']
                        ).ENTITIES['is_dealer'].A.all()

        assert set(cn.Graph(sample_derrico).DEALERS.index) == {'Node A', 'Node B', 'Node C'}
        assert len(set(cn.Graph(df=sample_onegrouper,
                 source='lender',
                 target='borrower',
                 amount='amount',
                 grouper='date').DEALERS))==2


    def test_Warnings(self):
        cn.Graph(df=sample_warning,
                 source='lender',
                 target='borrower',
                 amount='amount',
                 grouper='date')

        cn.SUPPRESS_WARNINGS = True

        cn.Graph(df=sample_warning,
                 source='lender',
                 target='borrower',
                 amount='amount',
                 grouper='date')


