"""  Created on 11/10/2022::
------------- tests.py -------------

**Authors**: L. Mingarelli
"""

from compnet.algo import CompNet, compression_factor
from compnet._testing.sample import (sample0, sample_bilateral,
                                     sample_noncons2, sample_noncons4)

class TestCompression:

    def test_describe(self):

        CompNet(sample_bilateral).describe()
        assert (CompNet(sample_bilateral).describe(ret=True) == [30, 15, 15]).all()

    def test_compress_bilateral(self):
        net = CompNet(sample_bilateral)
        bil_compr = net.compress(type='bilateral')

        assert (bil_compr.AMOUNT == [5, 15]).all()
        assert (CompNet(bil_compr).net_flow == CompNet(sample_bilateral).net_flow).all()

        assert (CompNet(sample_noncons2).compress(type='bilateral').AMOUNT == [10, 5, 20]).all()

    def test_compress_NC_ED(self):
        dsc = CompNet(sample_noncons4).describe(ret=True)
        ncedc = CompNet(sample_noncons4).compress(type='NC-ED')

        cmpr_dsc = CompNet(ncedc).describe(ret=True)
        # Check Null Excess
        assert cmpr_dsc['Excess size'] == 0
        # Check Conserved Compressed size
        assert cmpr_dsc['Compressed size'] == dsc['Compressed size'] == cmpr_dsc['Gross size']

    def test_compress_NC_MAX(self):
        dsc = CompNet(sample_noncons4).describe(ret=True)
        ncmaxc = CompNet(sample_noncons4).compress(type='NC-MAX')

        cmpr_dsc = CompNet(ncmaxc).describe(ret=True)
        # Check Null Excess
        assert cmpr_dsc['Excess size'] == 0
        # Check Conserved Compressed size
        assert cmpr_dsc['Compressed size'] == dsc['Compressed size'] == cmpr_dsc['Gross size']

    def test_compression_factor(self):
        import numpy as np, pylab as plt

        compressed = CompNet(sample_bilateral).compress(type='bilateral')
        ps = np.array(list(np.linspace(0.1, 15.01, 100)) + [16] )
        cfs = [compression_factor(sample_bilateral, compressed, p=p) for p in ps]
        plt.axhline(cfs[-1], color='k')
        plt.plot(ps, cfs, color='red')
        plt.show()
        assert (np.array(cfs)>=cfs[-1]).all()

        ps = np.array(list(np.linspace(0.5, 20, 200)))
        compressed1 = CompNet(sample_noncons4).compress(type='nc-ed')
        compressed2 = CompNet(sample_noncons4).compress(type='nc-max')
        cfs1 = [compression_factor(sample_noncons4, compressed1, p=p, _max_comp_p=200)
                for p in ps]
        cfs2 = [compression_factor(sample_noncons4, compressed2, p=p, _max_comp_p=200)
                for p in ps]

        plt.axhline(cfs1[-1], color='k')
        plt.axhline(cfs2[-1], color='gray', linestyle='--')
        plt.plot(ps, cfs1, color='blue')
        plt.plot(ps, cfs2, color='red')
        # plt.xlim(0, 16)
        plt.show()










