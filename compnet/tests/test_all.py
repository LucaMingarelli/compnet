"""  Created on 23/07/2022::
------------- test_all.py -------------

**Authors**: L. Mingarelli
"""
import pandas as pd, numpy as np


from compnet.tests.sample.sample0 import (sample0, sample_bilateral, sample_cycle, sample_entangled,
                                  sample_nested_cycle1, sample_nested_cycle2, sample_nested_cycle3, sample_nested_cycle4,
                                  sample_noncons1, sample_noncons1_compressed, sample_noncons2, sample_noncons2_compressed,
                                  sample_noncons2_compressed, sample_noncons3, sample_noncons3_compressed, sample_noncons4, 
                                  sample_noncons4_compressed)

from compnet.algo import Graph, compression_factor, _market_desc, compressed_network_conservative

### Compare page 64 here: https://www.esrb.europa.eu/pub/pdf/wp/esrbwp44.en.pdf
sample_derrico = pd.DataFrame([['Node A','Node B', 5],
     ['Node B','Node C', 10],
     ['Node C','Node A', 20],
     ],columns=['SOURCE', 'TARGET' ,'AMOUNT'])

class Tests:
    def test_check_commonprob(self):
        c_comp = Graph(sample_derrico).compress(type='c')
        ncmax_comp = Graph(sample_derrico).compress(type='nc-max')
        nced__comp = Graph(sample_derrico).compress(type='nc-ed')








