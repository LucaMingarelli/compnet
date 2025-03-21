"""  Created on 10/10/2022::
------------- sample0.py -------------

**Authors**: L. Mingarelli
"""

import pandas as pd

sample0 = pd.DataFrame(
    [['A','B', 5],
     ['B','C', 15]
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT']
)

sample_bilateral = pd.DataFrame(
    [['A','B', 10],
     ['B','C', 15],
     ['B','A', 5],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT']
)

sample_purebilateral = pd.DataFrame(
    [['A','B', 10],
     ['A','D', 15],
     ['C','D', 5],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT']
)



sample_noncons1 = pd.DataFrame(
    [['A','B', 6],
     ['B','C', 5],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])
sample_noncons1_compressed = pd.DataFrame(
    [['A','B', 1],
     ['A','C', 5],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])


sample_noncons2 = pd.DataFrame(
    [['A','B', 10],
     ['B','C', 20],
     ['C','A', 5],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])
sample_noncons2_compressed = pd.DataFrame(
    [['A','C', 5],
     ['B','C', 10],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])

# _get_nodes_net_flow(sample_noncons2)
# _get_nodes_net_flow(compressed_network_non_conservative(sample_noncons2))

sample_noncons3 = pd.DataFrame(
    [['A','B', 5],
     ['B','C', 5],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])
sample_noncons3_compressed = pd.DataFrame(
    [['A','C', 5],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])


sample_noncons4 = pd.DataFrame(
    [['A','B', 4],
     ['B','C', 3],
     ['C','D', 5],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])

# compressed_network_non_conservative(sample_noncons4)
# _get_nodes_net_flow(sample_noncons4)
# _get_nodes_net_flow(sample_noncons4_compressed)

# THESE TWO BELOW ARE EQUIVALENTLY COMPRESSED
# TODO RESEARCH: Find all possible compressions -> which compression strategy does it minimise systemic risk?
# _get_nodes_net_flow(compressed_network_non_conservative(sample_noncons4))
sample_noncons4_compressed = pd.DataFrame(
    [['A','B', 1],
     ['A','D', 3],
     ['C','D', 2],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])


# compressed_network_non_conservative(df=sample_noncons2)




sample_cycle = pd.DataFrame(
    [['A','B', 1],
     ['A','C', 3],
     ['C','D', 2],
     ['D','A', 2],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])

sample_nested_cycle1 = pd.DataFrame(
    [['A','B', 1],
     ['B','C', 3],
     ['C','D', 2],
     ['D','A', 2],
     ['C','A', 4],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])

sample_nested_cycle2 = pd.DataFrame(
    [['A','B', 1],
     ['B','C', 3],
     ['B','D', 2],
     ['D','A', 2],
     ['C','A', 4],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])

sample_nested_cycle3 = pd.DataFrame(
    [['A','B', 2],
     ['B','C', 3],
     ['C','A', 2],
     ['D','A', 2],
     ['B','E', 4],
     ['E','D', 4],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])

sample_nested_cycle4 = pd.DataFrame(
    [['A','B', 1],
     ['B','F', 3],
     ['F','G', 2],
     ['G','E', 2],
     ['E','F', 4],
     ['G','C', 2],
     ['C','D', 1],
     ['D','A', 3],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])

sample_entangled = pd.DataFrame(
    [['A','B', 5],
     ['B','C', 10],
     ['C','A', 20],
     ['C','D', 10],
     ['D','B', 3],
     ],
    columns=['SOURCE', 'TARGET' ,'AMOUNT'])




sample_onegrouper = pd.DataFrame([['A', 'B', 15, '2025-02-10'],
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

sample_twogrouper = pd.DataFrame([['A', 'B', 10, '2025-02-10', 'ISIN_A'],
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

sample_warning = pd.DataFrame([['A', 'B', 15, '2025-02-10'],
                                    ['A', 'B', 20, '2025-02-11'],
                                    ['B', 'C', 15, '2025-02-11'],
                                    ['B', 'A', 6, '2025-02-11'],
                                    ['A', 'B', 25, '2025-02-12'],
                                    ['B', 'C', 15, '2025-02-12'],
                                    ['B', 'A', 7, '2025-02-12'],
                                    ],
                                   columns=['lender', 'borrower', 'amount', 'date'])

