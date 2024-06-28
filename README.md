# <img src="compnet/res/icons/Network_Compression.png" width="120px"/> *compnet* — Compression for Market Network data 

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/LucaMingarelli/compnet/tree/main.svg?style=svg&circle-token=5c008782a97bdc48aa09b6d25d815a563d572595)](https://dl.circleci.com/status-badge/redirect/gh/LucaMingarelli/compnet/tree/main)
[![version](https://img.shields.io/badge/version-0.0.1-success.svg)](#)
[![PyPI Latest Release](https://img.shields.io/pypi/v/compnet.svg)](https://pypi.org/project/compnet/)
[![License](https://img.shields.io/pypi/l/compnet.svg)](https://github.com/LucaMingarelli/compnet/blob/master/LICENSE.txt)

[//]: # ([![Downloads]&#40;https://static.pepy.tech/personalized-badge/compnet?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads&#41;]&#40;https://pepy.tech/project/compnet&#41;)


# About

***compnet*** is a package for market compression of network data.

It is based on xxx.


# How to get started

Given the network with edge list `el`,
start by constructing the *graph* representation $G$ via the class `compnet.Graph`:
```python
import pandas as pd
import compnet

el = pd.DataFrame([['A','B', 10],
                   ['B','C', 15],
                   ['B','A', 5],
                   ],
                  columns=['SOURCE', 'TARGET' ,'AMOUNT'])
g = compnet.Graph(el)
```

If the dataframe does not contain columns named `'SOURCE'`, `'TARGET'`, and `'AMOUNT'`,
the corresponding column names should be passed as well to `compnet.Graph` 
via the parameters `source`, `target`, and `amount`.

For example:
```python

el = pd.DataFrame([['A','B', 10],
                   ['B','C', 15],
                   ['B','A', 5],
                   ],
                  columns=['bank', 'counterpart' ,'notional'])
g = compnet.Graph(el, source='bank', target='counterpart', amount='notional')
```

Once the graph object `g` is created, it is possible to quickly inspect its properties as
```python
g.describe()
```
which returns the gross, compressed, and excess market sizes of the graph
```text
┌─────────────────┬──────────┐
│                 │   AMOUNT │
├─────────────────┼──────────┤
│ Gross size      │       30 │
│ Compressed size │       15 │
│ Excess size     │       15 │
└─────────────────┴──────────┘
```







## Grouping along additional dimensions
If an additional dimension exists...





















# Authors
Luca Mingarelli, 2022

[![Python](https://img.shields.io/static/v1?label=made%20with&message=Python&color=blue&style=for-the-badge&logo=Python&logoColor=white)](#)
