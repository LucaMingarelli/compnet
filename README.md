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

Given a dataframe `el` containing a network's edge list,
start by constructing the *graph* representation $G$ via the class `compnet.Graph`:
```python
import pandas as pd
import compnet as cn

el = pd.DataFrame([['A','B', 10],
                   ['B','C', 15],
                   ['B','A', 5],
                   ],
                  columns=['SOURCE', 'TARGET' ,'AMOUNT'])
g = cn.Graph(el)
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
g = cn.Graph(el, source='bank', target='counterpart', amount='notional')
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

Denoting by $A$ the weighted adjacency matrix of the network with elements $A_{ij}$, 
the *gross*, *compressed*, and *excess* market sizes are respectively defined as

$$
GMS = \sum_{i}\sum_{j} A_{ij}
$$

$$
CMS = \frac{1}{2}\sum_i\left|\sum_j \left(A_{ij} - A_{ji}\right) \right|
$$

$$
EMS = GMS - CMS
$$

Notice in particular that $\sum_j \left(A_{ij} - A_{ji}\right)$ represents the net position of node $i$.

----

At this point, it is possible to run a compression algorithm on `g` via the method `Graph.compress`.
For any two graphs one can further compute the **compression efficiency**

$$CE = 1 - \frac{EMS_2}{EMS_1} $$

with $EMS_j$ the *excess market size* of graph $j$.
Moreover, the **compression ratio of order p** for two adjacency matrices $A$ and $A^c$ is defined as

$$CR_p(A, A^c) = \frac{2}{N(N-1)}\frac{||L(A^c, N)||_p}{||L(A, N)||_p} $$

with $N$ the number of nodes and $||L(A, N)||_p$ the $p$-norm of xxx

$$||L(A, N)||_p = \left(  \frac{2}{N(N-1)} \sum_i \sum\_{j=i+1} |A\_{ij}|^p \right)^{1/p}$$


and the **compression factor of order p** for two adjacency matrices $A$ and $A^c$ as

$$CF_p(A, A^c) = 1 - CR_p.$$

Four options are currently available: `bilateral`, `c`, `nc-ed`, `nc-max`.

#### Bilateral compression
Bilateral compression compresses only edges between pairs of nodes.
In our example above there exists two edges (trades) in opposite directions
between node `A` and node `B`, which can be bilaterally compressed.

Running
```python
g_bc = g.compress(type='bilateral')
g_bc
```

returns the following bilaterally compressed graph object
```text
compnet.Graph object:
┌──────────┬──────────┬──────────┐
│ SOURCE   │ TARGET   │   AMOUNT │
├──────────┼──────────┼──────────┤
│ A        │ B        │        5 │
│ B        │ C        │       15 │
└──────────┴──────────┴──────────┘
```
with compression efficiency and factor
```text
Compression Efficiency CE = 0.667
Compression Factor CF(p=2) = 0.718
```





#### Conservative compression
Under conservative compression onlyexisting edges (trades) are reduced or removed. 
No new edge is added.

The resulting conservatively compressed graph is always a sub-graph of the original graph.
Moreover, the resulting conservatively compressed graph is always a directed acyclic graph (DAG).

The conservatively compressed graph can be obtained as 
```python
g_cc = g.compress(type='c')
g_cc
```

which in our example above returns
```text
compnet.Graph object:
┌──────────┬──────────┬──────────┐
│ SOURCE   │ TARGET   │   AMOUNT │
├──────────┼──────────┼──────────┤
│ A        │ B        │        5 │
│ B        │ C        │       15 │
└──────────┴──────────┴──────────┘
```
with compression efficiency and factor
```text
Compression Efficiency CE = 0.667
Compression Factor CF(p=2) = 0.718
```


#### Non-conservative ED compression
...

#### Maximal non-conservative compression
...


The non-conservative maximally compressed graph can be obtained as 
```python
g_cc = g.compress(type='nc-max')
g_cc
```

which in our example above returns
```text
compnet.Graph object:
┌──────────┬──────────┬──────────┐
│ SOURCE   │ TARGET   │   AMOUNT │
├──────────┼──────────┼──────────┤
│ B        │ C        │       10 │
│ A        │ C        │        5 │
└──────────┴──────────┴──────────┘
```
with compression efficiency and factor
```text
Compression Efficiency CE = 1.0
Compression Factor CF(p=2) = 0.801
```




## Grouping along additional dimensions
If an additional dimension exists...





















# Authors
Luca Mingarelli, 2022

[![Python](https://img.shields.io/static/v1?label=made%20with&message=Python&color=blue&style=for-the-badge&logo=Python&logoColor=white)](#)
