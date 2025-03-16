# <img src="compnet/res/icons/Network_Compression.png" width="120px"/> *compnet* — Compression for Market Network data 

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/LucaMingarelli/compnet/tree/main.svg?style=svg&circle-token=5c008782a97bdc48aa09b6d25d815a563d572595)](https://dl.circleci.com/status-badge/redirect/gh/LucaMingarelli/compnet/tree/main)
[![version](https://img.shields.io/badge/version-0.7.0-success.svg)](#)
[![PyPI Latest Release](https://img.shields.io/pypi/v/compnet.svg)](https://pypi.org/project/compnet/)
[![Downloads](https://static.pepy.tech/badge/compnet)](https://pepy.tech/project/compnet)
[![License](https://img.shields.io/pypi/l/compnet.svg)](https://github.com/LucaMingarelli/compnet/blob/master/LICENSE.md)
<a href="https://www.buymeacoffee.com/lucamingarelli" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-yellow.png" alt="Buy Me A Coffee" style="height: 30px !important;width: 109px !important;" ></a>

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

This data is also accessible as a `pandas.Series` via the attribute `g.properties`.

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

The net position of each node are also accessible as
`g.net_flow`, which returns

```text
A    -5.0
B   -10.0
C    15.0
```
Similarly, the gross amount for each node can be accessed as
`g.gross_flow`, which returns

```text
         OUT  IN  GROSS_TOTAL
ENTITY                       
A       10.0   5         15.0
B       20.0  10         30.0
C        0.0  15         15.0
```

----

At this point, it is possible to run a compression algorithm on `g` via the method `Graph.compress`.
For any two graphs one can further compute the **compression efficiency**

$$CE = 1 - \frac{EMS_2}{EMS_1} $$

with $EMS_j$ the *excess market size* of graph $j$.
Moreover, the **compression ratio of order p** for two adjacency matrices $A$ and $A^c$ is defined as

$$CR_p(A, A^c) = \frac{||L(A^c, N)||_p}{||L(A, N)||_p} $$

with $N$ the number of nodes and $||L(A, N)||_p$ the $p$-norm of the average absolute weight:

$$||L(A, N)||_p = \left(  \frac{1}{N(N-1)} \sum\_{i\ne j} |A\_{ij}|^p \right)^{1/p}$$


Notice that $L(A, N)=\frac{1}{N(N-1)} \sum\_{i\ne j} |A\_{ij}|$ 
is a measure of the overall connectivity of 
the network: 
it quantifies, on average, 
how *strongly* nodes are connected.
If considering an unweighted network 
(i.e. $A_{ij}\in \\{0,1\\}$),
then $L(A, N)$ corresponds to 
the density of the network, 
that is the fraction of possible links that are actually present.
In the case of weighted networks instead, 
$L(A, N)$ represents the average strength 
or intensity of the connections, 
taking into account the magnitude of each weight.


The **compression factor of order p** 
for two adjacency matrices $A$ and $A^c$ is then defined as

$$CF_p(A, A^c) = 1 - CR_p.$$

Four options for compression are currently available: `bilateral`, `c`, `nc-ed`, `nc-max`.


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
Under conservative compression only existing edges (trades) are reduced or removed. 
No new edge is added.

The resulting conservatively compressed graph is always a sub-graph of the original graph.
Moreover, the resulting conservatively compressed graph is always a directed acyclic graph (DAG), 
since all loops within the graph are removed.


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


#### Non-conservative Equally-Distributed compression
Under non-conservative compression previously non-existent edges may be introduced. 

Non-conservative compression allows to achieve an after-compression 
GMS equal to the CMS, thus removing all excess intermediation amounts in the network.

However, there is no unique solution to this problem.

The equally-distributed approach provides the simplest possible solution, 
by distributing flows from nodes with negative net flows 
to nodes with positive net flows on a pro-rata basis, 
that is distributing flows equally. 

The non-conservatively equally-distributed compressed graph 
can be obtained as 
```python
g_cc = g.compress(type='nc-ed')
g_nced
```

which in our example above returns
```text
compnet.Graph object:
┌────────┬───────────────┬────────────┐
│ SOURCE │ TARGET        │   AMOUNT   │
├────────┼───────────────┼────────────┤
│ A      │ C             │          5 │
│ B      │ C             │         10 │
└────────┴───────────────┴────────────┘
```
with compression efficiency and factor
```text
Compression Efficiency CE = 1.0
Compression Factor CF(p=2) = 0.402
```


#### Maximal non-conservative compression

An alternative solution to the non-conservative compression problem is 
achieved by minimising the number of links and maximising their concentration.

This solution is in a sense diametrically opposed to the previous
equally-distributed solution.
While both solutions achieve a post-compression GMS equal to the network's CMS, 
the present maximal non-conservative approach achieves in general
a lower compression factor at any order $p\ge 1$.


The non-conservative maximally compressed graph can be obtained as 
```python
g_ncmax = g.compress(type='nc-max')
g_ncmax
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
Compression Factor CF(p=2) = 0.402
```

Although in this case both the equally-distributed and maximal 
compressions yield the same result, this needs not be the case in general.

Considering for instance the network
```python
el = pd.DataFrame([['A','B', 4],
                   ['B','C', 3],
                   ['C','D', 5],
                   ],
                  columns=['SOURCE', 'TARGET' ,'AMOUNT'])
g = cn.Graph(el)
```
one finds the following equally-distributed compressed network
```text
compnet.Graph object:
┌──────────┬──────────┬──────────┐
│ SOURCE   │ TARGET   │   AMOUNT │
├──────────┼──────────┼──────────┤
│ A        │ B        │ 0.666667 │
│ A        │ D        │ 3.33333  │
│ C        │ B        │ 0.333333 │
│ C        │ D        │ 1.66667  │
└──────────┴──────────┴──────────┘
```
with compression efficiency and factor
```text
Compression Efficiency CE = 1.0
Compression Factor CF(p=2) = 0.463
```
Maximally non-conservative compression yields instead

```text
compnet.Graph object:
┌──────────┬──────────┬──────────┐
│ SOURCE   │ TARGET   │   AMOUNT │
├──────────┼──────────┼──────────┤
│ A        │ D        │        4 │
│ C        │ D        │        1 │
│ C        │ B        │        1 │
└──────────┴──────────┴──────────┘

```
with compression efficiency and factor
```text
Compression Efficiency CE = 1.0
Compression Factor CF(p=2) = 0.4
```


## Grouping along additional dimensions

When considering networks with additional dimensions or layers, 
such as time, collateral type, market sub-segments, etc. 
`compnet.Graph` allows to describe and perform compression on each such layer independently
via the parameter `grouper`, taking either a single field (as `str`) 
or multiple ones (as `list`) if grouping along multiple dimensions is necessary.

For instance, one might consider the market described by the tensor $A^\tau$
with elements $A^\tau_{ij}$, where $\tau$ indexes time with daily frequency.
This can be represented for example by the following edge list:
```python

el = pd.DataFrame([['A','B', 15, '2025-02-10'],
                   ['B','C', 15, '2025-02-10'],
                   ['B','A',  5, '2025-02-10'],
                   ['A','B', 20, '2025-02-11'],
                   ['B','C', 15, '2025-02-11'],
                   ['B','A',  6, '2025-02-11'],
                   ['A','B', 25, '2025-02-12'],
                   ['B','C', 15, '2025-02-12'],
                   ['B','A',  7, '2025-02-12'],
                   ],
                  columns=['lender', 'borrower' ,'amount', 'date'])
```

Creating the graph object as usual via
```python
g = cn.Graph(el, source='lender', target='borrower', amount='amount', grouper='date')
```

and requesting its description as

```python
g.describe()
```

one is presented with the following output describing 
the time evolution of the network's gross, compressed, and excess sizes:

```text
┌────────────┬──────────────┬───────────────────┬───────────────┐
│ date       │   Gross size │   Compressed size │   Excess size │
├────────────┼──────────────┼───────────────────┼───────────────┤
│ 2025-02-10 │           35 │                15 │            20 │
│ 2025-02-11 │           41 │                15 │            26 │
│ 2025-02-12 │           47 │                18 │            29 │
└────────────┴──────────────┴───────────────────┴───────────────┘
```

As before, this data is also accessible as a `pandas.DataFrame` via the attribute `g.properties`.

## Central clearing

The method `centrally_clear` allows to clear all positions 
through a common counterparty specified via the parameter `ccp_name`,
introducing a new entity should `ccp_name` not be part of the list of entities already. 



For instance

```python
el = pd.DataFrame([['A','B', 4],
                   ['B','C', 3],
                   ['C','D', 5],
                   ],
                  columns=['SOURCE', 'TARGET' ,'AMOUNT'])
cn.Graph(el).centrally_clear()
```

returns 

```
┌──────────┬──────────┬──────────┐
│ SOURCE   │ TARGET   │   AMOUNT │
├──────────┼──────────┼──────────┤
│ CCP      │ B        │        4 │
│ CCP      │ C        │        3 │
│ CCP      │ D        │        5 │
│ A        │ CCP      │        4 │
│ B        │ CCP      │        3 │
│ C        │ CCP      │        5 │
└──────────┴──────────┴──────────┘
```

By definition, the central clearing operation doubles the Graph's GMS, 
while CMS is invariant.

It is also possible to return directly the bilaterally compressed graph as
```python
cn.Graph(el).centrally_clear(net=True)
```

which yields

```
┌──────────┬──────────┬──────────┐
│ SOURCE   │ TARGET   │   AMOUNT │
├──────────┼──────────┼──────────┤
│ A        │ CCP      │        4 │
│ C        │ CCP      │        2 │
│ CCP      │ B        │        1 │
│ CCP      │ D        │        5 │
└──────────┴──────────┴──────────┘
```

Any grouper specified on `Graph` is automatically accounted for.













# Author
Luca Mingarelli, 2022

[![Python](https://img.shields.io/static/v1?label=made%20with&message=Python&color=blue&style=for-the-badge&logo=Python&logoColor=white)](#)
