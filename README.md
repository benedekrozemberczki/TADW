TADW
============================================
<p align="justify">
An implementation of "Network Representation Learning with Rich Text Information". Text Attribtued Deep Walk (TADW) is a node embedding algorithm which learns an embedding of nodes and fuses the node representations with node attributes. The procedure places nodes in an abstract feature space where information about a fixed order procimity is preserved and attributes of neighbours within the proximity are also part of the representation. TADW learns the joint feature-proximal representations using regularized non-negative matrix factorization. In our implementation we assumed that the proximity matrix used in the approximation is sparse, hence the solution runtime can be linear in the number of nodes for low proximity. For a large proximity order value (which is larger than the graph diameter) the runtime is quadratic. The model can assume that the node-feature matrix is sparse or that it is dense, which changes the runtime considerably.
  
<div style="text-align:center"><img src ="fscnmf.png" ,width=720/></div>

This repository provides an implementation for TADW as described in the paper:
> Network Representation Learning with Rich Text Information.
> Yang Cheng, Liu Zhiyuan, Zhao Deli, Sun Maosong and Chang Edward Y
> IJCAI, 2015.
> https://www.ijcai.org/Proceedings/15/Papers/299.pdf


### Requirements

The codebase is implemented in Python 2.7. package versions used for development are just below.
```
networkx          1.11
tqdm              4.19.5
numpy             1.13.3
pandas            0.20.3
texttable         1.2.1
scipy             1.1.0
argparse          1.1.0
```

### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. Sample graphs for the `Wikipedia Chameleons` and `Wikipedia Giraffes` are included in the  `input/` directory. 

The feature matrix can be stored two ways:

If the feature matrix is a **sparse binary** one it is stored as a json. Nodes are keys of the json and features are the values. For each node feature column ids are stored as elements of a list. The feature matrix is structured as:

```javascript
{ 0: [0, 1, 38, 1968, 2000, 52727],
  1: [10000, 20, 3],
  2: [],
  ...
  n: [2018, 10000]}
```
If the feature matrix is **dense** it is assumed that it is stored as csv with coma separators. It has a header, the first collumn contains node identifiers and it is sorted by these identifers. It should look like this:

| **NODE ID**| **Feature 1** | **Feature 2** | **Feature 3** | **Feature 4** |
| --- | --- | --- | --- |--- |
| 0 | 0 |0 |1 |1 |
| 1 | 1 |1 |0 |1 |
| 2 | 0 |0 |1 |1 |
| 3 | 1 |1 |1 |1 |
| ... | ... |... |... |... |
| n | 1 |0 |1 |1 |

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --edge-path STR           Input graph path.           Default is `input/chameleon_edges.csv`.
  --feature-path STR        Input Features path.        Default is `input/chameleon_features.json`.
  --output-path STR         Embedding path.             Default is `output/chameleon_tadw.csv`.
```

#### Model options

```
  --dimensions INT         Number of embeding dimensions.                     Default is 32.
  --order INT              Order of adjacency matrix powers.                  Default is 2.
  --iterations INT         Number of gradient descent interations.            Default is 20.
  --alpha FLOAT            Learning rate.                                     Default is 10**-6.
  --lambd FLOAT            Regularization term coefficient.                   Default is 1000.0.  
  --lower-control FLOAT    Overflow control parameter.                        Default is 10**-15.
  --features STR           Structure of the feature matrix.                   Default is `sparse`. 
```

### Examples

The following commands learn a graph embedding and write the embedding to disk. The node representations are ordered by the ID.

Creating a sparse TADW embedding of the default dataset with the default hyperparameter settings. Saving the embedding at the default path.

```
python src/main.py
```
Creating a TADW embedding of the default dataset with 128x2 dimensions and approximation order 1.

```
python src/main.py --dimensions 128 --order 1
```

Creating a TADW  embedding with high regularization.

```
python src/main.py --lambd 2000
```

Creating an embedding of an other dataset with dense features the `Wikipedia Giraffes`. Saving the output in a custom folder.

```
python src/main.py --edge-path input/dog_edges.csv --feature-path input/dog_features.csv --output-path output/dog_fscnmf.csv --features dense
```
