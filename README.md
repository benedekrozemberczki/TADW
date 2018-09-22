TADW
============================================
<p align="justify">
An implementation of "Network Representation Learning with Rich Text Information". Text Attribtued Deep Walk (TADW) is a node embedding algorithm which learns an embedding of nodes and fuses the node representations with node attributes. The procedure places nodes in an abstract feature space where information about a fixed order procimity is preserved and attributes of neighbours within the proximity are also part of the representation. TADW learns the joint feature-proximal representations using regularized non-negative matrix factorization. In our implementation we assumed that the proximity matrix used in the approximation is sparse, hence the solution runtime can be linear in the number of nodes for low proximity. For a large proximity order value (which is larger than the graph diameter) the runtime is quadratic. The model can assume that the node-feature matrix is sparse or that it is dense.
  
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

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for the `Wikipedia Giraffes` is included in the  `input/` directory.

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --edge-path STR           Input graph path.           Default is `input/giraffe_edges.csv`.
  --feature-path STR        Input Features path.        Default is `input/giraffe_features.csv`.
  --output-path STR         Embedding path.             Default is `output/giraffe_fscnmf.csv`.
```

#### Model options

```
  --dimensions INT         Number of embeding dimensions.                     Default is 32.
  --order INT              Order of adjacency matrix powers.                  Default is 2.
  --iterations INT         Number of gradient descent interations.            Default is 20.
  --alpha FLOAT            Learning rate.                                     Default is 10**-8.
  --lambd FLOAT            Regularization term coefficient.                   Default is 1000.0.  
  --lower-control FLOAT    Overflow control parameter.                        Default is 10**-15.  
```

### Examples

The following commands learn a graph embedding and write the embedding to disk. The node representations are ordered by the ID.

Creating a TADW embedding of the default dataset with the default hyperparameter settings. Saving the embedding at the default path.

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

Creating an embedding of an other dataset the `Wikipedia Dogs`. Saving the output in a custom folder.

```
python src/main.py --edge-path input/dog_edges.csv --feature-path input/dog_features.csv --output-path output/dog_fscnmf.csv
```
