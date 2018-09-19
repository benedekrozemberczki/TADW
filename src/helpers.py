import argparse
import networkx as nx
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
from texttable import Texttable

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Wiki Giraffes.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """

    parser = argparse.ArgumentParser(description = "Run TADW.")


    parser.add_argument('--edge-path',
                        nargs = '?',
                        default = './input/giraffe_edges.csv',
	                help = 'Input edges.')

    parser.add_argument('--feature-path',
                        nargs = '?',
                        default = './input/giraffe_features.csv',
	                help = 'Input features.')

    parser.add_argument('--output-path',
                        nargs = '?',
                        default = './output/giraffe_tadw.csv',
	                help = 'Output embedding.')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 32,
	                help = 'Number of dimensions. Default is 32.')

    parser.add_argument('--order',
                        type = int,
                        default = 2,
	                help = 'Target matrix approximation order. Default is 2.')

    parser.add_argument('--iterations',
                        type = int,
                        default = 20,
	                help = 'Number of gradient descent iterations. Default is 20.')

    parser.add_argument('--lambd',
                        type = float,
                        default = 1000.0,
	                help = 'Regularization term coefficient. Default is 1000.')

    parser.add_argument('--alpha',
                        type = float,
                        default = 10**-8,
	                help = 'Learning rate. Default is 10^-8.')


    parser.add_argument('--lower-control',
                        type = float,
                        default = 10**-15,
	                help = 'Overflow control. Default is 10**-15.')
    
    return parser.parse_args()

def normalize_adjacency(graph):
    """
    """
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    A = sparse.csr_matrix(nx.adjacency_matrix(graph),dtype=np.float32)
    degs = sparse.csr_matrix(sparse.coo_matrix((degs,(ind,ind)),shape=A.shape,dtype=np.float32))
    A = A.dot(degs)
    return A

def read_graph(edge_path, order):
    """
    """
    print("Target matrix creation started.")
    graph = nx.from_edgelist(pd.read_csv(edge_path).values.tolist())
    A = normalize_adjacency(graph)
    if order > 1:
        powered_A, out_A = A, A
        
        for power in tqdm(range(order-1)):
            powered_A = powered_A.dot(A)
            out_A = out_A + powered_A
    else:
        out_A = A
    print("Factorization started.")
    return out_A

def read_features(feature_path):
    """
    """
    features = pd.read_csv(feature_path)
    X = np.array(features)[:,1:]
    return X

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)

    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),v] for k,v in args.iteritems()])
    print t.draw()
