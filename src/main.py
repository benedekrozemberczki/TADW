"""Running TADW."""

from tadw import DenseTADW, SparseTADW
from helpers import parameter_parser, read_graph, read_features
from helpers import read_sparse_features, tab_printer

def learn_model(args):
    """
    Method to create adjacency matrix powers, read features, and learn embedding.
    :param args: Arguments object.
    """
    A = read_graph(args.edge_path, args.order)
    if args.features == "dense":
        X = read_features(args.feature_path)
        model = DenseTADW(A, X, args)
    elif args.features == "sparse":
        X = read_sparse_features(args.feature_path)
        model = SparseTADW(A, X, args)
    model.optimize()
    model.save_embedding()

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    learn_model(args)
