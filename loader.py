import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CitationFull, WikipediaNetwork, Actor, WebKB, LINKXDataset

""" folder for downloaded datasets """
if not os.path.isdir('dataset'):
    os.makedirs('dataset')

def loader(name):
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='dataset', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root='dataset', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root='dataset', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['DBLP']:
        dataset = CitationFull(root='dataset', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['Chameleon', 'Squirrel']:
        preProcDs = WikipediaNetwork(
            root='dataset', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root='dataset', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index

    elif name in ['Film']:
        dataset = Actor(root='dataset', transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['Texas', 'Cornell', 'Wisconsin']:
        dataset = WebKB(root='dataset', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['Penn94', 'Cornell5']:
        dataset = LINKXDataset(root='dataset', name=name)
        data = dataset[0]
    
    return data