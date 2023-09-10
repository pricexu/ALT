# Homophilic graphs
python main.py --dataset Cora --alt adj_norm
python main.py --dataset Citeseer --alt adj_norm --MLPA 0.01
python main.py --dataset Pubmed --alt adj_norm --MLPA 0
python main.py --dataset DBLP --alt adj_norm
python main.py --dataset Computers --alt adj
python main.py --dataset Photo --alt adj_norm
python main.py --dataset CS --alt adj_norm
python main.py --dataset Physics --alt adj_norm

# Heterophilic graphs
python main.py --dataset Chameleon --alt adj_norm --MLPA 0.05
python main.py --dataset Squirrel --alt adj_norm --MLPA 0.05
python main.py --dataset Texas --alt adj --amp 0.6 --wd 5e-4
python main.py --dataset Wisconsin --alt adj --wd 5e-4 --MLPA 0 --lr 0.1
python main.py --dataset Cornell --alt adj --lr 0.08 --wd 7e-4 --MLPA 0.1 --seed 9
python main.py --dataset Film --alt adj
python main.py --dataset Cornell5 --alt adj_norm --MLPA 0.05 --wd 0
python main.py --dataset Penn94 --alt adj_norm --MLPA 0.05 --wd 0
