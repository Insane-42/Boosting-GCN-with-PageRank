# Boosting GCN with PageRank 

Pytorch 1.0 with python 3.7.

Course project for *Graph Neural Networks*, PKU

# Benchmark

| dataset       | Citeseea | Cora | Pubmed | 
|---------------|----------|------|--------|
| GCN(official) | 70.3     | 81.5 | 79.0   |
| Our model     | 72.2     | 82.1 | 79.4   |

Hints:
To achieve 82.1% on `Cora` dataset, the model needs 400 epochs. For other datasets, it needs 200 epochs to achieve optimal performance.

It trains relatively slower on `Pubmed` dataset.


# How to run experiment
```
python train.py
```


