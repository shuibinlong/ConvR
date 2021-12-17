# ConvR
The pytorch implementation of ConvR model from ACL 2019 paper *Adaptive Convolution for Multi-Relational Learning*

Paper: [Adaptive Convolution for Multi-Relational Learning](https://aclanthology.org/N19-1103.pdf)

Note: This is an unofficial reproduction version of the model.

## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch](https://github.com/pytorch/pytorch) using [Anaconda](https://www.continuum.io/downloads).
2. Install the requirements `pip install -r requirements.txt`
3. You can now run the model

## Running a model

Parameters are configured in `configs/ConvR.json`, all the hyperparameters in the configuration file come from the paper.

Start training command:
```
$ python main.py -c configs/ConvR.json
```

## Citation

```
@inproceedings{jiang-etal-2019-adaptive,
    title = "Adaptive Convolution for Multi-Relational Learning",
    author = "Jiang, Xiaotian  and
      Wang, Quan  and
      Wang, Bin",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1103",
    doi = "10.18653/v1/N19-1103",
    pages = "978--987",
    abstract = "We consider the problem of learning distributed representations for entities and relations of multi-relational data so as to predict missing links therein. Convolutional neural networks have recently shown their superiority for this problem, bringing increased model expressiveness while remaining parameter efficient. Despite the success, previous convolution designs fail to model full interactions between input entities and relations, which potentially limits the performance of link prediction. In this work we introduce ConvR, an adaptive convolutional network designed to maximize entity-relation interactions in a convolutional fashion. ConvR adaptively constructs convolution filters from relation representations, and applies these filters across entity representations to generate convolutional features. As such, ConvR enables rich interactions between entity and relation representations at diverse regions, and all the convolutional features generated will be able to capture such interactions. We evaluate ConvR on multiple benchmark datasets. Experimental results show that: (1) ConvR performs substantially better than competitive baselines in almost all the metrics and on all the datasets; (2) Compared with state-of-the-art convolutional models, ConvR is not only more effective but also more efficient. It offers a 7{\%} increase in MRR and a 6{\%} increase in Hits@10, while saving 12{\%} in parameter storage.",
}
```
