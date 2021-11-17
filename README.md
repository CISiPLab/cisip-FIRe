# Fast Image Retrieval
[![Documentation Status](https://readthedocs.org/projects/fast-image-retrieval/badge/?version=latest)](https://fast-image-retrieval.readthedocs.io/en/latest/?badge=latest)

Documentation: https://fast-image-retrieval.readthedocs.io/en/latest/

## Introduction
Fast Image Retrieval is an open source image retrieval project release by Center of Image and Signal Processing Lab (CISiP Lab), Universiti Malaya. This framework implements most of the major binary hashing methods, together with different popular backbone networks and public datasets.

### Major features
- **One for All**

    Herein, we unified (i) various binary hashing methods, (ii) different backbone, and (iii) multiple datasets under a single framework to ease the research and benchmarking in this domain. It supports popular binary hashing methods, e.g. [HashNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf), [GreedyHash](https://papers.nips.cc/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html), [DPN](https://www.ijcai.org/proceedings/2020/115), [OrthoHash](https://arxiv.org/abs/2109.14449), etc.
- **Modularity**

    We break the framework into parts so that one can easily implement their own method by joining up the components.

## License
This project is released under [BSD 3-Clause License](https://github.com/CISiPLab/fast-image-retrieval/blob/main/LICENSE).

## Changelog
Please refer to [Changelog](https://fast-image-retrieval.readthedocs.io/en/latest/misc.html#changelog) for more detail.
## Implemented method/backbone/datasets
[//]: <> (This is copied from docs/implementation.md)
### Backbone
1. Alexnet
2. VGG{16}
3. ResNet{18,34,50,101,152}

### Loss (Method)
#### Supervised
|Method|Config Template|Loss Name|
|---|---|---|
|[ADSH](https://arxiv.org/abs/1707.08325)|adsh.yaml|adsh|
|[BiHalf](https://arxiv.org/abs/2012.12334)|bihalf-supervised.yaml|bihalf-supervised|
|Cross Entropy|ce.yaml|ce|
|[CSQ](https://arxiv.org/abs/1908.00347)|csq.yaml|csq|
|[DBDH](https://www.sciencedirect.com/science/article/abs/pii/S0925231220306032)|dbdh.yaml|dbdh|
|[DFH](https://arxiv.org/abs/1909.00206)|dfh.yaml|dfh|
|[DPN](https://www.ijcai.org/proceedings/2020/115)|dpn.yaml|dpn|
|[DPSH](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)|dpsh.yaml|dpsh|
|[DTSH](https://arxiv.org/abs/1612.03900)|dtsh.yaml|dtsh|
|[GreedyHash](https://papers.nips.cc/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html)|greedyhash.yaml|greedyhash|
|[HashNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)|hashnet.yaml|hashnet|
|JMLH|jmlh.yaml|jmlh|
|MIHash|mihash.yaml|mihash|
|[OrthoCos(OrthoHash)](https://arxiv.org/abs/2109.14449)|orthocos.yaml|orthocos|
|[OrthoArc(OrthoHash)](https://arxiv.org/abs/2109.14449)|orthoarc.yaml|orthoarc|
|SDH|sdh.yaml|sdh|
|SDH-C|sdhc.yaml|sdhc|
#### Unsupervised
|Method|Config Template|Loss Name|
|---|---|---|
|BiHalf|bihalf.yaml|bihalf|
|CIBHash|cibhash.yaml|cibhash|
|[GreedyHash](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)|greedyhash-unsupervised.yaml|greedyhash-unsupervised|
|SSDH|ssdh.yaml|ssdh|
|TBH|tbh.yaml|tbh|
#### Shallow (Non-Deep learning methods)
|Method|Config Template|Loss Name|
|---|---|---|
|ITQ|itq.yaml|itq|
|LsH|lsh.yaml|lsh|
|PCAHash|pca.yaml|pca|
|SH|sh.yaml|sh|



### Datasets
|Dataset|Name in framework|
|---|---|
|ImageNet100|imagenet|
|NUS-WIDE|nuswide|
|MS-COCO|coco|
|MIRFLICKR/Flickr25k|mirflickr|
|Stanford Online Product|sop|
|Cars dataset|cars|
|CIFAR10|cifar10|

## Installation
Please head up to [Get Started Docs](https://fast-image-retrieval.readthedocs.io/en/latest/get_started.html) for guides on setup conda environment and installation.

## Tutorials
Please head up to [Tutorials Docs](https://fast-image-retrieval.readthedocs.io/en/latest/tutorials.html) for guidance.

## Reference

If you find this framework useful in your research, please consider cite this project.

```
@inproceedings{dpn2020,
  title={Deep Polarized Network for Supervised Learning of Accurate Binary Hashing Codes.},
  author={Fan, Lixin and Ng, Kam Woh and Ju, Ce and Zhang, Tianyu and Chan, Chee Seng},
  booktitle={IJCAI},
  pages={825--831},
  year={2020}
}

@inproceedings{orthohash2021,
  title={One Loss for All: Deep Hashing with a Single Cosine Similarity based Learning Objective},
  author={Hoe, Jiun Tian and Ng, Kam Woh and Zhang, Tianyu and Chan, Chee Seng and Song, Yi-Zhe and Xiang, Tao},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## Contributing
We welcome the contributions to improve this project. Please file your suggestions/issues by creating new issues or send us a pull request for your new changes/improvement/features/fixes.
