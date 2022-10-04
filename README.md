<div align="left">
<img width="450" alt="screener" src="https://user-images.githubusercontent.com/23725126/143378891-79b49584-700c-45f2-994c-1c077c7418c3.png"><br><br>
</div>
Released September, 2021

-----------------

[![Documentation Status](https://readthedocs.org/projects/fast-image-retrieval/badge/?version=latest)](https://fast-image-retrieval.readthedocs.io/en/latest/?badge=latest)

Documentation: https://fast-image-retrieval.readthedocs.io/en/latest/

## Introduction
Fast Image Retrieval (FIRe) is an open source image retrieval project release by Center of Image and Signal Processing Lab (CISiP Lab), Universiti Malaya. This framework implements most of the major binary hashing methods, together with different popular backbone networks and public datasets.

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
|Method|Config Template|Loss Name|64bit ImageNet AlexNet (mAP@1K)|
|---|---|---|---|
|[ADSH](https://ieeexplore.ieee.org/abstract/document/8825992)|[adsh.yaml](configs/templates/adsh.yaml)|adsh|0.645|
|[BiHalf](https://arxiv.org/abs/2012.12334)|[bihalf-supervised.yaml](configs/templates/bihalf-supervised.yaml)|bihalf-supervised|0.684|
|Cross Entropy|[ce.yaml](configs/templates/ce.yaml)|ce|0.434|
|[CSQ](https://arxiv.org/abs/1908.00347)|[csq.yaml](configs/templates/csq.yaml)|csq|0.686|
|[DFH](https://arxiv.org/abs/1909.00206)|[dfh.yaml](configs/templates/dfh.yaml)|dfh|0.689|
|[DPN](https://www.ijcai.org/proceedings/2020/115)|[dpn.yaml](configs/templates/dpn.yaml)|dpn|0.692|
|[DPSH](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)|[dpsh.yaml](configs/templates/dpsh.yaml)|dpsh|0.599|
|[DTSH](https://arxiv.org/abs/1612.03900)|[dtsh.yaml](configs/templates/dtsh.yaml)|dtsh|0.608|
|[GreedyHash](https://papers.nips.cc/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html)|[greedyhash.yaml](configs/templates/greedyhash.yaml)|greedyhash|0.667|
|[HashNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)|[hashnet.yaml](configs/templates/hashnet.yaml)|hashnet|0.588|
|[JMLH](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CEFRL/Shen_Embarrassingly_Simple_Binary_Representation_Learning_ICCVW_2019_paper.pdf)|[jmlh.yaml](configs/templates/jmlh.yaml)|jmlh|0.664|
|[OrthoCos(OrthoHash)](https://arxiv.org/abs/2109.14449)|[orthocos.yaml](configs/templates/orthocos.yaml)|orthocos|**0.701**|
|[OrthoArc(OrthoHash)](https://arxiv.org/abs/2109.14449)|[orthoarc.yaml](configs/templates/orthoarc.yaml)|orthoarc|0.698|
|SDH-C|[sdhc.yaml](configs/templates/sdhc.yaml)|sdhc|0.639|
#### Unsupervised
|Method|Config Template|Loss Name|64bit ImageNet AlexNet (mAP@1K)|
|---|---|---|---|
|[BiHalf](https://arxiv.org/abs/2012.12334)|[bihalf.yaml](configs/templates/bihalf.yaml)|bihalf|0.403|
|[CIBHash](https://www.ijcai.org/proceedings/2021/0133.pdf)|[cibhash.yaml](configs/templates/cibhash.yaml)|cibhash|0.322|0.686401
|[GreedyHash](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)|[greedyhash-unsupervised.yaml](configs/templates/greedyhash-unsupervised.yaml)|greedyhash-unsupervised|0.407|
|[SSDH](https://ieeexplore.ieee.org/document/8101524)|[ssdh.yaml](configs/templates/ssdh.yaml)|ssdh|0.146|
|[TBH](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Auto-Encoding_Twin-Bottleneck_Hashing_CVPR_2020_paper.pdf)|[tbh.yaml](configs/templates/tbh.yaml)|tbh|0.324|
#### Shallow (Non-Deep learning methods)
|Method|Config Template|Loss Name|64bit ImageNet AlexNet (mAP@1K)|
|---|---|---|---|
|[IMH](https://ieeexplore.ieee.org/abstract/document/7047876)|[imh.yaml](configs/templates/imh.yaml)|imh|0.467|
|[ITQ](https://slazebni.cs.illinois.edu/publications/ITQ.pdf)|[itq.yaml](configs/templates/itq.yaml)|itq|0.402|
|[LsH](https://dl.acm.org/doi/10.1145/276698.276876)|[lsh.yaml](configs/templates/lsh.yaml)|lsh|0.206|
|PCAHash|[pca.yaml](configs/templates/pca.yaml)|pca|0.405|
|[SH](https://papers.nips.cc/paper/2008/hash/d58072be2820e8682c0a27c0518e805e-Abstract.html)|[sh.yaml](configs/templates/sh.yaml)|sh|0.350|

```{warning}
Shallow methods only works with descriptor datasets. We will upload the descriptor datasets and 
```

### Datasets
|Dataset|Name in framework|
|---|---|
|ImageNet100|imagenet100|
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
