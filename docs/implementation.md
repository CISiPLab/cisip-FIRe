# Backbone
1. Alexnet
2. VGG{16}
3. ResNet{18,34,50,101,152}

# Loss (Method)
## Supervised
|Method|Config Template|Loss Name|64bit ImageNet AlexNet (mAP@1K)|
|---|---|---|---|
|[ADSH](https://arxiv.org/abs/1707.08325)|adsh.yaml|adsh|0.645|
|[BiHalf](https://arxiv.org/abs/2012.12334)|bihalf-supervised.yaml|bihalf-supervised|0.684|
|Cross Entropy|ce.yaml|ce|0.434|
|[CSQ](https://arxiv.org/abs/1908.00347)|csq.yaml|csq|0.686|
|[DBDH](https://www.sciencedirect.com/science/article/abs/pii/S0925231220306032)|dbdh.yaml|dbdh|Not Implemented|
|[DFH](https://arxiv.org/abs/1909.00206)|dfh.yaml|dfh|0.689|
|[DPN](https://www.ijcai.org/proceedings/2020/115)|dpn.yaml|dpn|0.692|
|[DPSH](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)|dpsh.yaml|dpsh|0.599|
|[DTSH](https://arxiv.org/abs/1612.03900)|dtsh.yaml|dtsh|0.608|
|[GreedyHash](https://papers.nips.cc/paper/2018/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html)|greedyhash.yaml|greedyhash|0.667|
|[HashNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)|hashnet.yml|hashnet|0.588|
|JMLH|jmlh.yaml|jmlh|0.664|
|MIHash|mihash.yaml|mihash|x|
|[OrthoCos(OrthoHash)](https://arxiv.org/abs/2109.14449)|orthocos.yaml|orthocos|0.701182|
|[OrthoArc(OrthoHash)](https://arxiv.org/abs/2109.14449)|orthoarc.yaml|orthoarc|0.697745|
|SDH|sdh.yaml|sdh|x|
|SDH-C|sdhc.yaml|sdhc|0.639|
## Unsupervised
|Method|Config Template|Loss Name|64bit ImageNet AlexNet (mAP@1K)|
|---|---|---|---|
|BiHalf|bihalf.yaml|bihalf|0.403|
|CIBHash|cibhash.yaml|cibhash|0.322|0.686401
|[GreedyHash](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)|greedyhash-unsupervised.yaml|greedyhash-unsupervised|0.407|
|SSDH|ssdh.yaml|ssdh|0.146|
|TBH|tbh.yaml|tbh|0.324|
## Shallow (Non-Deep learning methods)
|Method|Config Template|Loss Name|64bit ImageNet AlexNet (mAP@1K)|
|---|---|---|---|
|ITQ|itq.yaml|itq|0.402|
|LsH|lsh.yaml|lsh|0.206|
|PCAHash|pca.yaml|pca|0.405|
|SH|sh.yaml|sh|0.350|

```{warning}
Shallow methods only works with descriptor datasets. We will upload the descriptor datasets and 
```

# Datasets
|Dataset|Name in framework|
|---|---|
|ImageNet100|imagenet100|
|NUS-WIDE|nuswide|
|MS-COCO|coco|
|MIRFLICKR/Flickr25k|mirflickr|
|Stanford Online Product|sop|
|Cars dataset|cars|
|CIFAR10|cifar10|
