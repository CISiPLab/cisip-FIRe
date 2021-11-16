# Backbone
1. Alexnet
2. VGG{16}
3. ResNet{18,34,50,101,152}

# Loss (Method)
## Supervised
|Method|Config Template|Loss Name|
|---|---|---|
|ADSH|adsh.yaml|adsh|
|BiHalf|bihalf-supervised.yaml|bihalf-supervised|
|Cross Entropy|ce.yaml|ce|
|CSQ|csq.yaml|csq|
|DBDH|dbdh.yaml|dbdh|
|DFH|dfh.yaml|dfh|
|DPN|dpn.yaml|dpn|
|DPSH|dpsh.yaml|dpsh|
|DTSH|dtsh.yaml|dtsh|
|GreedyHash|greedyhash.yaml|greedyhash|
|HashNet|hashnet.yaml|hashnet|
|JMLH|jmlh.yaml|jmlh|
|MIHash|mihash.yaml|mihash|
|OrthoCos(OrthoHash)|orthocos.yaml|orthocos|
|OrthoArc(OrthoHash)|orthoarc.yaml|orthoarc|
|SDH|sdh.yaml|sdh|
|SDH-C|sdhc.yaml|sdhc|
## Unsupervised
|Method|Config Template|Loss Name|
|---|---|---|
|BiHalf|bihalf.yaml|bihalf|
|CIBHash|cibhash.yaml|cibhash|
|GreedyHash|greedyhash-unsupervised.yaml|greedyhash-unsupervised|
|SSDH|ssdh.yaml|ssdh|
|TBH|tbh.yaml|tbh|
## Shallow (Non-Deep learning methods)
|Method|Config Template|Loss Name|
|---|---|---|
|ITQ|itq.yaml|itq|
|LsH|lsh.yaml|lsh|
|PCAHash|pca.yaml|pca|
|SH|sh.yaml|sh|



# Datasets
|Dataset|Name in framework|
|---|---|
|ImageNet100|imagenet|
|NUS-WIDE|nuswide|
|MS-COCO|coco|
|MIRFLICKR/Flickr25k|mirflickr|
|Stanford Online Product|sop|
|Cars dataset|cars|
|CIFAR10|cifar10|
