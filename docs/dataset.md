# Prepare datasets
You have to download the datasets you needed to their path respectively. Datasets are stored in `data/` folder. For example, ImageNet100 is stored at `data/imagenet`. If `data/` folder is not yet created, you may create it in the project root by: 
```bash
mkdir data
```
```{warning}
We do not own these datasets, please read and agree to the original authors' license before download and use.
```
## CIFAR10
There is no need of extra downloading for CIFAR10 dataset. 
The dataset will be downloaded automatically at the first time of running.

## ImageNet100
This is a subset/variant of original ImageNet dataset, we do not own the dataset. 
Please head to [ImageNet official website](https://www.image-net.org/)
to read and accept their term of uses and license before download the dataset 
[here](https://entuedu-my.sharepoint.com/:f:/g/personal/jiuntian001_e_ntu_edu_sg/EuuKVcqoIgZPtWsKpC5BmnwB90sJUFhPZGBxrXvglgmxEw) (updated Aug 2022). This is a subset with most common 100 classes.

Download and extract the archive at `data/` directory:
```bash
# at <project_root>/data/
tar -xzf imagenet.tar.gz
```
 
## NUS-WIDE
This is a subset/variant of original NUS-WIDE dataset, we do not own the dataset. 
Please head to [NUS-WIDE official website](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)
to read and accept their term of uses and license before download the dataset 
[here](https://entuedu-my.sharepoint.com/:f:/g/personal/jiuntian001_e_ntu_edu_sg/EuuKVcqoIgZPtWsKpC5BmnwB90sJUFhPZGBxrXvglgmxEw) or [here](https://www.kaggle.com/datasets/xinleili/nuswide) (updated Feb 2023). This is a subset with 21 most common classes.

Download and extract the archive at `data/` directory:
```bash
# at <project_root>/data/
tar -xzf nuswide_v2_256.tar.gz
```

## MS-COCO
This is a subset/variant of original MS-COCO dataset, we do not own the dataset. 
Please head to [MS-COCO official website](https://cocodataset.org/#termsofuse)
to read and accept their term of uses and license before download the dataset 
[here](https://entuedu-my.sharepoint.com/:f:/g/personal/jiuntian001_e_ntu_edu_sg/EuuKVcqoIgZPtWsKpC5BmnwB90sJUFhPZGBxrXvglgmxEw) (updated Aug 2022).

Download and extract the archive at `data/` directory:
```bash
# at <project_root>/data/
tar -xzf coco.tar.gz
```

## MIRFlickr-25000
This is a subset/variant of original MIRFLICKR dataset, we do not own the dataset. 
Please head to [MIRFLICKR official website](https://press.liacs.nl/mirflickr/)
to read and accept their term of uses and license before download the dataset 
[here](https://entuedu-my.sharepoint.com/:f:/g/personal/jiuntian001_e_ntu_edu_sg/EuuKVcqoIgZPtWsKpC5BmnwB90sJUFhPZGBxrXvglgmxEw) (updated Aug 2022).

Download and extract the archive at `data/` directory:
```bash
# at <project_root>/data/
tar -xzf mirflickr.tar.gz
```

## Stanford Online Product
This is a subset/variant of original Stanford Online Product dataset, we do not own the dataset. 
Please head to [Stanford Online Product official website](https://cvgl.stanford.edu/projects/lifted_struct/)
to read and accept their term of uses and license before download the dataset 
[here](https://entuedu-my.sharepoint.com/:f:/g/personal/jiuntian001_e_ntu_edu_sg/EuuKVcqoIgZPtWsKpC5BmnwB90sJUFhPZGBxrXvglgmxEw) (updated Aug 2022).

Download and extract the archive at `data/` directory:
```bash
# at <project_root>/data/
tar -xzf sop.tar.gz
```



## References
We thank the following repository for the processed datasets:
1. https://github.com/TreezzZ/DSDH_PyTorch (imagenet100,nuswide)
2. https://github.com/thuml/HashNet/tree/master/pytorch (coco)
