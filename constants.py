descriptors_data_folder = {
    'cifar10_alexnet_1': 'data/cifar10_1_descriptor',
    'cifar10_vgg16_1': 'data/cifar10_1_vgg16_descriptor',
    'cifar10_resnet18_1': 'data/cifar10_resnet18_descriptor',
    'cifar10_alexnet_2': 'data/cifar10_2_descriptor',
    'cifar10_vgg16_2': 'data/cifar10_2_vgg16_descriptor',
    'cifar10_alexnet_3': 'data/cifar10_3_descriptor',
    'imagenet100_alexnet': 'data/imagenet100_descriptor',
    'imagenet100_vgg16': 'data/imagenet100_vgg16_descriptor',
    'imagenet100_resnet18': 'data/imagenet100_resnet18_descriptor',
    'mirflickr_alexnet': 'data/mirflickr_alexnet_descriptor',
    'mirflickr_vgg16': 'data/mirflickr_vgg16_descriptor',
    'mirflickr_resnet18': 'data/mirflickr_resnet18_descriptor',
    'nuswide_vgg16': 'data/nuswide_vgg16_descriptor',
    'coco_alexnet': 'data/coco_descriptor',
    'coco_vgg16': 'data/coco_vgg16_descriptor',
    'coco_resnet18': 'data/coco_resnet18_descriptor',
    'coco_vgg16_pca_128': 'data/coco_vgg16_pca_128_descriptor',  # 2 * nbits
    'sop_resnet18': 'data/sop_resnet18_descriptor',
    'sop_alexnet': 'data/sop_alexnet_descriptor',
    'sop_vgg16': 'data/sop_vgg16_descriptor',
    # for non one-hot dataset, remember add the dfolder at configs.py:non_onehot_dataset
    'sop_instance_resnet18': 'data/sop_instance_resnet18_descriptor',
    'sop_instance_alexnet': 'data/sop_instance_alexnet_descriptor',
    'sop_instance_vgg16': 'data/sop_instance_vgg16_descriptor',
}

losses = {
    'supervised': ['greedyhash', 'jmlh', 'dpn', 'orthocos', 'ce', 'bihalf-supervised', 'orthoarc',
                   'sdhc', 'csq', 'adsh'],
    'pairwise': ['hashnet', 'dbdh', 'dpsh', 'mihash', 'sdh', 'dfh', 'dtsh'],
    'unsupervised': ['greedyhash-unsupervised', 'bihalf', 'ssdh'],
    'autoencoder': [],
    'adversarial': ['tbh'],
    'shallow': ['itq', 'pca', 'lsh', 'sh', 'imh'],
    'contrastive': ['cibhash']
}

datasets = {
    'class': ['imagenet100', 'nuswide', 'cifar10', 'imagenet50a', 'imagenet50b', 'cars', 'cifar10_II', 'landmark',
              'landmark200', 'landmark500', 'gldv2delgembed', 'roxford5kdelgembed', 'descriptor', 'sop',
              'sop_instance', 'food101'],
    'multiclass': ['nuswide', 'coco', 'mirflickr'],
}

supported_model = {
    'greedyhash': ['gh'],
    'jmlh': ['jmlh'],
    'dpn': ['dpn'],
    'ce': ['ce'],
    'ceq': ['ce'],
    'cea': ['ce'],
    'orthoarc': ['orthohash'],
    'orthocos': ['orthohash'],
    'greedyhash-unsupervised': ['norm-unsupervised'],
    'bihalf': ['norm-unsupervised'],
    'bihalf-supervised': ['ce'],
    'sdh': ['ce', 'dpn'],
    'sdhc': ['ce'],
    'csq': ['dpn'],
    'dfh': ['dpn'],
    'dbdh': ['dpn', 'single'],
    'dpsh': ['dpn'],
    'hashnet': ['dpn'],
    'dtsh': ['dpn'],
    'adsh': ['orthohash'],
    'ssdh': ['norm-unsupervised'],
    'tbh': ['tbh'],
    'pca': ['linear'],
    'itq': ['linear'],
    'lsh': ['linear'],
    'sh': ['linear'],
    'imh': ['linear'],
    'cibhash': ['cibhash'],
    'mihash': ['single', 'dpn']
}
