import logging

from datasets.cifar_dataset import build_cifar10_dataloader
from datasets.custom_dataset import build_custom_dataloader
from datasets.fmnist_dataset import build_fmnist_dataloader
from datasets.mnist_dataset import build_mnist_dataloader
from datasets.svhn_dataset import build_svhn_dataloader
from datasets.visa_dataset import build_visa_dataloader
from datasets.waterbirds_dataset import build_waterbirds_dataloader
from datasets.brain_dataset import build_brain_dataloader
from datasets.isic_dataset import build_isic_dataloader
from datasets.aptos_dataset import build_aptos_dataloader
from datasets.wbc_dataset import build_wbc_dataloader

logger = logging.getLogger("global")


def build(cfg, training, distributed, category):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "cifar10":
        data_loader = build_cifar10_dataloader(cfg, training, distributed)
    elif dataset == "fashion-mnist":
        data_loader = build_fmnist_dataloader(cfg, training, distributed)
    elif dataset == "svhn":
        data_loader = build_svhn_dataloader(cfg, training, distributed)
    elif dataset == 'waterbirds':
        data_loader = build_waterbirds_dataloader(cfg, training, distributed)
    elif dataset == 'brain':
        data_loader = build_brain_dataloader(cfg, training, distributed)
    elif dataset == 'isic':
        data_loader = build_isic_dataloader(cfg, training, distributed)
    elif dataset == 'aptos':
        data_loader = build_aptos_dataloader(cfg, training, distributed)
    elif dataset == 'wbc':
        data_loader = build_wbc_dataloader(cfg, training, distributed)
    elif dataset == 'mnist':
        data_loader = build_mnist_dataloader(cfg, training, distributed)
    elif dataset == 'fmnist':
        data_loader = build_fmnist_dataloader(cfg, training, distributed)
    elif dataset == 'visa':
        data_loader = build_visa_dataloader(cfg, training, distributed, category)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True, category=None):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, training=True, distributed=distributed, category=category)
    print("train loader len", len(train_loader))

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, training=False, distributed=distributed, category=category)
    logger.info("build dataset done")
    if cfg_dataset.get('type', None) in ['waterbirds', 'brain', 'isic', 'aptos', 'wbc', 'mnist', 'fmnist', 'visa']:
        print("test loader len", len(test_loader[0]), len(test_loader[1]))
        return train_loader, test_loader[0], test_loader[1]
    else:
        print("test loader len", len(test_loader))
        return train_loader, test_loader



