import os
from torchvision import datasets, transforms
from PIL import Image

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from sampler import MultiScaleImageFolder


# from typing import Any, Callable, cast, Dict, List, Optional, Tuple
# from typing import Union
#
# from PIL import Image
# # IMAGENET_DEFAULT_MEAN = (0.445, )
# # IMAGENET_DEFAULT_STD = (0.269, )
# IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
#
#
# def pil_loader(path: str) -> Image.Image:
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, "rb") as f:
#         img = Image.open(f)
#         return img.convert("L")
#
#
# # TODO: specify the return type
# def accimage_loader(path: str) -> Any:
#     import accimage
#     try:
#         return accimage.Image(path)
#     except OSError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)
#
#
# def default_loader(path: str) -> Any:
#     from torchvision import get_image_backend
#
#     if get_image_backend() == "accimage":
#         return accimage_loader(path)
#     else:
#         
#         return pil_loader(path)
#
#
# class MyImageFolder(datasets.DatasetFolder):
#     def __init__(
#         self,
#         root: str,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         loader: Callable[[str], Any] = default_loader,
#         is_valid_file: Optional[Callable[[str], bool]] = None,
#     ):
#         super().__init__(
#             root,
#             loader,
#             IMG_EXTENSIONS if is_valid_file is None else None,
#             transform=transform,
#             target_transform=target_transform,
#             is_valid_file=is_valid_file,
#         )
#         self.imgs = self.samples


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if is_train and args.multi_scale_sampler:
            dataset = MultiScaleImageFolder(root, args)
        else:
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # This should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter if args.color_jitter > 0 else None,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if args.three_aug:  # --aa should not be "" to use this as it actually overrides the auto-augment
            print(f"Using 3-Augments instead of Rand Augment")
            cur_augs = transform.transforms
            three_aug = transforms.RandomChoice([transforms.Grayscale(num_output_channels=3),
                                                 transforms.RandomSolarize(threshold=192.0),
                                                 transforms.GaussianBlur(kernel_size=(5, 9))])
            final_transforms = cur_augs[0:2] + [three_aug] + cur_augs[2:]
            transform = transforms.Compose(final_transforms)
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # Warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize((args.input_size, args.input_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # To maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=Image.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
