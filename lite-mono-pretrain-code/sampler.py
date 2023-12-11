from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import math
import random
import numpy as np
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
from typing import Tuple
from typing import Optional, Union


class MultiScaleSamplerDDP(Sampler):
    def __init__(self, base_im_w: int, base_im_h: int, base_batch_size: int, n_data_samples: int,
                 min_crop_size_w: int = 160, max_crop_size_w: int = 320,
                 min_crop_size_h: int = 160, max_crop_size_h: int = 320,
                 n_scales: int = 5, is_training: bool = True, distributed=True) -> None:
        # min. and max. spatial dimensions
        min_im_w, max_im_w = min_crop_size_w, max_crop_size_w
        min_im_h, max_im_h = min_crop_size_h, max_crop_size_h

        # Get the GPU and node related information
        if not distributed:
            num_replicas = 1
            rank = 0
        else:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

        # adjust the total samples to avoid batch dropping
        num_samples_per_replica = int(math.ceil(n_data_samples * 1.0 / num_replicas))
        total_size = num_samples_per_replica * num_replicas
        img_indices = [idx for idx in range(n_data_samples)]
        img_indices += img_indices[:(total_size - n_data_samples)]
        assert len(img_indices) == total_size

        self.shuffle = True if is_training else False
        if is_training:
            self.img_batch_pairs = _image_batch_pairs(base_im_w, base_im_h, base_batch_size, num_replicas, n_scales, 32,
                                                      min_im_w, max_im_w, min_im_h, max_im_h)
        else:
            self.img_batch_pairs = [(base_im_h, base_im_w, base_batch_size)]

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.batch_size_gpu0 = base_batch_size

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_pairs)
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]
        else:
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]

        start_index = 0
        while start_index < self.n_samples_per_replica:
            curr_h, curr_w, curr_bsz = random.choice(self.img_batch_pairs)

            end_index = min(start_index + curr_bsz, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != curr_bsz:
                batch_ids += indices_rank_i[:(curr_bsz - n_batch_samples)]
            start_index += curr_bsz

            if len(batch_ids) > 0:
                batch = [(curr_h, curr_w, b_id) for b_id in batch_ids]
                yield batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.n_samples_per_replica


def _image_batch_pairs(crop_size_w: int,
                       crop_size_h: int,
                       batch_size_gpu0: int,
                       n_gpus: int,
                       max_scales: Optional[float] = 5,
                       check_scale_div_factor: Optional[int] = 32,
                       min_crop_size_w: Optional[int] = 160,
                       max_crop_size_w: Optional[int] = 320,
                       min_crop_size_h: Optional[int] = 160,
                       max_crop_size_h: Optional[int] = 320,
                       *args, **kwargs) -> list:
    """
        This function creates batch and image size pairs.  For a given batch size and image size, different image sizes
        are generated and batch size is adjusted so that GPU memory can be utilized efficiently.

    :param crop_size_w: Base Image width (e.g., 224)
    :param crop_size_h: Base Image height (e.g., 224)
    :param batch_size_gpu0: Batch size on GPU 0 for base image
    :param n_gpus: Number of available GPUs
    :param max_scales: Number of scales. How many image sizes that we want to generate between min and max scale factors.
    :param check_scale_div_factor: Check if image scales are divisible by this factor.
    :param min_crop_size_w: Min. crop size along width
    :param max_crop_size_w: Max. crop size along width
    :param min_crop_size_h: Min. crop size along height
    :param max_crop_size_h: Max. crop size along height
    :param args:
    :param kwargs:
    :return: a sorted list of tuples. Each index is of the form (h, w, batch_size)
    """

    width_dims = list(np.linspace(min_crop_size_w, max_crop_size_w, max_scales))
    if crop_size_w not in width_dims:
        width_dims.append(crop_size_w)

    height_dims = list(np.linspace(min_crop_size_h, max_crop_size_h, max_scales))
    if crop_size_h not in height_dims:
        height_dims.append(crop_size_h)

    image_scales = set()

    for h, w in zip(height_dims, width_dims):
        # ensure that sampled sizes are divisible by check_scale_div_factor
        # This is important in some cases where input undergoes a fixed number of down-sampling stages
        # for instance, in ImageNet training, CNNs usually have 5 downsampling stages, which downsamples the
        # input image of resolution 224x224 to 7x7 size
        h = make_divisible(h, check_scale_div_factor)
        w = make_divisible(w, check_scale_div_factor)
        image_scales.add((h, w))

    image_scales = list(image_scales)

    img_batch_tuples = set()
    n_elements = crop_size_w * crop_size_h * batch_size_gpu0
    for (crop_h, crop_y) in image_scales:
        # compute the batch size for sampled image resolutions with respect to the base resolution
        _bsz = max(batch_size_gpu0, int(round(n_elements/(crop_h * crop_y), 2)))

        _bsz = make_divisible(_bsz, n_gpus)
        _bsz = _bsz if _bsz % 2 == 0 else _bsz - 1  # Batch size must be even
        img_batch_tuples.add((crop_h, crop_y, _bsz))

    img_batch_tuples = list(img_batch_tuples)
    return sorted(img_batch_tuples)


def make_divisible(v: Union[float, int],
                   divisor: Optional[int] = 8,
                   min_value: Optional[Union[float, int]] = None) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MultiScaleImageFolder(ImageFolder):
    def __init__(self, root, args) -> None:
        self.args = args
        ImageFolder.__init__(self, root=root, transform=None, target_transform=None, is_valid_file=None)

    def get_transforms(self, size: int):
        imagenet_default_mean_and_std = self.args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        resize_im = size > 32
        transform = create_transform(
            input_size=size,
            is_training=True,
            color_jitter=self.args.color_jitter,
            auto_augment=self.args.aa,
            interpolation=self.args.train_interpolation,
            re_prob=self.args.reprob,
            re_mode=self.args.remode,
            re_count=self.args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(size, padding=4)

        return transform

    def __getitem__(self, batch_indexes_tup: Tuple):
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        transforms = self.get_transforms(size=int(crop_size_w))

        path, target = self.samples[img_index]
        sample = self.loader(path)
        if transforms is not None:
            sample = transforms(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
