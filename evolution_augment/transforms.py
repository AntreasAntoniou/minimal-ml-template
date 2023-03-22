import torch
from torchvision import transforms


class RandAugmentV2(transforms.RandAugment):
    _SUPPORTED_OPS = {
        'Identity', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
        'Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize',
        'Solarize', 'AutoContrast', 'Equalize'
    }

    def __init__(self,
                 op_list,
                 num_ops,
                 magnitude,
                 num_magnitude_bins,
                 interpolation=transforms.InterpolationMode.NEAREST,
                 fill=None):
        super(RandAugmentV2,
              self).__init__(num_ops=num_ops,
                             magnitude=magnitude,
                             num_magnitude_bins=num_magnitude_bins,
                             interpolation=interpolation,
                             fill=fill)
        if isinstance(op_list, (list, tuple)):
            self.op_list = set(op_list)
        else:
            raise ValueError(
                'Expected `op_list` to be of type `{{list, tuple}}` but got {}'
                .format(type(op_list)))
        for op in op_list:
            if op not in RandAugmentV2._SUPPORTED_OPS:
                raise ValueError(
                    'Got unsupport OP type, supported OPs are: {}'.format(
                        RandAugmentV2._SUPPORTED_OPS))

    def _augmentation_space(self, num_bins, image_size):
        augmentation_space = super(RandAugmentV2, self)._augmentation_space(
            num_bins, image_size)
        return {
            k: v
            for k, v in augmentation_space.items() if k in self.op_list
        }


class EvolutionAugment:

    def __init__(self, augmentation_ops, num_candidates, min_augmentations,
                 max_augmentations, max_magnitude, num_magnitude_bins,
                 interpolation, normalize=True, image_means=None, image_stds=None):
        self.augmentation_ops = augmentation_ops
        self.num_candidates = num_candidates
        self.min_augmentations = min_augmentations
        self.max_augmentations = max_augmentations
        self.max_magnitude = max_magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.normalize = normalize
        self.image_means = image_means
        self.image_stds = image_stds
        self._to_tensor = transforms.ToTensor()

        if normalize:
            if image_means is None:
                raise ValueError('`image_means` cannot be None when `normalize=True`')

            if image_stds is None:
                raise ValueError('`image_stds` cannot be None when `normalize=True`')

            self.normalize_fn = transforms.Normalize(
                mean=image_means, std=image_stds)


    def _generate_candidate(self):
        num_ops = torch.randint(low=self.min_augmentations,
                                high=self.max_augmentations + 1,
                                size=(1, )).item()
        magnitude = torch.randint(low=0, high=self.max_magnitude, size=(1, ))
        candidate = RandAugmentV2(op_list=self.augmentation_ops,
                                  num_ops=num_ops,
                                  magnitude=int(magnitude),
                                  num_magnitude_bins=self.num_magnitude_bins)
        return (candidate, magnitude)

    def _generate_population(self):
        population = []
        for _ in range(self.num_candidates):
            population += [self._generate_candidate()]

        return population

    def __call__(self, sample):
        population = self._generate_population()
        augmented_images = []
        magnitudes = []
        for candidate, magnitude in population:
            augmented_image = self._to_tensor(candidate(sample['img'][0]))
            augmented_images += [self.normalize_fn(augmented_image)]
            magnitudes += [magnitude]

        sample['augmented_images'] = [torch.stack(augmented_images, dim=0)]
        sample['magnitudes'] = [torch.stack(magnitudes, dim=0)]
        return sample
