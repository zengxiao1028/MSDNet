resnet50_imagenet50_configs = [
    ('imagenet50 80',   3.07,   103,    0.8935,     301,    10.606),
    ('imagenet50 50',   1.79,   32,     0.8661,     193,    3.12),
    ('imagenet50 20',   0.74,   6,      0.801,      97,     0.53),
    ('imagenet50 b10',  0.31,   2,      0.6363,     65,     0.149),
    ('imagenet50 b20',  0.27,   2,      0.625,      55,     0.117)
]

# name GFlops, load time, acc, inference time, model size
resnet50_scene_configs = [
    ('scene32 90',      4.0,        144,    0.7979,     378,    15.61),
    ('scene32 70',      2.33,       65,     0.7847,     255,    6.467),
    ('scene32 40',      1.05,       18,     0.7763,     140,     1.706),
    ('scene32 10',      0.66,       4,      0.7406,     88,     0.365),
    ('scene32 b20',     0.27,       2,      0.6869,     55,     0.117)
]

# name, GFlops, load_time, acc, inference_time, model_size
resnet50_imagenet100_configs = [
    ('imagenet100 100',     5.32,       240,        0.9008,     502,    23.793),
    ('imagenet100 90',      4,          144,        0.8909,     378,    15.61),
    ('imagenet100 80',      3.07,       103,        0.879,      301,    10.606),
    ('imagenet100 70',      2.33,       65,         0.8622,     255,    6.467),
    ('imagenet100 60',      2.03,       42,         0.7192,     217,    4.407),
]

            # name, GFlops, load_time, acc, inference_time, model_size
VGG4096_cifar10_configs = [
    ('cifar10 VGG16-E40p',      7.38,       403,     0.8693,     748,    48.996),
    ('cifar10 VGG16-E30p',      2.79,       223,     0.8680,     367,    27.055),
    ('cifar10 VGG16-E20p(fake)', 2.14,      165,     0.8616,     226,    3.216),
    ('cifar10 VGG16-E10p(fake)', 1.51,      123,     0.8479,     169,    2.06),
    ('cifar10 VGG16-E05p',      0.82,       52,      0.8358,     138,    6.625),
]

VGG512_GTSRB_configs = [
    ('GTSRB VGG16-E25p',    2.14,   33,     0.9853,     226,    3.216),
    ('GTSRB VGG16-E05p',    0.81,   10,     0.9777,     112,    0.982),
    ('GTSRB VGG16-E01p',    0.11,   3,      0.9591,     52,     0.343),
    ('GTSRB VGG16-E00p',    0.06,   3,      0.9520,     48,     0.233),
    ('GTSRB VGG16-E005p',   0.04,   3,      0.9443,     42,     0.18),
]