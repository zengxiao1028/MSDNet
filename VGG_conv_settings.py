"""
trim_setting for various model. The number in the list indicates the conv layer filter sizes
"""
VGG_conv_100 = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]  # 13 conv layers
VGG_conv_75p = [48, 64, 128, 128, 256, 256, 256, 384, 384, 384, 384, 384, 384]
VGG_conv_50p = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
VGG_conv_40p = [32, 48, 96, 112, 224, 224, 224, 224, 224, 224, 224, 224, 224]
VGG_conv_30p = [32, 32, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128]
VGG_conv_25p = [32, 32, 64, 64, 96, 96, 96, 96, 96, 96, 96, 96, 96]
VGG_conv_20p = [24, 32, 64, 64, 96, 96, 80, 80, 80, 80, 80, 80, 80]
VGG_conv_15p = [16, 32, 64, 64, 80, 80, 80, 72, 72, 72, 72, 72, 72]
VGG_conv_12p = [16, 32, 64, 64, 80, 72, 72, 64, 64, 64, 64, 64, 64]
VGG_conv_10p = [16, 32, 64, 64, 64, 64, 64, 48, 48, 48, 48, 48, 48]
VGG_conv_05p = [16, 24, 48, 48, 48, 48, 48, 32, 32, 32, 32, 32, 32]  # a ruthless/brutal cut!
VGG_conv_03p = [12, 20, 40, 40, 40, 40, 40, 28, 28, 28, 28, 28, 28]
VGG_conv_02p = [12, 12, 28, 28, 28, 28, 28, 18, 18, 18, 18, 18, 16]
