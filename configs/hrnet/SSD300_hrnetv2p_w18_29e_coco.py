_base_ = './SSD300_hrnetv2p_w18_29e_coco.py'
# model settings
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(18, 36)),
            stage3=dict(num_channels=(18, 36, 72)),
            stage4=dict(num_channels=(18, 36, 72, 144)))),
    neck=dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256))
