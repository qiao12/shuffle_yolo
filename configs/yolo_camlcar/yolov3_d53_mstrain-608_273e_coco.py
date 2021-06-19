_base_ = '../_base_/default_runtime.py'

pth_path = '/home/calmcar/work_dir/epoch_17.pth'
# model settings
norm_cfg = dict(
    type='BN',
    requires_grad=True,
)

act_cfg = dict(
    type='ReLU',
    inplace=True
)
model = dict(
    type='YOLOV3',
    # pretrained=pth_path,
    backbone=dict(type='ShuffleNetV2', widen_factor=1.0, out_indices=(1, 2)),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=2,
        in_channels=[512, 256],
        out_channels=[256, 128]),
    bbox_head=dict(
        type='YOLOCubeHead',
        num_classes=21,
        in_channels=[256, 128],
        out_channels=[512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        ],
            strides=[32, 16]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/calmcar/data/20200914/'
img_norm_cfg = dict(mean=[91.87, 89.15, 95.04], std=[61.81, 59.86, 63.88], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(608, 608)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'cube_train.json',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=train_pipeline,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'cube_val.json',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=test_pipeline,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'cube_test.json',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=test_pipeline,
        filter_empty_gt=False))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,  # same as burn-in in darknet
    warmup_ratio=1.0/100,
    step=[218, 246])
# runtime settings

# total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=90)
evaluation = dict(interval=2, metric=['bbox'],save_best='bbox_mAP',)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = "/home/calmcar/work_dir_yolo_608/epoch_70.pth"
workflow = [('train', 1)]