_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
weight=5.0
tau=1.0
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = '/home/calmcar/github_repository/mmdetection/~/work_dir_center_lite/epoch_130.pth',
    distill_cfg = [
        dict(student_module = 'backbone.maxpool',
                         teacher_module = 'backbone.stem.1.0.downsample.1',
                         output_hook = True,
                         methods=[
                             dict(type='ChannelWiseDivergence',
                                       name='out_cnvs',
                                       student_channels = 64,
                                       teacher_channels = 256,
                                       tau = tau,
                                       weight =weight,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.deconv_layers.5.activate',
                         teacher_module = 'backbone.out_convs.0',
                         output_hook = True,
                         methods=[dict(type='ChannelWiseDivergence',
                                       name='hourglass_modules',
                                       student_channels = 64,
                                       teacher_channels = 64,
                                       tau = tau,
                                       weight =weight,
                                       )
                                ]
                        ),
                    # dict(student_module = 'backbone.layer4',
                    #      teacher_module = 'backbone.hourglass_modules.0.low2.low2.low2.low2.low2.1.relu',
                    #      output_hook = True,
                    #
                    #      methods=[dict(type='ChannelWiseDivergence',
                    #                    tau = tau,
                    #                    name='hourglass_modules_inter',
                    #                    student_channels = 256,
                    #                    teacher_channels = 256,
                    #                    weight =weight,
                    #                    )
                    #             ]
                    #     ),
        #
        #             dict(student_module = 'bbox_head.wh_head.2',
        #                  teacher_module = 'bbox_head.wh_head.2',
        #                  output_hook = True,
        #
        #                  methods=[dict(type='ChannelWiseDivergence',
        #                                tau = tau,
        #                                name='loss_cw_fpn_1',
        #                                student_channels = 256,
        #                                teacher_channels = 256,
        #                                weight =weight,
        #                                )
        #                         ]
        #                 ),
        #
        #             dict(student_module = 'bbox_head.offset_head.2',
        #                  teacher_module = 'bbox_head.offset_head.2',
        #                  output_hook = True,
        #
        #                  methods=[dict(type='ChannelWiseDivergence',
        #                                tau = tau,
        #                                name='loss_cw_fpn_0',
        #                                student_channels = 256,
        #                                teacher_channels = 256,
        #                                weight =weight,
        #                                )
        #                         ]
        #                 ),


                   ]
    )

teacher_cfg = '/home/calmcar/github_repository/mmdetection/configs/centernet-calmcar/centernet_resnet18_dcnv2_140e_coco.py'
student_cfg = '/home/calmcar/github_repository/mmdetection/configs/centernet-calmcar/centernet_resnet18_lite.py'
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[90, 120])


# lr_config = dict(
#     policy='cosine',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 100,
#     by_epoch=False,
#     target_lr=1e-5)
runner = dict(max_epochs=140)
# runner = dict(type='EpochBasedRunner', max_epochs=140)
# Avoid evaluation and saving weights too frequently
evaluation = dict(interval=5, metric='bbox')
resume_from = "/home/calmcar/github_repository/mmdetection/work_dir_hourglass2res18_2/epoch__15.pth"
checkpoint_config = dict(interval=5)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,)
