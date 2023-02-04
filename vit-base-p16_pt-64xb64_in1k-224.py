model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=5,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
        hidden_dim=3072),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=5, prob=1.0)))
dataset_type = 'MyDataset'
img_norm_cfg = dict(
    mean=[135.56433, 101.25855, 92.16169],
    std=[45.25038, 41.994884, 38.90942],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[135.56433, 101.25855, 92.16169],
        std=[45.25038, 41.994884, 38.90942],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[135.56433, 101.25855, 92.16169],
        std=[45.25038, 41.994884, 38.90942],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='MyDataset',
        data_prefix='/home/data1/yxd/mmclassification/',
        ann_file='/home/data1/yxd/mmclassification/data/meta/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[135.56433, 101.25855, 92.16169],
                std=[45.25038, 41.994884, 38.90942],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='MyDataset',
        data_prefix='/home/data1/yxd/mmclassification/',
        ann_file='/home/data1/yxd/mmclassification/data/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[135.56433, 101.25855, 92.16169],
                std=[45.25038, 41.994884, 38.90942],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='MyDataset',
        data_prefix='/home/data1/yxd/mmclassification/',
        ann_file='/home/data1/yxd/mmclassification/data/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[135.56433, 101.25855, 92.16169],
                std=[45.25038, 41.994884, 38.90942],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy', metric_options=dict(topk=1))
paramwise_cfg = dict(
    custom_keys=dict({
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }))
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.3,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/vit-base-p16_0205_0350'
gpu_ids = range(0, 2)
