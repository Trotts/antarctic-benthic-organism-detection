NUM_EPOCHS = 200

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=NUM_EPOCHS, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=NUM_EPOCHS/2),
    dict(
        type='MultiStepLR',
        begin=0,
        end = NUM_EPOCHS,
        by_epoch=True,
        milestones=[100, 150],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # retinanet uses 0.01 as the default learning rate, did not learn 0.01. 0.001 learns. 
    # (see mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py)
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)