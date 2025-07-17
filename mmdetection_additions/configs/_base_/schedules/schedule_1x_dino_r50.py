"MMDetection scheduler config used for training the DINO model during WSBD experimentation."

NUM_EPOCHS = 200

# training schedule for 1x
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=NUM_EPOCHS, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# learning rate
param_scheduler = [
    dict(
        type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=NUM_EPOCHS / 2
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=NUM_EPOCHS,
        by_epoch=True,
        milestones=[100, 150],
        gamma=0.1,
    ),
]

# optimizer
# Source: mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py
# Default optimizer for DINO
# Note: Uses AdamW not SGD. Assuming this is the optimal, so keeping.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.0001  # 0.0002 for DeformDETR
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
