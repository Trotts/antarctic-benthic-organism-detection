default_scope = "mmdet"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", save_best="auto", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=10,
        rule="greater",
        min_delta=0.001,
    ),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)

log_level = "INFO"
load_from = None
resume = False
