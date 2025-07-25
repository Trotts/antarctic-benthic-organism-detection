"MMDetection config for the Co-DETR model used during WSBD experimentation."

model = dict(
    type="ConditionalDETR",
    num_queries=300,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1,
    ),
    encoder=dict(  # DetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
        ),
    ),
    decoder=dict(  # DetrTransformerDecoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, attn_drop=0.1, cross_attn=False
            ),
            cross_attn_cfg=dict(
                embed_dims=256, num_heads=8, attn_drop=0.1, cross_attn=True
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
        ),
        return_intermediate=True,
    ),
    positional_encoding=dict(num_feats=128, normalize=True),
    bbox_head=dict(
        type="ConditionalDETRHead",
        num_classes=25,
        embed_dims=256,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=100),
)
