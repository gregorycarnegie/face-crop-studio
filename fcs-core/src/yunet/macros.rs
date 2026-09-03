#[macro_export]
macro_rules! backbone_block {
    ($stage:literal, $conv:literal, $dw:literal, $db:literal) => {
        $crate::yunet::StageBlock {
            point_weight: concat!("backbone.model", $stage, ".conv", $conv, ".conv1.weight"),
            point_bias: concat!("backbone.model", $stage, ".conv", $conv, ".conv1.bias"),
            depth_weight: $dw,
            depth_bias: $db,
        }
    };
}

#[macro_export]
macro_rules! neck_block {
    ($idx:literal, $dw:literal, $db:literal) => {
        $crate::yunet::StageBlock {
            point_weight: concat!("neck.lateral_convs.", $idx, ".conv1.weight"),
            point_bias: concat!("neck.lateral_convs.", $idx, ".conv1.bias"),
            depth_weight: $dw,
            depth_bias: $db,
        }
    };
}

#[macro_export]
macro_rules! head_block {
    ($type:literal, $level:literal) => {
        $crate::yunet::HeadBlock {
            conv1_weight: concat!(
                "bbox_head.multi_level_",
                $type,
                ".",
                $level,
                ".conv1.weight"
            ),
            conv1_bias: concat!("bbox_head.multi_level_", $type, ".", $level, ".conv1.bias"),
            conv2_weight: concat!(
                "bbox_head.multi_level_",
                $type,
                ".",
                $level,
                ".conv2.weight"
            ),
            conv2_bias: concat!("bbox_head.multi_level_", $type, ".", $level, ".conv2.bias"),
        }
    };
}

#[macro_export]
macro_rules! detection_head {
    ($level:literal) => {
        $crate::yunet::DetectionHeadConfig {
            cls: head_block!("cls", $level),
            obj: head_block!("obj", $level),
            bbox: head_block!("bbox", $level),
            kps: head_block!("kps", $level),
        }
    };
}
