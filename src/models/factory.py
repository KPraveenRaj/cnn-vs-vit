"""Model construction via timm. Nothing clever — model identity is a
controlled variable in this study, so no architectural surgery happens here.

build_model     pretrained backbone + fresh classification head (fine-tuning)
feature_extractor  frozen backbone emitting pooled features (num_classes=0),
                   used by linear_probe to cache features once per
                   model x dataset (features are fraction-independent).
"""
import timm


def build_model(timm_name: str, num_classes: int):
    return timm.create_model(timm_name, pretrained=True, num_classes=num_classes)


def feature_extractor(timm_name: str):
    model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
