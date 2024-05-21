import torch
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import v2 as T
from torchvision.models.detection import MaskRCNN
from PIL import Image


def get_transform(train=True):
    transform_list = [
        T.PILToTensor(),
    ]

    if train:
        apply_prob = 0.15
        transform_list.extend([
            T.RandomHorizontalFlip(apply_prob),
            T.RandomVerticalFlip(apply_prob),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=apply_prob),
            T.RandomApply([T.RandomRotation(degrees=(-5, 5))], p=apply_prob),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=apply_prob),
            # T.RandomApply([T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0))], p=apply_prob),
            T.RandomApply([T.RandomAffine(degrees=0, translate=(0.2, 0.2),
                                          scale=(0.8, 1.2), shear=(-5, 5))], p=apply_prob),
            # T.RandomPerspective(distortion_scale=0.25, p=apply_prob),
            # T.RandomErasing(p=apply_prob, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        ])

    transform_list.extend([
        T.ToDtype(torch.float, scale=True),
        T.ToPureTensor(),
    ])

    return T.Compose(transform_list)


def init_untrained_model(num_classes) -> MaskRCNN:
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def init_trained_model(model_path: Path):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = init_untrained_model(3)
    if str(model_path).endswith("gz"):
        import gzip
        with gzip.open(model_path, 'rb') as f_in:
            loaded_model = torch.load(f_in, map_location=torch.device(device))
    else:
        loaded_model = torch.load(model_path, map_location=torch.device(device))
    state_dictionary = loaded_model.state_dict()
    model.load_state_dict(state_dictionary)
    model.to(device)

    return model


def run_prediction(image_path, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = T.PILToTensor()(Image.open(image_path).convert('RGB'))
    eval_transform = get_transform(train=False)
    model.eval()

    with torch.no_grad():
        x = eval_transform(image)
        x = x.to(device)
        predictions = model([x, ])
        prediction = predictions[0]
        # Remove tensor from GPU memory
        del x
    torch.cuda.empty_cache()
    return prediction
