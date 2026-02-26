import torchvision
from torchvision import transforms

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

MASK_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (256, 256),
        interpolation=torchvision.transforms.InterpolationMode.NEAREST
    ),
    transforms.PILToTensor()
])