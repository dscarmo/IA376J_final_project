from torchvision import transforms

def get_transform(transform_str):
    transform = None
    
    if transform_str == "baseline": 
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((512, 256)),
                                        transforms.ToTensor()])
    
    return transform, transform
