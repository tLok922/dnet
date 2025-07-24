import math
import torch
import torch.nn.functional as F
from dataset_edge import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import joint_transforms_edge
import cv2

def get_features(image, patch_size, overlap, encoder, downscale_factor=32):
    # extracts & returns distortion features Fd

    device = next(encoder.parameters()).device
    #image = image.unsqueeze(0)
    image = image.to(device)

    patch_stride = patch_size-overlap
    B,C,H,W = image.shape
    # print(image.shape)
    
    #TODO: apply reflective padding to image
    # pad_H = (math.ceil((H-patch_size)/stride)+1)*stride+patch_size-H
    # pad_W = (math.ceil((W-patch_size)/stride)+1)*stride+patch_size-W
    # image_padded = F.pad(image, (0, pad_W, 0, pad_H), mode='reflect') #prueba -> fix code
    padding_height_to_be_added = (math.ceil((H-patch_size)/patch_stride))*patch_stride+patch_size-H
    padding_width_to_be_added = (math.ceil((W-patch_size)/patch_stride))*patch_stride+patch_size-W
    image_padded = F.pad(image, (overlap//2, padding_width_to_be_added+overlap//2,  overlap//2, padding_height_to_be_added+overlap//2), mode="reflect")
    _,_, padded_height, padded_width = image_padded.shape

    #TODO: split into patches
    patches = F.unfold(image_padded,kernel_size=patch_size,stride=patch_stride)
    _B,_,n_patches = patches.shape
    
    # print(patches.shape)
    #useful info: https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
    patches = patches.transpose(1,2).contiguous().view(B*n_patches,C,patch_size,patch_size)


    f5 = encoder(patches) #f5,f4,f3,f2,f1 = encoder(patches) 
    _, out_channels, feature_patch_height, feature_patch_width = f5.shape
    # print(f5.shape)
    # print(f4.shape)
    # print(f3.shape)
    # print(f2.shape)
    # print(f1.shape)

    f5 = f5.view(B, n_patches, out_channels* feature_patch_height * feature_patch_width).transpose(1,2)

    n_patches_row = (padded_height-patch_size)//patch_stride+1
    n_patches_col = (padded_width-patch_size)//patch_stride+1
    combined_f5_height = n_patches_row*feature_patch_height
    combined_f5_width = n_patches_col*feature_patch_width

    feature_map = F.fold(f5, output_size=(combined_f5_height,combined_f5_width),kernel_size=(feature_patch_height, feature_patch_width), stride=(feature_patch_height,feature_patch_width))

    # TODO: count overlaps & avg result -> feature_map
    ones = torch.ones((B, (out_channels*feature_patch_height*feature_patch_width), n_patches), device=device)
    overlap_count = F.fold(ones, output_size=(combined_f5_height,combined_f5_width),kernel_size=(feature_patch_height, feature_patch_width), stride=(feature_patch_height,feature_patch_width))
    feature_map = feature_map/overlap_count.clamp(min=1e-6)
    # print(feature_map.shape)

    final_height=H // downscale_factor
    final_width=W // downscale_factor
    feature_map=feature_map[:,:,:final_height,:final_width]
    return feature_map #Fd

class ImageFolderExtendsFeatureMap(ImageFolder):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None,edge_transform=None, encoder=None, patch_size=256, overlap=32, downscale_factor=32):
        super(ImageFolderExtendsFeatureMap, self).__init__(root, joint_transform, img_transform, target_transform, edge_transform)
        self.encoder=encoder
        self.patch_size=patch_size
        self.overlap=overlap
        self.downscale_factor=downscale_factor
        self.scale=384

    def __getitem__(self, index):
        img, target, edge = super(ImageFolderExtendsFeatureMap, self).__getitem__(index)
        with torch.no_grad():
            feature_map = get_features(img.unsqueeze(0),self.patch_size, self.overlap, self.encoder, downscale_factor=self.downscale_factor)
        feature_map = F.interpolate(feature_map, size=(self.scale, self.scale), mode='bilinear').squeeze(0)

        return img, target, edge, feature_map

def get_dataloaders(train_root, test_root, encoder, scale=384, batch_size=12, patch_size=256, overlap=32, downscale_factor=32):
    joint_transform = joint_transforms_edge.Compose([
        joint_transforms_edge.RandomRotate(),
        joint_transforms_edge.Resize((scale, scale))
    ])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    edge_transform = transforms.ToTensor()

    train_set = ImageFolderExtendsFeatureMap(train_root, joint_transform=joint_transform, img_transform=img_transform, target_transform=target_transform, edge_transform=edge_transform, encoder=encoder, patch_size=patch_size, overlap=overlap, downscale_factor=downscale_factor)
    val_set = ImageFolderExtendsFeatureMap(test_root, joint_transform=joint_transform, img_transform=img_transform, target_transform=target_transform, edge_transform=edge_transform, encoder=encoder, patch_size=patch_size, overlap=overlap, downscale_factor=downscale_factor)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,  shuffle=False)
    return train_loader, val_loader
