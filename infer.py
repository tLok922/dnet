#!/home/bsft21/tinloklee2/miniconda3/envs/depth/bin/python
import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import skimage.io
from config import testing_root, save_testing_path, dnet_path, encoder_path, ckpt_path
from misc import check_mkdir, crf_refine

from model.dnet import DNet
from model.dnet import EncoderNet

from dataloader import get_features
import torch.nn.functional as F

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = {
        'snapshot': '100',
        'scale': 416,
        'crf': True
    }

    net_path = dnet_path
    encoder_path# = encoder_path
    # to_test = {'PMD': testing_root}
    to_test = {'MSD': testing_root}
    save_folder_path = save_testing_path
    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    to_tensor_transform = transforms.ToTensor()
    # detectron
    #to_test = {'MSD': external_testing}

    to_pil = transforms.ToPILImage()
    net = DNet().to(device)
    net.load_state_dict(torch.load(net_path), strict=True)
    encoder = EncoderNet([1,1,1,1,2])
    encoder.load_state_dict(torch.load(encoder_path), strict=True)
    
    net.eval()
    encoder.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image'))]
            start = time.time()
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                #check_mkdir(os.path.join("result", name))
                # check_mkdir(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot'])))

                img = Image.open(os.path.join(root, 'image', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).to(device)
                img_f = img
                # TODO: get_features does not work for very large images (MINC dataset)!
                if w < 384 or h < 384:
                    # rescale proportionally
                    scale_fac = max(384 / w, 384 / h)
                    new_w, new_h = int(w * scale_fac), int(h * scale_fac)
                    img_f = img.resize((new_w, new_h))
                elif w > 1920 or h > 1920:
                    scale_fac = max(1920 / w, 1920 / h)
                    new_w, new_h = int(w * scale_fac), int(h * scale_fac)
                    img_f = img.resize((new_w, new_h))
                
                with torch.no_grad():
                    features = get_features(to_tensor_transform(img_f).unsqueeze(0), 256, 32, encoder, downscale_factor=32)
                    features = F.interpolate(features, size=(384, 384), mode='bilinear', align_corners=False)
                    features = features.to(device) 
                    
                f_4, f_3, f_2, f_1, edge, final = net(img_var,features)
                #f_4, f_3, f_2, f_1, edge, final, edge2, edge3, edge4 = net(img_var)
                # output = f.data.squeeze(0).cpu()
                # edge = e.data.squeeze(0).cpu()

                f_4 = f_4.data.squeeze(0).cpu()
                f_3 = f_3.data.squeeze(0).cpu()
                f_2 = f_2.data.squeeze(0).cpu()
                f_1 = f_1.data.squeeze(0).cpu()
                edge = edge.data.squeeze(0).cpu()
                final = final.data.squeeze(0).cpu()

                f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                edge = np.array(transforms.Resize((h, w))(to_pil(edge)))
                final = np.array(transforms.Resize((h, w))(to_pil(final)))

                if args['crf']:
                    final = crf_refine(np.array(img.convert('RGB')), final)

                img_path = os.path.join(save_folder_path, img_name[:-4] + ".png")
                print(img_path)
                Image.fromarray(final).save(
                    os.path.join(img_path))

            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
