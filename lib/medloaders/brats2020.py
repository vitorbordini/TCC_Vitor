import glob
import os

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset

import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import create_sub_volumes
from lib.utils.general import PSNR


class MICCAIBraTS2020(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, mode, dataset_path='./datasets', classes=5, crop_dim=(200, 200, 150), split_idx=220,
                 samples=10,
                 load=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param split_idx: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + 'MICCAI_BraTS2020_TrainingData/'
        self.full_vol_dim = (240, 240, 155)  # slice, width, height
        self.crop_size = crop_dim
        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        self.samples = samples
        self.full_volume = None
        self.classes = classes
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)
        std = 0.065
        self.save_name = self.root + '/brats2020/brats2020-list-' + mode + '-samples-' + str(samples) + '.txt'
        self.noise_train = augment3D.Noise(mean=0, std=std*args.noise_train,type=args.noise_type)
        self.noise_val = augment3D.Noise(mean=0, std=std*args.noise_val,type=args.noise_type)
        self.noise_test = augment3D.Noise(mean=0, std=std*args.noise_test,type=args.noise_type)

        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            #self.affine = img_loader.load_affine_matrix(list_IDsT1[0])
            return

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
        self.sub_vol_path = self.root + '/brats2020/generated/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)
        
        list_IDsT1 = sorted(glob.glob(self.training_path + '*/*t1.nii.gz'))
        list_IDsT1ce = sorted(glob.glob(self.training_path + '*/*t1ce.nii.gz'))
        list_IDsT2 = sorted(glob.glob(self.training_path + '*/*t2.nii.gz'))
        list_IDsFlair = sorted(glob.glob(self.training_path + '*/*flair.nii.gz'))
        labels = sorted(glob.glob(self.training_path + '*/*_seg.nii.gz'))

        if self.mode == 'train':
            print('Brats2020, Total data:', len(list_IDsT1))
            list_IDsT1 = list_IDsT1[:split_idx[0]]
            list_IDsT1ce = list_IDsT1ce[:split_idx[0]]
            list_IDsT2 = list_IDsT2[:split_idx[0]]
            list_IDsFlair = list_IDsFlair[:split_idx[0]]
            labels = labels[:split_idx[0]]
            self.list = create_sub_volumes(list_IDsT1, list_IDsT1ce, list_IDsT2, list_IDsFlair, labels,
                                           dataset_name="brats2020", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold)

        elif self.mode == 'val':
            list_IDsT1 = list_IDsT1[split_idx[0]:split_idx[1]]
            list_IDsT1ce = list_IDsT1ce[split_idx[0]:split_idx[1]]
            list_IDsT2 = list_IDsT2[split_idx[0]:split_idx[1]]
            list_IDsFlair = list_IDsFlair[split_idx[0]:split_idx[1]]
            labels = labels[split_idx[0]:split_idx[1]]
            self.list = create_sub_volumes(list_IDsT1, list_IDsT1ce, list_IDsT2, list_IDsFlair, labels,
                                           dataset_name="brats2020", mode=mode, samples=samples,
                                           full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold)
        else:
            list_IDsT1 = list_IDsT1[split_idx[1]:]
            list_IDsT1ce = list_IDsT1ce[split_idx[1]:]
            list_IDsT2 = list_IDsT2[split_idx[1]:]
            list_IDsFlair = list_IDsFlair[split_idx[1]:]
            labels =labels[split_idx[1]:]
            print(len(list_IDsT1))
            self.list = create_sub_volumes(list_IDsT1, list_IDsT1ce, list_IDsT2, list_IDsFlair, labels,
                                       dataset_name="brats2020", mode=mode, samples=samples,
                                       full_vol_dim=self.full_vol_dim, crop_size=self.crop_size,
                                       sub_vol_path=self.sub_vol_path, th_percent=self.threshold)
            # Todo inference code here

        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)
        #return len(self.list_IDsT1)

    def __getitem__(self, index):
        get_psnr = False
        f_t1, f_t1ce, f_t2, f_flair, f_seg = self.list[index]
        img_t1, img_t1ce, img_t2, img_flair, img_seg = np.load(f_t1), np.load(f_t1ce), np.load(f_t2), np.load(
            f_flair), np.load(f_seg)
        #f_t1, f_t1ce, f_t2, f_flair, f_seg = self.list_IDsT1[index],self.list_IDsT1ce[index],self.list_IDsT2[index],self.list_IDsFlair[index],self.labels[index]
        #img_t1, img_t1ce, img_t2, img_flair, img_seg = np.squeeze(nib.load(f_t1).get_fdata()), np.squeeze(nib.load(f_t1ce).get_fdata()),np.squeeze(nib.load(f_t2).get_fdata()), np.squeeze(nib.load(f_flair).get_fdata()),np.squeeze(nib.load(f_seg).get_fdata())
        
        if self.mode == 'train' and self.augmentation:
            [img_t1, img_t1ce, img_t2, img_flair], img_seg = self.transform([img_t1, img_t1ce, img_t2, img_flair],img_seg)

        if self.noise_train.std > 0 and self.mode == 'train':
            get_psnr = True
            [noise_img_t1, noise_img_t1ce, noise_img_t2, noise_img_flair] = self.noise_train([img_t1, img_t1ce, img_t2, img_flair])
        if self.noise_val.std > 0 and self.mode == 'val':
            get_psnr = True
            [noise_img_t1, noise_img_t1ce, noise_img_t2, noise_img_flair] = self.noise_val([img_t1, img_t1ce, img_t2, img_flair])
        if self.noise_test.std >0 and self.mode == 'test':
            get_psnr = True
            [noise_img_t1, noise_img_t1ce, noise_img_t2, noise_img_flair] = self.noise_test([img_t1, img_t1ce, img_t2, img_flair])
            
        if get_psnr:
            psnr = (PSNR(img_t1,noise_img_t1) + PSNR(img_t1ce,noise_img_t1ce) + 
        PSNR(img_t2,noise_img_t2) + PSNR(img_flair,noise_img_flair))/4
            return torch.FloatTensor(noise_img_t1.copy()).unsqueeze(0), torch.FloatTensor(noise_img_t1ce.copy()).unsqueeze(
                0), torch.FloatTensor(noise_img_t2.copy()).unsqueeze(0), torch.FloatTensor(noise_img_flair.copy()).unsqueeze(0), torch.FloatTensor(img_seg.copy()),psnr
        else:
            psnr = 0
            return torch.FloatTensor(img_t1.copy()).unsqueeze(0), torch.FloatTensor(img_t1ce.copy()).unsqueeze(
                0), torch.FloatTensor(img_t2.copy()).unsqueeze(0), torch.FloatTensor(img_flair.copy()).unsqueeze(0), torch.FloatTensor(img_seg.copy()),psnr