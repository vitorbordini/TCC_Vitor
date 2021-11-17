import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import nibabel as nib
import glob

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
from lib.utils.general import fast_hist,prepare_input,PSNR
from lib.augment3D.noise import Noise
from lib.medloaders.brats2020 import MICCAIBraTS2020

def create_image(args,noise_level,mode,img,label,output,count):
    # Define a single layer for plotting
    layer = 50
    img = img[:,:,layer]
    output = output[:,:,layer]
    label = label[:,:,layer]
    
    # Define a dictionary of class labels
    classes_dict = {
        'Non-enhancing tumor': 2.,
        'Enhancing tumor': 4. 
    }
    
    color_dict = {
        'Edema': [0,255,0],
        'Non-enhancing tumor': [0,0,255],
        'Enhancing tumor': [127,127,127] 
    }
    # Set up for plotting
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    img = np.where(img<np.mean(img),255,0)
    image_out = np.zeros((img.shape[0],img.shape[1],3)).astype(np.uint8)
    image_target = np.zeros((img.shape[0],img.shape[1],3)).astype(np.uint8)
    
    image_out[img<150] = [255,255,255]
    image_target[img<150] = [255,255,255]
    
    for i,key in enumerate(classes_dict.keys()):
      mask_target = np.where(label == classes_dict[key], 255, 0)
      mask_out = np.where(output == classes_dict[key], 255, 0)
      image_target[mask_target == 255,:] = color_dict[key]
      image_out[mask_out == 255,:] = color_dict[key]

    ax[0].imshow(image_target)
    ax[0].set_title("Resposta desejada", fontsize=45)
    ax[0].axis('off')
    
    ax[1].imshow(image_out)
    ax[1].set_title("Saída do modelo", fontsize=45)
    ax[1].axis('off')
    plt.tight_layout()
    
    if noise_level >0:
        plt.savefig(args.model_path + "figures/{}/{}_{}_{}_{}_{}".format(args.noise_type,args.noise_type,args.model,noise_level,mode,count))
    else:
        plt.savefig(args.model_path + "figures/{}_{}_0_{}_{}".format(args.noise_type,args.model,mode,count))
    plt.close()
        
def get_images(args,model,generator,mode):
    for batch_idx, input_number in tqdm(enumerate(generator)):
        with torch.no_grad():
            input_tensor, target = prepare_input(input_tuple=input_number, args=args)
            input_tensor.requires_grad = False
            output = model(input_tensor)
            target = target.cpu().numpy()
            output = torch.argmax(output,dim=1).cpu().numpy()
            tensor = input_tensor[0][0].cpu().numpy()
            create_image(args=args,noise_level=args.noise_test,mode=mode,img=tensor,label=target[0],output=output[0],count=batch_idx)
        

def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)
    if not os.path.exists(args.model_path + "figures/"):
        os.makedirs(args.model_path + "figures/")
    
    if not os.path.exists(args.model + "figures/{}/".format(args.noise_type)):
        os.makedirs(args.model + "figures/"+ args.noise_type)
    model,_ = medzoo.create_model(args)
    model.load_state_dict(torch.load(args.model_path + 'model.pt')['model_state_dict'])
    
    split = (0.6,0.8,1)
    total_data = 335
    split_idx = int(split[0] * total_data),int(split[1]* total_data)
    path = 'drive/MyDrive/2020-12-BRICS/TCC_VITOR/datasets/brats2020/'
    
    loader = MICCAIBraTS2020(args, 'test', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=args.samples_val, load=args.loadData)
    
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    generator = DataLoader(loader, **params)
    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")
        get_images(args,model,generator,mode="val")
        #noise_level_analysis(args,model,mode="train")
        


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="brats2020")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--loadData', type=bool, default=True)
    parser.add_argument('--augmentation', type=bool, default=True)
    parser.add_argument('--normalization', type=str, default='brats') 
    #possible values: max,mean,max_min,brats and full_volume mean

    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=146)
    parser.add_argument('--samples_val', type=int, default=41)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--terminal_show_freq', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--inModalities', type=int, default=4)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--noise_train', type=int, default=0)
    parser.add_argument('--noise_val', type=int, default=0)
    parser.add_argument('--noise_test', type=int, default=0)
    parser.add_argument('--noise_type', type=str, default="gaussian")
    parser.add_argument('--model_path', type=str,
                        default='/content/drive/MyDrive/2020-12-BRICS/TCC_VITOR/MedicalZooPytorch/runs/')
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    args.tb_log_dir = '/content/drive/MyDrive/2020-12-BRICS/TCC_VITOR/MedicalZooPytorch/runs/'
    args.model = args.model_path.split('/')[-2].split('_')[0]
    return args


if __name__ == '__main__':
    main()
