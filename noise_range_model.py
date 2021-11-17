import argparse
import os
import numpy as np
import torch
import pandas as pd
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.utils.hausdorff as haus
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader



# Lib files
import lib.utils as utils
from lib.utils.general import fast_hist,prepare_input,PSNR
from lib.augment3D.noise import Noise
from tqdm import tqdm
from lib.medloaders.brats2020 import MICCAIBraTS2020

def noise_level_analysis(args,model,mode):
    split = (0.6,0.8,1)
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    total_data = 335
    split_idx = int(split[0] * total_data),int(split[1]* total_data)
    noise_levels = np.arange(1,11,9)
    print(noise_levels)
    classes = ["Fundo","NCR/NET","ED","ET"]
    epsilon = 1e-5
    hist = np.zeros((args.classes, args.classes))
    scores_per_noise = np.zeros((args.classes,noise_levels.shape[0]))
    haus_mean = np.zeros((args.classes,noise_levels.shape[0]))
    psnrs = np.zeros(noise_levels.shape[0])
    path = 'drive/MyDrive/2020-12-BRICS/TCC_VITOR/datasets/brats2020/'
    for i,noise_level in tqdm(enumerate(noise_levels)):
        psnr = 0
        if mode =='train':
            args.noise_train=noise_level
            loader= MICCAIBraTS2020(args, 'train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=args.samples_train, load=args.loadData)
        elif mode == 'val':
            args.noise_val=noise_level
            loader = MICCAIBraTS2020(args, 'val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=args.samples_val, load=args.loadData)
        elif mode == 'test':
            args.noise_test=noise_level
            loader = MICCAIBraTS2020(args, 'test', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=args.samples_val, load=args.loadData)
        if not args.loadData:
            args.loadData = True
        generator = DataLoader(loader, **params)
        for batch_idx, input_tuple in enumerate(generator):
            with torch.no_grad():
                try:
                    input_tensor, target = prepare_input(input_tuple=input_tuple, args=args)
                    psnr += input_tuple[-1][0]
                    input_tensor.requires_grad = False
                    output = model(input_tensor)
                    target = target.cpu().squeeze().numpy()
                    width,height,depth = target.shape
                    output = torch.argmax(output,dim=1).squeeze().cpu().numpy()
                    for classe in range(1,4):
                        output_mask = np.where(output == classe, 1, 0)
                        target_mask = np.where(target == classe, 1, 0)
                        
                        output_mask = np.reshape(output_mask,(width*depth,height)).tolist()
                        target_mask = np.reshape(target_mask,(width*depth,height)).tolist()
                        haus_mean[classe-1,i] += haus.averaged_hausdorff_distance(output_mask,target_mask)
                    hist += fast_hist(target.flatten(), output.flatten(), args.classes)
                except RuntimeError as e:
                    print(str(e))
        
        dice = 2*(np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) + epsilon)
        scores_per_noise[:-1,i] = dice[1:]
        scores_per_noise[-1,i] = np.mean(scores_per_noise[:-1,i])
        hist_norm = np.round(hist/hist.sum(axis=1,keepdims=1)*100,decimals=2)
        pd.DataFrame(hist_norm,columns=classes,index=classes).to_csv(args.model_path + "hist_{}_{}_{}.csv".format(mode,noise_level,args.noise_type))
        psnrs[i] = psnr/len(generator)
        haus_mean[-1,i]=np.mean(haus_mean[:-1,i])
    
    print(haus_mean)
    print(scores_per_noise)
    print(psnrs)
    
        
    y = np.round(np.polyfit(psnrs,scores_per_noise[-1],deg=1),decimals=4)
    y_haus = np.round(np.polyfit(psnrs,haus_mean[-1],deg=1),decimals=4)
    fig,ax = plt.subplots(nrows=1,ncols=2)
    for classe in range(scores_per_noise.shape[0]):
        data = scores_per_noise[classe]
        data_haus = haus_mean[classe]
        #print(data,psnrs)
        ax[0].scatter(psnrs,data)
        ax[1].scatter(psnrs,data_haus)
    
    ax[0].set_title("Score Dice")
    ax[0].plot(psnrs,y[0]*psnrs+y[1])
    ax[0].legend(["y={}*x+{}".format(y[0],np.round(y[1],decimals=2))] + classes[1:] + ["mean"],loc="lower right")
    ax[0].set_ylabel("Dice score")
    ax[0].set_xlabel("PSNR")
    ax[0].set_ylim([0.3,0.85])
    
    ax[1].set_title("Distancia Hausdorff")
    ax[1].plot(psnrs,y_haus[0]*psnrs+y_haus[1])
    ax[1].legend(["y={}*x+{}".format(y_haus[0],np.round(y_haus[1],decimals=2))] + classes[1:] + ["mean"],loc="upper right")
    ax[1].set_ylabel("Hausdorff distance")
    ax[1].set_xlabel("PSNR")
    ax[1].set_ylim([0.0,50.0])
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    wspace=0.55, 
                    hspace=0.1)
    plt.savefig(args.model_path + "noise_graph_{}_{}.eps".format(args.noise_type,args.model))

def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)

    model,_ = medzoo.create_model(args)
    model.load_state_dict(torch.load(args.model_path + 'model.pt')['model_state_dict'])
    
    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")
        noise_level_analysis(args,model,mode=args.mode)
        #noise_level_analysis(args,model,mode="train")
        


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="brats2020")
    parser.add_argument('--dim', nargs="+", type=int, default=(128, 128, 128))
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
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--model_path', type=str,
                        default='/content/drive/MyDrive/2020-12-BRICS/TCC_VITOR/runs/')
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    args.tb_log_dir = '/content/drive/MyDrive/2020-12-BRICS/TCC_VITOR/runs/'
    args.model = args.model_path.split('/')[-2].split('_')[0]
    return args


if __name__ == '__main__':
    main()
