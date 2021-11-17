import argparse
import os

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
# Lib files
import lib.utils as utils
from lib.utils.general import fast_hist,prepare_input
import numpy as np
import torch
import pandas as pd

def get_confusion_matrix(generator,args,model,mode):
    classes = ["Fundo","NCR/NET","ED","ET"]
    hist = np.zeros((args.classes, args.classes))
    for batch_idx, input_tuple in enumerate(generator):
        with torch.no_grad():
            try:
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=args)
                input_tensor.requires_grad = False
                output = model(input_tensor)
                target = target.cpu()
                output = torch.argmax(output,dim=1).cpu()
                hist += fast_hist(target.numpy().flatten(), output.numpy().flatten(), args.classes)
            except RuntimeError as e:
                print(str(e))
    hist = np.round(hist/hist.sum(axis=1)*100,decimals=2)
    print(hist)
    pd.DataFrame(hist,columns=classes,index=classes).to_csv(args.model_path + "hist_{}.csv".format(mode))

def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
    path = 'drive/MyDrive/2020-12-BRICS/TCC_VITOR/MedicalZooPytorch/datasets/MICCAI_BraTS_2020_Data_Training/')
    model,_ = medzoo.create_model(args)
    model.load_state_dict(torch.load(args.model_path + 'model.pt')['model_state_dict'])
    
    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")
        get_confusion_matrix(val_generator,args,model,mode="val")
        get_confusion_matrix(training_generator,args,model,mode="train")
        


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=8)
    parser.add_argument('--dataset_name', type=str, default="brats2020")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--loadData', type=bool, default=True)
    parser.add_argument('--augmentation', type=bool, default=True)
    parser.add_argument('--normalization', type=str, default='brats') 
    #possible values: max,mean,max_min,brats and full_volume mean

    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=130)
    parser.add_argument('--samples_val', type=int, default=15)
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
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET','DMFnet','RESIDUALUNET'))
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    args.tb_log_dir = '/content/drive/MyDrive/2020-12-BRICS/TCC_VITOR/MedicalZooPytorch/runs/'
    return args


if __name__ == '__main__':
    main()
