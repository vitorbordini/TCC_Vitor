import argparse
import os

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss, create_loss
from lib.train.trainer import Trainer

import torch
def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)

    training_generator, val_generator, full_volume = medical_loaders.generate_datasets(args,
    path = 'drive/MyDrive/2020-12-BRICS/TCC_VITOR/datasets/brats2020/')
    
    model, optimizer = medzoo.create_model(args)
    if args.model_path is not None:
        print('Loading weights from {}...'.format(args.model_path))
        model.load_state_dict(torch.load(args.model_path + 'model.pt')['model_state_dict'])
        print('Done')
        
    criterion = create_loss(args.loss,weight=torch.tensor([0.1, 1, 1, 1]).cuda()) if args.loss != 'DiceLoss' else DiceLoss(classes=args.classes, 
    weight=torch.tensor([0.1, 1, 1, 1]).cuda())
    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                      valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=8)
    parser.add_argument('--loss', type=str, default='DiceLoss')
    #SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'BCEDiceLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss',
    #                'PixelWiseCrossEntropyLoss', 'GeneralizedDiceLoss', 'DiceLoss', 'TagsAngularLoss', 'MSELoss',
    #                'SmoothL1Loss', 'L1Loss', 'WeightedSmoothL1Loss']
    parser.add_argument('--dataset_name', type=str, default="brats2020")
    parser.add_argument('--dim', nargs="+", type=int, default=(128,128,128))#(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=250)
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
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--noise_train', type=int, default=0)
    parser.add_argument('--noise_val', type=int, default=0)
    parser.add_argument('--noise_test', type=int, default=0)
    parser.add_argument('--noise_type', type=str, default="gaussian")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET','DMFnet','RESIDUALUNET'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='/content/drive/MyDrive/2020-12-BRICS/TCC_VITOR/runs/')
    parser.add_argument('--model_path', type=str,
                        default=None)
    args = parser.parse_args()

    args.tb_log_dir = '/content/drive/MyDrive/2020-12-BRICS/TCC_VITOR/runs/'
    args.save = args.tb_log_dir + args.model + "_" + args.opt + "_" + args.loss+ "_" + str(args.noise_train) + "_" + str(args.noise_val)+ "_" + args.noise_type
    
    return args


if __name__ == '__main__':
    main()
