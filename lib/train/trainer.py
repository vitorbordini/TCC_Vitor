import numpy as np
import torch

from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter
from lib.utils.early import EarlyStopping
import lib.utils as utils
from lib.utils.general import poly_lr_scheduler,fast_hist
import pandas as pd

class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)
        self.name_model = args.model + "_" + args.opt + "_" + args.loss+ "_" + str(args.noise_train) + "_" + str(args.noise_val)+ "_" + args.noise_type
        self.stop = EarlyStopping(patience=50,path=args.log_dir + self.name_model+"/"+ "model.pt")

        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 0

    def training(self):
        classes = ["Fundo","NCR/NET","ED","ET"]
        for epoch in range(self.start_epoch, self.args.nEpochs):
            hist_train = self.train_epoch(epoch)

            hist_val = self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            

            self.writer.write_end_of_epoch(epoch)
            self.stop(val_loss,self.model,self.optimizer)
            if self.stop.counter == 0:
                pd.DataFrame(hist_train,columns=classes,index=classes).to_csv(self.args.log_dir + self.name_model + "/" + "hist_train.csv")
                pd.DataFrame(hist_val,columns=classes,index=classes).to_csv(self.args.log_dir + self.name_model + "/" + "hist_val.csv")
                
            if self.stop.early_stop:
                break

            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        epsilon = 1e-5
        lr = poly_lr_scheduler(self.optimizer, self.args.lr, iter=epoch, max_iter=self.args.nEpochs)
        self.model.train()
        hist = np.zeros((self.args.classes, self.args.classes))
        losses = 0
        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            
            loss = self.criterion(output, target.long())
            if isinstance(loss,tuple):
                loss = loss[0]
            losses +=loss.item()
            loss.backward()
            self.optimizer.step()
            target = target.cpu().detach()
            output = torch.argmax(output,dim=1).cpu().detach()
            hist += fast_hist(target.flatten().numpy(), output.numpy().flatten(), self.args.classes)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')
        dice = 2*(np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) + epsilon)
        self.writer.update_scores(1, losses/len(self.train_data_loader), dice, 'train',
                                          epoch * self.len_epoch)
        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)
        #if epoch%50==0:
        #    self.writer.imshow(output,target,mode="val")
            
        return np.round(hist/hist.sum(axis=1)*100,decimals=2)
    def validate_epoch(self, epoch):
        epsilon = 1e-5
        self.model.eval()
        hist = np.zeros((self.args.classes, self.args.classes))
        losses = 0
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            
            with torch.no_grad():
                try:
                    input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                    input_tensor.requires_grad = False
                    output = self.model(input_tensor)
                    loss = self.criterion(output, target.long())
                    if isinstance(loss,tuple):
                        loss = loss[0]
                    losses +=loss.item()    
                    target = target.cpu()
                    output = torch.argmax(output,dim=1).cpu()
                    hist += fast_hist(target.numpy().flatten(), output.numpy().flatten(), self.args.classes)
                except RuntimeError as e:
                    print(str(e))
        dice = 2*(np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) + epsilon)
        self.writer.update_scores(1, losses/len(self.valid_data_loader), dice, 'val',
                                          epoch * self.len_epoch)
        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)
        #if epoch%50==0:
        #    self.writer.imshow(output,target,mode="val")
        return np.round(hist/hist.sum(axis=1)*100,decimals=2)
