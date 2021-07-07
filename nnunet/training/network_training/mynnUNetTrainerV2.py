from nnunet.utilities.visdomvisualiser import get_plotter
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from nnunet.training.loss_functions.dice_loss import DC_and_ClDC_loss

class mynnUNetTrainerV2(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 250
        self.freeze = False
        self.unfreeze = 0
        self.usevisdom=False
        #task = self.dataset_directory.split("/")[-1]
        #job_name = self.experiment_name
        #self.my_name = task+"_"+job_name
        # setup the visualizer
        # define config element

    def set_epochs(self, epochs = 250):
        self.max_num_epochs = epochs

    def set_lr(self, lr = 1e-2):
        self.initial_lr = lr

    def set_freeze(self, unfreeze=1):
        self.freeze = True
        self.unfreeze = unfreeze

    def set_visdom(self):
        self.usevisdom=True

    def set_clloss(self):

        self.loss = DC_and_ClDC_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'k': 5})

    def initialize(self, training=True, force_load_plans=False):
        '''
        Print keys to visdom and set number of epochs
        '''
        timestamp = datetime.now()
        if self.usevisdom and training:
            try:
                self.plotter = get_plotter(self.model_name)
                self.plotter.plot_text("Initialising this model: %s <br> on %d_%d_%d_%02.0d_%02.0d_%02.0d" %
                                        (self.model_name,timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                        timestamp.second), plot_name="Welcome")
            except:
                print("Unable to connect to visdom.")

        super().initialize(training, force_load_plans)
        if self.freeze:
            self.initialize_optimizer_and_scheduler_freezing()

        if training and self.usevisdom:
            try:
                self.plotter.plot_text("EPOCHS: %s <br> LEARNING RATE: %s <br> TRAINING KEYS: %s <br> VALIDATION KEYS: %s" % (str(self.max_num_epochs),str(self.initial_lr),str(self.dataset_tr.keys()),str(self.dataset_val.keys())), plot_name="Dataset_Info")
            except:
                print("Unable to connect to visdom.")
        
    
    def plot_to_visdom(self):
        ########### this is a copy of network_trainer.plot_progress
        """
        Should probably by improved
        :return:
        """
        
        font = {'weight': 'normal',
                'size': 8}

        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()

        x_values = list(range(self.epoch + 1))

        ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

        ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

        if len(self.all_val_losses_tr_mode) > 0:
            ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
        if len(self.all_val_eval_metrics) == len(x_values):
            ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("evaluation metric")
        ax.legend()
        ax2.legend(loc=9)
        self.plotter.plot_matplotlib(plt, "theplot")
        #fig.savefig(join(self.output_folder, "progress.png"))
        plt.close()
        
    def run_training(self):
        ret = super().run_training()
        if self.usevisdom:
            try: 
                self.plotter.close_client()
            except:
                print("Unable to connect to visdom.")
        return ret


    def on_epoch_end(self):
        """
        do everything and also update visdom
        :return:
        """
        continue_training = super().on_epoch_end()
        if self.usevisdom:
            try:
                self.plot_to_visdom()
                self.plotter.plot_text("Best epoch: %s<br> With eval criterion: %s <br>" % (self.best_epoch_based_on_MA_tr_loss, self.best_val_eval_criterion_MA), plot_name="Best_epoch")
            except:
                print("Unable to connect to visdom.")
        
        return continue_training


    ## ------------ added by Camila

    def reinit_optim(self):
        '''Required if load pretrained weights'''
        if self.freeze:
            self.initialize_optimizer_and_scheduler_freezing()

    def initialize_optimizer_and_scheduler_freezing(self):
        assert self.network is not None, "self.initialize_network must be called first"

        ## somehow look for all layers
        for param in self.network.parameters():
            param.requires_grad = False #Freeze it all

        #should do this for last self.unfreeze number of layers
        #not sure if last convtranspose before last computational block should be excluded?
        # for param in self.network.tu[-1].parameters()
        #     param.requires_grad = True 
        #last computational block
        for param in self.network.conv_blocks_localization[-1].parameters():
            param.requires_grad = True 
        #last segmentation layer
        for param in self.network.seg_outputs[-1].parameters():
            param.requires_grad = True 

        # print("\n\n ------------------ not frozen --------------------")
        # for unf in filter(lambda p: p.requires_grad,self.network.parameters()):
        #     print(unf)
        # print("\n\n\n")

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,self.network.parameters()), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    ## ------------- end added by Camila
