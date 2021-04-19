from nnunet.utilities.visdomvisualiser import get_plotter
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import matplotlib
import matplotlib.pyplot as plt

class mynnUNetTrainerV2(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 250

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        self.my_name = task+"_"+job_name
        # setup the visualizer
        # define config element
        self.plotter = get_plotter(self.my_name)

    def set_epochs(self, epochs = 250):
        self.max_num_epochs = epochs

    def set_lr(self, lr = 1e-2):
        self.initial_lr = lr

    def initialize(self, training=True, force_load_plans=False):
        '''
        Print keys to visdom and set number of epochs
        '''
        super().initialize(training, force_load_plans)
        if training:
            self.plotter.plot_text("EPOCHS: %s <br> LEARNING RATE: %s <br> TRAINING KEYS: %s <br> VALIDATION KEYS: %s" % (str(self.max_num_epochs),str(self.initial_lr),str(self.dataset_tr.keys()),str(self.dataset_val.keys())), plot_name="Dataset_Info")
        
    
    def plot_to_visdom(self):
        ########### this is a copy of network_trainer.plot_progress
        """
        Should probably by improved
        :return:
        """
        
        font = {'weight': 'normal',
                'size': 4.5}

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
        self.plotter.plot_matplotlib(plt, self.my_name+"eval")
        #fig.savefig(join(self.output_folder, "progress.png"))
        plt.close()
        


    def on_epoch_end(self):
        """
        do everything and also update visdom
        :return:
        """
        continue_training = super().on_epoch_end()
        self.plot_to_visdom()
        self.plotter.plot_text("Best epoch: %s<br> With eval criterion: %s <br>" % (self.best_epoch_based_on_MA_tr_loss, self.best_val_eval_criterion_MA), plot_name="Best_epoch")
        return continue_training



    