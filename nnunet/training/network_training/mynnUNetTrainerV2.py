from nnunet.utilities.visdomvisualiser import get_plotter
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import sys
from nnunet.training.loss_functions.dice_loss import DC_and_ClDC_loss, DC_and_ClDC_and_CE_loss, Only_CE_loss
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_patch_size
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import numpy as np
from torch import nn

class mynnUNetTrainerV2(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 250
        self.freeze = False
        self.unfreeze = 0
        self.usevisdom=False
        self.useclloss=False
        self.useceloss=False
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
        self.useclloss=True
        self.loss = DC_and_ClDC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'k': 5}, {})
        #self.loss = DC_and_ClDC_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'k': 5})

    def set_celoss(self):
        self.useceloss=True
        self.loss = Only_CE_loss({})
        

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

        #super().initialize(training, force_load_plans)
        ## ------- nnunettrainerv2 nodeepsupervision
        """
        removed deep supervision
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                assert self.deep_supervision_scales is None
                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    classes=None,
                                                                    pin_memory=self.pin_memory)

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

        # -----------------------

        if self.freeze:
            self.initialize_optimizer_and_scheduler_freezing()

        if training and self.usevisdom:
            try:
                self.plotter.plot_text("EPOCHS: %s <br> LEARNING RATE: %s <br> TRAINING KEYS: %s <br> VALIDATION KEYS: %s" % (str(self.max_num_epochs),str(self.initial_lr),str(self.dataset_tr.keys()),str(self.dataset_val.keys())), plot_name="Dataset_Info")
            except:
                print("Unable to connect to visdom.")
        
    ## --------- from nodeepsupervision
    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    ## --------- from nodeepsupervision
    def setup_DA_params(self):
        """
        we leave out the creation of self.deep_supervision_scales, so it remains None
        :return:
        """
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

    ## --------- from nodeepsupervision
    def run_online_evaluation(self, output, target):
        return nnUNetTrainer.run_online_evaluation(self, output, target)


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

        for p in self.all_parts_tr_losses:
            matplotlib.rc('font', **font)
            fig = plt.figure(figsize=(7.5, 6))
            ax = fig.add_subplot(111)
            ax.plot(x_values, self.all_parts_tr_losses[p], color='b', ls='-', label="loss_tr_%s"%p)
            ax.plot(x_values, self.all_parts_val_losses[p], color='r', ls='-', label="loss_val_%s, train=False"%p)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
            self.plotter.plot_matplotlib(plt, "%s"%p)
            plt.close()

        matplotlib.rc('font', **font)
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111)
        colors = ['b','r','g']
        col=0
        for p in self.all_parts_tr_losses:
            ax.plot(x_values, self.all_parts_tr_losses[p], color=colors[col], ls="-", label="loss_tr_%s"%p)
            ax.plot(x_values, self.all_parts_val_losses[p], color=colors[col], ls="--", label="loss_val_%s, train=False"%p)
            col+=1
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        self.plotter.plot_matplotlib(plt, "alllosses")
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
        self.plot_all_losses()
        return continue_training


    ## ------------ added by Camila

    def plot_all_losses(self):
        """
        copy of plot
        """
        for p in self.all_parts_tr_losses:
            try:
                font = {'weight': 'normal',
                        'size': 18}

                matplotlib.rc('font', **font)

                fig = plt.figure(figsize=(30, 24))
                ax = fig.add_subplot(111)
                #ax2 = ax.twinx()

                x_values = list(range(self.epoch + 1))

                ax.plot(x_values, self.all_parts_tr_losses[p], color='b', ls='-', label="loss_tr_%s"%p)

                ax.plot(x_values, self.all_parts_val_losses[p], color='r', ls='-', label="loss_val_%s, train=False"%p)

                ax.set_xlabel("epoch")
                ax.set_ylabel("loss")
                #ax2.set_ylabel("evaluation metric")
                ax.legend()
                #ax2.legend(loc=9)

                fig.savefig(join(self.output_folder, "progress%s_%s.png" % (self.model_name, p)))
                plt.close()
            except IOError:
                self.print_to_log_file("failed to plot: ", sys.exc_info())

        try:
            font = {'weight': 'normal',
                        'size': 18}
            matplotlib.rc('font', **font)
            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            colors = ['b','r','g']
            col=0
            for p in self.all_parts_tr_losses:
                ax.plot(x_values, self.all_parts_tr_losses[p], color=colors[col], ls="-", label="loss_tr_%s"%p)
                ax.plot(x_values, self.all_parts_val_losses[p], color=colors[col], ls="--", label="loss_val_%s, train=False"%p)
                col+=1
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
            fig.savefig(join(self.output_folder, "progress%s_alllosses.png" % (self.model_name)))
            plt.close()
        except:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

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
        for param in self.network.tu[-1].parameters():
            print(param)
            param.requires_grad = True 
        #last computational block
        for param in self.network.conv_blocks_localization[-1].parameters():
            print(param)
            param.requires_grad = True 
        #last segmentation layer
        for param in self.network.seg_outputs[-1].parameters():
            print(param)
            param.requires_grad = True 

        # print("\n\n ------------------ not frozen --------------------")
        # for unf in filter(lambda p: p.requires_grad,self.network.parameters()):
        #     print(unf)
        # print("\n\n\n")

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,self.network.parameters()), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None


    ## ----------- Same method as network_trainer.py but keeping losses
    # def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
    #     """
    #     gradient clipping improves training stability

    #     :param data_generator:
    #     :param do_backprop:
    #     :param run_online_evaluation:
    #     :return:
    #     """
    #     data_dict = next(data_generator)
    #     data = data_dict['data']
    #     target = data_dict['target']

    #     data = maybe_to_torch(data)
    #     target = maybe_to_torch(target)

    #     if torch.cuda.is_available():
    #         data = to_cuda(data)
    #         target = to_cuda(target)

    #     self.optimizer.zero_grad()

    #     if self.fp16:
    #         with autocast():
    #             output = self.network(data)
    #             del data
    #             l = self.loss(output, target)

    #         if do_backprop:
    #             self.amp_grad_scaler.scale(l).backward()
    #             self.amp_grad_scaler.unscale_(self.optimizer)
    #             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
    #             self.amp_grad_scaler.step(self.optimizer)
    #             self.amp_grad_scaler.update()
    #     else:
    #         output = self.network(data)
    #         del data
    #         l, self.lossparts = self.loss(output, target)

    #         if do_backprop:
    #             l.backward()
    #             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
    #             self.optimizer.step()

    #     if run_online_evaluation:
    #         self.run_online_evaluation(output, target)

    #     del target

    #     return l.detach().cpu().numpy()


    ## ------------- end added by Camila
