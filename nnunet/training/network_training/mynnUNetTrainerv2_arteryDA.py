#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

## -------------------------------- Camila

from typing import Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.data_augmentation_forArteries import get_arteries_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader3D, DataLoader2D
from nnunet.training.network_training.mynnUNetTrainerV2 import mynnUNetTrainerV2
from torch import nn
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, default_2D_augmentation_params, get_patch_size


class mynnUNetTrainerV2_arteryDA(mynnUNetTrainerV2):
    ## ------- method copied from nnunetTrainerV2_NoDA
    def setup_DA_params(self):       
        """
        we leave out the creation of self.deep_supervision_scales, so it remains None
        :return:
        """
        # if self.threeD:
        #     self.data_aug_params = default_3D_augmentation_params
        #     self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        #     self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        #     self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        #     if self.do_dummy_2D_aug:
        #         self.data_aug_params["dummy_2D"] = True
        #         self.print_to_log_file("Using dummy2d data augmentation")
        #         self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        # else:
        #     self.do_dummy_2D_aug = False
        #     if max(self.patch_size) / min(self.patch_size) > 1.5:
        #         default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
        #     self.data_aug_params = default_2D_augmentation_params
        # self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        # if self.do_dummy_2D_aug:
        #     self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
        #                                                      self.data_aug_params['rotation_x'],
        #                                                      self.data_aug_params['rotation_y'],
        #                                                      self.data_aug_params['rotation_z'],
        #                                                      self.data_aug_params['scale_range'])
        #     self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        #     patch_size_for_spatialtransform = self.patch_size[1:]
        # else:
        #     self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
        #                                                      self.data_aug_params['rotation_y'],
        #                                                      self.data_aug_params['rotation_z'],
        #                                                      self.data_aug_params['scale_range'])
        #     patch_size_for_spatialtransform = self.patch_size
        super().setup_DA_params()
        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params["do_rotation"] = False
        self.data_aug_params["do_gamma"] = False
        self.data_aug_params["gamma_retain_stats"] = False
        self.data_aug_params['selected_seg_channels'] = [0]



    ## ------- method copied from nnunetTrainerV2_NoDA
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent
                                 , pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.patch_size, self.patch_size, self.batch_size,
                                 transpose=self.plans.get('transpose_forward'),
                                 oversample_foreground_percent=self.oversample_foreground_percent
                                 , pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  transpose=self.plans.get('transpose_forward'),
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)

        self.setup_DA_params()
        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            self.tr_gen, self.val_gen = get_arteries_augmentation(self.dl_tr, self.dl_val, self.patch_size,
                                                            params=self.data_aug_params,
                                                            deep_supervision_scales=self.deep_supervision_scales,
                                                            pin_memory=self.pin_memory)

    ## ------ this method I copied from nnunetTrainerV2_NoDA
    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction

        """        
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.do_ds = ds
        return ret

