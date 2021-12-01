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
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader3D, DataLoader2D
from nnunet.training.network_training.mynnUNetTrainerV2 import mynnUNetTrainerV2
from torch import nn


class mynnUNetTrainerV2_noDA(mynnUNetTrainerV2):
    ## ------- method copied from nnunetTrainerV2_NoDA
    def setup_DA_params(self):
        super().setup_DA_params()
        # important because we need to know in validation and inference that we did not mirror in training
        self.data_aug_params["do_mirror"] = False
        self.data_aug_params["mirror_axes"] = tuple()

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
        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            self.tr_gen, self.val_gen = get_no_augmentation(self.dl_tr, self.dl_val,
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
        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                  "do_mirroring was set to False")
        do_mirroring = False
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.do_ds = ds
        return ret

