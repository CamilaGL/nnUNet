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

from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    GammaTransform, MirrorTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


def get_arteries_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_train=None, seeds_val=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None, pin_memory=True, regions=None,
                            use_nondetMultiThreadedAugmenter: bool = False):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    # if params.get("dummy_2D") is not None and params.get("dummy_2D"):
    #     ignore_axes = (0,)
    #     tr_transforms.append(Convert3DTo2DTransform())
    # else:
    #     ignore_axes = None

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=False, 
        do_rotation=False,
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=False, p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())


    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    # if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
    #     tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
    #     if params.get("cascade_do_cascade_augmentations") is not None and params.get(
    #             "cascade_do_cascade_augmentations"):
    #         if params.get("cascade_random_binary_transform_p") > 0:
    #             tr_transforms.append(ApplyRandomBinaryOperatorTransform(
    #                 channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
    #                 p_per_sample=params.get("cascade_random_binary_transform_p"),
    #                 key="data",
    #                 strel_size=params.get("cascade_random_binary_transform_size"),
    #                 p_per_label=params.get("cascade_random_binary_transform_p_per_label")))
    #         if params.get("cascade_remove_conn_comp_p") > 0:
    #             tr_transforms.append(
    #                 RemoveRandomConnectedComponentFromOneHotEncodingTransform(
    #                     channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
    #                     key="data",
    #                     p_per_sample=params.get("cascade_remove_conn_comp_p"),
    #                     fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
    #                     dont_do_if_covers_more_than_X_percent=params.get(
    #                         "cascade_remove_conn_comp_fill_with_other_class_p")))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    # if deep_supervision_scales is not None:
    #     if soft_ds:
    #         assert classes is not None
    #         tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
    #     else:
    #         tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
    #                                                           output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)

    
    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                    params.get("num_cached_per_thread"),
                                                    seeds=seeds_train, pin_memory=pin_memory)
    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    # import IPython;IPython.embed()
    batchgenerator_train.restart()

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
    #     val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    # if deep_supervision_scales is not None:
    #     if soft_ds:
    #         assert classes is not None
    #         val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
    #     else:
    #         val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
    #                                                            output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    
    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                max(params.get('num_threads') // 2, 1),
                                                params.get("num_cached_per_thread"),
                                                seeds=seeds_val, pin_memory=pin_memory)
    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
    batchgenerator_val.restart()

    return batchgenerator_train, batchgenerator_val

