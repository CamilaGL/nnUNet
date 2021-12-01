import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_name", type=str, default=None, help="Task Name")
    parser.add_argument("-b", "--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("-p", "--patch_size",type=int, required=False, default=64,
                        help="Pixels per axis")
    parser.add_argument("-d", "--downsampling",type=int, required=False, default=4,
                        help="Number of poolings to downsample")

    args = parser.parse_args()
    task_name = args.task_name
    batchsize=args.batch_size
    patchsize=args.patch_size
    pool=args.downsampling
    plans_fname = join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_plans_3D.pkl')
    plans = load_pickle(plans_fname)
    index3dfr = len(plans['plans_per_stage'])-1 #fullres index
    plans['plans_per_stage'][index3dfr]['batch_size'] = batchsize
    plans['plans_per_stage'][index3dfr]['patch_size'] = np.array((patchsize, patchsize, patchsize))
    plans['plans_per_stage'][index3dfr]['num_pool_per_axis'] = [pool, pool, pool]
    plans['plans_per_stage'][index3dfr]['pool_op_kernel_sizes'] = [[2, 2, 2] for n in range(pool)]
    plans['plans_per_stage'][index3dfr]['conv_kernel_sizes'] = [[3, 3, 3] for n in range(pool+1)]
    save_pickle(plans, join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_ps%d_bs%d_ds%d_plans_3D.pkl'%(patchsize, batchsize, pool)))

if __name__ == "__main__":
    main()

