import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_video
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import re
import sys
from datetime import datetime
import pandas as pd
import projects.transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font',family='Arial', size=6)
import smtplib, ssl
# import pytorch3d.transforms as pyd


def calculate_accuracy(outputs_demo, demo, seq_lengths):
    pred_demo = torch.nn.functional.softmax(outputs_demo, dim=-1).max(-1)[1]
    demo = demo.unsqueeze(-1).expand(pred_demo.shape)
    acc_batch = [torch.nn.functional.interpolate((
        demo == pred_demo)[i, 0:length].float().unsqueeze(0).unsqueeze(0), 100) 
        for i, length in enumerate(seq_lengths.long())]
    acc_batch = torch.stack(acc_batch).squeeze()
    
    return acc_batch


def calculate_traj_accuracy(outputs_tracker, traj, seq_lengths):
    pose_error = torch.linalg.norm(outputs_tracker[:,:,0:3] - traj[:,:,0:3], dim=2)
    quat_pred = outputs_tracker[:,:,[6,3,4,5]]
    quat_true = traj[:,:,[6,3,4,5]]
    qdiff = pyd.quaternion_multiply(pyd.quaternion_invert(quat_pred), quat_true)
    qdiff = 2 * torch.atan2(torch.sqrt(torch.sum(qdiff[:,:,1:]**2, 2)), qdiff[:,:,0]).abs()

    pose_error = [torch.nn.functional.interpolate(
        pose_error[i, 0:length].float().unsqueeze(0).unsqueeze(0), 100) 
        for i, length in enumerate(seq_lengths.long())]
    quat_error = [torch.nn.functional.interpolate(
        qdiff[i, 0:length].float().unsqueeze(0).unsqueeze(0), 100) 
        for i, length in enumerate(seq_lengths.long())]
    pose_error = torch.stack(pose_error).squeeze()
    quat_error = torch.stack(quat_error).squeeze()
    
    return pose_error, quat_error


def calculate_trajpos_accuracy(outputs_tracker, traj, seq_lengths):
    pose_error = torch.linalg.norm(outputs_tracker[:,:,0:3] - traj[:,:,0:3], dim=2)
    pose_error = [torch.nn.functional.interpolate(
        pose_error[i, 0:length].float().unsqueeze(0).unsqueeze(0), 100) 
        for i, length in enumerate(seq_lengths.long())]

    pose_error = torch.stack(pose_error).squeeze()
    
    return pose_error


def plot_accuracy(acc, writer, epoch):
    acc_mean_temporal = acc.mean(0)
    acc_mean = acc.mean() 
    fig = plt.figure()
    step = 100/len(acc_mean_temporal)
    plt.fill_between(np.arange(1, len(acc_mean_temporal)+1)*step, 0,
            acc_mean_temporal.cpu())
    plt.ylim([0.0, 1.0])
    writer.add_figure('Temporal mean validation accuracy', fig, epoch)
    writer.add_scalar('Mean validation accuracy', acc_mean, epoch)
    
    
def plot_tracker_accuracy(acc_pose, acc_quat, writer, epoch):
    acc_quat = 180 * acc_quat / np.pi
    fig, axs = plt.subplots(2)
    axs[0].plot(acc_pose.mean(0).cpu())
    axs[1].plot(acc_quat.mean(0).cpu())
    axs[0].legend(['Position error'])
    axs[1].legend(['Orientation error'])
    writer.add_figure('Pose mean validation error', fig, epoch)
    writer.add_scalar('Position mean validation error', acc_pose.mean(), epoch)
    writer.add_scalar('Orientation mean validation error', acc_quat.mean(), epoch)
    

def plot_trackerpos_accuracy(acc_pose, acc_quat, writer, epoch):
    fig, axs = plt.subplots(1)
    axs.plot(acc_pose.mean(0).cpu())
    axs.legend(['Position error'])
    writer.add_figure('Pose mean validation error', fig, epoch)
    writer.add_scalar('Position mean validation error', acc_pose.mean(), epoch)


def model_tester(val_indices, 
                 outputs_demo, demo, outputs_tracker, traj, 
                 writer, seq_lengths, epoch,
                 rgb, dep):
    pred_demo = torch.nn.functional.softmax(outputs_demo, dim=-1).max(-1)[1]
    demo = demo.unsqueeze(-1).expand(pred_demo.shape)
    for idx in val_indices:
        fig, axs = plt.subplots(2, figsize=(6.4, 9.0))
        
        outputs_demo = torch.nn.functional.softmax(outputs_demo, dim=-1)                    
        axs[0].plot(outputs_demo[idx][0:int(seq_lengths[idx])].cpu())
        [line.set_color('r') for line in axs[0].lines]
        axs[0].lines[int(demo[idx][0].cpu())].set_color('g')
        axs[1].plot(outputs_tracker[idx][0:int(seq_lengths[idx])].cpu())
        axs[1].set_prop_cycle(None)
        axs[1].plot(traj[idx][0:int(seq_lengths[idx])].cpu(), '--')
        axs[1].legend(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        axs[0].set_xticks(np.arange(0, seq_lengths[idx].cpu()))
        axs[1].set_xticks(np.arange(0, seq_lengths[idx].cpu()))
                         
        fig.tight_layout()
        writer.add_figure('Predictions/sample_' + str(idx), fig, epoch)
        
        if epoch == 0:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            imgs = TF.normalize(rgb[idx], -mean/std, 1.0/std)
            writer.add_video('Videos/rgb_sample_' + str(idx), imgs.unsqueeze(0))
            imgs = TF.normalize(dep[idx], -mean/std, 1.0/std)
            writer.add_video('Videos/dep_sample_' + str(idx), imgs.unsqueeze(0))
            

def model_tester_demo(val_indices, 
                 outputs_demo, demo, 
                 writer, seq_lengths, epoch,
                 rgb=None, dep=None):
    pred_demo = torch.nn.functional.softmax(outputs_demo, dim=-1).max(-1)[1]
    demo = demo.unsqueeze(-1).expand(pred_demo.shape)
    for idx in val_indices:
        fig, axs = plt.subplots(1, figsize=(6.4, 4.5))
        
        outputs_demo = torch.nn.functional.softmax(outputs_demo, dim=-1)                    
        axs.plot(outputs_demo[idx][0:int(seq_lengths[idx])].cpu())
        [line.set_color('r') for line in axs.lines]
        axs.lines[int(demo[idx][0].cpu())].set_color('g')
        axs.set_xticks(np.arange(0, seq_lengths[idx].cpu()))
                         
        fig.tight_layout()
        writer.add_figure('Predictions/sample_' + str(idx), fig, epoch)
        
        if epoch == 0 and rgb is not None:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            imgs = TF.normalize(rgb[idx], -mean/std, 1.0/std)
            writer.add_video('Videos/rgb_sample_' + str(idx), imgs.unsqueeze(0))
            imgs = TF.normalize(dep[idx], -mean/std, 1.0/std)
            writer.add_video('Videos/dep_sample_' + str(idx), imgs.unsqueeze(0))
            
            
def calculate_tracker_accuracy(outputs_tracker, traj):
    pose_error = torch.linalg.norm(outputs_tracker[:,0:3] - traj[:,0:3], dim=1)
    quat_pred = outputs_tracker[:,[6,3,4,5]]
    quat_true = traj[:,[6,3,4,5]]
    qdiff = pyd.quaternion_multiply(pyd.quaternion_invert(quat_pred), quat_true)
    qdiff = 2 * torch.atan2(torch.sqrt(torch.sum(qdiff[:,1:]**2, 1)), qdiff[:,0]).abs()
    quat_error = qdiff
    
    return pose_error, quat_error


def plot_tracker_boxplot(acc_pose, acc_quat, writer, epoch):
    acc_quat = 180 * acc_quat / np.pi
    fig, axs = plt.subplots(1,2)
    axs[0].boxplot(acc_pose.cpu().numpy(), whis=(5,95), showfliers=False)
    axs[1].boxplot(acc_quat.cpu().numpy(), whis=(5,95), showfliers=False)
    axs[0].legend(['Position error'])
    axs[1].legend(['Orientation error'])
    writer.add_figure('Pose mean validation error', fig, epoch)
    writer.add_scalar('Position mean validation error', acc_pose.mean(), epoch)
    writer.add_scalar('Orientation mean validation error', acc_quat.mean(), epoch)
    
    
def plot_fancy_tracker_boxplot(acc_pose, acc_quat, writer, epoch):
    import seaborn as sns
    acc_quat = 180 * acc_quat / np.pi
    fig, axs = plt.subplots(1,2)
    
    axs[0] = sns.boxplot(data=acc_pose.cpu().numpy(), whis=[5,95])
    axs[0] = sns.stripplot(data=acc_pose.cpu().numpy(), color="orange", jitter=0.2, size=2.5)
    axs[1] = sns.boxplot(data=acc_quat.cpu().numpy(), whis=[5,95])
    axs[1] = sns.stripplot(data=acc_quat.cpu().numpy(), color="orange", jitter=0.2, size=2.5)
    axs[0].legend(['Position error'])
    axs[1].legend(['Orientation error'])
    writer.add_figure('Fancy pose mean validation error', fig, epoch)
    
    
def plot_dec(rgb, rgb_rec, writer, epoch):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    rgb = TF.normalize(rgb, -mean/std, 1.0/std)
    rgb_rec = TF.normalize(rgb_rec, -mean/std, 1.0/std)
    for idx in range(0, 4):
        writer.add_image('Images/rgb_' + str(idx), rgb[idx], epoch)
        writer.add_image('Images/rgb_reconstructed_' + str(idx), rgb_rec[idx], epoch)
    

def interpolate_classes(arr, seq_lengths):
    if len(arr.shape) != 1:
        # arr = [torch.nn.functional.interpolate(
        #     arr[i, 0:length].float().view(1, 4, -1), 100, mode='linear', align_corners=False)
        #     for i, length in enumerate(seq_lengths.long())]

        arr = [F.interpolate(F.softmax(
            arr[i, 0:int(length)], dim=-1).max(-1)[1].float().unsqueeze(0).unsqueeze(0), 
            100, mode='nearest').squeeze()
            for i, length in enumerate(seq_lengths)]
        arr = torch.stack(arr)
    else:
        arr = arr.unsqueeze(1).expand(-1, 100)
    return arr
    
    
def plot_fancy_accuracy(acc, writer, epoch):
    acc_mean_temporal = acc.mean(0)
    fig = plt.figure()
    step = 100/len(acc_mean_temporal)
    plt.fill_between(np.arange(1, len(acc_mean_temporal)+1)*step, 0,
            acc_mean_temporal, alpha=0.4, color='tab:blue')
    plt.plot(np.arange(1, len(acc_mean_temporal)+1)*step, acc_mean_temporal, 'b')
    plt.ylim([0.0, 1.1])
    plt.xlabel('Percentage of input video')
    plt.ylabel('Slot prediction accuracy')
    writer.add_figure('Temporal range-mean validation accuracy', fig, epoch)
    fig.savefig('accuracy.pdf')
    np.save('accuracy.npy', acc_mean_temporal)
    
        
def plot_confusion_matrix(demo_true, demo_pred, writer, epoch):
    from sklearn.metrics import confusion_matrix as cm
    import seaborn as sns

    conf_mat_40 = cm(demo_true[:, 40], demo_pred[:, 40])
    conf_mat_70 = cm(demo_true[:, 70], demo_pred[:, 70])
    conf_mat_99 = cm(demo_true[:, 99], demo_pred[:, 99])
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    cm_perc_40 = conf_mat_40 / conf_mat_40.sum(axis=1, keepdims=True)
    cm_perc_70 = conf_mat_70 / conf_mat_70.sum(axis=1, keepdims=True)
    cm_perc_99 = conf_mat_99 / conf_mat_99.sum(axis=1, keepdims=True)
    g0 = sns.heatmap(cm_perc_40, annot=True, ax=axs[0], cbar=False, cmap='Blues_r', square=True, fmt='.1%', xticklabels=['1','2','3','4'], yticklabels=['1','2','3','4'])
    g1 = sns.heatmap(cm_perc_70, annot=True, ax=axs[1], cbar=False, cmap='Blues_r', square=True, fmt='.1%', xticklabels=['1','2','3','4'], yticklabels=['1','2','3','4'])
    g2 = sns.heatmap(cm_perc_99, annot=True, ax=axs[2], cbar=False, cmap='Blues_r', square=True, fmt='.1%', xticklabels=['1','2','3','4'], yticklabels=['1','2','3','4'])
    g0.set_title('After 40 % of motion')
    g0.set_xlabel('Predicted target slot $k$')
    g0.set_ylabel('Actual target slot $k$')
    g1.set_title('After 70 % of motion')
    g2.set_title('After 100 % of motion')
    plt.tight_layout()
    
    writer.add_figure('Confusion matrix', fig, epoch)
    fig.savefig('conf_matrices.pdf')
        

def plot_confusion_matrix50(demo_true, demo_pred, writer, epoch):
    from sklearn.metrics import confusion_matrix as cm
    import seaborn as sns

    conf_mat_25 = cm(demo_true[:, 25], demo_pred[:, 25])
    conf_mat_50 = cm(demo_true[:, 50], demo_pred[:, 50])
    conf_mat_75 = cm(demo_true[:, 75], demo_pred[:, 75])
    conf_mat_99 = cm(demo_true[:, 99], demo_pred[:, 99])
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    cm_perc_25 = conf_mat_25 / conf_mat_25.sum(axis=1, keepdims=True)
    cm_perc_50 = conf_mat_50 / conf_mat_50.sum(axis=1, keepdims=True)
    cm_perc_75 = conf_mat_75 / conf_mat_75.sum(axis=1, keepdims=True)
    cm_perc_99 = conf_mat_99 / conf_mat_99.sum(axis=1, keepdims=True)
    g0 = sns.heatmap(cm_perc_25, annot=True, ax=axs[0], cbar=False, cmap='Blues_r', square=True, fmt='.1%', xticklabels=['1','2','3','4'], yticklabels=['1','2','3','4'])
    g1 = sns.heatmap(cm_perc_50, annot=True, ax=axs[1], cbar=False, cmap='Blues_r', square=True, fmt='.1%', xticklabels=['1','2','3','4'], yticklabels=['1','2','3','4'])
    g2 = sns.heatmap(cm_perc_75, annot=True, ax=axs[2], cbar=False, cmap='Blues_r', square=True, fmt='.1%', xticklabels=['1','2','3','4'], yticklabels=['1','2','3','4'])
    g3 = sns.heatmap(cm_perc_99, annot=True, ax=axs[3], cbar=False, cmap='Blues_r', square=True, fmt='.1%', xticklabels=['1','2','3','4'], yticklabels=['1','2','3','4'])
    g0.set_xlabel('Predicted target slot')
    g1.set_xlabel('Predicted target slot')
    g2.set_xlabel('Predicted target slot')
    g3.set_xlabel('Predicted target slot')
    g0.set_ylabel('Actual target slot')
    g0.set_title('After 25 % of motion')
    g1.set_title('After 50 % of motion')
    g2.set_title('After 75 % of motion')
    g3.set_title('After 100 % of motion')
    # plt.tight_layout()
    
    writer.add_figure('Confusion matrix', fig, epoch)
    fig.savefig('conf_matrices.pdf')


def model_tester_pro(val_indices, 
                 outputs_demo, demo, 
                 writer, seq_lengths, epoch,
                 rgb, dep, traj, resume_plot=None):

    pred_demo = torch.nn.functional.softmax(outputs_demo, dim=-1).max(-1)[1]
    demo = demo.unsqueeze(-1).expand(pred_demo.shape)
    outputs_demo = torch.nn.functional.softmax(outputs_demo, dim=-1)                    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    heights = [0.75, 0.6, 0.7, 0.6]
    labels = ['$k$=1', '$k$=2', '$k$=3', '$k$=4']
    x = np.arange(len(labels))
    width = 0.5
    MF = int(min(seq_lengths[val_indices]))
    SL = len(val_indices)
    NR = 4 # number of rows
    
    matplotlib.style.use('seaborn')
    from matplotlib import gridspec
    gs = gridspec.GridSpec(SL*NR, MF, height_ratios=heights*SL,
        wspace=0.03, hspace=0.03)
    if resume_plot is not None:
        import pickle
        fig = pickle.load(open(resume_plot, 'rb'))
        plt.figure(fig.number)
    else:
        fig = plt.figure(figsize=(1.8*MF, 1.8*sum(heights)*SL))

    for ii, idx in enumerate(val_indices):
        rgbs = TF.normalize(rgb[idx], -mean/std, 1.0/std)
        deps = TF.normalize(dep[idx], -mean/std, 1.0/std)
        # fig, axs = plt.subplots(1, figsize=(6.4, 4.5))
        i_grid = 0
        fr_min = int(seq_lengths[idx]-MF)
        fr_max = int(seq_lengths[idx])
        for i_frame in range(fr_min, fr_max):
            if resume_plot is None:
                # Plot RGB images (1st row)
                ax_rgb = plt.subplot(gs[NR*ii, i_grid])
                plt.imshow(deps[i_frame].cpu().permute(1,2,0))
                plt.axis('off')
                extent = ax_rgb.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted())
                plt.savefig('dep_{}_frame_{}.png'.format(idx, i_frame), dpi=500, bbox_inches=extent)
                # Plot probs (2nd row)
                ax_vid_pred = plt.subplot(gs[NR*ii+1, i_grid])
                bars = plt.bar(x, outputs_demo[idx][i_frame].cpu(), width)
                plt.gca().set_xticks(x)
                # plt.gca().set_xticklabels(labels)
                if i_grid == 0:
                    pass
                    # for nn, bar in enumerate(bars):
                    #     plt.gca().annotate('$k$={}'.format(nn+1), 
                    #         xy=(bar.get_x(),bar.get_height()+0.05),
                    #         fontsize=9)
                    # plt.gca().set_ylabel('Predicted $\mathbf{p}$ \n(HandNet)')
                    # plt.gca().yaxis.set_tick_params(direction='in', pad=2)
                else:
                    plt.gca().set_xticklabels([])
                    plt.gca().set_yticklabels([])
                # plt.gca().set_xlabel('Target slot label $k$')
                [bar.set_color('tab:red') for bar in bars]
                bars[int(demo[idx][0].cpu())].set_color('tab:green')
                plt.gca().set_ylim([0.0, 1.0])
                extent = ax_vid_pred.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted())
                plt.savefig('rgb_{}_pred_{}.png'.format(idx, i_frame), dpi=500, bbox_inches=extent)
            else:
                ax_traj = plt.subplot(gs[NR*ii+2, i_grid])
                traj = traj.cpu()
                plt.plot(traj[idx, 0:i_frame+1, 2],
                         traj[idx, 0:i_frame+1, 1], '-s',
                         markersize=5
                )
                plt.xlim([traj[idx,:fr_max,2].min()-0.02, traj[idx,:fr_max,2].max()+0.02])
                plt.ylim([traj[idx,:fr_max,1].min()-0.02, traj[idx,:fr_max,1].max()+0.02])
                plt.gca().invert_xaxis()
                if i_grid == 0:
                    pass
                    # plt.gca().xaxis.set_tick_params(direction='in', pad=-15)
                    # plt.gca().yaxis.set_tick_params(direction='in', pad=2)
                    # # plt.gca().set_xlabel('x')
                    # plt.gca().set_ylabel('y', rotation=0)
                else:
                    plt.xticks(visible=False)
                    plt.yticks(visible=False)
                extent = ax_traj.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
                plt.savefig('traj_{}_input_{}.png'.format(idx, i_frame), dpi=500, bbox_inches=extent)

                # Plot OptiNet probs (4th row)
                ax_opti_pred = plt.subplot(gs[NR*ii+3, i_grid])
                bars = plt.bar(x, outputs_demo[idx][i_frame].cpu(), width)
                plt.gca().set_xticks(x)
                plt.gca().set_xticklabels([])
                if i_grid == 0:
                    pass
                    # for nn, bar in enumerate(bars):
                    #     plt.gca().annotate('$k$={}'.format(nn+1), 
                    #         xy=(bar.get_x(),bar.get_height()+0.05),
                    #         fontsize=9)
                    # plt.gca().set_ylabel('Predicted $\mathbf{p}$ \n(OptiNet)')
                    # plt.gca().yaxis.set_tick_params(direction='in', pad=2)
                else:
                    plt.gca().set_xticklabels([])
                    plt.gca().set_yticklabels([])
                # plt.gca().set_xlabel('Target slot label $k$')
                [bar.set_color('tab:red') for bar in bars]
                bars[int(demo[idx][0].cpu())].set_color('tab:green')
                plt.gca().set_ylim([0.0, 1.0])
                extent = ax_opti_pred.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted())
                plt.savefig('traj_{}_pred_{}.png'.format(idx, i_frame), dpi=500, bbox_inches=extent)
            i_grid += 1

    if resume_plot is None:
        import pickle
        pickle.dump(fig, open('/tmp/traj_fig.pkl', 'wb'))