import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
sns.set_style("darkgrid")
import pandas as pd

def calculate_pos_accuracy(outputs, targets, seq_lengths):
    dims = torch.tensor(outputs.shape)
    targets = targets.unsqueeze(1).expand(dims.tolist())

    pose_error = torch.linalg.norm(outputs[:,:,0:3] - targets[:,:,0:3], dim=2)
    return pose_error

def model_tester_position(val_indices, 
                 outputs, targets, 
                 writer, seq_lengths, epoch):
    for idx in val_indices:
        fig, axs = plt.subplots(1, figsize=(6, 4.1))
        
        out = outputs[idx][0:int(seq_lengths[idx])].cpu()
        true = targets[idx].unsqueeze(0).expand(out.shape).cpu()
        plt.plot(out.numpy())
        axs.set_prop_cycle(None)
        plt.plot(true.numpy(), '--')
        axs.legend(['X', 'Y', 'Z'])
        plt.xlabel('Number of processed input samples')
        plt.ylabel('Goal position [m]')

        axs.set_xticks(np.arange(0, seq_lengths[idx].cpu()))

        fig.tight_layout()
        # plt.savefig('sample_' + str(idx) + '.png')
        writer.add_figure('Predictions/sample_' + str(idx), fig, epoch)

        # out = outputs[idx][0:int(seq_lengths[idx])].cpu().numpy()
        # true = targets[idx].unsqueeze(0).expand(out.shape).cpu().numpy()
        # df_out = pd.DataFrame(data=out, columns=['X', 'Y', 'Z'])
        # df_out['Frame number'] = df_out.index
        # df_out['Type'] = 'Predicted'
        # df_true = pd.DataFrame(data=true, columns=['X', 'Y', 'Z'])
        # df_true['Frame number'] = df_true.index
        # df_true['Type'] = 'True'
        # df_all = pd.concat([df_out, df_true]).reset_index(drop=True)
        # lineplot = sns.lineplot(data=df_all, x='Frame number', y='X') #, x='Frame number', y='X', hue='Type')
        # fig = lineplot.get_figure()

def plot_fancy_accuracy(acc, writer, epoch):
    plt.clf()
    plt.style.use('seaborn')

    acc_mean_temporal = acc.mean(0)
    fig = plt.figure(figsize=(6, 4))
    step = 100/len(acc_mean_temporal)
    plt.fill_between(np.arange(1, len(acc_mean_temporal)+1)*step,
        np.quantile(acc, 0.25, axis=0), np.quantile(acc, 0.75, axis=0),
        alpha=0.4, color='tab:blue')
    plt.fill_between(np.arange(1, len(acc_mean_temporal)+1)*step,
        np.quantile(acc, 0.05, axis=0), np.quantile(acc, 0.95, axis=0),
        alpha=0.2, color='tab:blue')
    plt.plot(np.arange(1, len(acc_mean_temporal)+1)*step, acc_mean_temporal, 'tab:blue')
    plt.xlabel('Percentage of processed motion [%]')
    plt.ylabel('Handover position error [m]')
    plt.gca().grid(b=True, which='major', alpha=0.3, linewidth=2.0, linestyle='-')
    plt.legend(['Mean error', '50 % error range', '90 % error range'])
    fig.savefig('accuracy.pdf')
    writer.add_figure('Temporal mean validation error', fig, epoch)
    # np.save('accuracy.npy', acc_mean_temporal)

def model_tester_pro(val_indices, 
                 outputs, targets, 
                 writer, seq_lengths, epoch,
                 rgb, dep, traj):

    heights = [0.75, 0.6, 0.7, 0.6]
    MF = int(min(seq_lengths[val_indices]))
    SL = len(val_indices)
    NR = 3 # number of rows
    
    from matplotlib import gridspec
    gs = gridspec.GridSpec(SL*NR, MF, height_ratios=heights*SL,
        wspace=0.03, hspace=0.03)
    fig = plt.figure(figsize=(1.8*MF, 1.8*sum(heights)*SL))

    for ii, idx in enumerate(val_indices):
        rgbs = rgb[idx]
        trjs = traj[idx]
        i_grid = 0
        fr_min = int(seq_lengths[idx]-MF)
        fr_max = int(seq_lengths[idx])
        for i_frame in range(fr_min, fr_max):
            # Plot RGB images (1st row)
            ax_rgb = plt.subplot(gs[NR*ii, i_grid])
            plt.imshow(rgbs[i_frame].cpu().permute(1,2,0))
            plt.axis('off')
            extent = ax_rgb.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
            
            # Plot trajectories (2nd row)
            ax_vid_pred = plt.subplot(gs[NR*ii+1, i_grid])
            ax = plt.axes(projection ='3d')
            ax.plot3D(trjs[0:i_frame+1,0],
                      trjs[0:i_frame+1,1],
                      trjs[0:i_frame+1,2], 'red')
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.gca().set_zticklabels([])
            if i_grid == 0:
                pass
            else:
                plt.gca().set_xticklabels([])
                plt.gca().set_yticklabels([])
            plt.gca().set_ylim([0.0, 1.0])
            extent = ax_vid_pred.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
            # plt.savefig('rgb_{}_pred_{}.png'.format(idx, i_frame), dpi=500, bbox_inches=extent)

            # Plot predictions (3rd row)
            ax_vid_pred = plt.subplot(gs[NR*ii+2, i_grid])
            # TODO

            i_grid += 1

def model_tester_new(val_indices, traj, outputs, targets, rgb):

    plt.style.use('default')

    heights = [0.75, 0.8, 0.7]
    MF = min([len(x) for x in traj])
    SL = len(val_indices)
    NR = 3 # number of rows
    STEP = 2
    
    from matplotlib import gridspec
    gs = gridspec.GridSpec(SL*NR, MF, height_ratios=heights*SL,
        wspace=0.05, hspace=0.03)
    fig = plt.figure(figsize=(1.8*MF, 1.8*sum(heights)*SL))

    for ii, idx in enumerate(val_indices):
        rgbs = rgb[ii]
        trjs = traj[ii].cpu().numpy()
        i_grid = 0
        fr_min = int(len(traj[ii])-MF)
        fr_max = int(len(traj[ii]))
        for i_frame in range(fr_min + STEP, fr_max, STEP):
            # Plot RGB images (1st row)
            ax_rgb = plt.subplot(gs[NR*ii, i_grid])
            plt.imshow(rgbs[i_frame].cpu().permute(1,2,0))
            plt.axis('off')
            extent = ax_rgb.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
            
            # Plot trajectories (2nd row)
            ax_vid_pred = plt.subplot(gs[NR*ii+1, i_grid], projection='3d')
            ax_vid_pred.plot3D(trjs[0:i_frame+1,0],
                               trjs[0:i_frame+1,1],
                               trjs[0:i_frame+1,2], 'red')
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.gca().set_zticklabels([])
            plt.gca().set_xlabel('X', size=6, labelpad=-15)
            plt.gca().set_ylabel('Y', size=6, labelpad=-15)
            plt.gca().set_zlabel('Z', size=6, labelpad=-15)
            if i_grid == 0:
                pass
            else:
                plt.gca().set_xticklabels([])
                plt.gca().set_yticklabels([])
                plt.gca().set_zticklabels([])
            plt.gca().set_xlim([trjs[:,0].min(), trjs[:,0].max()])
            plt.gca().set_ylim([trjs[:,1].min(), trjs[:,1].max()])
            plt.gca().set_zlim([trjs[:,2].min(), trjs[:,2].max()])
            # extent = ax_vid_pred.get_window_extent().transformed(
            #     fig.dpi_scale_trans.inverted())
            # plt.savefig('rgb_{}_pred_{}.png'.format(idx, i_frame), dpi=500, bbox_inches=extent)

            # Plot predictions (3rd row)
            outputs[ii] = outputs[ii].cpu()
            targets[ii] = targets[ii].cpu()
            targets[ii] = targets[ii].expand(len(outputs[ii][0]), 3)
            plt.subplot(gs[NR*ii+2, i_grid])
            plt.plot(outputs[ii][0][0:i_frame], linewidth=1.0)
            plt.gca().set_prop_cycle(None)
            plt.plot(targets[ii][0:], '--', linewidth=1.0)
            plt.gca().set_xlim([0, fr_max])
            plt.gca().set_ylim([outputs[ii][0].min()-0.1, outputs[ii][0].max()+0.1])

            x_minor_ticks = np.arange(0, fr_max, 3)

            plt.gca().tick_params(which='both', axis='x', direction='in', pad=-11)
            plt.gca().set_xticks(x_minor_ticks, minor=True)
            plt.gca().tick_params(axis='y', direction='in', pad=-11)
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.tick_params(
                axis='x',
                which='major',
                bottom=False,
                top=False,
                labelbottom=False)
            plt.grid(which='minor', alpha=0.5, linewidth=0.2)
            plt.grid(which='both', axis='y', alpha=0.5, linewidth=0.2)

            for axis in ['top','bottom','left','right']:
                plt.gca().spines[axis].set_linewidth(0.4)
                plt.gca().spines[axis].set_alpha(0.6)

            if i_grid == 0:
                plt.legend(['X', 'Y', 'Z'], prop={'size': 6})

            i_grid += 1

    fig.savefig('examples.pdf')

def plot_fancy_tracker_boxplot(acc_position, writer, epoch):
    fig, axs = plt.subplots(1,2)
    
    axs[0] = sns.boxplot(data=acc_pose.cpu().numpy(), whis=[5,95])
    axs[0] = sns.stripplot(data=acc_pose.cpu().numpy(), color="orange", jitter=0.2, size=2.5)
    axs[0].legend(['Position error'])
    axs[1].legend(['Orientation error'])
    writer.add_figure('Pose mean validation error', fig, epoch)

def plot_fancy_xyz_boxplots(outputs, targets):
    plt.clf()
    plt.cla()
    plt.style.use('seaborn')
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 9.6))
    step = 100/outputs.shape[1]

    targets = targets.unsqueeze(1).expand(-1, outputs.shape[1], 3)
    x_error = (outputs[:,:,0] - targets[:,:,0]).abs().mean(0)
    y_error = (outputs[:,:,1] - targets[:,:,1]).abs().mean(0)
    z_error = (outputs[:,:,2] - targets[:,:,2]).abs().mean(0)
    axs[0].grid(alpha=0.3, linewidth=2.0)
    axs[0].plot(np.arange(1, outputs.shape[1]+1)*step, x_error, color=sns.color_palette()[0])
    axs[0].plot(np.arange(1, outputs.shape[1]+1)*step, y_error, color=sns.color_palette()[1])
    axs[0].plot(np.arange(1, outputs.shape[1]+1)*step, z_error, color=sns.color_palette()[2])
    axs[0].legend(['X error', 'Y error', 'Z error'])
    axs[0].set_xlabel('Percentage of processed motion [%]')
    axs[0].set_ylabel('Mean handover position error [m]')

    x_box_error = (outputs[:,-1,0] - targets[:,-1,0]).abs() * 1000
    y_box_error = (outputs[:,-1,1] - targets[:,-1,1]).abs() * 1000
    z_box_error = (outputs[:,-1,2] - targets[:,-1,2]).abs() * 1000
    axs[1] = sns.boxplot(data=[x_box_error, y_box_error, z_box_error], 
        showfliers=False, width=0.3, whis=[5,95])
    axs[1].set_ylabel('Handover position error [mm]')
    axs[1].set_xticklabels(['X error', 'Y error', 'Z error'])
    axs[1].set_axisbelow(True)
    axs[1].grid(alpha=0.3, linewidth=2.0)

    fig.savefig('boxplots.pdf')