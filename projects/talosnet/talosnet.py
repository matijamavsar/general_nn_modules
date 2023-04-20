from genericpath import isdir
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.nn import functional as F
from projects.talosnet.utils import *
from projects.talosnet.data import PlotVideoTrajDataset
import projects.talosnet.traj_transforms as transforms


class MSEPoseLoss():
    def __init__(self, sig_mse_weight=5):
        self.sig_mse_weight = sig_mse_weight
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, output, target):
        dims = torch.tensor(output.shape)
        target = target.unsqueeze(1).expand(dims.tolist())

        x = torch.arange(0, dims[1]).float()
        weights = self.sigmoid(self.sig_mse_weight * 
            ((x / torch.max(x)) - 0.5)).type_as(output)
        loss = torch.sum((target - output)**2, 0)
        sample_loss = torch.sum(
            torch.sum(loss, -1) * weights) / torch.prod(dims)

        return sample_loss


class TalosNet(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.automatic_optimization = True

        from .model import TalosNet
        self.model = TalosNet(args.rnn_type, args.use_residual)

        if args.resume_path is not None:
            print('Loading model from', args.resume_path)
            self.model.load_state_dict(torch.load(args.resume_path)['state_dict'])

        # Loss functions
        self.loss_class = MSEPoseLoss(sig_mse_weight=args.sig_mse_weight)
        self.optimizer = torch.optim.Adam(
            self.parameters(), args.lr, 
            [0.9, 0.999], 0.001, self.args.weight_decay, False)

        self.writer = SummaryWriter()
        self.writer.add_text('Model', str(self.model))

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['state_dict'] = self.model.state_dict()

    def forward(self, x):
        out = self.model.forward(x)
        return out

    def shared_step(self, batch):
        data, lengths, old_lengths = batch[0], batch[1], batch[2]
        traj_raw = data['pos_raw_camera'].to(self.args.dev)
        traj_filt = data['pos_filt_camera'].to(self.args.dev)
        traj_ip = data['pos_ip_camera'].to(self.args.dev)
        traj_fip = data['pos_fip_camera'].to(self.args.dev)
        seq_lengths = old_lengths['pos_ip_camera'].to(self.args.dev)
        
        # inputs = torch.cat((traj_filt, traj_fip))
        inputs = traj_fip
        targets = traj_fip[:, -1]

        outputs = self.forward(inputs)
        loss = self.loss_class.forward(
            outputs, targets)
        
        return loss, outputs, targets, seq_lengths

    def training_step(self, train_batch, batch_idx):
        loss, _, _, _ = self.shared_step(train_batch)
        return loss

    def training_epoch_end(self, training_step_outputs):
        loss = []
        for i in training_step_outputs:
            loss.append(i['loss'])
        loss = torch.stack(loss).mean()
        self.writer.add_scalar('Epoch training loss', loss, self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        loss, outputs, targets, seq_lengths = self.shared_step(
            val_batch)
        acc = calculate_pos_accuracy(outputs, targets, seq_lengths)
 
        # Test the model on some validation samples from first batch
        if ((self.current_epoch % self.args.test_interval == 0) and
            (batch_idx == 0)):
            val_indices = [10, 20, 30, 40]
            model_tester_position(val_indices, outputs, targets, self.writer,
                seq_lengths, self.current_epoch)

        return loss, acc

    def validation_epoch_end(self, validation_step_outputs):
        # Plot accuracy on the validation dataset
        loss, acc = [], []
        for i, j in validation_step_outputs:
            loss.append(i)
            acc.append(j)
        loss = torch.stack(loss).mean()
        acc = torch.cat(acc)

        self.writer.add_scalar('Epoch validation loss', loss, self.current_epoch)
        self.writer.add_scalar('Epoch accuracy', acc.mean(), self.current_epoch)
        plot_fancy_accuracy(acc.cpu(), self.writer, self.current_epoch)
        self.log('val_error', acc.mean())
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        _, outputs, targets, seq_lengths = self.shared_step(test_batch)
        acc = calculate_pos_accuracy(outputs, targets, seq_lengths)

        # Test the model on some samples from first batch
        if ((self.current_epoch % self.args.test_interval == 0) and
            (batch_idx == 0)):
            val_indices = [10, 66]

            tranlist = transforms.TrajRandomSample([19, 19], 0)
            plot_ds = PlotVideoTrajDataset(
                self.args.plot_set,
                self.args.color_video_regex,
                self.args.depth_video_regex,
                self.args.data_regex, 
                self.args.camera_regex,
                tranlist)
            plot_trj = []
            plot_out = []
            plot_tar = []
            plot_rgb = []
            for idx in val_indices:
                traj_fip = plot_ds[idx]['pos_fip_camera'].unsqueeze(0)
                traj_fip = traj_fip.to(torch.float32).to(self.args.dev)
                plot_out.append(self.forward(traj_fip))
                plot_trj.append(traj_fip.squeeze())
                plot_tar.append(traj_fip[:,-1])
                plot_rgb.append(plot_ds[idx]['color_video'])
            # model_tester_pro(val_indices, outputs, targets, rgbs,
            #     self.writer, self.current_epoch)
            model_tester_new(val_indices, plot_trj, plot_out, plot_tar, plot_rgb)

        return acc, outputs, targets

    def test_epoch_end(self, test_step_outputs):
        acc, outputs, targets = [], [], []
        for i, j, k in test_step_outputs:
            acc.append(i)
            outputs.append(j)
            targets.append(k)
        acc = torch.cat(acc).cpu()
        outputs = torch.cat(outputs).cpu()
        targets = torch.cat(targets).cpu()

        plot_fancy_xyz_boxplots(outputs, targets)
        plot_fancy_accuracy(acc, self.writer, self.current_epoch)
        
        self.log('test_error', acc.mean())

    def configure_optimizers(self):
        return self.optimizer