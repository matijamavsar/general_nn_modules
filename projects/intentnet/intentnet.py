import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from ..utils import *
from ..losses import *

class IntentNet(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        
        self.args = args

        from ..networks import IntentNet
        self.model = IntentNet(args.use_googlenet).to(args.dev)
        if args.resume_path is not None:
            print('Loading model from', args.resume_path)
            self.load_state_dict(torch.load(args.resume_path, 'cpu')['state_dict'])

        self.loss_class = Losses(sig_nll_weight=args.sig_nll_weight,
            dev=args.dev)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=args.lr,
                                             weight_decay=args.weight_decay)

        self.writer = SummaryWriter()
        self.writer.add_text('Optimizer', str(self.optimizer))
        self.writer.add_text('Model', str(self.model))

    def forward(self, x):
        out_demo, out_tracker = self.model.forward(x)
        return out_demo, out_tracker

    def shared_step(self, batch):
        data, lengths = batch[0], batch[1]
        rgb = data['rgb_video'].to(self.args.dev)
        dep = data['depth_video'].to(self.args.dev)
        traj = data['traj'].to(self.args.dev)
        demo = data['demo'].to(self.args.dev)
        seq_lengths = lengths['rgb_video'].to(self.args.dev)
        
        inputs = torch.cat((rgb, dep.mean(2, keepdim=True)), axis=2)
        
        outputs_demo, outputs_tracker = self.forward(inputs)
        loss_demo = self.loss_class.weighted_cross_entropy_loss(
            outputs_demo, demo, seq_lengths)
        # loss_tracker = self.loss_class.mse_loss(outputs_tracker, traj, seq_lengths)
        
        loss = loss_demo
        return loss, outputs_demo, demo, seq_lengths, rgb, dep

    def training_step(self, train_batch, batch_idx):
        loss, _, _, _, _, _ = self.shared_step(train_batch)
        return loss

    def training_epoch_end(self, training_step_outputs):
        loss = []
        for i in training_step_outputs:
            loss.append(i['loss'])
        loss = torch.stack(loss).mean()
        self.writer.add_scalar('Epoch training loss', loss, self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        loss, outputs_demo, demo, seq_lengths, rgb, dep = self.shared_step(
            val_batch)
        acc = calculate_accuracy(outputs_demo, demo, seq_lengths)
 
        # Test the model on some validation samples from first batch
        if ((self.current_epoch % self.args.test_interval == 0) and
            (batch_idx == 0)):
            val_indices = [0, 1, 2, 3]
            model_tester_demo(val_indices, outputs_demo, demo, self.writer,
                seq_lengths, self.current_epoch, rgb, dep)

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
        plot_accuracy(acc, self.writer, self.current_epoch)
        self.log('val_acc', acc.mean())
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        _, outputs_demo, demo, seq_lengths, rgb, dep = self.shared_step(test_batch)
        acc = calculate_accuracy(outputs_demo, demo, seq_lengths)
        demo_true = interpolate_classes(demo, seq_lengths)
        demo_pred = interpolate_classes(outputs_demo, seq_lengths)
        
        # Test the model on some samples from first batch
        if ((self.current_epoch % self.args.test_interval == 0) and
            (batch_idx == 0)):
            val_indices = [0, 20, 77, 43]
            # model_tester_demo(val_indices, outputs_demo, demo, self.writer,
            #     seq_lengths, self.current_epoch, rgb, dep)
            model_tester_pro(val_indices, outputs_demo, demo, self.writer,
                seq_lengths, self.current_epoch, rgb, dep, None, None)

        return acc, demo_true, demo_pred

    def test_epoch_end(self, test_step_outputs):
        acc, demo_true, demo_pred = [], [], []
        for i, j, k in test_step_outputs:
            acc.append(i)
            demo_true.append(j)
            demo_pred.append(k)
        acc = torch.cat(acc).cpu()
        demo_true = torch.cat(demo_true).cpu()
        demo_pred = torch.cat(demo_pred).cpu()
        # plot_confusion_matrix(demo_true, demo_pred, self.writer, self.current_epoch)
        plot_confusion_matrix50(demo_true, demo_pred, self.writer, self.current_epoch)
        plot_fancy_accuracy(acc, self.writer, self.current_epoch)
        self.log('test_acc', acc.mean())

    def configure_optimizers(self):
        return self.optimizer

