import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import normalize
from .rimednet.dmp_integrator import DMP_integrate_batch_2


class Losses():
    
    def __init__(self, sig_nll_weight, sig_mse_weight=10,
                 w_g=None, w_tau=None, extend=None, c_weight=None):
        self.sig_nll_weight = torch.tensor(sig_nll_weight)
        self.sig_mse_weight = torch.tensor(sig_mse_weight)
        self.w_g = w_g
        self.w_tau = w_tau
        if self.w_g is not None:
            print("Setting w_g to {} and w_tau to {}".format(w_g, w_tau))
            self.w_g = torch.tensor(w_g)
            self.w_tau = torch.tensor(w_tau)
        self.loss_nll = torch.nn.CrossEntropyLoss(reduction='none', 
            weight=torch.tensor(c_weight).cuda())
        self.loss_mse = torch.nn.MSELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.extend = extend
    
    def weighted_cross_entropy_loss(self, output, target, lengths):
        dims = torch.tensor(output.shape)

        if self.extend:
            x = torch.arange(0, dims[1]).float()
            sample_loss = []
            weights = self.sigmoid(self.sig_nll_weight * 
                ((x / torch.max(x)) - 0.5)).type_as(output)
            for step in range(dims[1]):
                sample_loss.append(self.loss_nll(output[:, step], target).mean())
            sample_loss = (torch.stack(sample_loss) * weights).sum() / dims[1]
        else:
            sample_loss = torch.tensor(0.0)
            for i in range(int(dims[0])):
                x = torch.arange(0, lengths[i]).float()
                weights = self.sigmoid(self.sig_nll_weight * ((x / x.max()) - 0.5))
                for j, weight in enumerate(weights):
                    sample_loss += (weight * self.loss_nll(output[i,j].unsqueeze(0),
                        target[i].unsqueeze(0).long())).squeeze()
                sample_loss = sample_loss / len(weights)
            sample_loss = sample_loss / dims[0]
        
        return sample_loss
        
    def weighted_mse_loss(self, output, target, lengths):
        dims = torch.tensor(output.shape)
        target = target.unsqueeze(1).expand(dims.tolist())

        if self.extend:
            # Calculate loss with insane ultra efficiency
            sigmoid = nn.Sigmoid()
            x = torch.arange(0, dims[1]).float()
            weights = sigmoid(self.sig_mse_weight * 
                ((x / torch.max(x)) - 0.5)).type_as(output)
            loss = torch.sum((target - output)**2, 0)
            loss[:, 7:14] *= self.w_g
            loss[:, -1] *= self.w_tau
            sample_loss = torch.sum(
                torch.sum(loss, -1) * weights) / torch.prod(dims)

        else:
            sample_loss = torch.tensor(0.0)
            for i in range(int(dims[0])):
                x = torch.arange(0, lengths[i]).float()
                weights = self.sigmoid(self.sig_mse_weight * ((x / x.max()) - 0.5))
                for j, weight in enumerate(weights):
                    sample_diff = target[i,j] - output[i,j]
                    if self.w_g is not None:
                        sample_diff[7:14] *= self.w_g
                        sample_diff[-1] *= self.w_tau
                    sample_loss += (weight * torch.mean(sample_diff**2))
                sample_loss = sample_loss / len(weights)
            sample_loss = sample_loss / dims[0]
        
        return sample_loss
    
    def mse_loss(self, output, target, lengths):
        dims = torch.tensor(output.shape)
        
        sample_loss = torch.tensor(0.0)
        for i in range(int(dims[0])):
            sample_loss += self.loss_mse(output[i], target[i]).squeeze()
        sample_loss = sample_loss / dims[0]
        
        return sample_loss


class TemporallyWeightedMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self, sig_mse_weight=10,
                 w_g=None, w_tau=None, extend=None):
        super(TemporallyWeightedMSELoss, self).__init__()

        self.sig_mse_weight = torch.tensor(sig_mse_weight)
        self.w_g = w_g
        self.w_tau = w_tau
        if self.w_g is not None:
            print("Setting w_g to {} and w_tau to {}".format(w_g, w_tau))
            self.w_g = torch.tensor(w_g)
            self.w_tau = torch.tensor(w_tau)
        self.sigmoid = nn.Sigmoid()
        self.extend = extend

    def forward(self, output, target, lengths):
        dims = torch.tensor(output.shape)
        target = target.unsqueeze(1).expand(dims.tolist())

        tc = target.clone()
        oc = output.clone()
        oc[:,:,3:7] = normalize(output[:,:,3:7], dim=2)
        tc[:,:,3:7] = normalize(target[:,:,3:7], dim=2)
        oc[:,:,10:14] = normalize(output[:,:,10:14], dim=2)
        tc[:,:,10:14] = normalize(target[:,:,10:14], dim=2)

        if self.extend:
            # Calculate loss with insane ultra efficiency
            x = torch.arange(0, dims[1]).float()
            weights = self.sigmoid(self.sig_mse_weight * 
                ((x / torch.max(x)) - 0.5)).type_as(output)
            loss = torch.sum((tc - oc)**2, 0)
            loss[:, 7:14] *= self.w_g
            loss[:, -1] *= self.w_tau
            sample_loss = torch.sum(
                torch.sum(loss, -1) * weights) / torch.prod(dims)

        else:
            sample_loss = torch.tensor(0.0).type_as(output)
            for i in range(int(dims[0])):
                x = torch.arange(0, lengths[i]).float()
                weights = self.sigmoid(self.sig_mse_weight * 
                    ((x / torch.max(x)) - 0.5)).type_as(output)
                for j, weight in enumerate(weights):
                    sample_diff = tc[i,j] - oc[i,j]
                    if self.w_g is not None:
                        sample_diff[7:14] *= self.w_g
                        sample_diff[-1] *= self.w_tau
                    sample_loss += (weight * torch.mean(sample_diff**2))
                sample_loss = sample_loss / len(weights)
            sample_loss = sample_loss / dims[0]
        
        return sample_loss


class TemporallyWeightedMSETrajLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, N_basis, dof, tau, dt, sig_mse_weight=10, 
        device='cuda:0', w_g=None, w_tau=None, extend=None):
        super(TemporallyWeightedMSETrajLoss, self).__init__()

        self.device = device
        self.extend = extend
        self.N_basis = torch.tensor(N_basis)
        self.dof = torch.tensor(dof)
        self.tau = torch.tensor(tau)
        self.dt = torch.tensor(dt)
        self.sig_mse_weight = torch.tensor(sig_mse_weight)
        self.sigmoid = nn.Sigmoid()
        if w_g is not None:
            print("Setting w_g to {} and w_tau to {}".format(w_g, w_tau))
            self.w_g = torch.tensor(w_g)
            self.w_tau = torch.tensor(w_tau)


    def compute_weights(self, frame_count):
        x = torch.arange(0, frame_count).float()
        weights = self.sigmoid(self.sig_mse_weight * 
            ((x / torch.max(x)) - 0.5))
        return weights
        

    def forward(self, output, target, target_dmp, lengths):
        dims = torch.tensor(output.shape)

        # traj = DMP_integrate_batch_2(torch.cat((
        #     output, target_dmp.unsqueeze(1)), axis=1), 
        #     self.N_basis, self.dof, self.tau, self.dt, self.device)[1]

        # traj_pred = traj[:, 0:dims[1]]
        # target = traj[:, -1]

        traj_pred = DMP_integrate_batch_2(output, self.N_basis, 
            self.dof, self.tau, self.dt, self.device)[1]

        tpc = traj_pred.clone()
        tpc[:,:,:,3:7] = normalize(traj_pred[:,:,:,3:7], dim=3)

        if self.extend:
            x = torch.arange(0, dims[1]).float().type_as(traj_pred)
            weights = self.sigmoid(self.sig_mse_weight * 
                ((x / torch.max(x)) - 0.5))
            # TODO: dodaj sqrt?
            traj_loss = ((tpc - target.unsqueeze(1))**2).sum((0,-1))
            traj_loss[:, -1] *= self.w_g
            traj_loss = traj_loss.sum(1)

            tau_diff = output[:,:,-1] - target_dmp.unsqueeze(1)[:,:,-1]
            tau_loss = (tau_diff**2).sum(0) * self.w_tau
            loss = ((traj_loss + tau_loss) * weights).sum() / torch.prod(dims)
        else:
            loss = torch.tensor(0.).to(self.device)
            for i in range(int(dims[0])):
                frame_count = int(lengths[i].item())
                weights = self.compute_weights(frame_count)
                sample_loss = torch.zeros(frame_count).to(self.device)
                tau_loss = torch.zeros(frame_count).to(self.device)

                for j in range(frame_count):
                    # Comparing trajectories spacially
                    sample_loss[j] = torch.sum((target[i, :] - traj_pred[i, j, :])**2) / target.shape[1]
                    
                    # Comparing trajectories temporally
                    t_tau = target_dmp[:, -1][i]
                    i_tau = output[i, j, -1]
                    tau_loss[j] = (t_tau - i_tau)**2
            
                sample_loss = torch.sum(sample_loss * weights) / frame_count
                loss = loss + sample_loss

            loss = loss / dims[0]
        
        return loss


class MMDLoss(torch.nn.modules.loss._Loss):

    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def gaussian_kernel(self, source, target, 
            kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])

        max_len = max(source.shape[1], target.shape[1])
        while (source.shape[1] < max_len):
            source = torch.cat((source, source[:,-1].unsqueeze(1)), axis=1)
        while(target.shape[1] < max_len):
            target = torch.cat((target, target[:,-1].unsqueeze(1)), axis=1)

        total = torch.cat([source, target], dim=0)
        total = total.view(total.shape[0], -1)
        total0 = total.unsqueeze(0).expand(int(total.size(0)),
            int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)),
            int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, 
            kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
