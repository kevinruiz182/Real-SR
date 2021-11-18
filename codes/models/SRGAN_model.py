import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel as P
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss

logger = logging.getLogger('base')

import numpy as np
import torch.nn.functional as F
### Quality mesuare ###
## LPIPS
import LPIPS.models.dist_model as dm
model_LPIPS = dm.DistModel()
model_LPIPS.initialize(model='net-lin',net='alex',use_gpu=True)
L_FM = 1            # Scaling params for the feature matching loss
L_LPIPS = 1e-3      # Scaling params for the LPIPS loss
L_ADV = 1e-3        # Scaling params for the Adv loss

### U-Net Discriminator ###
# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, wide=True,
                preactivation=True, activation=nn.LeakyReLU(0.1, inplace=False), downsample=nn.AvgPool2d(2, stride=2)):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                            kernel_size=1, padding=0)

        self.bn1 = self.which_bn(self.hidden_channels)
        self.bn2 = self.which_bn(out_channels)

    def forward(self, x):
        if self.preactivation:
            h = self.activation(x)
        else:
            h = x
        h = self.bn1(self.conv1(h))
        if self.downsample:
            h = self.downsample(h)

        return h #+ self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1, inplace=False),
                upsample=nn.Upsample(scale_factor=2, mode='nearest')):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                            kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(out_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
        h = self.bn1(self.conv1(h))
        return h


class UnetD(torch.nn.Module):
    def __init__(self):
        super(UnetD, self).__init__()

        self.enc_b1 = DBlock(3, 64, preactivation=False)
        self.enc_b2 = DBlock(64, 128)
        self.enc_b3 = DBlock(128, 192)
        self.enc_b4 = DBlock(192, 256)
        self.enc_b5 = DBlock(256, 320)
        self.enc_b6 = DBlock(320, 384)

        self.enc_out = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.dec_b1 = GBlock(384, 320)
        self.dec_b2 = GBlock(320*2, 256)
        self.dec_b3 = GBlock(256*2, 192)
        self.dec_b4 = GBlock(192*2, 128)
        self.dec_b5 = GBlock(128*2, 64)
        self.dec_b6 = GBlock(64*2, 32)

        self.dec_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        e1 = self.enc_b1(x)
        e2 = self.enc_b2(e1)
        e3 = self.enc_b3(e2)
        e4 = self.enc_b4(e3)
        e5 = self.enc_b5(e4)
        e6 = self.enc_b6(e5)

        e_out = self.enc_out(F.leaky_relu(e6, 0.1))
        # print(e1.size())
        # print(e2.size())
        # print(e3.size())
        # print(e4.size())
        # print(e5.size())
        # print(e6.size())

        d1 = self.dec_b1(e6)
        d2 = self.dec_b2(torch.cat([d1, e5], 1))
        d3 = self.dec_b3(torch.cat([d2, e4], 1))
        d4 = self.dec_b4(torch.cat([d3, e3], 1))
        d5 = self.dec_b5(torch.cat([d4, e2], 1))
        d6 = self.dec_b6(torch.cat([d5, e1], 1))

        d_out = self.dec_out(F.leaky_relu(d6, 0.1))

        return e_out, d_out, [e1,e2,e3,e4,e5,e6], [d1,d2,d3,d4,d5,d6]

class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        if self.is_train:

            self.model_D = UnetD().cuda()
            self.netG.train()
            # self.netD.train()
            self.model_D.train()

        # define losses, optimizer and scheduler
        if self.is_train:

            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0

            self.model_optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))

            self.optimizers.append(self.model_optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):

        def rand_bbox(size, lam):
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2

        def Huber(input, target, delta=0.01, reduce=True):
            abs_error = torch.abs(input - target)
            quadratic = torch.clamp(abs_error, max=delta)

            # The following expression is the same in value as
            # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
            # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
            # This is necessary to avoid doubling the gradient, since there is already a
            # nonzero contribution to the gradient from the quadratic term.
            linear = (abs_error - quadratic)
            losses = 0.5 * torch.pow(quadratic, 2) + delta * linear

            if reduce:
                return torch.mean(losses)
            else:
                return losses

        for p in self.model_D.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L.detach())

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix

            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

            # LPIPS loss
            loss_LPIPS, _ = model_LPIPS.forward_pair(self.var_ref*2-1, self.fake_H.detach()*2-1)
            loss_LPIPS = torch.mean(loss_LPIPS) * L_LPIPS

            # FM and GAN losses
            e_S, d_S, e_Ss, d_Ss = self.model_D( self.fake_H.detach() )
            _, _, e_Hs, d_Hs = self.model_D( self.var_ref )

            # FM loss
            loss_FMs = []
            for f in range(6):
                loss_FMs += [Huber(e_Ss[f], e_Hs[f])]
                loss_FMs += [Huber(d_Ss[f], d_Hs[f])]
            loss_FM = torch.mean(torch.stack(loss_FMs)) * L_FM

            # GAN loss
            loss_Advs = []
            loss_Advs += [torch.nn.ReLU()(1.0 - e_S).mean() * L_ADV]
            loss_Advs += [torch.nn.ReLU()(1.0 - d_S).mean() * L_ADV]
            loss_Adv = torch.mean(torch.stack(loss_Advs))

            l_g_total += loss_LPIPS + loss_FM + loss_Adv

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.model_D.parameters():
            p.requires_grad = True

        self.model_optimizer_D.zero_grad()

        e_S, d_S, _, _ = self.model_D( self.fake_H.detach() )
        e_H, d_H, _, _ = self.model_D( self.var_ref )

        # D Loss, for encoder end and decoder end
        loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_S).mean()
        loss_D_Enc_H = torch.nn.ReLU()(1.0 - e_H).mean()

        loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_S).mean()
        loss_D_Dec_H = torch.nn.ReLU()(1.0 - d_H).mean()

        loss_D = loss_D_Enc_H + loss_D_Dec_H

        # CutMix for consistency loss
        batch_S_CutMix = self.fake_H.detach().clone()

        # probability of doing cutmix
        p_mix = step / 100000
        if p_mix > 0.5:
            p_mix = 0.5

        if torch.rand(1) <= p_mix:
            r_mix = torch.rand(1)   # real/fake ratio

            bbx1, bby1, bbx2, bby2 = rand_bbox(batch_S_CutMix.size(), r_mix)
            batch_S_CutMix[:, :, bbx1:bbx2, bby1:bby2] = self.var_ref[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_S_CutMix.size()[-1] * batch_S_CutMix.size()[-2]))

            e_mix, d_mix, _, _ = self.model_D( batch_S_CutMix )

            loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_mix).mean()
            loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_mix).mean()

            d_S[:,:,bbx1:bbx2, bby1:bby2] = d_H[:,:,bbx1:bbx2, bby1:bby2]
            loss_D_Cons = F.mse_loss(d_mix, d_S)

            loss_D += loss_D_Cons
            self.log_dict['loss_D_Cons'] = torch.mean(loss_D_Cons).item()

        loss_D += loss_D_Enc_S + loss_D_Dec_S
        loss_D.backward()
        self.model_optimizer_D.step()

        #set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
                # self.log_dict['l_g_mean_color'] = l_g_mean_color.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['loss_Adv'] = loss_Adv.item()
            self.log_dict['loss_LPIPS'] = loss_LPIPS.item()
            self.log_dict['loss_FM'] = loss_FM.item()

        # for monitoring
        self.log_dict['loss_D'] = loss_D.item()
        self.log_dict['e_H'] = torch.mean(e_H).item()
        self.log_dict['e_S'] = torch.mean(e_S).item()
        self.log_dict['d_H'] = torch.mean(d_H).item()
        self.log_dict['d_S'] = torch.mean(d_S).item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def back_projection(self):
        lr_error = self.var_L - torch.nn.functional.interpolate(self.fake_H,
                                                                scale_factor=1/self.opt['scale'],
                                                                mode='bicubic',
                                                                align_corners=False)
        us_error = torch.nn.functional.interpolate(lr_error,
                                                   scale_factor=self.opt['scale'],
                                                   mode='bicubic',
                                                   align_corners=False)
        self.fake_H += self.opt['back_projection_lamda'] * us_error
        torch.clamp(self.fake_H, 0, 1)

    def test_chop(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.forward_chop(self.var_L)
        self.netG.train()

    def forward_chop(self, *args, shave=10, min_size=160000):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = self.opt['scale']
        n_GPUs = min(torch.cuda.device_count(), 4)
        args = [a.squeeze().unsqueeze(0) for a in args]

        # height, width
        h, w = args[0].size()[-2:]
        # print('len(args)', len(args))
        # print('args[0].size()', args[0].size())

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]
        # print('len(x_chops)', len(x_chops))
        # print('x_chops[0].size()', x_chops[0].size())

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                # print(len(x))
                # print(x[0].size())
                y = P.data_parallel(self.netG, *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:

            # print(x_chops[0].size())
            for p in zip(*x_chops):
                # print('len(p)', len(p))
                # print('p[0].size()', p[0].size())
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1:
            y = y[0]

        return y

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.model_D)
            if isinstance(self.model_D, nn.DataParallel) or isinstance(self.model_D,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.model_D.__class__.__name__,
                                                 self.model_D.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.model_D.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.model_D, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.model_D, 'D', iter_step)
