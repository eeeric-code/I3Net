import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.optim.lr_scheduler as lrs

###### optim ######
def select_optim(opt,net):
    lr = opt.lr    
    if opt.optim == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                    lr=lr, weight_decay=opt.wd, momentum=0.9)
        # print('================== SGD lr = %.6f ==================' % lr)

    elif opt.optim == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                      lr=lr, betas=(opt.beta1,opt.beta2), eps=opt.eps)
        # print('================== Adam lr = %.6f ==================' % lr)

    elif opt.optim == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                      lr=lr, weight_decay=opt.wd)
        # print('================== AdamW lr = %.6f ==================' % lr)
    
    return optimizer

###### optim ######
def select_scheduler(opt,optimizer):
    if opt.schedule == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=opt.lr_decay,
            gamma=opt.gamma
        )
    if opt.schedule == 'cos_lr':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Tmax, \
        #                                                eta_min=opt.lr / opt.lr_gap)
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=opt.max_epoch,
                                      lr_min=opt.lr/10,
                                      warmup_lr_init=opt.lr/100,
                                      warmup_t=int(opt.max_epoch * opt.warmup_epoch),
                                      cycle_limit=1,
                                      t_in_epochs=False,
        )

    elif opt.schedule == 'Tmin':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=opt.patience, threshold=0.000001)
    elif opt.schedule == 'Tmax':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                               patience=opt.patience, threshold=0.000001)

    return scheduler