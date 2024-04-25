import config
args, unparsed = config.get_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from tqdm import tqdm

from data import trainSet
from util_evaluation import calc_psnr,calc_ssim
import datetime
from select_model import select_model
import optim
import val
from select_loss import Select_Loss

####################################################################
# seed
GLOBAL_SEED = 777
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

args.ckpt_dir = 'experiments/'+args.model+'/'+args.ckpt_dir
os.makedirs(args.ckpt_dir,exist_ok=True)

if len(args.gpu_id) > 1:
    args.parallel = True

# data
trainset = trainSet(data_root=args.traindata_path,args=args)
# batch_size = args.batch_size*len(device_ids)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                shuffle=False,num_workers=args.num_workers, pin_memory=False)

# model
model = select_model(args)

if args.resume:
    print('load weight')
    load_ckpt = torch.load(args.ckpt,map_location=torch.device('cpu'))
    model.load_state_dict(load_ckpt['state_dict'])
    print('load weight success')
    args.start_epoch = load_ckpt['epoch']

model = model.cuda()

#### optim ####
optimizer = optim.select_optim(args,model)
# optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
scheduler = optim.select_scheduler(args,optimizer)

#### loss ####
loss_function = Select_Loss(args).cuda()



########################### train ###################################
# log
with open(args.ckpt_dir + '/logs.txt',mode='a+') as f:
    s = "START EXPERIMENT\n"
    f.write(s)
    for i,j in args.__dict__.items():
        f.write(str(i)+' : '+str(j)+'\n')

# amp
if args.amp:
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

model.train()
for epoch in tqdm(range(args.start_epoch,args.max_epoch)):
    loss_epoch = 0
    psnr_epoch = 0

    for iter, hr in tqdm(enumerate(dataloader)):
                 
        if len(hr.shape) == 5:
            gt = torch.cat([i for i in hr],0) #[bz,h,w,7]
        else: gt = hr
        lr = gt[...,::args.upscale] # [bz,h,w,4]
        if torch.cuda.is_available():
            gt = Variable(gt.cuda())
            lr = Variable(lr.cuda())
        
        optimizer.zero_grad()

        if args.amp:
            with autocast():
                sr = model(lr)
                loss_iter = loss_function(sr,gt)
                scaler.scale(loss_iter).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            sr = model(lr)
            loss_iter = loss_function(sr,gt)
            loss_iter.backward()
            optimizer.step()

        psnr_iter = 0
        for bz in range(gt.shape[0]):
            psnr_iter += calc_psnr(gt[bz, :, :, :],sr[bz, :, :, :]).item()
        psnr_iter /= (bz+1)  

        #### log ####    
        lr_tmp = optimizer.state_dict()['param_groups'][0]['lr']
        log = r"epoch[{}/{}] iter[{}/{}] psnrTr:{:.6f} lossTr:{:.12f} lr:{:.12f}"\
            .format(epoch+1, args.max_epoch , \
                iter+1, len(dataloader),\
                psnr_iter,loss_iter,lr_tmp)
        now = str(datetime.datetime.now())
        print(now+' '+log)

        loss_epoch += loss_iter
        psnr_epoch += psnr_iter

    loss_epoch /= (iter+1)
    psnr_epoch /= (iter+1)

    #### lr schedule ####
    if args.schedule == 'step':
        scheduler.step()
    elif args.schedule == 'cos_lr':
        #### torch.cos_lr ####
        # scheduler.step()
        #### timm.cos_lr ####
        scheduler.step_update(epoch)
    elif args.schedule == 'Tmin':
        scheduler.step(loss_epoch)
    elif args.schedule == 'Tmax':
        scheduler.step(psnr_epoch)

    epoch +=1


    log = r"epoch[{}/{}] psnrTr:{:.6f} lossTr:{:.12f} lr:{:.12f}"\
    .format(epoch, args.max_epoch , \
        psnr_epoch,loss_epoch,lr_tmp)
    now = str(datetime.datetime.now())
    print(now+' '+log)
    with open(args.ckpt_dir + '/logs.txt',mode='a+') as f:
        f.write('\n'+now+log)

    if epoch >int(0.99*args.max_epoch):
        os.makedirs(args.ckpt_dir+'/pth',exist_ok=True)
        try:
            torch.save({'epoch': epoch, 'state_dict': model.module.state_dict()}, args.ckpt_dir + '/pth/' + str(epoch).zfill(4) + '.pth')
        except:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, args.ckpt_dir + '/pth/' + str(epoch).zfill(4) + '.pth')


# val_opt = True
# if val_opt:
#     val.val(args=args,model=model)