# save each slice of each volume

from medpy.io import load
import numpy as np
import nibabel as nib
import os
import pickle
datapath = 'xxx/data_volume/Task10_Colon/imagesTr/'
volume_list = os.listdir(datapath)
savepath = 'xxx/data_slice/Task10_Colon/imagesTr/'
os.makedirs(savepath,exist_ok=True)

for i,volumename in enumerate(volume_list):
    volume_path = datapath + volumename
    # pt
    data = pickle.load(open(volume_path,'rb'))
    volnp = data['image'].astype("uint16")
    spacing = data['spacing']

    savefile = savepath + volumename.split('.')[0]
    os.makedirs(savefile,exist_ok=True)

    for frame in range(volnp.shape[2]):
        vol_slice = volnp[:,:,frame]
        savename = savefile + '/' + volumename.split('.')[0] + '_slice_' + str(frame).zfill(3) + '.pt'
        pickle.dump({'image':vol_slice,'spacing':spacing},open(savename, 'wb'))

    print(f'{i} / {len(volume_list)}  ' + savefile)

print('done')
