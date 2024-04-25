# following SAINT https://github.com/cpeng93/SAINT, split train/test volume
# save volume.pt: {'image':image_h_w_s.dtype('uint16'),'spacing':(x,y,z)}, image_scale=0-4095

from medpy.io import load
import os, pickle
import numpy as np

test_set = pickle.load(open('./test_set.pt','rb'))
test_set = {k:test_set[k] for k in sorted(test_set.keys())}

src_path = 'xxx/Task03_Liver'
tgt_path = 'xxx/data_volume/Task03_Liver'

os.makedirs(tgt_path,exist_ok=True)
os.makedirs(os.path.join(tgt_path,'imagesTr'),exist_ok=True)
os.makedirs(os.path.join(tgt_path,'imagesTs'),exist_ok=True)


for inst in ['imagesTr','imagesTs']:
    file_dir = os.path.join(src_path, inst)
    patients = os.listdir(file_dir)
    for id,patient in enumerate(patients):
        if not '._' in patient:
            img, header = load(os.path.join(file_dir, patient))

            spacing = header.get_voxel_spacing()
            img = np.clip(img,-1024,img.max())
            img = img - img.min()
            img = np.clip(img,0,4095)
            img = img.astype("uint16")
            data = {'image': img, 'spacing': spacing}
            if not patient.split('.')[0] in test_set:
                pickle.dump(data, open(os.path.join(tgt_path,'imagesTr', patient.replace('.nii.gz','.pt')), 'wb'))
                print(f"{id}/{len(patients)}:volume finished, " + patient, img.shape)

            else:
                pickle.dump(data, open(os.path.join(tgt_path,'imagesTs', patient.replace('.nii.gz','.pt')), 'wb'))
                print(f"{id}/{len(patients)}:volume finished, " + patient, img.shape)

print('done')