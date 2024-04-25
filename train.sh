python -W ignore main.py \
    --upscale 2 \
    --lr_slice_patch 4 \
    --traindata_path xxx/data_slice/Task10_Colon/imagesTr \
    --testdata_path xxx/data_volume/Task10_Colon/imagesTs \
    --ckpt_dir Colon_x2 \
    --batch_size 6 \
    --gpu_id '0' \
    --model 'i3net' \
    --num_workers 4