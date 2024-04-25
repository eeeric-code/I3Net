python test.py \
    --upscale 2 \
    --lr_slice_patch 4 \
    --testdata_path xxx/data_volume/Task10_Colon/imagesTs \
    --gpu_id '0' \
    --model i3net \
    --ckpt experiments/i3net/Colon_x2/pth/1500.pth \
    --ckpt_dir Colon_x2 \
    --num_workers 4