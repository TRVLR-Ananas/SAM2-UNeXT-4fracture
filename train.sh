CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\pretrained\\sam2_hiera_large.pt" \
--dinov2_path "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\pretrained\\model.safetensors" \
--train_image_path "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TrainImageMask\\TrainImage\\" \
--train_mask_path "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TrainImageMask\\TrainMask\\" \
--save_path "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\Checkpoint\\" \
--epoch 50 \
--lr 0.0002 \
--batch_size 1