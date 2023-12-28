# python3 train_net.py --config-file configs/densepose_rcnn_R_50_s1x_sds_img.yaml --num-gpus 4 --resume OUTPUT_DIR "./output/densepose_R_50_s1x_4gpu_180000_0.01lr_4batch_sds-img-at-1e-7_new/"

# python3 train_net.py --config-file configs/densepose_rcnn_R_50_s1x_pds_img.yaml --num-gpus 4 --resume OUTPUT_DIR "./output/densepose_R_50_s1x_4gpu_180000_0.01lr_4batch_pds-img-at-1e-7_new/"

# python3 train_net.py --config-file configs/densepose_rcnn_R_50_s1x_pds_img.yaml --num-gpus 4 --resume MODEL.ROI_DENSEPOSE_SDS_HEAD.CFGDIR "./configs/ldm/ldm-kl-112x112x3-cond-pds-2.yaml" MODEL.ROI_DENSEPOSE_SDS_HEAD.SDS_LOSS_WEIGHT 0.0000001 OUTPUT_DIR "./output/densepose_R_50_s1x_4gpu_180000_0.01lr_4batch_pds-img-roundI-at-1e-7_new/" 

python3 train_net.py --config-file configs/densepose_rcnn_R_50_s1x_sds_img.yaml --num-gpus 4 --resume MODEL.ROI_DENSEPOSE_SDS_HEAD.SDS_LOSS_WEIGHT 0.0000001 OUTPUT_DIR "./output/densepose_R_50_s1x_4gpu_180000_0.01lr_4batch_sdsnew-img-dp-mask2-uncond0-gumbel-20_new_260k/" 