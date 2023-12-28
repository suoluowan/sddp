#! /bin/bash
# python3 train_net.py --config-file configs/densepose_rcnn_R_101_FPN_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/densepose_R_101_FPN_s1x/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_101_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/densepose_R_101_s1x/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/densepose_R_101_FPN_DL_s1x/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_50_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/densepose_R_50_s1x_4gpu-4batch_180000_0.005lr/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/densepose_R_50_FPN_DL_s1x/"
# python3 train_net.py --config-file configs/HRNet/densepose_rcnn_HRFPN_HRNet_w32_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/densepose_rcnn_HRFPN_HRNet_w32_s1x/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_50_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/densepose_R_50_s1x_4gpu_130000_0.01lr_4batch/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_50_FPN_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/densepose_R_50_FPN_s1x_4batch_0.01lr/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/densepose_R_50_FPN_DL_s1x_4batch_0.01lr/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_101_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/densepose_R_101_s1x_4batch_0.01lr/"
# python3 train_net.py --config-file configs/densepose_rcnn_R_101_FPN_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/densepose_R_101_FPN_s1x_4batch_0.01lr/"



# python3 train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 SOLVER.IMS_PER_BATCH 128 OUTPUT_DIR "./output/densepose_R_101_FPN_DL_s1x_32batch_0.01lr/"

python3 train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml --num-gpus 4 --resume SOLVER.IMS_PER_BATCH 32 SOLVER.MAX_ITER 90000 SOLVER.STEPS 60000,80000 SOLVER.BASE_LR 0.1 OUTPUT_DIR "./output/densepose_R_101_FPN_DL_s1x_8batch_0.1lr_90000/"

# python3 train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 SOLVER.IMS_PER_BATCH 32 SOLVER.MAX_ITER 90000 SOLVER.STEPS 60000,80000 OUTPUT_DIR "./output/densepose_R_101_FPN_DL_s1x_8batch_0.01lr_90000/"

# python3 train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml --num-gpus 4 --resume SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/densepose_R_101_FPN_DL_s1x_4batch_0.01lr/"