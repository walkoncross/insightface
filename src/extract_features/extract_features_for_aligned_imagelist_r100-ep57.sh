export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

nohup python extract_features_for_aligned_imagelist.py \
    --model=/disk2/zhaoyafei/face-model-packages/face-insight/model-r100-ms1m-zyf-0221/slim/model-r100-slim,57 \
    --image-list=/disk2/data/FACE/face-idcard-1M/face-idcard-1M-image-list.txt \
    --image-dir=/disk2/data/FACE/face-idcard-1M/face-idcard-1M-mtcnn-aligned-112x112/aligned_imgs \
    --save-dir=/disk2/data/FACE/face-idcard-1M/features/insightface-r100-ep57-zyf \
    --batch-size=180 \
    --image-size=3,112,112 \
    --add-flip \
    --gpu=3 \
    --save-format=.bin \
    --flip-sim > nohup-extract-log-r100-ep57.txt &

#    --use-mean \