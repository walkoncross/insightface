export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

nohup python extract_features_for_aligned_imagelist.py \
    --model=../../models/model-r34-amf/model,0 \
    --image-list=/disk2/data/FACE/face-idcard-1M/face-idcard-1M-image-list.txt \
    --image-dir=/disk2/data/FACE/face-idcard-1M/face-idcard-1M-mtcnn-aligned-112x112/aligned_imgs \
    --save-dir=/disk2/data/FACE/face-idcard-1M/features/insightface-r34-amf \
    --batch-size=256\
    --image-size=3,112,112 \
    --add-flip \
    --gpu=0 \
    --save-format=.bin \
    --flip-sim > nohup-extract-log.txt &

#    --use-mean \