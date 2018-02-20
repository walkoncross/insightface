set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python extract_features_for_aligned_imagelist.py ^
    --model=C:/zyf/dnn_models/face_models/insight-face/model-r34-amf/model,0 ^
    --image-list=../../test_data/face_chips_list.txt ^
    --image-dir=../../test_data/face_chips_112x112 ^
    --save-dir=./rlt-features-r34-amf ^
    --batch-size=4 ^
    --image-size=3,112,112 ^
    --add-flip ^
    --gpu=0 ^
    --save-format=npy ^
    --flip-sim

:    --use-mean ^
