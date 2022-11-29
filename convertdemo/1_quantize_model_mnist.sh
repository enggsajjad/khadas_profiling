#!/bin/bash

NAME=mnist
ACUITY_PATH=../bin/

tensorzone=${ACUITY_PATH}tensorzonex

echo "-------------------- QUANTIZATION SCRIPT"

$tensorzone \
    --action quantization \
    --dtype float \
    --source text \
    --source-file dataset/dataset1.txt \
    --channel-mean-value '127 127 127 255' \
    --model-input ${NAME}.json \
    --model-data ${NAME}.data \
    --quantized-dtype asymmetric_affine-u8 \
    --quantized-rebuild-all \
    --batch-size 1 \
#   --quantized-rebuild \
#    --epochs 5

#Note: 
#	1.--quantized-dtype asymmetric_affine-u8 , you can set dynamic_fixed_point-i8 asymmetric_affine-u8 dynamic_fixed_point-i16(s905d3 not support point-i16) perchannel_symmetric_affine-i8(only for t965d4/t982ar301)
#	2.default batch-size(100),epochs(1) ,the numbers of pictures in data/validation_tf.txt must equal to batch-size*epochs,if you set the epochs >1
#	3.Other parameters settings, Refer to sectoin 3.4(Step 2) of the  <Model_Transcoding and Running User Guide_V0.8> documdent


