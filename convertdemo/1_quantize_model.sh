#!/bin/bash

NAME=yolov3
ACUITY_PATH=../bin/

tensorzone=${ACUITY_PATH}tensorzonex

<<COMMENT
$tensorzone \
    --action quantization \
    --dtype float32 \
    --source text \
    --source-file data/validation_tf.txt \
    --channel-mean-value '0 0 0 256' \
    --model-input ${NAME}.json \
    --model-data ${NAME}.data \
    --quantized-dtype dynamic_fixed_point-i8 \
    --quantized-rebuild \
#    --batch-size 2 \
#    --epochs 5
COMMENT

$tensorzone \
    --action quantization \
    --dtype float32 \
    --source text \
    --source-file data/validation_tf.txt \
    --channel-mean-value '0 0 0 256' \
    --reorder-channel '2 1 0' \
    --model-input ${NAME}.json \
    --model-data ${NAME}.data \
    --quantized-dtype dynamic_fixed_point-i8 \
    --quantized-rebuild \
#    --batch-size 2 \
#    --epochs 5

#Note: 
#	1.--quantized-dtype asymmetric_affine-u8 , you can set dynamic_fixed_point-i8 asymmetric_affine-u8 dynamic_fixed_point-i16(s905d3 not support point-i16) perchannel_symmetric_affine-i8(only for t965d4/t982ar301)
#	2.default batch-size(100),epochs(1) ,the numbers of pictures in data/validation_tf.txt must equal to batch-size*epochs,if you set the epochs >1
#	3.Other parameters settings, Refer to sectoin 3.4(Step 2) of the  <Model_Transcoding and Running User Guide_V0.8> documdent


