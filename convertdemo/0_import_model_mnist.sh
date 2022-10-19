#!/bin/bash

NAME=mnist
ACUITY_PATH=../bin/
#in=$1
imgSize="$1"
convert_onnx=${ACUITY_PATH}convertonnx

#inputlist="3,${imgSize},${imgSize}"
inputlist="${imgSize}"

$convert_onnx \
   --onnx-model  ./network/${NAME}.onnx \
   --inputs "input" \
   --input-size-list ${inputlist} \
   --outputs "output" \
   --input-dtype-list "float" \
   --net-output ${NAME}.json \
   --data-output ${NAME}.data  \
   --size-with-batch "1"
