#!/bin/bash

echo "-------------------- EXPORT CASE"

NAME=mnist
ACUITY_PATH=../bin/

export_ovxlib=${ACUITY_PATH}ovxgenerator

$export_ovxlib \
    --model-input ${NAME}.json \
    --data-input ${NAME}.data \
    --reorder-channel '0 1 2' \
    --channel-mean-value '127 127 127 255' \
    --export-dtype quantized \
    --optimize VIPNANOQI_PID0X88  \
    --viv-sdk ${ACUITY_PATH}vcmdtools \
    --pack-nbg-unify  \

# $export_ovxlib \
#     --model-input ${NAME}.json \
#     --data-input ${NAME}.data \
#     --model-quantize ${NAME}-markus-mod.quantize \
#     --reorder-channel '0 1 2' \
#     --channel-mean-value '0 0 0 256' \
#     --export-dtype quantized \
#     --optimize VIPNANOQI_PID0X88  \
#     --viv-sdk ${ACUITY_PATH}vcmdtools \
#     --pack-nbg-unify  \



#Note:
#	 --optimize VIPNANOQI_PID0XB9  
#	when exporting nbg case for different platforms, the paramsters are different.
#   you can set VIPNANOQI_PID0X7D	VIPNANOQI_PID0X88	VIPNANOQI_PID0X99
#				VIPNANOQI_PID0XA1	VIPNANOQI_PID0XB9	VIPNANOQI_PID0XBE	VIPNANOQI_PID0XE8
#	Refer to sectoin 3.4(Step 3) of the  <Model_Transcoding and Running User Guide_V0.8> documdent


rm -rf nbg_unify_${NAME}

mv ../*_nbg_unify nbg_unify_${NAME}

cd nbg_unify_${NAME}

mv network_binary.nb ${NAME}.nb

cd ..

#save normal case demo export.data 
mkdir -p ${NAME}_normal_case_demo
mv  *.h *.c .project .cproject *.vcxproj BUILD *.linux *.export.data ${NAME}_normal_case_demo

# delete normal_case demo source
#rm  *.h *.c .project .cproject *.vcxproj  BUILD *.linux *.export.data

rm *.data ${NAME}.quantize *.json


