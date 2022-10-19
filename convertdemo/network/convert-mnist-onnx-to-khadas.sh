#!/bin/bash
#imgSize=28
echo -e "\e[35m>>>>> Entering convert-mnist-onnx-to-khadas.sh ... <<<<<<<\e[37m"

CONVERT_PATH="/acuity-toolkit/convertdemo/"
cd $CONVERT_PATH

source ./network/imgSize.config

bash 0_import_model_mnist.sh ${imgSize} && bash 1_quantize_model_mnist.sh && bash 2_export_case_code_mnist.sh

echo -e "\e[35m>>>>> Exiting convert-mnist-onnx-to-khadas.sh ... <<<<<<<\e[37m"
