#!/bin/bash

echo "Setting up golabl variables for profiling..."
export VIVANTE_SDK_DIR=/home/khadas/hussain/Just_for_get_op_time/data/vcmdtools

export LD_LIBRARY_PATH=/home/khadas/hussain/Just_for_get_op_time/data/drivers_64_exportdata

export VIV_VX_DEBUG_LEVEL=1

export CNN_PERF=1

export NN_LAYER_DUMP=1

echo "display the set variables...."
echo $VIVANTE_SDK_DIR
echo $LD_LIBRARY_PATH
echo $VIV_VX_DEBUG_LEVEL
echo $CNN_PERF
echo $NN_LAYER_DUMP


