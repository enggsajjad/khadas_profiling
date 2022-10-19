#!/bin/bash

######################################################################
#
# ENVIRONMENT SETTINGS ON HOST UBUNTU.
#
######################################################################


######################################################################
#
# !!!MUST BE CHANGED!!!
#
######################################################################

#----------------------------- REMOTE KHADAS -----------------------------
# Remote Khadas Adreess
export KHADAS_ADDRESS=10.10.254.119
# test input image
#export INPUT_IMAGE=2.bmp.jpg
#export INPUT_IMAGE=rand_0.jpg
#----------------------------- UBUNTU HOST -----------------------------





######################################################################
#
# !!!!OPTIONAL!!!! (USED TO EASE THE FLOW IMPLEMTATION)
#
######################################################################

#----------------------------- UBUNTU HOST -----------------------------
# Ubuntu Host Adreess
export HOST_ADDRESS=10.10.254.178
# Ubuntu Host User
export HOST_USER=sajjad
# the path for the conversion scripts folder
#export CONV_SCRIPTS_PATH=/home/sajjad/sajjad/npusdk2_4662/aml_npu_sdk/acuity-toolkit/demo
PDIR="$(dirname "${PWD}")"
#export CONV_SCRIPTS_PATH=${PWD}/convertdemo
export CONV_SCRIPTS_PATH=${PDIR}/convertdemo
# the folder where NBG conversion files are created
export CONV_NBG_FOLDER=nbg_unify_mnist
# the NN model appliction name
export NN_NAME=mnist
# the NN model appliction onnx files
export NN_MODEL=${NN_NAME}.onnx
# the path of the folders where default workflow scripts nbg_unify and normal_case makefile/build scripts are located
#export WORKFLOW_PROJECT=${PWD}
export WORKFLOW_PROJECT=${PDIR}
# the folders where default nbg_unify makefile/build scripts are located
export NBG_TEMPLATE_SCRIPTS=template_nbg_unify
# test input image
#export INPUT_IMAGE_PATH=${PWD}/convertdemo/data/bmp
#----------------------------- UBUNTU HOST NOTEBOOK TO ONNX -----------------------------
# the path for the jupyter notebook
#export NOTEBOOK_PATH=/home/sajjad/mnist_notebook
#export NOTEBOOK_PATH=${PWD}/notebook
export NOTEBOOK_PATH=./notebook
# jupyter notebook name
export NOTEBOOK_NAME=MnistProfile

#----------------------------- REMOTE KHADAS -----------------------------

# Remote Khadas User
export KHADAS_USER=khadas
# the path on remote khadas where NBG conversion folder is to be created and executed
export REMOTE_PROJECT_DIR=/home/khadas
# the path on remote khadas where NBG conversion folder is to be created and executed
#export NBG_DIR=nbg_unify_mnist
export NBG_DIR=${CONV_NBG_FOLDER}

