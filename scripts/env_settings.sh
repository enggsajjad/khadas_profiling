#!/bin/bash

######################################################################
#
# SETTINGS ON HOST UBUNTU.
#
######################################################################


#----------------------------- UBUNTU HOST -----------------------------
# Ubuntu Host Adreess
export HOST_ADDRESS=10.10.254.178
# Ubuntu Host User
export HOST_USER=sajjad
# the path for the conversion scripts folder
export CONV_SCRIPTS_PATH=/home/sajjad/sajjad/npusdk2_4662/aml_npu_sdk/acuity-toolkit/demo
# the folder where NBG conversion files are created
export CONV_NBG_FOLDER=nbg_unify_mnist
# the NN model appliction name
export NN_NAME=mnist
# the NN model appliction onnx files
export NN_MODEL=${NN_NAME}.onnx
# the path of the folders where default workflow scripts nbg_unify and normal_case makefile/build scripts are located
export WORKFLOW_PROJECT=/home/sajjad/sajjad/scripts
# the folders where default nbg_unify makefile/build scripts are located
export NBG_SCRIPTS=nbg_unify

#----------------------------- UBUNTU HOST NOTEBOOK TO ONNX -----------------------------
# the path for the jupyter notebook
export NOTEBOOK_PATH=/home/sajjad/mnist_notebook
# jupyter notebook name
export NOTEBOOK_NAME=Mnist

#----------------------------- REMOTE KHADAS -----------------------------
# Remote Khadas Adreess
export KHADAS_ADDRESS=10.10.254.119
# Remote Khadas User
export KHADAS_USER=khadas
# the path on remote khadas where NBG conversion folder is to be created and executed
export REMOTE_PROJECT_DIR=/home/khadas/hussain
# the path on remote khadas where NBG conversion folder is to be created and executed
export NBG_DIR=nbg_unify_mnist

