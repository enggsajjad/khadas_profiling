import math
import torch
import torch.nn as nn
from builder import *

class slice_reshape_operation(nn.Module):
    def __init__(self, op_name, C_in, C_out, img_w, img_h, expansion, stride):
        super(slice_reshape_operation, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.C_in = C_in
        self.op = PRIMITIVES[op_name](C_in, C_out, expansion, stride)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.select(3,0)
        x = x.select(2,0)
        x = torch.reshape(x,(batch_size, self.C_in,  self.img_h, self.img_w))
        #x = x.reshape(batch_size, self.C_in,  self.img_h, self.img_w)
        x = self.op(x)

        return x
        

op_name = "ir_k3_re"
C_in = 128
C_out = 128
img_w = 320
img_h = 320
expansion = 6
stride = 1
#generate input to test
input_x = torch.rand(1,img_w*img_h*C_in,1,3)
op = slice_reshape_operation(op_name, C_in, C_out, img_w, img_h, expansion, stride)

y = op(input_x)

torch.onnx.export(op,  # model being run
                  input_x,  
                  "test.onnx",
                  export_params=True,  # store the trained parameter weights inside the model file
                  # the ONNX version to export the model to
                  opset_version=7,
                  verbose=True,
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={"input": {0: "batch", 1:"channel",2: "width",3:"height"}}
                 )