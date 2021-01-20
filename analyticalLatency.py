import os
import sys
import time
import math
import math
from models import *
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from tqdm import tqdm

def computeCycles(numInput, numTime, numFilter, arrX, arrY):
        ## Utilization and Max Bandwidth
        numFolds = 0 ## To compute avearge
        weightedUtili = 0.0
        avgUtilization = 0.0
        maxBandwidth = 0.0
        maxReadBandwidth = 0.0
        maxWriteBandwidth = 0.0

        cycles = 0
        cycles = (numInput//arrX) * (numFilter//arrY) * (numTime + arrX + arrY - 1)
        numFolds = (numInput//arrX) * (numFilter//arrY)
        weightedUtili = (numInput//arrX) * (numFilter//arrY) * 1.0

        if numFolds != 0:
            maxReadBandwidth = arrX * numTime + numTime * arrY
            maxWriteBandwidth = arrX * arrY

        if numInput % arrX > 0:
            cycles = cycles + (numFilter//arrY) * (numTime + (numInput % arrX) + arrY - 1)
            numFolds += (numFilter//arrY)
            weightedUtili += (numFilter//arrY) * ((numInput % arrX)*arrY/(arrX * arrY))

            maxReadBandwidth = max(maxReadBandwidth, (numInput % arrX) * numTime + numTime * arrY)
            maxWriteBandwidth = max(maxWriteBandwidth, (numInput % arrX)*arrY)

        if numFilter % arrY > 0:
            cycles = cycles + (numInput//arrX) * (numTime + arrX + (numFilter % arrY) - 1)
            numFolds += (numInput//arrX)
            weightedUtili += (numInput//arrX) * (arrX*(numFilter % arrY)/(arrX * arrY))

            maxReadBandwidth = max(maxReadBandwidth, arrX * numTime + numTime * (numFilter % arrY) )
            maxWriteBandwidth = max(maxWriteBandwidth, arrX*(numFilter % arrY))

        if numInput % arrX > 0 and numFilter % arrY > 0:
            cycles = cycles + (numTime + (numInput % arrX) + (numFilter % arrY) - 1)
            numFolds += 1
            weightedUtili += 1 * ((numInput % arrX)*(numFilter % arrY)/(arrX * arrY))

            maxReadBandwidth = max(maxReadBandwidth, (numInput % arrX) * numTime + numTime * (numFilter % arrY))
            maxWriteBandwidth = max(maxWriteBandwidth, (numInput % arrX)*(numFilter % arrY))

        avgUtilization = weightedUtili/numFolds
        return cycles, avgUtilization, maxReadBandwidth, maxWriteBandwidth

class Bandwidth:
    def __init__(self):
        self.layerReadBW = []
        self.layerWriteBW = []
        self.readBW = 0
        self.writeBW = 0

    def update(self, read, write):
        self.layerReadBW.append(read)
        self.layerWriteBW.append(write)
        self.readBW = max(read, self.readBW)
        self.writeBW = max(write, self.writeBW)

def gemmCycles(dimension_rows, dimension_cols, ifmap_h, ifmap_w, filt_h, filt_w,
            num_channels, strides, num_filt, batch_size = 1):
        N = batch_size
        H = ifmap_h
        W = ifmap_w
        C = num_channels
        M = num_filt
        R = filt_h
        S = filt_w
        Stride = strides
        arrX = dimension_rows
        arrY = dimension_cols

        E = (H - R + Stride)//Stride
        F = (W - S + Stride)//Stride

        ## Reduce to Mat mul of A x B and  B X C - Forward Pass (M x RSC with RSC x NEF to get M x NEF)
        ## Assuming M1: numFilter * numTime, M2: numTime * numInput
        numInput = N * E * F
        numTime  = R * S * C
        numFilter= M

        fcycles, favgUtilization, fmaxReadBW, fmaxWriteBW = computeCycles(numInput, numTime, numFilter, arrX, arrY)

        ## Input Gradients (RSC x M with M x NEF to get RSC x NEF)
        numInput  = N * E * F
        numTime   = M
        numFilter = R * S * C

        inp_grad_cycles, inp_grad_utilization, imaxReadBW, imaxWriteBW = computeCycles(numInput, numTime, numFilter, arrX, arrY)

        ## Weight Gradients (M x NEF with NEF x RSC to get M x RSC)
        numInput  = R * S * C
        numTime   = N * E * F
        numFilter = M

        wgt_grad_cycles, wgt_grad_utilization, wmaxReadBW, wmaxWriteBW = computeCycles(numInput, numTime, numFilter, arrX, arrY)
        #Vin: Returning only Forward Pass bandwidths for now -- if required, we can others
        return math.ceil(fcycles), favgUtilization, math.ceil(inp_grad_cycles), inp_grad_utilization, math.ceil(wgt_grad_cycles), wgt_grad_utilization, fmaxReadBW, fmaxWriteBW

class Latency:
    def __init__(self):
        self.flayerwise = []
        self.ilayerwise = []
        self.wlayerwise = []
        self.time = 0
        self.pointwiseConv = 0
        self.depthwiseConv = 0
        self.otherConv = 0

class Utilization:
    def __init__(self):
        self.layerwiseUtilization = []
        self.avgUtilization = 0.0
        self.layercount = 0
        self.sum = 0.0

    def update(self, utili):
        self.layerwiseUtilization.append(utili)
        self.sum += utili
        self.layercount += 1
        self.avgUtilization = self.sum/self.layercount

    def updateFuSe(self, utili):
        self.layerwiseUtilization.append(utili)
        self.sum += 0.5 * utili
        self.layercount += 0.5
        self.avgUtilization = self.sum/self.layercount

class ForwardHook:
    def __init__(self, arraySizeX, arraySizeY, hardware, numNPU = 1, parallel_strategy = None):
        self.latency = Latency()
        self.utilize = Utilization()
        self.bandwidth = Bandwidth()
        self.arraySizeX = arraySizeX
        self.arraySizeY = arraySizeY
        self.numNPU = numNPU
        self.parallel_strategy = parallel_strategy
        assert hardware == 'FuSe'or hardware == 'Systolic'
        self.hardware = hardware

    def __call__(self, module, module_in, module_out):
        if isinstance(module, nn.Conv2d):
            inT = module_in[0]
            inDim_h, inDim_w = (inT.shape[2], inT.shape[3])
            batch_size = 1
            if self.parallel_strategy == "Data":
                batch_size = inT.shape[0] // self.numNPU
            inC = module.in_channels
            outC = module.out_channels
            k_h, k_w = module.kernel_size
            s_h, s_w = module.stride
            p_h, p_w = module.padding
            g = module.groups
            inDim_h = inDim_h + 2*p_h
            inDim_w = inDim_w + 2*p_w
            outDim_h = (inDim_h - k_h + s_h) / s_h
            outDim_w = (inDim_w - k_w + s_w) / s_w

            # Groups == 1. Normal Convolution. Maps as GEMM op on Systolic and FuSe.
            if g == 1:
                t, u, it, iu, wt, wu, r, w = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY,
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=inC,strides=s_h, num_filt=outC, batch_size = batch_size)
                # print('Group=1 ', inDim_h, inDim_w, k_h, k_w, inC, outC, t)
                self.utilize.update(u)
                self.bandwidth.update(r,w)
                if k_h == 1 and k_w == 1:
                    self.latency.pointwiseConv += t
                else:
                    self.latency.otherConv += t
                self.latency.flayerwise.append(t)
                self.latency.ilayerwise.append(it)
                self.latency.wlayerwise.append(wt)

            # Groups != 1. Therefore its a Depthwise Convolution. (PitFall)
            else:
                # If Systolic Hardware: Do Poor Utiliation GEMM. With 1 channel and 1 filter.
                if self.hardware == 'Systolic':
                    t, u, it, iu, wt, wu, r, w = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY,
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=1,strides=s_h, num_filt=1, batch_size = batch_size)
                    t = t*outC
                    it = it*outC
                    wt = wt*outC
                    self.utilize.update(u)
                    self.bandwidth.update(r,w)
                    self.latency.depthwiseConv += t
                    self.latency.flayerwise.append(t)
                    self.latency.ilayerwise.append(it)
                    self.latency.wlayerwise.append(wt)

                elif self.hardware == 'FuSe':
                    # On FuSe, If its spatial DW conv, do poor utilization GEMM
                    # Else with FuSe networks, do FuseConv
                    if k_h != 1 and k_w != 1:
                        t, u, it, iu, wt, wu, r, w = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY,
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=1,strides=s_h, num_filt=1, batch_size = batch_size)
                        t = t*outC
                        it = it*outC
                        wt = wt*outC
                        self.utilize.update(u)
                        self.bandwidth.update(r,w)
                        self.latency.depthwiseConv += t
                        self.latency.flayerwise.append(t)
                        self.latency.ilayerwise.append(it)
                        self.latency.wlayerwise.append(wt)
                    # Case: 1 x K kernel
                    elif k_h == 1:
                        # No of 1D convs performed.
                        # No of Folds in X direction.
                        # No of Folds in Y direction.
                        # Time for one Fold  (Compute happens only in Y direction.0)
                        # accommodating batch-size
                        num1Dconv = batch_size * inDim_h * outC
                        numFoldsX = num1Dconv/self.arraySizeX
                        numFoldsY = inDim_w/self.arraySizeY
                        oneFoldTime = self.arraySizeY + k_w

                        t = math.ceil((math.ceil(numFoldsX)/s_w)*(oneFoldTime*math.ceil(numFoldsY)))

                        #Calculating Weight Gradients (Convolution between Output gradients and Input (Transposed?))
                        numInput  = k_h * k_w
                        numTime   = batch_size * outDim_h * outDim_w
                        numFilter = 1

                        wt, wu, rb, wb = computeCycles(numInput, numTime, numFilter, self.arraySizeX, self.arraySizeY)

                        wt = wt*outC

                        ## Calculating Input Gradients
                        num1Dconvinp = batch_size * outDim_h * inC
                        numFoldsXinp = num1Dconvinp/self.arraySizeX
                        numFoldsYinp = outDim_w/self.arraySizeY
                        oneFoldTime = self.arraySizeY + k_w

                        it = math.ceil((math.ceil(numFoldsXinp)/s_w)*(oneFoldTime*math.ceil(numFoldsYinp)))

                        self.latency.depthwiseConv += t
                        self.latency.flayerwise.append(t)
                        self.latency.wlayerwise.append(wt)
                        self.latency.ilayerwise.append(it)

                        ## Utilization
                        outDim_h = inDim_h/s_h
                        outDim_w = (inDim_w-k_w+1)/s_w

                        if outDim_h >= self.arraySizeX:
                            if outDim_w >= self.arraySizeY:
                                u = 1.0
                                r = arraySizeX * (arraySizeY + k_w)
                                w = arraySizeX * outDim_w
                            else:
                                u = outDim_w/self.arraySizeY
                                r = arraySizeX * (outDim_w + k_w)
                                w = arraySizeX * outDim_w
                        else:
                            if outDim_w >= self.arraySizeY:
                                u = outDim_h/self.arraySizeX
                                r = outDim_h * (arraySizeY + k_w)
                                w = outDim_h * arraySizeY
                            else:
                                u = (outDim_h/self.arraySizeX)*(outDim_w/self.arraySizeY)
                                r = outDim_h * (outDim_w + k_w)
                                w = outDim_h * outDim_w

                        self.utilize.updateFuSe(u)
                        self.bandwidth.update(r,w)
                    # Case : K x 1 kernel
                    # Accommodating batch size
                    elif k_w == 1:
                        num1Dconv = batch_size * inDim_w * outC
                        numFoldsX = num1Dconv/self.arraySizeY
                        numFoldsY = inDim_h/self.arraySizeX
                        oneFoldTime = self.arraySizeX + k_h
                        t = math.ceil((math.ceil(numFoldsX)/s_h)*(oneFoldTime*math.ceil(numFoldsY)))

                        ## Calculating Weight Gradients
                        numInput  = k_h * k_w
                        numTime   = batch_size * outDim_h * outDim_w
                        numFilter = 1

                        wt, wu, rb, wb = computeCycles(numInput, numTime, numFilter, self.arraySizeX, self.arraySizeY)

                        wt = wt*outC

                        ## Calculating Input Gradients
                        num1Dconvinp = batch_size * outDim_w * inC
                        numFoldsXinp = num1Dconvinp/self.arraySizeX
                        numFoldsYinp = outDim_h/self.arraySizeY
                        oneFoldTime = self.arraySizeY + k_h

                        it = math.ceil((math.ceil(numFoldsXinp)/s_h)*(oneFoldTime*math.ceil(numFoldsYinp)))

                        self.latency.depthwiseConv += t
                        self.latency.flayerwise.append(t)
                        self.latency.wlayerwise.append(wt)
                        self.latency.ilayerwise.append(it)


                        ## Utilization
                        outDim_h = (inDim_h-k_h+1)/s_h
                        outDim_w = (inDim_w)/s_w

                        if outDim_w >= self.arraySizeX:
                            if outDim_h >= self.arraySizeY:
                                u = 1.0
                                r = arraySizeY * (arraySizeX + k_h)
                                w = arraySizeX * arraySizeY
                            else:
                                u = outDim_h/self.arraySizeY
                                r = arraySizeY * (outDim_h + k_h)
                                w = arraySizeX * outDim_h
                        else:
                            if outDim_h >= self.arraySizeY:
                                u = outDim_w/self.arraySizeX
                                r = outDim_w * (arraySizeX + k_h)
                                w = outDim_w * arraySizeX
                            else:
                                u = (outDim_w/self.arraySizeX)*(outDim_h/self.arraySizeY)
                                r = outDim_w * (outDim_h + k_h)
                                w = outDim_w * outDim_h
                        self.utilize.updateFuSe(u)
                        self.bandwidth.update(r,w)

            self.latency.time += t

        elif isinstance(module, nn.Linear):
            inT = module_in[0]
            inDim_h, inDim_w = (inT.shape[0], inT.shape[1])
            #assert inDim_h == 1 ## Why is this assert there?
            inC = module.in_features
            outC = module.out_features
            t, u, it, iu, wt, wu, r, w = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY,
                                ifmap_h=1, ifmap_w=1,
                                filt_h=1, filt_w=1,
                                num_channels=inC,strides=1, num_filt=outC)
            ## Not updating the layerwise time as of now! ~Vin
            self.utilize.update(u)
            self.bandwidth.update(r,w)
            self.latency.otherConv += t
            self.latency.time += t

    def clear(self):
        self.latency = Latency()
        self.utilize = Utilization()
        self.bandwidth = Bandwidth()

def getModelLatency(model, x, arraySizeX=8, arraySizeY=8, hardware='Systolic', numNPU=1, parallel_strategy="Data"):
    fhookfn = ForwardHook(arraySizeX, arraySizeY, hardware, numNPU, parallel_strategy)
    #bhookfn = ForwardHook(arraySizeX, arraySizeY, hardware, numNPU, parallel_strategy)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(fhookfn)
            #layer.register_backward_hook(bhookfn)
        elif isinstance(layer, nn.Linear):
            layer.register_forward_hook(fhookfn)
            #layer.register_backward_hook(bhookfn)
    model(x)
    latency = fhookfn.latency.time
    fwd_pass = fhookfn.latency.flayerwise
    wgt_grad = fhookfn.latency.wlayerwise
    inp_grad = fhookfn.latency.ilayerwise
    #back_inp_grad = bhookfn.latency.inplayerwise
    #back_wgt_grad = bhookfn.latency.wgtlayerwise
    fhookfn.clear()
    #bhookfn.clear()
    return latency, fwd_pass, wgt_grad, inp_grad
    #return latency, fwd_pass, back_inp_grad, back_wgt_grad

def getModelBandw(model, x, arraySizeX=8, arraySizeY=8, hardware='Systolic'):
    hookfn = ForwardHook(arraySizeX, arraySizeY, hardware)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hookfn)
        elif isinstance(layer, nn.Linear):
            layer.register_forward_hook(hookfn)
    model(x)
    readBWlist = hookfn.bandwidth.layerReadBW
    writeBWlist = hookfn.bandwidth.layerWriteBW
    hookfn.clear()
    return readBWlist, writeBWlist

def getModelUtili(model, x, arraySizeX=8, arraySizeY=8, hardware='Systolic', numNPU=1, parallel_strategy = "Data"):
    hookfn = ForwardHook(arraySizeX, arraySizeY, hardware, numNPU, parallel_strategy)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hookfn)
        elif isinstance(layer, nn.Linear):
            layer.register_forward_hook(hookfn)
    model(x)
    layerUtili = hookfn.utilize.layerwiseUtilization
    avgUtili = hookfn.utilize.avgUtilization
    hookfn.clear()
    return layerUtili, avgUtili

# def getModelLatencyBreakdown(model, x, mode='analytical', arraySize=8):
#     hookfn = ForwardHook(arraySize, mode)
#     for layer in model.modules():
#         if isinstance(layer, nn.Conv2d):
#             layer.register_forward_hook(hookfn)
#         elif isinstance(layer, nn.Linear):
#             layer.register_forward_hook(hookfn)
#     model(x)
#     totalLatency = hookfn.time
#     otherConvLatency = hookfn.otherConv
#     pointConvLatency = hookfn.pointwiseConv
#     depthConvLatency = hookfn.depthwiseConv
#     linearLatency = hookfn.linear
#     hookfn.clear()
#     return otherConvLatency, pointConvLatency, depthConvLatency, linearLatency


import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.orca.config.use_xvfb = True

def bandwidth_and_utilization(net, x, arrayX, arrayY, hardware='Systolic'):
    namestr = 'Layer'

    read,write = getModelBandw(net, x, arrayX, arrayY, hardware)

    layerUtili, avgUtil = getModelUtili(net, x, arrayX, arrayY, hardware, numNPU=1)


    layerN = [namestr+str(i) for i in range(len(layerUtili))]

    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
            go.Bar(x = layerN, y = read,  name = "Read Bandwidth"),
         #   go.Bar(x = layerN, y = write, name = "Write Bandwidth"),
            secondary_y=False,
    )

    figure.add_trace(
            go.Bar(x = layerN, y = write, name = "Write Bandwidth"),
            secondary_y = False,
    )

    figure.add_trace(
            go.Scatter(x = layerN, y = layerUtili, line=dict(color='grey', width=4, dash='dashdot'), name = "Utilization"),
            secondary_y=True,
    )

    figure.update_xaxes(title_text="MobileNetV3-Large")
    figure.update_yaxes(title_text="Bandwidth", secondary_y=False)
    figure.update_yaxes(title_text="Utilization", secondary_y=True)
    figure.update_layout(width = 1000, height = 500)

    figure.write_image('bandwidth_util.png')




def main():
    net = MobileNetV3Friendly('large', 1000)
    x = torch.randn([1,3,224,224])

    bandwidth_and_utilization(net,x, arrayX=256, arrayY=256, hardware='FuSe')
    #x = torch.randn([1,3,224,224])
    #hardware = 'Systolic' ## or 'FuSe'
    #arraySizeX = 256
    #arraySizeY = 256
    #from models import *
    #supernet = [MobileNetV1(1000), MobileNetV2(1000), MnasNet(1000), MobileNetV3('small', 1000), MobileNetV3('large', 1000)]
    #supernetf1 = [MobileNetV1Friendly(1000), MobileNetV2Friendly(1000), MnasNetFriendly(1000), MobileNetV3Friendly('small', 1000), MobileNetV3Friendly('large', 1000)]
    #l = 0
    #numNPU = 1
    #for net in supernet:
    #    l, fl, wl, il = getModelLatency(net, x, arraySizeX, arraySizeY, 'Systolic', numNPU=numNPU)
    #    print(l, sum(fl), sum(wl), sum(il))
    #    a, b = getModelBandw(net, x, arraySizeX, arraySizeY, hardware)
    #    # print( getModelLatency(net, x, arraySizeX, arraySizeY, hardware))
    #    #a, b = getModelUtili(net, x, arraySizeX, arraySizeY, hardware)
    #    #print(b)
    #print("Friendly")
    #for net in supernetf1:
    #    ll, fl, wl, il = getModelLatency(net, x, arraySizeX, arraySizeY, 'FuSe', numNPU=numNPU)
    #    print(ll, sum(fl), sum(wl), sum(il), l/ll)
    #    c, d = getModelBandw(net, x, arraySizeX, arraySizeY, 'FuSe')
    #    # print( getModelLatency(net, x, arraySizeX, arraySizeY, hardware))
    #    #a, b = getModelUtili(net, x, arraySizeX, arraySizeY, 'FuSe')
    #    #print(b)

    #for i in range(len(a)):
    #    if a[i] != c[i]:
    #        print(a[i], c[i], b[i], d[i])

if __name__ == '__main__':
    main()
