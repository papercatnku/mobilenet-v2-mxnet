import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
# recursive convolution block
class RecurConvBlock(nn.HybridBlock):
    def __init__(self, untied_c, tied_c, recur_n=3, **kwargs):
        super(RecurConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.untied_conv = nn.Conv2D(
            channels=untied_c,
            kernel_size=3,
            strides=(1,1),
            padding=(1,1)
            )
            self.tied_conv = nn.Conv2D(
                channels=tied_c,
                kernel_size=3,
                strides=(1,1),
                padding=(1,1)
            )
        self.recur_n = recur_n
        return
        
    def hybrid_forward(self, F, x):
        out =F.relu(self.untied_conv(x))
        for _ in range(self.recur_n):
            out = F.relu(self.tied_conv(out))
        return out

# Used for mobienet structure
def ConvBlock(channels, kernel, stride, pad):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, kernel, strides=stride, padding=pad, use_bias=False),
        nn.BatchNorm(scale=True),
        nn.Activation('relu')
    )
    return out

def Conv1x1(channels, is_linear=False):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, 1, padding=0, use_bias=False),
        nn.BatchNorm(scale=True)
    )
    if not is_linear:
        out.add(nn.Activation('relu'))
    return out

def DWise(channels, kernel, stride, pad):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, kernel, strides=stride, padding=pad, groups=channels, use_bias=False),
        nn.BatchNorm(scale=True),
        nn.Activation('relu')
    )
    return out

class InvertedResidual(nn.HybridBlock):
    def __init__(
        self,
        expansion_factor,
        num_filter_in,
        num_filter_out,
        kernel,
        stride,
        pad,
        same_shape=True,
        **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.stride = stride
        with self.name_scope(): 
            self.bottleneck = nn.HybridSequential()
            self.bottleneck.add(
                Conv1x1(num_filter_in*expansion_factor),
                DWise(num_filter_in*expansion_factor, kernel, self.stride, pad),
                Conv1x1(num_filter_out, is_linear=True)
            )
            if self.stride == 1 and not self.same_shape:
                self.conv_res = Conv1x1(num_filter_out)
    def hybrid_forward(self, F, x):
        out = self.bottleneck(x)
        #if self.stride == 1 and self.same_shape:
        #    out = F.elemwise_add(out, x)
        if self.stride == 1:
            if not self.same_shape:
                x = self.conv_res(x)
            out = F.elemwise_add(out, x)
        return out

class InvResiSeq(nn.HybridBlock):
    def __init__(
        self,
        num_filter_in,
        num_filter_out,
        n_blocks,
        kernel=(3,3),
        stride=(2,2),
        pad=(1,1),
        expansion_factor=6,
        **kwargs
        ):
        super(InvResiSeq, self).__init__(**kwargs)
        self.num_filter_in = num_filter_in
        self.num_filter_out = num_filter_out
        self.n = n_blocks
        with self.name_scope():
            self.seq = nn.HybridSequential()
            self.seq.add(
                InvertedResidual(
                    expansion_factor=expansion_factor,
                    num_filter_in=num_filter_in,
                    num_filter_out=num_filter_out,
                    kernel=kernel,
                    stride=stride,
                    pad=pad,
                    same_shape=False)
                    )
            for _ in range(n_blocks-1):
                self.seq.add(
                    InvertedResidual(
                        expansion_factor=expansion_factor,
                        num_filter_in=num_filter_in,
                        num_filter_out=num_filter_out,
                        kernel=kernel,
                        stride=(1,1),
                        pad=pad,
                        same_shape=False)
                        )
        return

    def hybrid_forward(self, F, x):
        out = self.seq(x)
        return out

def getMnetV2(first_conv_param, last_conv_param, inv_resi_params_ls, num_classes=1000):
    net = nn.HybridSequential(prefix='mnet-stem-')
    # first conv
    first_conv = ConvBlock(**first_conv_param)
    with net.name_scope():
        net.add(first_conv)
    # inverted residual unit
    inv_resi_params_ls = [
        InvResiSeq(**param) for param in inv_resi_params_ls
    ]
    with net.name_scope():
        net.add(*inv_resi_params_ls)
    # last conv
    last_conv = ConvBlock(**last_conv_param)
    with net.name_scope():
        net.add(last_conv)
    #  avg pooling, flatten, and dense layer
    with net.name_scope():
        net.add(
            nn.GlobalAvgPool2D(),
            nn.Flatten(),
            nn.Dense(units=num_classes)
            )
    # 

    return net

if __name__ == '__main__':
    first_conv_param = {
        'channels' : 32,
        'kernel' : (3,3),
        'stride' : (2,2),
        'pad' : (1,1)
        }
    last_conv_param = {
        'channels' : 1280,
        'kernel' : (1,1),
        'stride' : (1,1),
        'pad' : (0,0)
        }
    inv_resi_params_ls = [
        {
            'num_filter_in' : 32,
            'num_filter_out' : 16,
            'n_blocks' : 1,
            'kernel':(3,3),
            'stride':(1,1),
            'pad':(1,1),
            'expansion_factor':6
        },
        {
            'num_filter_in' : 16,
            'num_filter_out' : 24,
            'n_blocks' : 2,
            'kernel':(3,3),
            'stride':(2,2),
            'pad':(1,1),
            'expansion_factor':6
        },
        {
            'num_filter_in' : 24,
            'num_filter_out' : 32,
            'n_blocks' : 3,
            'kernel':(3,3),
            'stride':(2,2),
            'pad':(1,1),
            'expansion_factor':6
        },
        {
            'num_filter_in' : 32,
            'num_filter_out' : 64,
            'n_blocks' : 4,
            'kernel':(3,3),
            'stride':(1,1),
            'pad':(1,1),
            'expansion_factor':6
        },
        {
            'num_filter_in' : 64,
            'num_filter_out' : 96,
            'n_blocks' : 3,
            'kernel':(3,3),
            'stride':(2,2),
            'pad':(1,1),
            'expansion_factor':6
        },
        {
            'num_filter_in' : 96,
            'num_filter_out' : 160,
            'n_blocks' : 3,
            'kernel':(3,3),
            'stride':(2,2),
            'pad':(1,1),
            'expansion_factor':6
        },
        {
            'num_filter_in' : 160,
            'num_filter_out' : 320,
            'n_blocks' : 3,
            'kernel':(3,3),
            'stride':(1,1),
            'pad':(1,1),
            'expansion_factor':6
        }
    ]

    net = getMnetV2(
    first_conv_param=first_conv_param,
    last_conv_param=last_conv_param,
    inv_resi_params_ls=inv_resi_params_ls,
    num_classes=2)
    
    sym = net(mx.sym.Variable('data'))
    mx.viz.plot_network(symbol=net(mx.sym.Variable('data')),shape={'data':(1,3,224,224)}).view()
