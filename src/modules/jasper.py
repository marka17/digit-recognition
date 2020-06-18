import torch
from torch import nn

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if type(m) == nn.Conv1d or type(m) == MaskedConv1d:
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif type(m) == nn.BatchNorm1d:
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class MaskedConv1d(nn.Conv1d):
    """1D convolution with sequence masking
    """
    __constants__ = ["use_conv_mask"]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, use_conv_mask=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride,
                         padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.use_conv_mask = use_conv_mask

    def get_seq_len(self, lens):

        return ((lens + 2 * self.padding - self.dilation *
                 (self.kernel_size - 1) - 1) / self.stride + 1)

    def forward(self, inp):
        if self.use_conv_mask:
            x, lens = inp
            max_len = x.size(1)
            idxs = torch.arange(max_len).to(lens.dtype).to(lens.device).expand(len(lens), max_len)

            mask = idxs >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device).bool(), 0)
            del mask
            del idxs
            lens = self.get_seq_len(lens)
            return super().forward(x), lens

        else:
            return super().forward(inp)


class JasperBlock(nn.Module):
    __constants__ = ["use_conv_mask", "conv"]
    """
    Jasper Block: https://arxiv.org/pdf/1904.03288.pdf
    """

    def __init__(self, inplanes, planes, repeat=3, kernel_size=11, stride=1,
                 dilation=1, padding='same', dropout=0.2, activation=None,
                 residual=True, residual_panes=[], use_conv_mask=False):
        super().__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.use_conv_mask = use_conv_mask
        self.conv = nn.ModuleList()
        inplanes_loop = inplanes
        for _ in range(repeat - 1):
            self.conv.extend(
                self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                        stride=stride, dilation=dilation,
                                        padding=padding_val))
            self.conv.extend(
                self._get_act_dropout_layer(drop_prob=dropout, activation=activation))
            inplanes_loop = planes
        self.conv.extend(
            self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                    stride=stride, dilation=dilation,
                                    padding=padding_val))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(nn.ModuleList(
                    modules=self._get_conv_bn_layer(ip, planes, kernel_size=1)
                    )
                )

        self.out = nn.Sequential(
            *self._get_act_dropout_layer(drop_prob=dropout, activation=activation)
        )

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11,
                           stride=1, dilation=1, padding=0, bias=False):
        layers = [
            MaskedConv1d(in_channels, out_channels, kernel_size, stride=stride,
                         dilation=dilation, padding=padding, bias=bias,
                         use_conv_mask=self.use_conv_mask),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]
        return layers

    @staticmethod
    def _get_act_dropout_layer(drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [
            activation,
            nn.Dropout(p=drop_prob)
        ]
        return layers

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_):

        if self.use_conv_mask:
            xs, lens_orig = input_
        else:
            xs = input_
            lens_orig = 0

        # compute forward convolutions
        out = xs[-1]
        lens = lens_orig
        for i, l in enumerate(self.conv):
            if self.use_conv_mask and isinstance(l, MaskedConv1d):
                out, lens = l((out, lens))
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if j == 0 and self.use_conv_mask:
                        res_out, _ = res_layer((res_out, lens_orig))
                    else:
                        res_out = res_layer(res_out)
                out += res_out

        # compute the output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        if self.use_conv_mask:
            return out, lens
        else:
            return out
