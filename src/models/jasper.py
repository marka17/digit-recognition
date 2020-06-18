from torch import nn

from ..modules.jasper import JasperBlock, init_weights, jasper_activations


class JasperEncoder(nn.Module):
    __constants__ = ["use_conv_mask"]
    """
    Jasper encoder
    """

    def __init__(self, **kwargs):
        cfg = {}
        for key, value in kwargs.items():
            cfg[key] = value

        super().__init__()
        self._cfg = cfg

        activation = jasper_activations[cfg['encoder']['activation']]()
        self.use_conv_mask = cfg['encoder'].get('convmask', False)
        feat_in = cfg['input']['features'] * cfg['input'].get('frame_splicing', 1)
        init_mode = cfg.get('init_mode', 'xavier_uniform')

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in cfg['jasper']:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True

            encoder_layers.append(
                JasperBlock(feat_in, lcfg['filters'], repeat=lcfg['repeat'],
                            kernel_size=lcfg['kernel'], stride=lcfg['stride'],
                            dilation=lcfg['dilation'], dropout=lcfg['dropout'],
                            residual=lcfg['residual'], activation=activation,
                            residual_panes=dense_res, use_conv_mask=self.use_conv_mask))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if self.use_conv_mask:
            audio_signal, length = x
            return self.encoder(([audio_signal], length))
        else:
            return self.encoder([x])


class JasperDecoderForCTC(nn.Module):
    """Jasper decoder
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._feat_in = kwargs.get("feat_in")
        self._num_classes = kwargs.get("num_classes")
        init_mode = kwargs.get('init_mode', 'xavier_uniform')

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True), )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_output):
        out = self.decoder_layers(encoder_output[-1]).transpose(1, 2)
        return nn.functional.log_softmax(out, dim=2)


class Jasper(nn.Module):
    """Contains jasper encoder and decoder
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.transpose_in = kwargs.get("transpose_in", False)
        self.jasper_encoder = JasperEncoder(**kwargs.get("jasper_model_definition"))
        self.jasper_decoder = JasperDecoderForCTC(feat_in=kwargs.get("feat_in"),
                                                  num_classes=kwargs.get("num_classes"))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if self.jasper_encoder.use_conv_mask:
            t_encoded_t, t_encoded_len_t = self.jasper_encoder(x)
        else:
            if self.transpose_in:
                x = x.transpose(1, 2)
            t_encoded_t = self.jasper_encoder(x)

        out = self.jasper_decoder(t_encoded_t)
        if self.jasper_encoder.use_conv_mask:
            return out, t_encoded_len_t
        else:
            return out

    def infer(self, x):
        if self.jasper_encoder.use_conv_mask:
            return self.forward(x)
        else:
            ret = self.forward(x[0])
            return ret, len(ret)
