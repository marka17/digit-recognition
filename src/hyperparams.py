import attr


@attr.s
class Hyperparams:

    # featurizing
    sr: int = attr.ib(default=16000)
    n_fft: int = attr.ib(default=800)
    win_length: int = attr.ib(default=400)
    hop_length: int = attr.ib(default=160)
    n_mels: int = attr.ib(default=40)
