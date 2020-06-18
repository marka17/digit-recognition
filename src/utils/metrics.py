import Levenshtein

from ..data import ctc_decode


def calculate_cer(targets, decodings):
    cer = 0.0
    targets = (targets - 1).detach().cpu().tolist()

    pairs = []
    for target, d in zip(targets, decodings):
        target = "".join(map(str, target))
        decoding = ctc_decode(d.tolist())
        cer += min(1, Levenshtein.distance(target, decoding) / (len(decoding) + 1e-7))

        pairs.append(
            target + '\t\t' + decoding
        )

    return cer / len(target), pairs

