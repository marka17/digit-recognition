from typing import Dict


class DigitDictionary:
    """
    Main purpose of DigitDictionary is to shift digits right by 1
    because CTCLoss require 0 for blank symbol
    """

    def __init__(self, blank_symbol = 0):
        self.blank_symbol = blank_symbol

        self.digit2code: Dict = {d: d + 1 for d in range(10)}
        self.code2digit: Dict = {v: k for k, v in self.digit2code.items()}

    def encode(self, ):
        pass

    def decode(self):
        pass

    def ctc_decode(self):
        pass


def ctc_decode(sequence, black_symbol=0):
    current = []
    for i in range(0, len(sequence)):
        symbol = sequence[i]

        if i > 0 and symbol == sequence[i - 1]:
            continue

        # skip blank symbol
        if symbol == black_symbol:
            continue

        current.append(str(int(symbol) - 1))

    text = ''.join(current)
    return text
