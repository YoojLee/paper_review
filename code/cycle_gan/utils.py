import math

def calc_same_pad(i: int, k: int, s: int, d: int=1) -> int:
    """
    - Args
        i: input spatial dimension
        l: kernel size
        s: stride
        d: dilation
    """
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)