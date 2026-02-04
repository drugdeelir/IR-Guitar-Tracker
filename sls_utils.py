import numpy as np

def gray_to_binary(n):
    mask = n >> 1
    while mask != 0:
        n = n ^ mask
        mask = mask >> 1
    return n

def generate_gray_code_patterns(width, height):
    n_x = int(np.ceil(np.log2(width)))
    n_y = int(np.ceil(np.log2(height)))

    patterns_x = []
    for i in range(n_x):
        # bit i of gray code
        p = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            gray = x ^ (x >> 1)
            if (gray >> (n_x - 1 - i)) & 1:
                p[:, x] = 255
        patterns_x.append(p)
        patterns_x.append(255 - p) # Inverse

    patterns_y = []
    for i in range(n_y):
        p = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            gray = y ^ (y >> 1)
            if (gray >> (n_y - 1 - i)) & 1:
                p[y, :] = 255
        patterns_y.append(p)
        patterns_y.append(255 - p) # Inverse

    return patterns_x, patterns_y

def decode_gray_code(captures, threshold=20):
    # captures: list of (pattern, inverse_pattern) pairs
    # returns bitmask where bits are set if pattern > inverse_pattern + threshold
    bits = []
    for i in range(0, len(captures), 2):
        p = captures[i].astype(np.int16)
        inv = captures[i+1].astype(np.int16)

        bit = np.zeros(p.shape, dtype=np.uint8)
        # We only care about pixels where there is enough contrast
        valid = np.abs(p - inv) > threshold
        bit[p > inv] = 1
        bits.append((bit, valid))

    # Reconstruct value
    val = np.zeros(captures[0].shape, dtype=np.int32)
    total_valid = np.ones(captures[0].shape, dtype=bool)

    for i, (bit, valid) in enumerate(bits):
        val = (val << 1) | bit.astype(np.int32)
        total_valid &= valid

    # Convert gray to binary
    # This is a bit slow for whole image, better use a LUT or vectorization
    # But since we only have 10-11 bits, we can vectorize
    binary_val = val.copy()
    for shift in [16, 8, 4, 2, 1]:
        binary_val ^= (binary_val >> shift)

    return binary_val, total_valid
