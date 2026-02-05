import numpy as np

def gray_to_binary(n):
    mask = n >> 1
    while mask != 0:
        n = n ^ mask
        mask = mask >> 1
    return n

def generate_gray_code_patterns(width, height, max_bits=11):
    # Limit bits to ensure stripes are resolvable by camera (min ~5px)
    n_x = min(int(np.ceil(np.log2(width))), max_bits)
    n_y = min(int(np.ceil(np.log2(height))), max_bits)

    patterns_x = []
    for i in range(n_x):
        # bit i of gray code
        p = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            # Scale x to the range covered by n_x bits
            x_scaled = (x * (2**n_x)) // width
            gray = x_scaled ^ (x_scaled >> 1)
            if (gray >> (n_x - 1 - i)) & 1:
                p[:, x] = 255
        patterns_x.append(p)
        patterns_x.append(255 - p) # Inverse

    patterns_y = []
    for i in range(n_y):
        p = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            y_scaled = (y * (2**n_y)) // height
            gray = y_scaled ^ (y_scaled >> 1)
            if (gray >> (n_y - 1 - i)) & 1:
                p[y, :] = 255
        patterns_y.append(p)
        patterns_y.append(255 - p) # Inverse

    return patterns_x, patterns_y

def decode_gray_code(captures, target_range, threshold=5):
    if not captures or len(captures) == 0:
        return np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=bool)

    # Shape validation: ensure all captures have the same dimensions
    try:
        base_shape = captures[0].shape
    except (AttributeError, IndexError):
        return np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=bool)

    for i, cap in enumerate(captures):
        if cap is None or not hasattr(cap, 'shape') or cap.shape != base_shape:
            print(f"Error: Mismatched or invalid capture at index {i}. Expected {base_shape}")
            return np.zeros(base_shape, dtype=np.int32), np.zeros(base_shape, dtype=bool)

    # captures: list of (pattern, inverse_pattern) pairs
    # returns bitmask where bits are set if pattern > inverse_pattern + threshold
    bits = []
    for i in range(0, len(captures), 2):
        if i+1 >= len(captures): break
        p = captures[i].astype(np.int16)
        inv = captures[i+1].astype(np.int16)

        bit = np.zeros(p.shape, dtype=np.uint8)
        # We only care about pixels where there is enough contrast
        diff = np.abs(p - inv)
        valid = diff > threshold
        bit[p > inv] = 1
        bits.append((bit, valid))

    # Reconstruct value
    n_bits = len(bits)
    val = np.zeros(base_shape, dtype=np.int32)
    total_valid = np.ones(base_shape, dtype=bool)

    for i, (bit, valid) in enumerate(bits):
        val = (val << 1) | bit.astype(np.int32)
        total_valid &= valid

    # Convert gray to binary
    binary_val = val.copy()
    for shift in [16, 8, 4, 2, 1]:
        binary_val ^= (binary_val >> shift)

    # Scale back to target_range (width or height)
    # Using float for accuracy then rounding
    scale = target_range / (2**n_bits)
    scaled_val = (binary_val.astype(np.float32) * scale).astype(np.int32)

    return scaled_val, total_valid
