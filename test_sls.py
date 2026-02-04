import numpy as np
from sls_utils import generate_gray_code_patterns, decode_gray_code

def test_gray_code():
    w, h = 1280, 720
    patterns_x, patterns_y = generate_gray_code_patterns(w, h)

    # Simulate perfect capture
    captures_x = []
    for p in patterns_x:
        captures_x.append(p)

    captures_y = []
    for p in patterns_y:
        captures_y.append(p)

    decoded_x, valid_x = decode_gray_code(captures_x)
    decoded_y, valid_y = decode_gray_code(captures_y)

    # Check if decoded values match indices
    yy, xx = np.mgrid[0:h, 0:w]

    assert np.all(decoded_x == xx), "X decoding failed"
    assert np.all(decoded_y == yy), "Y decoding failed"
    assert np.all(valid_x), "X validation failed"
    assert np.all(valid_y), "Y validation failed"

    print("SLS Math Verification Passed!")

if __name__ == "__main__":
    test_gray_code()
