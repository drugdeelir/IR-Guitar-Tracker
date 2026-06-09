"""Headless smoke tests for the IR Guitar Tracker tracking pipeline.

These tests run without a camera or projector. They verify the worker's
core tracking logic (blob history, Kalman, rigid constraint, optical flow,
homography smoothing, keystone) using synthetic data only.

Run with:  pytest tests/test_tracking.py -v
"""
import sys
import os
import math
import threading

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: keep tests headless — stub heavy Qt/camera dependencies before
# importing worker so the module loads without a display server.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# Minimal PyQt5 stubs so worker.py imports without a display
import types

_qt_stubs = {
    "PyQt5": types.ModuleType("PyQt5"),
    "PyQt5.QtCore": types.ModuleType("PyQt5.QtCore"),
    "PyQt5.QtGui": types.ModuleType("PyQt5.QtGui"),
}

_QtCore = _qt_stubs["PyQt5.QtCore"]
_QtCore.QObject = object
_QtCore.QThread = object
_QtCore.pyqtSignal = lambda *a, **kw: None  # signals become None (not emitted)

_QtGui = _qt_stubs["PyQt5.QtGui"]
_QtGui.QImage = object

for name, mod in _qt_stubs.items():
    sys.modules.setdefault(name, mod)

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2  # noqa: E402 — must come after stubs
import worker as _w  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_worker():
    """Return a Worker instance with signals neutered so tests don't need Qt."""
    w = object.__new__(_w.Worker)
    # Manually call the parts of __init__ we need, skipping Qt
    import collections, logging, time
    w.logger = logging.getLogger("test_worker")
    w._detection_params = _w.DetectionParams()
    w._dp = w._detection_params
    w.smoothed_points = []
    w._kalman_filters = []
    w._kalman_initialized = False
    w.expected_marker_count = 4
    w.tracking_lost_frames = 0
    w.max_lost_tracking_frames = 30
    w.smoothing_alpha = 0.05
    w._blob_history = []
    w._rigid_template = None
    w._rigid_enabled = True
    w._rigid_blend = 0.6  # legacy attr kept for safety
    w._flow_enabled = True
    w._flow_prev_gray = None
    w._flow_prev_pts = None
    w._flow_lk_params = dict(
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    w._prev_transforms = {}
    w._transform_smooth = 0.6
    w._transform_cache = collections.OrderedDict()
    w._transform_cache_max = 64
    w._use_opencl = False
    return w


def _four_corners(scale=100.0):
    """Return 4 marker points at the corners of a square centred at origin."""
    return [(scale, scale), (2*scale, scale), (2*scale, 2*scale), (scale, 2*scale)]


# ---------------------------------------------------------------------------
# 1. DetectionParams defaults & rigid_blend field
# ---------------------------------------------------------------------------

class TestDetectionParams:
    def test_defaults(self):
        dp = _w.DetectionParams()
        assert dp.blob_min_hits == 8
        assert dp.blob_max_age == 15
        assert dp.kalman_static_pn == 0.005
        assert dp.fog_contrast_threshold == 18.0

    def test_rigid_blend_field_exists_with_correct_default(self):
        dp = _w.DetectionParams()
        assert hasattr(dp, "rigid_blend")
        assert abs(dp.rigid_blend - 0.6) < 1e-9

    def test_rigid_blend_is_mutable(self):
        dp = _w.DetectionParams()
        dp.rigid_blend = 0.3
        assert abs(dp.rigid_blend - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# 2. set_rigid_blend clamps correctly
# ---------------------------------------------------------------------------

class TestSetRigidBlend:
    def _worker(self):
        return _make_worker()

    def test_normal_value(self):
        w = self._worker()
        w.set_rigid_blend(0.4)
        assert abs(w._dp.rigid_blend - 0.4) < 1e-9

    def test_clamp_below_zero(self):
        w = self._worker()
        w.set_rigid_blend(-0.5)
        assert w._dp.rigid_blend == 0.0

    def test_clamp_above_one(self):
        w = self._worker()
        w.set_rigid_blend(1.5)
        assert w._dp.rigid_blend == 1.0

    def test_zero_disables_blend(self):
        w = self._worker()
        w.set_rigid_blend(0.0)
        assert w._dp.rigid_blend == 0.0


# ---------------------------------------------------------------------------
# 3. _is_default_warp / keystone detection
# ---------------------------------------------------------------------------

class TestIsDefaultWarp:
    def _worker(self):
        w = _make_worker()
        w.warp_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        return w

    def test_default_is_true(self):
        w = self._worker()
        assert w._is_default_warp() is True

    def test_moved_corner_is_false(self):
        w = self._worker()
        w.warp_points = [[0.05, 0.0], [1, 0], [1, 1], [0, 1]]
        assert w._is_default_warp() is False

    def test_tiny_float_error_still_default(self):
        w = self._worker()
        w.warp_points = [[1e-8, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        assert w._is_default_warp() is True

    def test_all_corners_moved(self):
        w = self._worker()
        w.warp_points = [[0.1, 0.1], [0.9, 0.0], [1.0, 0.9], [0.0, 1.0]]
        assert w._is_default_warp() is False


# ---------------------------------------------------------------------------
# 4. Keystone matrix geometry
# ---------------------------------------------------------------------------

class TestKeystoneMatrix:
    """Verify that the keystone warp matrix shrinks content to the expected quad."""

    def _compute_ks_M(self, warp_points, out_w=1920, out_h=1080):
        ks_src = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
        ks_dst = np.float32([[p[0] * out_w, p[1] * out_h] for p in warp_points])
        return cv2.getPerspectiveTransform(ks_src, ks_dst)

    def test_identity_warp_points_gives_identity_matrix(self):
        M = self._compute_ks_M([[0, 0], [1, 0], [1, 1], [0, 1]])
        corners_in = np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]]).reshape(-1, 1, 2)
        corners_out = cv2.perspectiveTransform(corners_in, M).reshape(-1, 2)
        np.testing.assert_allclose(corners_out, corners_in.reshape(-1, 2), atol=0.5)

    def test_top_left_corner_moved_in(self):
        """Moving TL corner to (0.1, 0.1) should map TL of source there."""
        wp = [[0.1, 0.1], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        M = self._compute_ks_M(wp)
        tl_in = np.float32([[0, 0]]).reshape(-1, 1, 2)
        tl_out = cv2.perspectiveTransform(tl_in, M).reshape(-1, 2)[0]
        np.testing.assert_allclose(tl_out, [192.0, 108.0], atol=1.0)

    def test_uniform_inset_shrinks_all_corners(self):
        inset = 0.05
        wp = [
            [inset, inset], [1 - inset, inset],
            [1 - inset, 1 - inset], [inset, 1 - inset],
        ]
        M = self._compute_ks_M(wp)
        src_corners = np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]]).reshape(-1, 1, 2)
        dst_corners = cv2.perspectiveTransform(src_corners, M).reshape(-1, 2)
        w, h = 1920, 1080
        expected = np.float32([
            [inset * w, inset * h], [(1-inset)*w, inset*h],
            [(1-inset)*w, (1-inset)*h], [inset*w, (1-inset)*h],
        ])
        np.testing.assert_allclose(dst_corners, expected, atol=1.0)


# ---------------------------------------------------------------------------
# 5. Rigid body constraint
# ---------------------------------------------------------------------------

class TestRigidBodyConstraint:
    def _worker_with_template(self, pts):
        w = _make_worker()
        arr = np.array(pts, dtype=np.float32)
        w._rigid_template = arr - arr.mean(axis=0)
        return w

    def test_no_op_when_disabled(self):
        pts = _four_corners()
        w = self._worker_with_template(pts)
        w._rigid_enabled = False
        out = w._apply_rigid_constraint(pts)
        assert out == pts

    def test_no_op_without_template(self):
        w = _make_worker()
        pts = _four_corners()
        out = w._apply_rigid_constraint(pts)
        assert out == pts

    def test_pure_translation_returns_same_shape(self):
        pts = _four_corners()
        w = self._worker_with_template(pts)
        shifted = [(x + 20, y + 10) for x, y in pts]
        out = w._apply_rigid_constraint(shifted)
        assert len(out) == 4

    def test_blend_zero_leaves_points_unchanged(self):
        pts = _four_corners()
        w = self._worker_with_template(pts)
        w._dp.rigid_blend = 0.0
        shifted = [(x + 30, y - 15) for x, y in pts]
        out = w._apply_rigid_constraint(shifted)
        for (ox, oy), (sx, sy) in zip(out, shifted):
            assert abs(ox - sx) < 1e-6 and abs(oy - sy) < 1e-6

    def test_blend_one_pulls_fully_onto_fit(self):
        """blend=1.0: output should sit on the similarity-transformed template."""
        pts = _four_corners()
        w = self._worker_with_template(pts)
        w._dp.rigid_blend = 1.0
        shifted = [(x + 15, y + 5) for x, y in pts]
        out = w._apply_rigid_constraint(shifted)
        # After pulling to the rigid fit, the output should form a square
        dists = []
        cx = sum(p[0] for p in out) / 4
        cy = sum(p[1] for p in out) / 4
        for px, py in out:
            dists.append(math.hypot(px - cx, py - cy))
        # All distances from centroid should be equal (rigid body)
        assert max(dists) - min(dists) < 0.5

    def test_count_mismatch_returns_unchanged(self):
        pts = _four_corners()
        w = self._worker_with_template(pts)
        three_pts = pts[:3]
        out = w._apply_rigid_constraint(three_pts)
        assert out == three_pts


# ---------------------------------------------------------------------------
# 6. Blob history — alpha, velocity, local search prediction
# ---------------------------------------------------------------------------

class TestBlobHistory:
    # _blob_history entry format: (cx, cy, peak, age, hits, vx, vy)
    # _update_blob_history takes list of (cx, cy, score) tuples.

    def _worker_with_history(self, cx, cy, hits=10):
        w = _make_worker()
        w._blob_history_radius = _w._BLOB_HISTORY_RADIUS
        w._blob_min_hits = _w._BLOB_MIN_HITS
        w._blob_max_age = _w._BLOB_MAX_AGE
        # Seed one blob with enough hits to be reported as stable
        w._blob_history = [(float(cx), float(cy), 200.0, 0, hits, 0.0, 0.0)]
        return w

    def test_alpha_smoothing_applied(self):
        """Position should move alpha fraction toward new detection each frame."""
        alpha = 0.4
        cx0, cy0 = 100.0, 200.0
        w = self._worker_with_history(cx0, cy0)
        # Keep delta well inside the 40px association radius
        new_cx, new_cy = 115.0, 210.0
        w._update_blob_history([(new_cx, new_cy, 200.0)])
        hcx, hcy, *_ = w._blob_history[0]
        expected_cx = cx0 + alpha * (new_cx - cx0)
        expected_cy = cy0 + alpha * (new_cy - cy0)
        assert abs(hcx - expected_cx) < 1.0
        assert abs(hcy - expected_cy) < 1.0

    def test_velocity_uses_raw_delta(self):
        """Velocity should reflect pre-smoothing displacement (not post-smooth delta)."""
        cx0, cy0 = 100.0, 100.0
        w = self._worker_with_history(cx0, cy0)
        new_cx, new_cy = 130.0, 100.0
        w._update_blob_history([(new_cx, new_cy, 200.0)])
        _, _, _, _, _, vx, vy = w._blob_history[0]
        raw_dx = new_cx - cx0  # 30px
        assert vx > 0.0
        assert abs(vx) < raw_dx  # weighted blend keeps it below the raw delta


# ---------------------------------------------------------------------------
# 7. Homography smoothing — _compute_transform blending
# ---------------------------------------------------------------------------

class TestHomographySmoothing:
    def _worker(self):
        return _make_worker()

    def test_first_call_returns_raw_matrix(self):
        w = self._worker()
        src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) * 100
        dst = src + np.float32([[5, 3], [5, 3], [5, 3], [5, 3]])
        M = w._compute_transform(src, dst, cache_key="k1")
        assert M is not None
        assert M.shape == (3, 3)

    def test_subsequent_call_blends(self):
        w = self._worker()
        src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        dst1 = src.copy()
        dst2 = src + 10.0
        w._compute_transform(src, dst1, cache_key="k2")
        M2 = w._compute_transform(src, dst2, cache_key="k2")
        # Translation should be between 0 and 10 (blended)
        assert M2 is not None
        tx = M2[0, 2]
        assert 0.0 < tx < 10.5


# ---------------------------------------------------------------------------
# 8. Marker spread — rotation invariance
# ---------------------------------------------------------------------------

class TestMarkerSpread:
    def test_square_spread(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        spread = _w.Worker._marker_spread(pts)
        expected = math.hypot(0.5, 0.5)  # distance from centroid to each corner
        assert abs(spread - expected) < 1e-6

    def test_rotation_does_not_change_spread(self):
        import math as _math
        pts = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
        spread0 = _w.Worker._marker_spread(pts)
        # Rotate 45 degrees around centroid
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        angle = _math.pi / 4
        rotated = [
            (cx + (p[0]-cx)*_math.cos(angle) - (p[1]-cy)*_math.sin(angle),
             cy + (p[0]-cx)*_math.sin(angle) + (p[1]-cy)*_math.cos(angle))
            for p in pts
        ]
        spread1 = _w.Worker._marker_spread(rotated)
        assert abs(spread1 - spread0) < 1e-4

    def test_empty_returns_zero(self):
        assert _w.Worker._marker_spread([]) == 0.0

    def test_single_point_returns_zero(self):
        assert _w.Worker._marker_spread([(5.0, 7.0)]) == 0.0


# ---------------------------------------------------------------------------
# 9. CueReader — opens, reads, releases without blocking
# ---------------------------------------------------------------------------

class TestCueReader:
    def test_bad_path_not_opened(self):
        r = _w.CueReader("/nonexistent/video.mp4", loop=False)
        assert not r.is_opened()
        r.release()

    def test_read_on_closed_returns_false(self, tmp_path):
        r = _w.CueReader(str(tmp_path / "no.mp4"), loop=False)
        ok, frame = r.read()
        assert not ok
        assert frame is None
        r.release()

    def test_release_is_idempotent(self):
        r = _w.CueReader("/nonexistent.mp4", loop=False)
        r.release()
        r.release()  # second call must not raise


# ---------------------------------------------------------------------------
# 10. OpenCL warp accelerator — CPU fallback
# ---------------------------------------------------------------------------

class TestWarpAccel:
    def _worker(self):
        w = _make_worker()
        w._use_opencl = False
        return w

    def test_cpu_warp_identity(self):
        w = self._worker()
        src = np.zeros((100, 100, 3), dtype=np.uint8)
        src[10:20, 10:20] = 200
        M = np.eye(3, dtype=np.float64)
        out = w._warp_perspective_accel(src, M, (100, 100))
        np.testing.assert_array_equal(out, src)

    def test_cpu_warp_translation(self):
        w = self._worker()
        src = np.zeros((200, 200, 3), dtype=np.uint8)
        src[50, 50] = [255, 0, 0]
        M = np.array([[1, 0, 10], [0, 1, 10], [0, 0, 1]], dtype=np.float64)
        out = w._warp_perspective_accel(src, M, (200, 200))
        # The white pixel should have moved 10px right and 10px down
        assert out[60, 60, 0] == 255
        assert out[50, 50, 0] == 0


# ---------------------------------------------------------------------------
# 11. set_warp_points round-trip
# ---------------------------------------------------------------------------

class TestSetWarpPoints:
    def test_round_trip(self):
        w = _make_worker()
        w.warp_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        new_pts = [[0.05, 0.05], [0.95, 0.0], [1.0, 1.0], [0.0, 1.0]]
        w.set_warp_points(new_pts)
        assert w.warp_points == new_pts

    def test_default_is_identity(self):
        w = _make_worker()
        w.warp_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        assert w._is_default_warp()


# ---------------------------------------------------------------------------
# 12. Thread safety: CueReader set_loop
# ---------------------------------------------------------------------------

class TestCueReaderThreadSafety:
    def test_set_loop_toggle(self, tmp_path):
        """Toggling loop mode from multiple threads must not raise."""
        r = _w.CueReader(str(tmp_path / "missing.mp4"), loop=True)
        errors = []

        def toggle():
            for _ in range(50):
                try:
                    r.set_loop(True)
                    r.set_loop(False)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        r.release()
        assert errors == []
