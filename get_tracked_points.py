    def get_tracked_points(self, frame, force_full=False, return_rejected=False):
        # Already called from TrackingThread, but let's ensure we use tracking_mutex for shared state
        if frame is None: return ([], []) if return_rejected else []
        h, w = frame.shape[:2]

        # Performance: Use GPU (UMat) for image processing
        u_frame = cv2.UMat(frame)
        with QMutexLocker(self.tracking_mutex):
            marker_cfg = self.marker_config
            marker_fp = self.marker_fingerprint if self.marker_fingerprint is not None else []

        roi_x, roi_y, roi_w, roi_h = 0, 0, w, h
        # Scale padding with resolution for wide setups
        # Increased to 6% for better high-velocity neck tracking
        dynamic_padding = max(self.roi_padding, int(w * 0.06))

        if not force_full and self.last_tracked_points is not None:
            # pts are normalized
            pts = np.array(self.last_tracked_points)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            # Convert to pixels for ROI calculation
            roi_x = max(0, int(min_x * w - dynamic_padding))
            roi_y = max(0, int(min_y * h - dynamic_padding))
            roi_w = min(w - roi_x, int((max_x - min_x) * w + 2 * dynamic_padding))
            roi_h = min(h - roi_y, int((max_y - min_y) * h + 2 * dynamic_padding))

        u_roi = cv2.UMat(u_frame, (roi_y, roi_y+roi_h), (roi_x, roi_x+roi_w))
        u_gray = cv2.cvtColor(u_roi, cv2.COLOR_BGR2GRAY)

        # 1. Projector Interference Masking (Feed-Forward)
        # Identify areas likely hit by bright projector light and suppress them.
        if self.h_c2p is not None:
            with QMutexLocker(self.last_projected_frame_mutex):
                if self.last_projected_frame_for_masking is not None:
                    ph, pw = self.last_projected_frame_for_masking.shape[:2]
                    S = np.array([[float(pw), 0, 0], [0, float(ph), 0], [0, 0, 1]], dtype=np.float32)
                    M_cam_to_proj = S @ self.h_c2p
                    T_roi = np.array([[1, 0, roi_x], [0, 1, roi_y], [0, 0, 1]], dtype=np.float32)
                    M_roi_to_proj = M_cam_to_proj @ T_roi
                    try:
                        # Performance: warp at lower resolution for interference masking
                        mask_w, mask_h = max(32, roi_w // 4), max(32, roi_h // 4)
                        T_scale = np.array([[mask_w/roi_w, 0, 0], [0, mask_h/roi_h, 0], [0, 0, 1]], dtype=np.float32)

                        # Use UMat for warped masking
                        u_proj_last = cv2.UMat(self.last_projected_frame_for_masking)
                        u_proj_small = cv2.warpPerspective(u_proj_last, T_scale @ M_roi_to_proj, (mask_w, mask_h))
                        u_proj_mask = cv2.resize(u_proj_small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

                        # Aggressive suppression (0.6)
                        u_suppression = cv2.multiply(u_proj_mask, 0.6)
                        u_gray = cv2.subtract(u_gray, u_suppression)
                    except: pass

        # 2. Advanced Pre-processing (GPU Accelerated)
        th_size = max(150, int(w / 30))
        th_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_size, th_size))
        u_tophat = cv2.morphologyEx(u_gray, cv2.MORPH_TOPHAT, th_kernel)

        u_gray_boosted = cv2.addWeighted(u_gray, 0.4, u_tophat, 0.6, 0)
        u_gray_boosted = self.tracking_clahe.apply(u_gray_boosted)

        if self.auto_threshold:
            _, u_t_global = cv2.threshold(u_gray_boosted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            u_t_local = cv2.adaptiveThreshold(u_gray_boosted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
            u_thresh = cv2.bitwise_and(u_t_global, u_t_local)
        else:
            effective_threshold = min(self.ir_threshold, 248)
            _, u_thresh = cv2.threshold(u_gray_boosted, effective_threshold, 255, cv2.THRESH_BINARY)

        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        u_thresh = cv2.morphologyEx(u_thresh, cv2.MORPH_OPEN, kernel_clean)
        u_thresh = cv2.dilate(u_thresh, kernel_clean, iterations=3)

        # Download from GPU to CPU for contour finding (OpenCV findContours needs CPU array)
        thresh = u_thresh.get()
        gray_boosted = u_gray_boosted.get()

        # Using findContours for efficient blob finding
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_points_raw = []
        rejected_points_raw = []

        # Min circularity: more lenient for still capture, strict for live tracking
        # Lowered to 0.60 to handle perspective distortion on large markers
        min_circ = 0.40 if force_full else 0.60

        for contour in contours:
            area = cv2.contourArea(contour)
            # Increased max_area to 15000 for high-res 4K+ camera feeds
            if 1 <= area < 15000:
                # Calculate circularity
                peri = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

                # Use intensity-weighted moments for sub-pixel centroid stability
                x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
                pad = 3
                x1 = max(0, x_b - pad)
                y1 = max(0, y_b - pad)
                x2 = min(roi_w, x_b + w_b + pad)
                y2 = min(roi_h, y_b + h_b + pad)

                dot_roi = gray_boosted[y1:y2, x1:x2]
                mask_roi = thresh[y1:y2, x1:x2]
                # Mask the grayscale ROI with the binary blob to ignore nearby noise/projector light
                # while preserving intensity weighting for sub-pixel precision.
                weighted_roi = cv2.bitwise_and(dot_roi, mask_roi)

                # REFINEMENT: Localized contrast boost for the blob area
                if np.max(weighted_roi) > 0:
                    weighted_roi = cv2.normalize(weighted_roi, None, 0, 255, cv2.NORM_MINMAX)

                M = cv2.moments(weighted_roi)

                if M["m00"] != 0:
                    cx_px = (M["m10"] / M["m00"]) + x1
                    cy_px = (M["m01"] / M["m00"]) + y1

                    # Sample peak intensity in the dot area
                    intensity = np.max(dot_roi)

                    cX = (cx_px + roi_x) / w
                    cY = (cy_px + roi_y) / h
                    candidate = {'pos': (cX, cY), 'intensity': intensity, 'area': area, 'circ': circularity}

                    if circularity >= min_circ and intensity >= (self.ir_threshold * 0.9):
                        detected_points_raw.append(candidate)
                    else:
                        rejected_points_raw.append(candidate)

        # Sort by intensity (brightest first) then area/circularity
        detected_points_raw.sort(key=lambda x: (x['intensity'], x['area'], x['circ']), reverse=True)
        detected_points_all = [x['pos'] for x in detected_points_raw]

        # Dot Persistence: only keep dots that were present in the last frame (or if searching)
        detected_points = []
        if self.confidence > 0.5:
            for p in detected_points_all:
                # Increased radii to account for high-velocity "neck" movement
                if any(np.linalg.norm(np.array(p) - np.array(prev_p)) < 0.01 for prev_p in self.last_raw_detections):
                    detected_points.append(p)
                elif any(np.linalg.norm(np.array(p) - np.array(lp)) < 0.05 for lp in (self.last_tracked_points or [])):
                    detected_points.append(p)
        else:
            detected_points = detected_points_all

        self.last_raw_detections = detected_points_all # Store all for next frame's comparison
        rejected_points = [x['pos'] for x in rejected_points_raw]

        if return_rejected:
            # We skip the tracking logic and return raw dots for calibration dialog
            return detected_points, rejected_points

        if marker_cfg and len(marker_cfg) >= 3 and len(detected_points) >= (len(marker_cfg) - 1):
            # OC-TOLERANCE: If we see at least 3 markers (and 1 is missing), we can estimate the guitar pose
            num_markers = len(marker_cfg)

            # Pre-calculate Kalman predictions for stickiness cost (helps handle "swallowed" markers)
            predictions = []
            if self.confidence > 0.3 and len(self.kalman_filters) == num_markers:
                for kf in self.kalman_filters:
                    pred = kf.predict()
                    predictions.append((pred[0, 0], pred[1, 0]))

            # If we were tracking, prioritize points near last known location
            if self.confidence > 0.2 and self.last_tracked_points and len(self.last_tracked_points) > 0:
                center = np.mean(self.last_tracked_points, axis=0)
                detected_points.sort(key=lambda p: np.linalg.norm(np.array(p) - center))

            num_markers = len(marker_cfg)
            # Limit search to top candidates to keep CPU usage low
            limit = 15
            points_to_check = detected_points[:limit]

            best_match = None
            best_matrix = None
            best_overall_error = 1000.0 # High initial cost

            # Try to find a match with N markers first, then N-1
            # We use list to avoid search_count being a generator or None
            # Defensive checks to avoid search crashes
            if marker_fp is None: marker_fp = []
            if points_to_check is None: points_to_check = []

            for search_count in [num_markers, num_markers - 1]:
                if search_count < 3: continue
                if len(points_to_check) < search_count: continue

                for indices in combinations(range(len(points_to_check)), search_count):
                    point_combo = [points_to_check[i] for i in indices]
                    if not point_combo or len(point_combo) < search_count: continue

                    # If we found all N markers
                    if search_count == num_markers:
                        src_pts = np.float32(marker_cfg).reshape(-1, 1, 2)

                        # Case 1: Already tracking - Use distance-based ordering from last known transform
                        if self.last_homography is not None and self.confidence > 0.3:
                            transformed_src = cv2.perspectiveTransform(src_pts, self.last_homography)
                            ordered_dst = []
                            temp_dst = list(point_combo)
                            for i in range(num_markers):
                                if i >= len(transformed_src) or not temp_dst: break
                                pred = transformed_src[i][0]
                                closest_idx = min(range(len(temp_dst)), key=lambda j: np.linalg.norm(np.array(temp_dst[j]) - pred))
                                ordered_dst.append(temp_dst.pop(closest_idx))

                            if len(ordered_dst) == num_markers:
                                ordered_dst_arr = np.float32(ordered_dst).reshape(-1, 1, 2)
                                if len(src_pts) >= 4:
                                    m, _ = cv2.findHomography(src_pts, ordered_dst_arr, 0)
                                else:
                                    m, _ = cv2.estimateAffinePartial2D(src_pts, ordered_dst_arr)
                                    if m is not None:
                                        h = np.eye(3, dtype=np.float32); h[:2, :] = m; m = h

                                if m is not None:
                                    proj = cv2.perspectiveTransform(src_pts, m)
                                    err = np.mean(np.linalg.norm(proj - ordered_dst_arr, axis=2))
                                    if err < 0.05: # Reasonable lock
                                        best_match = ordered_dst
                                        best_matrix = m
                                        best_overall_error = err
                                        break # Good enough

                        # Case 2: Searching - Try all permutations and find the most physically plausible one
                        if best_match is None and len(marker_cfg) <= 4:
                            for p in permutations(point_combo):
                                p_arr = np.float32(p).reshape(-1, 1, 2)
                                if len(src_pts) >= 4:
                                    m, _ = cv2.findHomography(src_pts, p_arr, 0)
                                else:
                                    m, _ = cv2.estimateAffinePartial2D(src_pts, p_arr)
                                    if m is not None:
                                        h = np.eye(3, dtype=np.float32); h[:2, :] = m; m = h

                                if m is not None:
                                    # Calculate reprojection error
                                    proj = cv2.perspectiveTransform(src_pts, m)
                                    err = np.mean(np.linalg.norm(proj - p_arr, axis=2))

                                    # Geometric sanity check: Check determinant to avoid mirrors/flips
                                    det = np.linalg.det(m[:2, :2])
                                    if det <= 0.1: continue # Quick prune

                                    # Fingerprint check: internal distances should match
                                    combo_distances = []
                                    for p1, p2 in combinations(p, 2):
                                        combo_distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
                                    combo_fp = sorted(combo_distances)

                                    # Normalized cost: err + fingerprint_mismatch
                                    if marker_fp and len(combo_fp) == len(marker_fp):
                                        fp_err = np.mean([abs(a - b) for a, b in zip(combo_fp, marker_fp)])
                                    else:
                                        fp_err = 0.5 # Penalty for mismatch in length/missing FP

                                    # Marker Stickiness: favor points closer to their predicted positions
                                    stickiness_err = 0
                                    if predictions and len(predictions) == num_markers:
                                        for i in range(num_markers):
                                            stickiness_err += np.linalg.norm(np.array(p[i]) - np.array(predictions[i]))
                                    elif self.last_tracked_points and len(self.last_tracked_points) == num_markers:
                                        for i in range(num_markers):
                                            stickiness_err += np.linalg.norm(np.array(p[i]) - np.array(self.last_tracked_points[i]))

                                    # Increased stickiness weight to prioritize temporal continuity over visual exactness
                                    total_cost = err + (fp_err * 6.0) + (stickiness_err * 10.0)

                                    if total_cost < best_overall_error:
                                        best_overall_error = total_cost
                                        best_match = list(p)
                                        best_matrix = m
                            if best_match and best_overall_error < 0.04: break

                    # If we found N-1 markers
                    elif search_count == num_markers - 1 and (self.last_homography is not None or predictions):
                        for missing_idx in range(num_markers):
                            subset_cfg = [marker_cfg[i] for i in range(num_markers) if i != missing_idx]
                            src_pts = np.float32(subset_cfg).reshape(-1, 1, 2)

                            # Use Kalman predictions if available for better alignment of the subset
                            if predictions and len(predictions) == num_markers:
                                subset_preds = [predictions[i] for i in range(num_markers) if i != missing_idx]
                                transformed_src = np.float32(subset_preds).reshape(-1, 1, 2)
                            else:
                                transformed_src = cv2.perspectiveTransform(src_pts, self.last_homography)
                            ordered_dst = []
                            temp_dst = list(point_combo)
                            for i in range(search_count):
                                if i >= len(transformed_src) or not temp_dst: break
                                pred = transformed_src[i][0]
                                closest_idx = min(range(len(temp_dst)), key=lambda j: np.linalg.norm(np.array(temp_dst[j]) - pred))
                                ordered_dst.append(temp_dst.pop(closest_idx))

                            if len(ordered_dst) == search_count:
                                ordered_dst_arr = np.float32(ordered_dst).reshape(-1, 1, 2)
                                if len(src_pts) >= 4:
                                    m, _ = cv2.findHomography(src_pts, ordered_dst_arr, 0)
                                else:
                                    m, _ = cv2.estimateAffinePartial2D(src_pts, ordered_dst_arr)
                                    if m is not None:
                                        h = np.eye(3, dtype=np.float32); h[:2, :] = m; m = h

                                if m is not None:
                                    proj = cv2.perspectiveTransform(src_pts, m)
                                    err = np.mean(np.linalg.norm(proj - ordered_dst_arr, axis=2))
                                    if err < 0.05:
                                        full_src = np.float32(marker_cfg).reshape(-1, 1, 2)
                                        predicted_full = cv2.perspectiveTransform(full_src, m).reshape(-1, 2)

                                        # RECOVERY: Look for the missing marker in the rejected list
                                        # This is crucial for the "tricky" neck marker
                                        final_points = list(predicted_full)
                                        missing_pred = predicted_full[missing_idx]

                                        all_cands = detected_points_raw + rejected_points_raw
                                        best_cand = None
                                        min_d = 0.03 # Search radius

                                        for cand in all_cands:
                                            cdist = np.linalg.norm(np.array(cand['pos']) - missing_pred)
                                            if cdist < min_d:
                                                min_d = cdist
                                                best_cand = cand['pos']

                                        if best_cand:
                                            final_points[missing_idx] = best_cand

                                        best_match = [tuple(p) for p in final_points]
                                        best_matrix = m
                                        best_overall_error = err
                                        break
                        if best_match: break
                if best_match: break

            if best_match and best_overall_error < 0.1: # More relaxed to ensure capture in noisy environments
                # Sanity Check: Filter out sudden jumps (reflections)
                if self.last_tracked_points is not None and self.confidence > 0.3:
                    prev_center = np.mean(self.last_tracked_points, axis=0)
                    curr_center = np.mean(best_match, axis=0)
                    # For a guitar, moving > 8% of screen width in one frame is unlikely
                    if np.linalg.norm(curr_center - prev_center) > 0.08:
                        self.confidence = max(0.0, self.confidence - 0.2)
                        pts = self.last_tracked_points if self.last_tracked_points is not None else []
                        return (pts, []) if return_rejected else pts

                # Kalman Correction
                kalman_pts = []
                for i, pt in enumerate(best_match):
                    if i >= len(self.kalman_filters): break
                    # Correct state with new observation
                    self.kalman_filters[i].correct(np.array([[np.float32(pt[0])], [np.float32(pt[1])]]))
                    # statePost is the state after correction
                    state = self.kalman_filters[i].statePost
                    kalman_pts.append((state[0, 0], state[1, 0]))

                if len(kalman_pts) < len(marker_cfg):
                    pts = self.last_tracked_points if self.last_tracked_points else detected_points
                    return (pts, []) if return_rejected else pts

                new_points_arr = np.array(kalman_pts, dtype=np.float32)

                # Decisive Reset: If current points are too far from the history
                if self.smoothed_points is not None and self.confidence > 0.5:
                    dist = np.linalg.norm(np.mean(new_points_arr, axis=0) - np.mean(self.smoothed_points, axis=0))
                    if dist > 0.06: # Significant jump detected (lowered for better stability)
                        self.confidence = max(0.0, self.confidence - 0.15)
                        pts = self.last_tracked_points if self.last_tracked_points is not None else []
                        return (pts, []) if return_rejected else pts

                # High-Distance Temporal Smoothing with Adaptive Alpha & Deadband
                if self.smoothed_points is None:
                    self.smoothed_points = new_points_arr
                    alpha = 1.0
                else:
                    # Calculate movement magnitude
                    move_dist = np.linalg.norm(np.mean(new_points_arr, axis=0) - np.mean(self.smoothed_points, axis=0))

                    # Deadband: if movement is microscopic, ignore it to prevent static jitter
                    # Increased for high-res stability
                    if move_dist < 0.0035:
                        pts = self.last_tracked_points if self.last_tracked_points is not None else []
                        return (pts, []) if return_rejected else pts

                    # Adaptive Alpha: lower when still, higher when moving
                    # base_alpha for stillness
                    base_alpha = (1.0 - self.smoothing_factor) * 0.08
                    # sensitivity to motion
                    motion_scale = 2.0
                    alpha = np.clip(base_alpha + move_dist * motion_scale, base_alpha, 0.9)

                    self.smoothed_points = self.smoothed_points * (1.0 - alpha) + new_points_arr * alpha

                # Moving Average over history
                self.history_points.append(self.smoothed_points.copy())
                if len(self.history_points) > self.history_len:
                    self.history_points.pop(0)

                avg_points = np.mean(self.history_points, axis=0)

                # Decisive snap: if we just regained tracking, don't average with old data
                if self.confidence < 0.3:
                    avg_points = new_points_arr
                    self.history_points = [new_points_arr.copy()]

                # RE-ESTIMATE HOMOGRAPHY from smoothed points to ensure stability of the mask
                src_pts = np.float32(marker_cfg).reshape(-1, 1, 2)
                dst_pts = np.float32(avg_points).reshape(-1, 1, 2)
                if len(src_pts) >= 4:
                    m, _ = cv2.findHomography(src_pts, dst_pts, 0)
                else:
                    m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if m is not None:
                        h = np.eye(3, dtype=np.float32); h[:2, :] = m; m = h

                if m is not None:
                    self.last_homography = m

                if self.confidence < 0.5 and (self.confidence + self.confidence_gain) >= 0.5:
                    self.status_update.emit("Tracking Status: LOCKED")

                self.confidence = min(1.0, self.confidence + self.confidence_gain)
                self.last_tracked_points = [tuple(p) for p in avg_points]
                return (self.last_tracked_points, []) if return_rejected else self.last_tracked_points

        if self.last_tracked_points is not None:
            if roi_x != 0 or roi_y != 0 or roi_w != w or roi_h != h:
                self.last_tracked_points = None
                res = self.get_tracked_points(frame, return_rejected=return_rejected)
                return res if res is not None else (([], []) if return_rejected else [])

        if self.confidence > 0.1 and (self.confidence - self.confidence_decay) <= 0.1:
            self.status_update.emit("Tracking Status: LOST")

        # Drop confidence faster if we don't see anything
        self.confidence = max(0.0, self.confidence - self.confidence_decay * 1.5)

        # If tracking is completely lost, clear history to avoid "jumps" when regained
        if self.confidence <= 0:
            self.history_points = []
            self.smoothed_points = None
            self.last_homography = None
            # Decisive Reset of Kalman filters to prevent old state from polluting new lock
            if self.marker_config:
                self.init_kalman(len(self.marker_config))

        if (self.confidence > 0.1 or self.tracking_freeze_enabled) and self.last_tracked_points is not None:
            return (self.last_tracked_points, []) if return_rejected else self.last_tracked_points

        return (detected_points, rejected_points) if return_rejected else detected_points

    def _update_warp_maps(self, w_render, h_render, w_proj, h_proj):
        if w_render <= 0 or h_render <= 0 or w_proj <= 0 or h_proj <= 0: return
        self.map_x = np.zeros((h_proj, w_proj), dtype=np.float32)
        self.map_y = np.zeros((h_proj, w_proj), dtype=np.float32)

        wp = np.array(self.warp_points)
        wp[:, 0] *= w_proj
