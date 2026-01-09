import numpy as np
from scipy.ndimage import maximum_filter


# code from https://github.com/openvinotoolkit/open_model_zoo/blob/9296a3712069e688fe64ea02367466122c8e8a3b/demos/common/python/models/open_pose.py#L135
class OpenPoseDecoder:

    BODY_PARTS_KPT_IDS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
                          (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17))
    BODY_PARTS_PAF_IDS = (12, 20, 14, 16, 22, 24, 0, 2, 4, 6, 8, 10, 28, 30, 34, 32, 36, 18, 26)

    def __init__(self, num_joints=18, skeleton=BODY_PARTS_KPT_IDS, paf_indices=BODY_PARTS_PAF_IDS,
                 max_points=100, score_threshold=0.1, min_paf_alignment_score=0.05, delta=0.5):
        self.num_joints = num_joints
        self.skeleton = skeleton
        self.paf_indices = paf_indices
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta

        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        batch_size, _, h, w = heatmaps.shape
        assert batch_size == 1, 'Batch size of 1 only supported'

        keypoints = self.extract_points(heatmaps, nms_heatmaps)
        pafs = np.transpose(pafs, (0, 2, 3, 1))

        if self.delta > 0:
            for kpts in keypoints:
                kpts[:, :2] += self.delta
                np.clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                np.clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])

        pose_entries, keypoints = self.group_keypoints(keypoints, pafs, pose_entry_size=self.num_joints + 2)
        poses, scores = self.convert_to_coco_format(pose_entries, keypoints)
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32)
            poses = poses.reshape((poses.shape[0], -1, 3))
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)

        return poses, scores

    def extract_points(self, heatmaps, nms_heatmaps):
        batch_size, channels_num, h, w = heatmaps.shape
        assert batch_size == 1, 'Batch size of 1 only supported'
        assert channels_num >= self.num_joints

        xs, ys, scores = self.top_k(nms_heatmaps)
        masks = scores > self.score_threshold
        all_keypoints = []
        keypoint_id = 0
        for k in range(self.num_joints):
            # Filter low-score points.
            mask = masks[0, k]
            x = xs[0, k][mask].ravel()
            y = ys[0, k][mask].ravel()
            score = scores[0, k][mask].ravel()
            n = len(x)
            if n == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            # Apply quarter offset to improve localization accuracy.
            x, y = self.refine(heatmaps[0, k], x, y)
            np.clip(x, 0, w - 1, out=x)
            np.clip(y, 0, h - 1, out=y)
            # Pack resulting points.
            keypoints = np.empty((n, 4), dtype=np.float32)
            keypoints[:, 0] = x
            keypoints[:, 1] = y
            keypoints[:, 2] = score
            keypoints[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            keypoint_id += n
            all_keypoints.append(keypoints)
        return all_keypoints

    def top_k(self, heatmaps):
        N, K, _, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        # Get positions with top scores.
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points:]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        # Keep top scores sorted.
        subind = np.argsort(-scores, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        scores = np.take_along_axis(scores, subind, axis=2)
        y, x = np.divmod(ind, W)
        return x, y, scores

    @staticmethod
    def refine(heatmap, x, y):
        h, w = heatmap.shape[-2:]
        valid = np.logical_and(np.logical_and(x > 0, x < w - 1), np.logical_and(y > 0, y < h - 1))
        xx = x[valid]
        yy = y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1], dtype=np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx], dtype=np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y

    @staticmethod
    def is_disjoint(pose_a, pose_b):
        pose_a = pose_a[:-2]
        pose_b = pose_b[:-2]
        return np.all(np.logical_or.reduce((pose_a == pose_b, pose_a < 0, pose_b < 0)))

    def update_poses(self, kpt_a_id, kpt_b_id, all_keypoints, connections, pose_entries, pose_entry_size):
        for connection in connections:
            pose_a_idx = -1
            pose_b_idx = -1
            for j, pose in enumerate(pose_entries):
                if pose[kpt_a_id] == connection[0]:
                    pose_a_idx = j
                if pose[kpt_b_id] == connection[1]:
                    pose_b_idx = j
            if pose_a_idx < 0 and pose_b_idx < 0:
                # Create new pose entry.
                pose_entry = np.full(pose_entry_size, -1, dtype=np.float32)
                pose_entry[kpt_a_id] = connection[0]
                pose_entry[kpt_b_id] = connection[1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                pose_entries.append(pose_entry)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                # Merge two poses are disjoint merge them, otherwise ignore connection.
                pose_a = pose_entries[pose_a_idx]
                pose_b = pose_entries[pose_b_idx]
                if self.is_disjoint(pose_a, pose_b):
                    pose_a += pose_b
                    pose_a[:-2] += 1
                    pose_a[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
                # Adjust score of a pose.
                pose_entries[pose_a_idx][-2] += connection[2]
            elif pose_a_idx >= 0:
                # Add a new limb into pose.
                pose = pose_entries[pose_a_idx]
                if pose[kpt_b_id] < 0:
                    pose[-2] += all_keypoints[connection[1], 2]
                pose[kpt_b_id] = connection[1]
                pose[-2] += connection[2]
                pose[-1] += 1
            elif pose_b_idx >= 0:
                # Add a new limb into pose.
                pose = pose_entries[pose_b_idx]
                if pose[kpt_a_id] < 0:
                    pose[-2] += all_keypoints[connection[0], 2]
                pose[kpt_a_id] = connection[0]
                pose[-2] += connection[2]
                pose[-1] += 1
        return pose_entries

    @staticmethod
    def connections_nms(a_idx, b_idx, affinity_scores):
        # From all retrieved connections that share starting/ending keypoints leave only the top-scoring ones.
        order = affinity_scores.argsort()[::-1]
        affinity_scores = affinity_scores[order]
        a_idx = a_idx[order]
        b_idx = b_idx[order]
        idx = []
        has_kpt_a = set()
        has_kpt_b = set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in has_kpt_a and j not in has_kpt_b:
                idx.append(t)
                has_kpt_a.add(i)
                has_kpt_b.add(j)
        idx = np.asarray(idx, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20):
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries = []
        # For every limb.
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue

            # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            # Sample points along every candidate limb vector.
            steps = (1 / (self.points_per_limb - 1) * vec_raw)
            points = steps * self.grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()

            # Compute affinity score between candidate limb vectors and part affinity field.
            part_pafs = pafs[0, :, :, paf_channel:paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_affinity_scores = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb

            # Get a list of limbs according to the obtained affinity score.
            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]

            # Suppress incompatible connections.
            a_idx, b_idx, affinity_scores = self.connections_nms(a_idx, b_idx, affinity_scores)
            connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                                   kpts_b[b_idx, 3].astype(np.int32),
                                   affinity_scores))
            if len(connections) == 0:
                continue

            # Update poses with new connections.
            pose_entries = self.update_poses(kpt_a_id, kpt_b_id, all_keypoints,
                                             connections, pose_entries, pose_entry_size)

        # Remove poses with not enough points.
        pose_entries = np.asarray(pose_entries, dtype=np.float32).reshape(-1, pose_entry_size)
        pose_entries = pose_entries[pose_entries[:, -1] >= 3]
        return pose_entries, all_keypoints

    @staticmethod
    def convert_to_coco_format(pose_entries, all_keypoints):
        num_joints = 17
        coco_keypoints = []
        scores = []
        for pose in pose_entries:
            if len(pose) == 0:
                continue
            keypoints = np.zeros(num_joints * 3)
            reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose[-2]
            for keypoint_id, target_id in zip(pose[:-2], reorder_map):
                if target_id < 0:
                    continue
                cx, cy, score = 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose[-1] - 1)))  # -1 for 'neck'
        return np.asarray(coco_keypoints), np.asarray(scores)


# Associative Embedding decoder for human-pose-estimation-0005 model
# Reference: https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/model_zoo/model_api/models/hpe_associative_embedding.py
class AssociativeEmbeddingDecoder:
    """
    Decoder for associative embedding-based pose estimation.
    Groups detected keypoints into individual poses using embedding similarity.
    
    Input format:
    - Heatmaps: shape (1, 17, H, W) - keypoint location probabilities for 17 COCO joints
    - Embeddings: shape (1, 17, H, W, 1) - associative embedding values for grouping
    - NMS pre-applied: negative values in heatmaps indicate filtered locations
    
    Output format:
    - List of poses: array of shape (N, 17, 3) where N is number of detected people
    - Each keypoint: (x, y, score) coordinates and confidence
    """
    
    def __init__(self, num_joints=17, max_points=100, score_threshold=0.25, 
                 delta=0.5, tag_threshold=1.0, detection_threshold=0.1,
                 max_num_people=30, adjust=True):
        """
        Args:
            num_joints: Number of keypoint types (17 for COCO)
            max_points: Maximum keypoints to extract per joint type
            score_threshold: Minimum score for a keypoint to be considered
            delta: Offset adjustment for keypoint localization
            tag_threshold: Maximum embedding distance for keypoints to be grouped
            detection_threshold: Minimum mean score for a pose to be kept
            max_num_people: Maximum number of people to detect
            adjust: Whether to apply sub-pixel refinement
        """
        self.num_joints = num_joints
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.delta = delta
        self.tag_threshold = tag_threshold
        self.detection_threshold = detection_threshold
        self.max_num_people = max_num_people
        self.adjust = adjust
    
    def __call__(self, heatmaps, embeddings):
        """
        Decode poses from heatmaps and embeddings.
        
        Args:
            heatmaps: Keypoint heatmaps, shape (1, 17, H, W)
            embeddings: Associative embeddings, shape (1, 17, H, W, 1)
        
        Returns:
            poses: Array of shape (N, 17, 3) with keypoints (x, y, score)
            scores: Array of shape (N,) with pose confidence scores
        """
        batch_size = heatmaps.shape[0]
        assert batch_size == 1, 'Batch size of 1 only supported'
        
        # Remove batch dimension
        heatmaps = heatmaps[0]  # (17, H, W)
        embeddings = embeddings[0]  # (17, H, W, 1)
        
        # Extract keypoint candidates from heatmaps
        all_keypoints, all_tags = self._extract_keypoints(heatmaps, embeddings)
        
        # Group keypoints into poses using embedding similarity
        poses, scores = self._group_keypoints(all_keypoints, all_tags)
        
        return poses, scores
    
    def _extract_keypoints(self, heatmaps, embeddings):
        """
        Extract keypoint candidates from heatmaps and their embedding tags.
        Handles pre-applied NMS (negative values are suppressed).
        
        Args:
            heatmaps: shape (num_joints, H, W)
            embeddings: shape (num_joints, H, W, 1)
        
        Returns:
            all_keypoints: List of arrays, each (N, 4) with [x, y, score, joint_id]
            all_tags: List of arrays, each (N,) with embedding values
        """
        num_joints, h, w = heatmaps.shape
        
        all_keypoints = []
        all_tags = []
        
        for joint_id in range(num_joints):
            heatmap = heatmaps[joint_id]
            embedding = embeddings[joint_id, :, :, 0]  # (H, W)
            
            # Handle NMS pre-applied: only consider non-negative values
            # Negative values indicate suppressed locations
            valid_mask = heatmap >= 0
            heatmap_valid = np.where(valid_mask, heatmap, 0)
            
            # Find local maxima above threshold
            candidates = self._find_peaks(heatmap_valid)
            
            if len(candidates) == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                all_tags.append(np.empty(0, dtype=np.float32))
                continue
            
            # Extract top candidates
            y_coords, x_coords = candidates
            scores = heatmap_valid[y_coords, x_coords]
            
            # Filter by score threshold
            valid_idx = scores > self.score_threshold
            x_coords = x_coords[valid_idx]
            y_coords = y_coords[valid_idx]
            scores = scores[valid_idx]
            
            # Limit to max_points
            if len(scores) > self.max_points:
                top_idx = np.argsort(scores)[-self.max_points:]
                x_coords = x_coords[top_idx]
                y_coords = y_coords[top_idx]
                scores = scores[top_idx]
            
            # Apply sub-pixel refinement if enabled
            if self.adjust and len(x_coords) > 0:
                x_coords, y_coords = self._refine_coordinates(
                    heatmap_valid, x_coords, y_coords, w, h
                )
            
            # Apply delta offset
            if self.delta > 0:
                x_coords = x_coords.astype(np.float32) + self.delta
                y_coords = y_coords.astype(np.float32) + self.delta
                x_coords = np.clip(x_coords, 0, w - 1)
                y_coords = np.clip(y_coords, 0, h - 1)
            
            # Extract embeddings for these keypoints
            tags = embedding[y_coords.astype(int), x_coords.astype(int)]
            
            # Pack keypoints: [x, y, score, joint_id]
            n = len(x_coords)
            keypoints = np.zeros((n, 4), dtype=np.float32)
            keypoints[:, 0] = x_coords
            keypoints[:, 1] = y_coords
            keypoints[:, 2] = scores
            keypoints[:, 3] = joint_id
            
            all_keypoints.append(keypoints)
            all_tags.append(tags)
        
        return all_keypoints, all_tags
    
    def _find_peaks(self, heatmap):
        """
        Find local maxima in heatmap using non-maximum suppression.
        
        Args:
            heatmap: 2D array (H, W)
        
        Returns:
            Tuple of (y_coords, x_coords) for peak locations
        """
        # Apply maximum filter to find local maxima
        max_filtered = maximum_filter(heatmap, size=3, mode='constant')
        
        # Peaks are locations where value equals max filtered value
        peaks = (heatmap == max_filtered) & (heatmap > 0)
        
        # Get coordinates
        y_coords, x_coords = np.where(peaks)
        
        return y_coords, x_coords
    
    def _refine_coordinates(self, heatmap, x, y, w, h):
        """
        Apply sub-pixel refinement to keypoint locations.
        
        Args:
            heatmap: 2D array (H, W)
            x, y: Keypoint coordinates (arrays)
            w, h: Heatmap dimensions
        
        Returns:
            Refined (x, y) coordinates
        """
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        # Find valid points (not on border)
        valid = (x > 0) & (x < w - 1) & (y > 0) & (y < h - 1)
        
        if not np.any(valid):
            return x, y
        
        x_valid = x[valid].astype(int)
        y_valid = y[valid].astype(int)
        
        # Compute gradients for sub-pixel refinement
        dx = np.sign(heatmap[y_valid, x_valid + 1] - heatmap[y_valid, x_valid - 1]) * 0.25
        dy = np.sign(heatmap[y_valid + 1, x_valid] - heatmap[y_valid - 1, x_valid]) * 0.25
        
        x[valid] = x[valid] + dx
        y[valid] = y[valid] + dy
        
        return x, y
    
    def _group_keypoints(self, all_keypoints, all_tags):
        """
        Group keypoints into individual poses using embedding similarity.
        
        Args:
            all_keypoints: List of keypoint arrays per joint type
            all_tags: List of embedding arrays per joint type
        
        Returns:
            poses: Array of shape (N, 17, 3) with (x, y, score) per keypoint
            scores: Array of shape (N,) with pose confidence scores
        """
        # Collect all detected keypoints with their tags
        all_detections = []
        for joint_id in range(self.num_joints):
            keypoints = all_keypoints[joint_id]
            tags = all_tags[joint_id]
            
            for i in range(len(keypoints)):
                all_detections.append({
                    'joint_id': int(keypoints[i, 3]),
                    'x': keypoints[i, 0],
                    'y': keypoints[i, 1],
                    'score': keypoints[i, 2],
                    'tag': tags[i]
                })
        
        if len(all_detections) == 0:
            return np.empty((0, 17, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        
        # Group detections by embedding similarity
        poses = self._match_by_tag(all_detections)
        
        # Convert to output format
        return self._format_poses(poses)
    
    def _match_by_tag(self, detections):
        """
        Match keypoints by embedding similarity to form poses.
        
        Args:
            detections: List of detection dicts with keys: joint_id, x, y, score, tag
        
        Returns:
            List of pose dicts, each containing keypoints for one person
        """
        if len(detections) == 0:
            return []
        
        # Group detections by joint type
        joint_detections = [[] for _ in range(self.num_joints)]
        for det in detections:
            joint_detections[det['joint_id']].append(det)
        
        # Initialize poses list
        poses = []
        
        # Use greedy matching: start with highest-scoring detections
        # and build poses by finding compatible keypoints
        used = set()
        
        # Sort all detections by score
        sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        for det in sorted_dets:
            det_id = id(det)
            if det_id in used or len(poses) >= self.max_num_people:
                continue
            
            # Start a new pose with this detection
            pose = {i: None for i in range(self.num_joints)}
            pose[det['joint_id']] = det
            pose_tag = det['tag']
            used.add(det_id)
            
            # Try to add compatible keypoints from other joint types
            for joint_id in range(self.num_joints):
                if joint_id == det['joint_id']:
                    continue
                
                # Find best matching detection for this joint
                best_det = None
                best_dist = float('inf')
                
                for candidate in joint_detections[joint_id]:
                    cand_id = id(candidate)
                    if cand_id in used:
                        continue
                    
                    # Compute embedding distance
                    tag_dist = abs(candidate['tag'] - pose_tag)
                    
                    if tag_dist < best_dist and tag_dist < self.tag_threshold:
                        best_dist = tag_dist
                        best_det = candidate
                
                # Add best match to pose
                if best_det is not None:
                    pose[joint_id] = best_det
                    used.add(id(best_det))
                    # Update pose tag as running average
                    pose_tag = (pose_tag + best_det['tag']) / 2
            
            poses.append(pose)
        
        return poses
    
    def _format_poses(self, poses):
        """
        Convert pose dictionaries to output array format.
        
        Args:
            poses: List of pose dicts
        
        Returns:
            poses_array: Array of shape (N, 17, 3)
            scores_array: Array of shape (N,)
        """
        if len(poses) == 0:
            return np.empty((0, 17, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        
        poses_list = []
        scores_list = []
        
        for pose in poses:
            # Build pose array: (17, 3) with [x, y, score]
            pose_array = np.zeros((self.num_joints, 3), dtype=np.float32)
            
            num_valid = 0
            total_score = 0.0
            
            for joint_id in range(self.num_joints):
                if pose[joint_id] is not None:
                    det = pose[joint_id]
                    pose_array[joint_id, 0] = det['x']
                    pose_array[joint_id, 1] = det['y']
                    pose_array[joint_id, 2] = det['score']
                    num_valid += 1
                    total_score += det['score']
                # else: remains [0, 0, 0] for missing keypoints
            
            # Filter poses with too few keypoints or low score
            if num_valid > 0:
                mean_score = total_score / num_valid
                if mean_score >= self.detection_threshold and num_valid >= 3:
                    poses_list.append(pose_array)
                    scores_list.append(mean_score * num_valid)  # Weighted by number of joints
        
        if len(poses_list) == 0:
            return np.empty((0, 17, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        
        return np.array(poses_list, dtype=np.float32), np.array(scores_list, dtype=np.float32)
