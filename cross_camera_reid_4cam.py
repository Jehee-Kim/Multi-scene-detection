"""
Cross-Camera Re-Identification Pipeline (4-Camera Version)
============================================================

YOLO11 detect + BoT-SORT tracking + OSNet feature matching + evidence reassignment + Global ID archive

변경점 (2-cam → 4-cam)
- VIDEO_SETS 초기값에서 비디오 세트를 선택해 실행
  · VIDEO_SETS[0] = [cam0, cam1, cam2, cam3]
  · VIDEO_SETS[1] = [cam0, cam1, cam2, cam3]
- Gallery 구조를 N-camera 범용으로 재설계
  · 초기 K frame(--initial_frames) 동안 모든 카메라를 동시에 보며 initial GID 부여
  · initial 이후 Camera 0 → update_cam_source, Camera 1-3 → update_cam_query
- 디스플레이는 2x2 그리드 (카메라 수에 따라 자동 조정)
- 나머지 Re-ID 로직(evidence decay, occlusion-aware, GID archive)은 기존 v10과 동일

설치
    pip install -U ultralytics opencv-python torch torchvision scipy numpy
    pip install torchreid
    pip install huggingface_hub

실행 예시
    # VIDEO_SETS[VIDEO_SET_INDEX]에 지정된 비디오 세트 실행
    python cross_camera_reid_4cam.py

    # 코드 상단의 VIDEO_SET_INDEX = 1 로 바꾸면 VIDEO_SETS[1] 실행
    # RUN_ALL_VIDEO_SETS = True 로 바꾸면 VIDEO_SETS 전체 순차 실행
"""

import argparse
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────────────────
YOLO_MODEL       = "yolov8n.pt"
OSNET_MODEL      = "osnet_x1_0"
OSNET_WEIGHTS    = "market1501"
OSNET_HF_REPO    = "MYerassyl/retail-heat-osnet"
OSNET_HF_FILENAME = "osnet_x1_0_market1501.pth"
PERSON_CLASS     = 0

MATCH_THRESHOLD       = 0.50
GALLERY_MAX_AGE       = 300
DISPLAY_W             = 640
DISPLAY_H             = 480
FRAME_STRIDE          = 20
INITIAL_FRAMES        = FRAME_STRIDE * 5  # 0이면 initial all-camera 단계 비활성화
# Initial 단계 전용 매칭 조건. dist = 1 - similarity 이므로 threshold는 낮을수록 엄격합니다.
INITIAL_MATCH_THRESHOLD = 0.35
INITIAL_DISTANCE_MARGIN = 0.25
INITIAL_MIN_FEATURES_TO_MATCH = 2
AVOID_CURRENT_FRAME_DUP_GID = True  # True이면 현재 프레임에서 이미 사용된 GID는 다음 후보로 넘깁니다.

# ── 비디오 세트 초기값 ───────────────────────────────────────────────────────
# args로 비디오 경로를 받지 않고, 여기서 직접 세트를 지정합니다.
# VIDEO_SETS[0]에는 카메라 0~3 영상 4개가 들어갑니다.
# VIDEO_SETS[1]에는 다른 비디오 세트의 카메라 0~3 영상 4개를 넣으면 됩니다.
VIDEO_SETS: List[List[str]] = [
    [
        "/home/knuvi/Multi-scene-detection/dataset/MOT2024/hospital/Camera_02.mp4",
        "/home/knuvi/Multi-scene-detection/dataset/MOT2024/hospital/Camera_18.mp4",
        "/home/knuvi/Multi-scene-detection/dataset/MOT2024/hospital/Camera_22.mp4",
        "/home/knuvi/Multi-scene-detection/dataset/MOT2024/hospital/Camera_29.mp4",
    ],
    [
        "/home/knuvi/Multi-scene-detection/dataset/MOT2024/Camera_0002.mp4",
        "/home/knuvi/Multi-scene-detection/dataset/MOT2024/Camera_0003.mp4",
        "/home/knuvi/Multi-scene-detection/dataset/MOT2024/Camera_0007.mp4",
        "/home/knuvi/Multi-scene-detection/dataset/MOT2024/Camera_0023.mp4",
    ]
]

# 기본으로 실행할 비디오 세트 인덱스입니다.
# 예: 0이면 VIDEO_SETS[0], 1이면 VIDEO_SETS[1]
VIDEO_SET_INDEX = 1

# True로 바꾸면 VIDEO_SETS에 들어있는 모든 세트를 순서대로 처리합니다.
RUN_ALL_VIDEO_SETS = False

MAX_FEATURES_PER_TRACK = 15
MIN_FEATURES_TO_MATCH  = 1
MIN_DET_CONF           = 0.30
MIN_BBOX_AREA          = 700
MIN_DET_BBOX_AREA      = 800
MATCH_CONFIRM_COUNT    = 2
EVIDENCE_DECAY         = 0.90
MIN_EVIDENCE           = 1.00
EVIDENCE_MARGIN        = 0.25
SWITCH_MARGIN          = 0.40
DISTANCE_MARGIN        = 0.03
YOLO_IMGSZ             = 1280
YOLO_CONF              = 0.18
YOLO_IOU               = 0.85
YOLO_AUGMENT           = False
CROP_PADDING           = 0.15
TOPK_PAIRWISE          = 5
MEAN_SCORE_WEIGHT      = 0.50
SET_SCORE_WEIGHT       = 0.50

SKIP_OCCLUDED_UPDATE    = True
OCCLUSION_IOU_THRESHOLD = 0.25
EDGE_MARGIN_RATIO       = 0.02
MIN_ASPECT_RATIO        = 0.20
MAX_ASPECT_RATIO        = 1.20
GLOBAL_MEMORY_MAX_AGE   = 900
REASSOC_DISTANCE_MARGIN = 0.02

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
PALETTE = [
    (220, 80,  80),  (80, 180,  80),  (80, 120, 220),
    (200, 160, 50),  (160, 80,  200), (50, 190,  190),
    (230, 120, 50),  (100, 200, 150), (180, 80,  150),
    (80, 160,  230), (210, 210,  80), (130, 100, 230),
]


def get_color(gid: int):
    return PALETTE[int(gid) % len(PALETTE)]


def l2_normalize(feat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(feat)
    if norm < 1e-12:
        return feat
    return feat / norm


# ── OSNet feature extractor ──────────────────────────────────────────────────
class OSNetExtractor:
    def __init__(self, model_name: str = OSNET_MODEL, weights: str = OSNET_WEIGHTS, device: str = "cpu"):
        self.device = torch.device(device)
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError as e:
            raise ImportError(
                "torchreid가 설치되어 있지 않습니다.\n"
                "  pip install torchreid\n"
                "또는:\n"
                "  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git\n"
            ) from e

        model_path = ""
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=OSNET_HF_REPO, filename=OSNET_HF_FILENAME)
            print(f"[INFO] OSNet weights : {model_path}")
        except Exception as e:
            print(f"[WARN] OSNet Market1501 weight 다운로드 실패 → ImageNet pretrained 사용. 정확도가 낮을 수 있습니다. ({e})")

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=str(self.device),
        )

    @torch.no_grad()
    def extract(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        h, w = bgr_crop.shape[:2]
        if h <= 0 or w <= 0:
            return None
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        feat = self.extractor([rgb])
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat)
        if feat.ndim == 2:
            feat = feat[0]
        return l2_normalize(feat.astype(np.float32))


# ── Tracker wrapper ───────────────────────────────────────────────────────────
class Tracker:
    def __init__(self, model_path: str, imgsz: int = YOLO_IMGSZ, conf: float = YOLO_CONF,
                 iou: float = YOLO_IOU, augment: bool = YOLO_AUGMENT,
                 tracker_name: str = "botsort.yaml"):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf  = conf
        self.iou   = iou
        self.augment = augment
        self.tracker_name = tracker_name

    def update(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, int, float]]:
        results = self.model.track(
            frame, persist=True, tracker=self.tracker_name,
            classes=[PERSON_CLASS], imgsz=self.imgsz,
            conf=self.conf, iou=self.iou, augment=self.augment, verbose=False,
        )
        detections = []
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        if boxes.id is None:
            return detections
        ids   = boxes.id.int().tolist()
        xyxy  = boxes.xyxy.int().tolist()
        confs = boxes.conf.float().tolist() if boxes.conf is not None else [1.0] * len(ids)
        for tid, box, conf in zip(ids, xyxy, confs):
            x1, y1, x2, y2 = box
            detections.append((int(tid), int(x1), int(y1), int(x2), int(y2), float(conf)))
        return detections


# ── Feature buffer ────────────────────────────────────────────────────────────
class TrackFeatureStore:
    def __init__(self, max_features: int = MAX_FEATURES_PER_TRACK):
        self.max_features = max_features
        self.buffers: Dict[Union[int, str], Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=max_features))
        self.last_seen: Dict[Union[int, str], int] = {}

    def update(self, key, feat: np.ndarray, frame_idx: int):
        self.buffers[key].append(l2_normalize(feat.astype(np.float32)))
        self.last_seen[key] = frame_idx

    def get_mean(self, key) -> Optional[np.ndarray]:
        buf = self.buffers.get(key)
        if not buf:
            return None
        return l2_normalize(np.mean(np.stack(list(buf)), axis=0).astype(np.float32))

    def get_features(self, key) -> List[np.ndarray]:
        return list(self.buffers.get(key, []))

    def count(self, key) -> int:
        buf = self.buffers.get(key)
        return len(buf) if buf else 0

    def keys(self):
        return list(self.buffers.keys())

    def expire(self, frame_idx: int, max_age: int):
        expired = [k for k, last in self.last_seen.items() if frame_idx - last > max_age]
        for k in expired:
            self.last_seen.pop(k, None)
            self.buffers.pop(k, None)


# ── Gallery (N-camera) ────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))


def pairwise_topk_similarity(feats_a: List[np.ndarray], feats_b: List[np.ndarray],
                              topk: int = TOPK_PAIRWISE) -> Optional[float]:
    if not feats_a or not feats_b:
        return None
    A = np.stack([l2_normalize(f.astype(np.float32)) for f in feats_a])
    B = np.stack([l2_normalize(f.astype(np.float32)) for f in feats_b])
    flat = (A @ B.T).reshape(-1)
    k = max(1, min(topk, flat.size))
    return float(np.mean(np.partition(flat, -k)[-k:]))


class Gallery:
    """
    N-camera 범용 Gallery.

    - initial phase : 초기 K frame 동안 cam0 기준 없이 모든 카메라 track을 같은 Global memory에 매칭한다.
    - regular phase : initial 이후 cam_idx=0은 source, cam_idx>0은 query로 동작한다.

    같은 카메라 안에서는 서로 다른 LID가 같은 GID를 동시에 가질 수 없고,
    다른 카메라끼리는 같은 사람이면 같은 GID를 공유할 수 있다.
    """

    def __init__(self, num_cams: int, threshold: float, max_age: int,
                 max_features: int = MAX_FEATURES_PER_TRACK,
                 min_features_to_match: int = MIN_FEATURES_TO_MATCH,
                 confirm_count: int = MATCH_CONFIRM_COUNT,
                 topk: int = TOPK_PAIRWISE,
                 mean_weight: float = MEAN_SCORE_WEIGHT,
                 set_weight: float = SET_SCORE_WEIGHT,
                 evidence_decay: float = EVIDENCE_DECAY,
                 min_evidence: float = MIN_EVIDENCE,
                 evidence_margin: float = EVIDENCE_MARGIN,
                 switch_margin: float = SWITCH_MARGIN,
                 distance_margin: float = DISTANCE_MARGIN,
                 memory_max_age: int = GLOBAL_MEMORY_MAX_AGE,
                 reassoc_margin: float = REASSOC_DISTANCE_MARGIN,
                 initial_threshold: float = INITIAL_MATCH_THRESHOLD,
                 initial_distance_margin: float = INITIAL_DISTANCE_MARGIN,
                 initial_min_features_to_match: int = INITIAL_MIN_FEATURES_TO_MATCH):

        self.num_cams = num_cams
        self.threshold = threshold
        self.max_age = max_age
        self.memory_max_age = memory_max_age
        self.reassoc_margin = reassoc_margin
        self.min_features_to_match = min_features_to_match
        self.confirm_count = confirm_count
        self.topk = topk
        ws = max(1e-6, mean_weight + set_weight)
        self.mean_weight = mean_weight / ws
        self.set_weight   = set_weight / ws
        self.evidence_decay  = evidence_decay
        self.min_evidence    = min_evidence
        self.evidence_margin = evidence_margin
        self.switch_margin   = switch_margin
        self.distance_margin = distance_margin
        self.initial_threshold = initial_threshold
        self.initial_distance_margin = initial_distance_margin
        self.initial_min_features_to_match = max(1, int(initial_min_features_to_match))

        # cam_idx → {local_id → feature store}
        self.stores: List[TrackFeatureStore] = [
            TrackFeatureStore(max_features) for _ in range(num_cams)
        ]

        # Global ID memory (shared across all cameras)
        self.store_gid = TrackFeatureStore(max_features * 2)
        self.gid_last_seen: Dict[int, int] = {}
        self.gid_last_cam:  Dict[int, int] = {}

        # cam0 local → GID (즉시 부여)
        self.src_gid: Dict[int, int] = {}
        self.gid_owner_src: Dict[int, int] = {}  # gid → src local_id

        # cam1~N : per-cam evidence/assignment
        # key: (cam_idx, local_id)
        self.q_current_gid: Dict[Tuple[int,int], int]            = {}
        self.q_temp_gid:    Dict[Tuple[int,int], int]            = {}
        self.q_evidence:    Dict[Tuple[int,int], Dict[int,float]] = defaultdict(lambda: defaultdict(float))
        self.q_hits:        Dict[Tuple[int,int], Dict[int,int]]   = defaultdict(lambda: defaultdict(int))
        self.q_last_best:   Dict[Tuple[int,int], Optional[float]] = {}

        # query camera one-to-one: (cam_idx, gid) → local_id
        # 같은 카메라 안에서는 하나의 GID를 하나의 LID만 소유하도록 관리한다.
        # 다른 카메라끼리는 같은 GID를 공유할 수 있다.
        self.gid_owner_q: Dict[Tuple[int,int], int] = {}

        self._next_gid = 0

    # ── helpers ──────────────────────────────────────────────────────────────
    def _new_gid(self) -> int:
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _temp_gid(self, cam_idx: int, local_id: int) -> int:
        key = (cam_idx, local_id)
        if key not in self.q_temp_gid:
            # 음수 temp ID: -(cam_idx * 10000 + local_id + 1) 로 충돌 없이 구분
            self.q_temp_gid[key] = -(cam_idx * 10000 + local_id + 1)
        return self.q_temp_gid[key]

    def _update_gid_memory(self, gid: int, feat: np.ndarray, frame_idx: int, cam_idx: int):
        if gid is None or gid < 0:
            return
        self.store_gid.update(gid, feat, frame_idx)
        self.gid_last_seen[gid] = frame_idx
        self.gid_last_cam[gid] = cam_idx

    def _dist(self, mean_a, feats_a, mean_b, feats_b) -> Optional[float]:
        if mean_a is None or mean_b is None:
            return None
        ms = cosine_sim(mean_a, mean_b)
        ss = pairwise_topk_similarity(feats_a, feats_b, self.topk)
        if ss is None:
            ss = ms
        return float(1.0 - (self.mean_weight * ms + self.set_weight * ss))

    def _track_to_gid_dist(self, store: TrackFeatureStore, lid: int, gid: int) -> Optional[float]:
        return self._dist(
            store.get_mean(lid), store.get_features(lid),
            self.store_gid.get_mean(gid), self.store_gid.get_features(gid),
        )

    def _rank_gid_candidates(self, store: TrackFeatureStore, lid: int,
                              cam_idx: int, exclude_active_same_cam: bool = True,
                              min_gid_features: Optional[int] = None,
                              exclude_gids: Optional[Set[int]] = None,
                              ) -> List[Tuple[int, float]]:
        candidates = []
        excluded = set(exclude_gids or [])
        required_gid_features = self.min_features_to_match if min_gid_features is None else max(1, int(min_gid_features))
        for gid in self.store_gid.keys():
            if gid < 0 or self.store_gid.count(gid) < required_gid_features:
                continue
            # 현재 프레임에서 이미 다른 detection이 사용한 GID는 후보에서 제외한다.
            # 이렇게 하면 1순위 GID가 이미 보이는 경우 자동으로 2순위/3순위 후보로 넘어간다.
            if gid in excluded:
                continue
            if exclude_active_same_cam and cam_idx == 0:
                if self.gid_owner_src.get(gid) is not None and self.gid_owner_src.get(gid) != lid:
                    continue
            if exclude_active_same_cam and cam_idx > 0:
                # 같은 카메라 내에서 이미 다른 LID가 쓰고 있는 GID는 후보에서 제외
                owner_lid = self.gid_owner_q.get((cam_idx, gid))
                if owner_lid is not None and owner_lid != lid:
                    continue
            dist = self._track_to_gid_dist(store, lid, gid)
            if dist is not None:
                candidates.append((gid, dist))
        candidates.sort(key=lambda x: x[1])
        return candidates

    def _release_src_gid(self, local_id: int, gid: Optional[int]):
        if gid is not None and gid >= 0 and self.gid_owner_src.get(gid) == local_id:
            self.gid_owner_src.pop(gid, None)

    def _assign_initial_gid(self, cam_idx: int, local_id: int,
                            gid: int, score: float = 0.0) -> bool:
        """
        Initial phase에서 cam0/source 기준 없이 GID를 배정한다.
        - 같은 카메라 내부: 하나의 GID는 하나의 LID만 소유 가능
        - 다른 카메라 간: 같은 GID 공유 가능
        """
        if gid is None or gid < 0:
            return False

        if cam_idx == 0:
            current = self.src_gid.get(local_id)
            owner_lid = self.gid_owner_src.get(gid)
            if owner_lid is not None and owner_lid != local_id:
                return False
            if current is not None and current != gid:
                self._release_src_gid(local_id, current)
            self.src_gid[local_id] = gid
            self.gid_owner_src[gid] = local_id
            return True

        key = (cam_idx, local_id)
        owner_key = (cam_idx, gid)
        owner_lid = self.gid_owner_q.get(owner_key)

        if owner_lid is not None and owner_lid != local_id:
            owner_score = self.q_evidence[(cam_idx, owner_lid)].get(gid, 0.0)
            # 같은 카메라에서 이미 쓰는 LID가 있으면 더 확실할 때만 교체
            if score <= owner_score + self.switch_margin:
                if key not in self.q_current_gid:
                    self.q_current_gid[key] = self._temp_gid(cam_idx, local_id)
                return False
            owner_track_key = (cam_idx, owner_lid)
            self.q_current_gid[owner_track_key] = self._temp_gid(cam_idx, owner_lid)
            self.gid_owner_q.pop(owner_key, None)

        current = self.q_current_gid.get(key)
        if current is not None and current != gid:
            self._release_q_gid(cam_idx, local_id, current)

        self.q_current_gid[key] = gid
        self.gid_owner_q[owner_key] = local_id

        # initial에서 부여한 GID가 regular query 단계로 넘어가도 바로 안정적으로 유지되도록 evidence를 seed한다.
        seeded_score = max(float(score), self.min_evidence)
        self.q_evidence[key][gid] = max(self.q_evidence[key].get(gid, 0.0), seeded_score)
        self.q_hits[key][gid] = max(self.q_hits[key].get(gid, 0), self.confirm_count)
        return True

    def update_initial(self, cam_idx: int, local_id: int,
                       feat: np.ndarray, frame_idx: int,
                       exclude_gids: Optional[Set[int]] = None) -> Tuple[int, Optional[float]]:
        """
        초기 K frame 전용 update.
        cam0을 기준 카메라로 고정하지 않고, 모든 카메라 track을 동일한 Global ID memory에 매칭한다.
        """
        self.stores[cam_idx].update(local_id, feat, frame_idx)
        excluded = set(exclude_gids or [])

        current_gid = self.get_gid(cam_idx, local_id)
        if current_gid is not None and current_gid >= 0 and current_gid not in excluded:
            self._update_gid_memory(current_gid, feat, frame_idx, cam_idx)
            return current_gid, None

        if self.stores[cam_idx].count(local_id) < self.initial_min_features_to_match:
            if cam_idx == 0:
                gid = self.src_gid.get(local_id)
                if gid is None:
                    gid = self._new_gid()
                    self._assign_initial_gid(cam_idx, local_id, gid, score=1.0)
            else:
                gid = self.q_current_gid.get((cam_idx, local_id), self._temp_gid(cam_idx, local_id))
                self.q_current_gid[(cam_idx, local_id)] = gid
            return gid, None

        cands = self._rank_gid_candidates(
            self.stores[cam_idx], local_id, cam_idx,
            min_gid_features=self.initial_min_features_to_match,
            exclude_gids=excluded,
        )
        best_gid = None
        best_dist = None
        second_dist = float("inf")
        if cands:
            best_gid, best_dist = cands[0]
            second_dist = cands[1][1] if len(cands) > 1 else float("inf")

        use_existing = (
            best_gid is not None and
            best_dist is not None and
            best_dist <= self.initial_threshold and
            (second_dist - best_dist) >= self.initial_distance_margin
        )

        if use_existing:
            gid = best_gid
            score = max(0.0, 1.0 - float(best_dist))
            assigned = self._assign_initial_gid(cam_idx, local_id, gid, score=score)
            if not assigned:
                gid = self._new_gid()
                self._assign_initial_gid(cam_idx, local_id, gid, score=1.0)
        else:
            gid = self._new_gid()
            self._assign_initial_gid(cam_idx, local_id, gid, score=1.0)

        self._update_gid_memory(gid, feat, frame_idx, cam_idx)
        return gid, best_dist

    # ── cam0 (source camera) ──────────────────────────────────────────────────
    def _select_existing_gid_for_src(self, local_id: int,
                                     exclude_gids: Optional[Set[int]] = None) -> Optional[int]:
        cands = self._rank_gid_candidates(self.stores[0], local_id, cam_idx=0, exclude_gids=exclude_gids)
        if not cands:
            return None
        best_gid, best_dist = cands[0]
        second_dist = cands[1][1] if len(cands) > 1 else float("inf")
        if best_dist <= self.threshold and (second_dist - best_dist) >= self.reassoc_margin:
            return best_gid
        return None

    def update_cam_source(self, local_id: int, feat: np.ndarray, frame_idx: int,
                          exclude_gids: Optional[Set[int]] = None) -> int:
        """cam_idx=0 전용. local track → GID 부여 후 Global memory 갱신."""
        self.stores[0].update(local_id, feat, frame_idx)
        excluded = set(exclude_gids or [])
        current_gid = self.src_gid.get(local_id)

        # 기존 GID가 이번 프레임에서 이미 사용 중이면 그대로 쓰지 않고,
        # 현재 프레임 사용 GID를 제외한 후보 중 가장 가까운 GID를 다시 찾는다.
        if current_gid is not None and current_gid not in excluded:
            gid = current_gid
        else:
            gid = self._select_existing_gid_for_src(local_id, exclude_gids=excluded)
            if gid is None:
                gid = self._new_gid()
            if current_gid is not None and current_gid != gid:
                self._release_src_gid(local_id, current_gid)
            self.src_gid[local_id] = gid
            self.gid_owner_src[gid] = local_id
        self._update_gid_memory(gid, feat, frame_idx, 0)
        return gid

    # ── query cameras (cam_idx >= 1) ──────────────────────────────────────────
    def _decay_evidence(self, cam_idx: int, local_id: int):
        key = (cam_idx, local_id)
        remove = []
        for gid in list(self.q_evidence[key].keys()):
            self.q_evidence[key][gid] *= self.evidence_decay
            if self.q_evidence[key][gid] < 1e-4:
                remove.append(gid)
        for g in remove:
            self.q_evidence[key].pop(g, None)
            self.q_hits[key].pop(g, None)

    def _evidence_rank(self, cam_idx: int, local_id: int) -> List[Tuple[int, float]]:
        key = (cam_idx, local_id)
        items = [(g, s) for g, s in self.q_evidence[key].items() if g >= 0]
        return sorted(items, key=lambda x: x[1], reverse=True)

    def _owner_conf(self, cam_idx: int, local_id: int, gid: int) -> float:
        return float(self.q_evidence[(cam_idx, local_id)].get(gid, 0.0))

    def _release_q_gid(self, cam_idx: int, local_id: int, gid: Optional[int]):
        if gid is not None and gid >= 0:
            owner_key = (cam_idx, gid)
            if self.gid_owner_q.get(owner_key) == local_id:
                self.gid_owner_q.pop(owner_key, None)

    def _try_assign_q_gid(self, cam_idx: int, local_id: int, candidate_gid: int) -> bool:
        key = (cam_idx, local_id)
        current = self.q_current_gid.get(key)
        if current == candidate_gid:
            return True

        owner_key = (cam_idx, candidate_gid)
        owner_lid = self.gid_owner_q.get(owner_key)
        my_conf = self._owner_conf(cam_idx, local_id, candidate_gid)

        # 같은 카메라 안에서 candidate_gid를 이미 다른 LID가 쓰고 있으면,
        # evidence가 충분히 더 강할 때만 빼앗고 아니면 할당하지 않는다.
        if owner_lid is not None and owner_lid != local_id:
            owner_track_key = (cam_idx, owner_lid)
            owner_conf = self._owner_conf(cam_idx, owner_lid, candidate_gid)
            if my_conf <= owner_conf + self.switch_margin:
                return False

            # 기존 same-cam owner를 temp GID로 되돌려 중복 GID를 제거
            self.q_current_gid[owner_track_key] = self._temp_gid(cam_idx, owner_lid)
            self.gid_owner_q.pop(owner_key, None)

        if current is not None:
            self._release_q_gid(cam_idx, local_id, current)

        self.q_current_gid[key] = candidate_gid
        self.gid_owner_q[owner_key] = local_id
        return True

    def _update_q_assignment(self, cam_idx: int, local_id: int, latest_best_gid: Optional[int]):
        key = (cam_idx, local_id)
        ranked = self._evidence_rank(cam_idx, local_id)
        if not ranked:
            if key not in self.q_current_gid:
                self.q_current_gid[key] = self._temp_gid(cam_idx, local_id)
            return

        best_gid, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        hits = self.q_hits[key].get(best_gid, 0)

        if (best_score < self.min_evidence or
                hits < self.confirm_count or
                best_score - second_score < self.evidence_margin):
            if key not in self.q_current_gid:
                self.q_current_gid[key] = self._temp_gid(cam_idx, local_id)
            return

        current = self.q_current_gid.get(key)
        temp    = self.q_temp_gid.get(key)
        if current is None or current == temp or current < 0:
            self._try_assign_q_gid(cam_idx, local_id, best_gid)
            return

        if current != best_gid:
            cur_score = self._owner_conf(cam_idx, local_id, current)
            if latest_best_gid == best_gid and best_score >= cur_score + self.switch_margin:
                self._try_assign_q_gid(cam_idx, local_id, best_gid)
        else:
            self._try_assign_q_gid(cam_idx, local_id, best_gid)

    def update_cam_query(self, cam_idx: int, local_id: int,
                         feat: np.ndarray, frame_idx: int,
                         exclude_gids: Optional[Set[int]] = None) -> Tuple[int, Optional[float]]:
        """cam_idx >= 1 전용. evidence 누적 후 GID 이어받기."""
        assert cam_idx >= 1, "cam_idx=0은 update_cam_source를 사용하세요."
        key = (cam_idx, local_id)
        self.stores[cam_idx].update(local_id, feat, frame_idx)
        self._decay_evidence(cam_idx, local_id)
        excluded = set(exclude_gids or [])

        if self.stores[cam_idx].count(local_id) < self.min_features_to_match:
            gid = self.q_current_gid.get(key, self._temp_gid(cam_idx, local_id))
            self.q_current_gid[key] = gid
            return gid, None

        cands = self._rank_gid_candidates(self.stores[cam_idx], local_id, cam_idx, exclude_gids=excluded)
        if not cands:
            gid = self.q_current_gid.get(key, self._temp_gid(cam_idx, local_id))
            self.q_current_gid[key] = gid
            return gid, None

        best_gid, best_dist = cands[0]
        second_dist = cands[1][1] if len(cands) > 1 else float("inf")
        self.q_last_best[key] = best_dist

        if best_dist <= self.threshold and (second_dist - best_dist) >= self.distance_margin:
            sim = max(0.0, 1.0 - best_dist)
            self.q_evidence[key][best_gid] += sim
            self.q_hits[key][best_gid] += 1

        self._update_q_assignment(cam_idx, local_id, latest_best_gid=best_gid)
        gid = self.q_current_gid.get(key, self._temp_gid(cam_idx, local_id))

        # 현재 프레임에서 이미 같은 GID가 보이면, 이번 프레임에 사용된 GID를 제외한
        # 후보 중 가장 가까운 GID로 한 번 더 대체를 시도한다.
        if gid is not None and gid >= 0 and gid in excluded:
            alt_cands = self._rank_gid_candidates(self.stores[cam_idx], local_id, cam_idx, exclude_gids=excluded)
            if alt_cands:
                alt_gid, alt_dist = alt_cands[0]
                alt_second = alt_cands[1][1] if len(alt_cands) > 1 else float("inf")
                if alt_dist <= self.threshold and (alt_second - alt_dist) >= self.distance_margin:
                    sim = max(0.0, 1.0 - alt_dist)
                    self.q_evidence[key][alt_gid] += sim
                    self.q_hits[key][alt_gid] += 1
                    if self._try_assign_q_gid(cam_idx, local_id, alt_gid):
                        gid = alt_gid
                        best_dist = alt_dist
            if gid in excluded:
                self._release_q_gid(cam_idx, local_id, gid)
                gid = self._temp_gid(cam_idx, local_id)
                self.q_current_gid[key] = gid

        if gid is not None and gid >= 0:
            if (self.q_hits[key].get(gid, 0) >= self.confirm_count and
                    self.q_evidence[key].get(gid, 0.0) >= self.min_evidence):
                self._update_gid_memory(gid, feat, frame_idx, cam_idx)

        return gid, best_dist

    def get_gid(self, cam_idx: int, local_id: int) -> Optional[int]:
        if cam_idx == 0:
            return self.src_gid.get(local_id)
        key = (cam_idx, local_id)
        return self.q_current_gid.get(key, self.q_temp_gid.get(key))

    def expire(self, frame_idx: int):
        for store in self.stores:
            store.expire(frame_idx, self.max_age)
        self.store_gid.expire(frame_idx, self.memory_max_age)

        valid_src = set(self.stores[0].keys())
        for lid in list(self.src_gid.keys()):
            if lid not in valid_src:
                old_gid = self.src_gid.pop(lid, None)
                if old_gid is not None and self.gid_owner_src.get(old_gid) == lid:
                    self.gid_owner_src.pop(old_gid, None)

        for ci in range(1, self.num_cams):
            valid = set(self.stores[ci].keys())
            for lid in list(k[1] for k in list(self.q_current_gid.keys()) if k[0] == ci):
                if lid not in valid:
                    key = (ci, lid)
                    old_gid = self.q_current_gid.pop(key, None)
                    self.q_temp_gid.pop(key, None)
                    self.q_evidence.pop(key, None)
                    self.q_hits.pop(key, None)
                    self.q_last_best.pop(key, None)
                    if old_gid is not None and old_gid >= 0:
                        owner_key = (ci, old_gid)
                        if self.gid_owner_q.get(owner_key) == lid:
                            self.gid_owner_q.pop(owner_key, None)

        valid_gid = set(self.store_gid.keys())
        for gid in list(self.gid_owner_src.keys()):
            if gid not in valid_gid:
                self.gid_owner_src.pop(gid, None)
        for owner_key in list(self.gid_owner_q.keys()):
            ci, gid = owner_key
            lid = self.gid_owner_q[owner_key]
            if gid not in valid_gid or lid not in set(self.stores[ci].keys()):
                self.gid_owner_q.pop(owner_key, None)
        for gid in list(self.gid_last_seen.keys()):
            if gid not in valid_gid:
                self.gid_last_seen.pop(gid, None)
                self.gid_last_cam.pop(gid, None)


# ── crop quality / occlusion filtering ────────────────────────────────────────
def bbox_iou(box_a: Tuple[int,int,int,int], box_b: Tuple[int,int,int,int]) -> float:
    ax1,ay1,ax2,ay2 = box_a
    bx1,by1,bx2,by2 = box_b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    area_a = max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b = max(0,bx2-bx1)*max(0,by2-by1)
    union = area_a + area_b - inter
    return float(inter/union) if union > 0 else 0.0


def find_occluded_track_ids(dets, iou_threshold=OCCLUSION_IOU_THRESHOLD) -> set:
    occluded = set()
    n = len(dets)
    for i in range(n):
        lid_i,x1_i,y1_i,x2_i,y2_i,_ = dets[i]
        for j in range(i+1,n):
            lid_j,x1_j,y1_j,x2_j,y2_j,_ = dets[j]
            if bbox_iou((x1_i,y1_i,x2_i,y2_i),(x1_j,y1_j,x2_j,y2_j)) >= iou_threshold:
                occluded.add(int(lid_i)); occluded.add(int(lid_j))
    return occluded


def filter_small_detections(dets, min_area: int):
    if min_area <= 0:
        return dets
    return [d for d in dets if max(0,d[3]-d[1])*max(0,d[4]-d[2]) >= min_area]


def valid_crop(frame, x1, y1, x2, y2, conf,
               edge_margin_ratio=EDGE_MARGIN_RATIO,
               min_aspect=MIN_ASPECT_RATIO,
               max_aspect=MAX_ASPECT_RATIO) -> Optional[np.ndarray]:
    if conf < MIN_DET_CONF:
        return None
    h,w = frame.shape[:2]
    bw = max(1, x2-x1); bh = max(1, y2-y1)
    if bw*bh < MIN_BBOX_AREA:
        return None
    asp = bw/bh
    if asp < min_aspect or asp > max_aspect:
        return None
    mx,my = int(w*edge_margin_ratio), int(h*edge_margin_ratio)
    if x1<=mx or y1<=my or x2>=w-mx or y2>=h-my:
        return None
    px,py = int(bw*CROP_PADDING), int(bh*CROP_PADDING)
    x1 = max(0, x1-px); x2 = min(w-1, x2+px)
    y1 = max(0, y1-py); y2 = min(h-1, y2+py)
    if x2<=x1 or y2<=y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


# ── 시각화 ────────────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, gid, local_id, cam_label="", only_gid: bool = False):
    color = get_color(gid if gid is not None and gid >= 0 else 0)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

    gid_text = "?" if gid is None or int(gid) < 0 else str(gid)
    label = f"GID:{gid_text}" if only_gid else f"LID:{local_id} GID:{gid_text}"
    font, fs, th_ = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw,th2),_ = cv2.getTextSize(label, font, fs, th_)

    tx = int(max(0, min(x1, w-tw-7)))
    ty = y1 - 6
    if ty - th2 - 4 < 0:
        ty = min(h-4, y2+th2+8)
    ty = int(max(th2+4, min(ty, h-4)))

    cv2.rectangle(frame, (tx, ty-th2-4), (tx+tw+6, ty+2), color, -1)
    cv2.putText(frame, label, (tx+3,ty-2), font, fs, (255,255,255), th_, cv2.LINE_AA)


def draw_cached(frame, cached, orig_w, orig_h, only_gid: bool = False):
    sx = DISPLAY_W / max(orig_w, 1)
    sy = DISPLAY_H / max(orig_h, 1)
    for (lid, x1, y1, x2, y2, gid) in cached:
        draw_box(frame, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), gid, lid, only_gid=only_gid)


def resize_frame(frame, w, h):
    return cv2.resize(frame, (w, h))


def make_display(frames_vis: List[np.ndarray], cam_labels: List[str],
                 frame_idx: int, stride: int, is_feat_frame: bool,
                 gid_maps: List[Dict], is_initial: bool = False) -> np.ndarray:
    """
    최대 4개 카메라를 2x2 그리드로 배치.
    카메라 수에 맞게 빈 슬롯은 검정 패딩.
    """
    n = len(frames_vis)
    cols = 2
    rows = (n + cols - 1) // cols

    tag = "INIT" if is_initial and is_feat_frame else ("OSNet" if is_feat_frame else f"track({stride})")
    all_gids = set()
    for gm in gid_maps:
        all_gids |= set(gm.values())
    pos_gids = {g for g in all_gids if g >= 0}

    panels = []
    for i, (vis, label) in enumerate(zip(frames_vis, cam_labels)):
        f = vis.copy()
        cv2.putText(f, label, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)
        cv2.putText(f, f"frame {frame_idx} [{tag}]", (10, DISPLAY_H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        if i == 0:
            cv2.putText(f, f"GIDs: {len(pos_gids)}", (10, DISPLAY_H-28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,220,100), 1)
        panels.append(f)

    # 빈 슬롯 채우기
    blank = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    while len(panels) < rows * cols:
        panels.append(blank.copy())

    sep_v = np.zeros((DISPLAY_H, 4, 3), dtype=np.uint8)
    sep_v[:] = (80,80,80)
    sep_h = np.zeros((4, DISPLAY_W*cols + 4*(cols-1), 3), dtype=np.uint8)
    sep_h[:] = (80,80,80)

    rows_img = []
    for r in range(rows):
        row_panels = panels[r*cols:(r+1)*cols]
        row_img = row_panels[0]
        for p in row_panels[1:]:
            row_img = np.hstack([row_img, sep_v, p])
        rows_img.append(row_img)

    if len(rows_img) == 1:
        return rows_img[0]

    result = rows_img[0]
    for ri in rows_img[1:]:
        result = np.vstack([result, sep_h, ri])
    return result


# ── 디렉토리 유틸 ─────────────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4",".avi",".mov",".mkv",".wmv",".m4v"}


def collect_videos(directory: str) -> list:
    d = Path(directory)
    if not d.is_dir():
        sys.exit(f"[ERROR] 디렉토리가 없습니다: {directory}")
    vids = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        sys.exit(f"[ERROR] 영상 파일이 없습니다: {directory}")
    return vids


def resolve_save_path(save_path: Optional[str], video_paths: List[str]) -> Optional[str]:
    if not save_path:
        return None
    p = Path(save_path)
    stem = "_vs_".join(Path(v).stem for v in video_paths[:4])
    if p.suffix.lower() not in VIDEO_EXTS:
        p.mkdir(parents=True, exist_ok=True)
        return str(p / f"result_{stem}.mp4")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def output_size(num_cams: int):
    """디스플레이 출력 영상 크기 계산."""
    cols = 2
    rows = (num_cams + cols - 1) // cols
    w = DISPLAY_W * cols + 4 * (cols - 1)
    h = DISPLAY_H * rows + 4 * max(0, rows - 1)
    return w, h


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run(
    video_paths: List[str],
    save_path: str = None,
    stride: int = FRAME_STRIDE,
    yolo_model: str = YOLO_MODEL,
    tracker_name: str = "botsort.yaml",
    imgsz: int = YOLO_IMGSZ,
    yolo_conf: float = YOLO_CONF,
    yolo_iou: float = YOLO_IOU,
    yolo_augment: bool = YOLO_AUGMENT,
    topk: int = TOPK_PAIRWISE,
    mean_weight: float = MEAN_SCORE_WEIGHT,
    set_weight: float = SET_SCORE_WEIGHT,
    confirm_count: int = MATCH_CONFIRM_COUNT,
    evidence_decay: float = EVIDENCE_DECAY,
    min_evidence: float = MIN_EVIDENCE,
    evidence_margin: float = EVIDENCE_MARGIN,
    switch_margin: float = SWITCH_MARGIN,
    distance_margin: float = DISTANCE_MARGIN,
    skip_occluded_update: bool = SKIP_OCCLUDED_UPDATE,
    occ_iou: float = OCCLUSION_IOU_THRESHOLD,
    edge_margin: float = EDGE_MARGIN_RATIO,
    min_aspect: float = MIN_ASPECT_RATIO,
    max_aspect: float = MAX_ASPECT_RATIO,
    memory_max_age: int = GLOBAL_MEMORY_MAX_AGE,
    reassoc_margin: float = REASSOC_DISTANCE_MARGIN,
    min_det_area: int = MIN_DET_BBOX_AREA,
    initial_frames: int = INITIAL_FRAMES,
    initial_threshold: float = INITIAL_MATCH_THRESHOLD,
    initial_distance_margin: float = INITIAL_DISTANCE_MARGIN,
    initial_min_features: int = INITIAL_MIN_FEATURES_TO_MATCH,
    show: bool = False,
    only_gid: bool = False,
    avoid_current_frame_dup_gid: bool = AVOID_CURRENT_FRAME_DUP_GID,
):
    num_cams = len(video_paths)
    assert 2 <= num_cams <= 4, "카메라는 2~4개이어야 합니다."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 카메라 수    : {num_cams}")
    print(f"[INFO] Device       : {device}")
    print(f"[INFO] Stride       : {stride}")
    print(f"[INFO] Initial      : {max(0, int(initial_frames))} frame(s) all-camera matching")
    print(f"[INFO] Threshold    : {MATCH_THRESHOLD}")
    print(f"[INFO] Init strict  : threshold={initial_threshold}, margin={initial_distance_margin}, min_features={initial_min_features}")
    for i, vp in enumerate(video_paths):
        print(f"[INFO] Camera {i}     : {vp}")

    print("[INFO] Loading YOLO + Tracker ...")
    trackers = [
        Tracker(yolo_model, imgsz=imgsz, conf=yolo_conf, iou=yolo_iou,
                augment=yolo_augment, tracker_name=tracker_name)
        for _ in range(num_cams)
    ]

    print("[INFO] Loading OSNet ...")
    extractor = OSNetExtractor(OSNET_MODEL, OSNET_WEIGHTS, device)

    gallery = Gallery(
        num_cams=num_cams,
        threshold=MATCH_THRESHOLD,
        max_age=GALLERY_MAX_AGE,
        topk=topk,
        mean_weight=mean_weight,
        set_weight=set_weight,
        confirm_count=confirm_count,
        evidence_decay=evidence_decay,
        min_evidence=min_evidence,
        evidence_margin=evidence_margin,
        switch_margin=switch_margin,
        distance_margin=distance_margin,
        memory_max_age=memory_max_age,
        reassoc_margin=reassoc_margin,
        initial_threshold=initial_threshold,
        initial_distance_margin=initial_distance_margin,
        initial_min_features_to_match=initial_min_features,
    )

    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open: {video_paths[i]}")

    orig_sizes = [
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for cap in caps
    ]

    save_path = resolve_save_path(save_path, video_paths)
    writer = None
    if save_path:
        out_w, out_h = output_size(num_cams)
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 20.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, float(fps), (out_w, out_h))
        print(f"[INFO] 저장 경로    : {save_path}  ({out_w}x{out_h})")

    cam_labels = [f"Camera {i}" for i in range(num_cams)]
    frame_idx  = 0
    paused     = False
    display    = None

    if show:
        print("[INFO] 시작 — Q: 종료  /  P·Space: 일시정지")
    else:
        print("[INFO] Headless 모드로 시작합니다.")

    while True:
        if not paused:
            frames = []
            for cap in caps:
                ret, frm = cap.read()
                if not ret:
                    print("[INFO] 영상 종료.")
                    goto_end = True
                    break
                frames.append(frm)
            else:
                goto_end = False

            if goto_end:
                break

            is_feat = (frame_idx % max(stride, 1) == 0)
            in_initial = (max(0, int(initial_frames)) > 0 and frame_idx < int(initial_frames))
            vis_frames = [resize_frame(f, DISPLAY_W, DISPLAY_H) for f in frames]
            caches   = [[] for _ in range(num_cams)]
            gid_maps = [{} for _ in range(num_cams)]
            # 이번 frame에서 이미 표시/할당된 positive GID.
            # 매칭 후보에서 이 GID들을 제외해서 같은 frame 중복 GID를 줄인다.
            used_gids_this_frame: Set[int] = set()

            for ci in range(num_cams):
                dets = trackers[ci].update(frames[ci])
                dets = filter_small_detections(dets, min_det_area)
                occluded = find_occluded_track_ids(dets, occ_iou) if skip_occluded_update else set()

                for (lid, x1, y1, x2, y2, conf) in dets:
                    gid = None
                    if is_feat and lid not in occluded:
                        crop = valid_crop(frames[ci], x1, y1, x2, y2, conf,
                                          edge_margin, min_aspect, max_aspect)
                        if crop is not None:
                            feat = extractor.extract(crop)
                            if feat is not None:
                                exclude_gids = used_gids_this_frame if avoid_current_frame_dup_gid else None
                                if in_initial:
                                    gid, _ = gallery.update_initial(ci, lid, feat, frame_idx, exclude_gids=exclude_gids)
                                elif ci == 0:
                                    gid = gallery.update_cam_source(lid, feat, frame_idx, exclude_gids=exclude_gids)
                                else:
                                    gid, _ = gallery.update_cam_query(ci, lid, feat, frame_idx, exclude_gids=exclude_gids)

                    if gid is None:
                        gid = gallery.get_gid(ci, lid)

                    # feature를 뽑지 않는 frame에서도 이미 같은 GID가 앞에서 표시됐다면
                    # 중복 표시를 피하기 위해 임시 ID(?)로 둔다. 실제 Gallery memory는 건드리지 않는다.
                    if (avoid_current_frame_dup_gid and gid is not None and gid >= 0
                            and gid in used_gids_this_frame):
                        gid = gallery._temp_gid(ci, lid) if ci > 0 else None

                    if gid is None:
                        continue
                    gid_maps[ci][lid] = gid
                    caches[ci].append((lid, x1, y1, x2, y2, gid))
                    if gid >= 0:
                        used_gids_this_frame.add(gid)

            if is_feat:
                gallery.expire(frame_idx)

            for ci in range(num_cams):
                draw_cached(vis_frames[ci], caches[ci], orig_sizes[ci][0], orig_sizes[ci][1], only_gid=only_gid)

            display = make_display(vis_frames, cam_labels, frame_idx, stride, is_feat, gid_maps, is_initial=in_initial)

            if writer:
                writer.write(display)

            frame_idx += 1

        if show and display is not None:
            cv2.imshow("Cross-Camera ReID (4-cam) | Q:quit  P:pause", display)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord("q"):
                print("[INFO] 사용자 종료.")
                break
            elif key in (ord("p"), 32):
                paused = not paused
                print("[INFO] " + ("일시정지" if paused else "재개"))

    for cap in caps:
        cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[INFO] 완료. 처리된 프레임 수: {frame_idx}")


# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Cross-Camera ReID Pipeline (2~4개 카메라)\n"
            "YOLO11 + BoT-SORT + OSNet + feature-set top-k + evidence reassignment\n\n"
            "비디오 경로는 args로 받지 않고, 코드 상단 VIDEO_SETS에서 지정합니다.\n"
            "VIDEO_SET_INDEX = 0 이면 VIDEO_SETS[0] 실행, 1이면 VIDEO_SETS[1] 실행."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── 출력 ──
    parser.add_argument("--save", default="./results", help="결과 영상 저장 경로 또는 폴더")
    parser.add_argument("--show", action="store_true", help="OpenCV 창 표시 (GUI 환경 필요)")
    parser.add_argument("--only_gid", action="store_true", help="박스 라벨에 LID 없이 GID만 표시합니다. 예: GID:3")
    parser.add_argument("--allow_current_frame_duplicate_gid", action="store_true",
                        help="현재 프레임에서 이미 사용된 GID도 후보로 허용합니다. 기본값은 중복 방지입니다.")
    parser.add_argument("--initial_frames", type=int, default=INITIAL_FRAMES,
                        help="초기 K frame 동안 cam0 기준 없이 모든 카메라를 같이 보며 GID를 먼저 부여합니다. 0이면 비활성화.")
    parser.add_argument("--initial_threshold", type=float, default=INITIAL_MATCH_THRESHOLD,
                        help="initial 단계 전용 거리 threshold. 낮을수록 더 엄격합니다. 예: 0.35")
    parser.add_argument("--initial_distance_margin", type=float, default=INITIAL_DISTANCE_MARGIN,
                        help="initial 단계 전용 1등-2등 후보 거리 차이 조건. 클수록 더 엄격합니다. 예: 0.08")
    parser.add_argument("--initial_min_features", type=int, default=INITIAL_MIN_FEATURES_TO_MATCH,
                        help="initial 단계에서 매칭에 사용하기 전 필요한 최소 feature 개수. 클수록 더 엄격하지만 느립니다.")

    # ── YOLO ──
    parser.add_argument("--yolo_model",  default=YOLO_MODEL,  help=f"YOLO 모델 (기본: {YOLO_MODEL})")
    parser.add_argument("--tracker",     default="botsort.yaml")
    parser.add_argument("--imgsz",       type=int,   default=YOLO_IMGSZ)
    parser.add_argument("--yolo_conf",   type=float, default=YOLO_CONF)
    parser.add_argument("--yolo_iou",    type=float, default=YOLO_IOU)
    parser.add_argument("--yolo_augment",action="store_true")
    parser.add_argument("--min_det_area",type=int,   default=MIN_DET_BBOX_AREA)

    # ── ReID ──
    parser.add_argument("--stride",    type=int,   default=FRAME_STRIDE)
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--confirm",   type=int,   default=MATCH_CONFIRM_COUNT)
    parser.add_argument("--topk",      type=int,   default=TOPK_PAIRWISE)
    parser.add_argument("--mean_weight",type=float,default=MEAN_SCORE_WEIGHT)
    parser.add_argument("--set_weight", type=float,default=SET_SCORE_WEIGHT)

    # ── Evidence ──
    parser.add_argument("--evidence_decay",  type=float, default=EVIDENCE_DECAY)
    parser.add_argument("--min_evidence",    type=float, default=MIN_EVIDENCE)
    parser.add_argument("--evidence_margin", type=float, default=EVIDENCE_MARGIN)
    parser.add_argument("--switch_margin",   type=float, default=SWITCH_MARGIN)
    parser.add_argument("--distance_margin", type=float, default=DISTANCE_MARGIN)

    # ── Occlusion / crop quality ──
    parser.add_argument("--no_skip_occluded", action="store_true")
    parser.add_argument("--occ_iou",    type=float, default=OCCLUSION_IOU_THRESHOLD)
    parser.add_argument("--edge_margin",type=float, default=EDGE_MARGIN_RATIO)
    parser.add_argument("--min_aspect", type=float, default=MIN_ASPECT_RATIO)
    parser.add_argument("--max_aspect", type=float, default=MAX_ASPECT_RATIO)

    # ── GID archive ──
    parser.add_argument("--memory_max_age", type=int,   default=GLOBAL_MEMORY_MAX_AGE)
    parser.add_argument("--reassoc_margin", type=float, default=REASSOC_DISTANCE_MARGIN)

    args = parser.parse_args()

    # threshold 전역 반영
    MATCH_THRESHOLD = args.threshold

    if not VIDEO_SETS:
        sys.exit("[ERROR] VIDEO_SETS가 비어 있습니다. 코드 상단 VIDEO_SETS에 비디오 경로를 넣어주세요.")

    if RUN_ALL_VIDEO_SETS:
        selected_video_sets = list(enumerate(VIDEO_SETS))
    else:
        if not (0 <= VIDEO_SET_INDEX < len(VIDEO_SETS)):
            sys.exit(f"[ERROR] VIDEO_SET_INDEX={VIDEO_SET_INDEX} 범위 오류. VIDEO_SETS 길이: {len(VIDEO_SETS)}")
        selected_video_sets = [(VIDEO_SET_INDEX, VIDEO_SETS[VIDEO_SET_INDEX])]

    for set_idx, video_paths in selected_video_sets:
        if not (2 <= len(video_paths) <= 4):
            sys.exit(f"[ERROR] VIDEO_SETS[{set_idx}]는 2~4개 비디오 경로를 가져야 합니다. 현재: {len(video_paths)}개")
        for vp in video_paths:
            if not Path(vp).is_file():
                sys.exit(f"[ERROR] 파일을 찾을 수 없습니다: {vp}")

        print(f"\n{'='*60}")
        print(f"[VIDEO_SET {set_idx}] " + "  ←→  ".join(Path(v).name for v in video_paths))
        print(f"{'='*60}")

        save_path = args.save
        if RUN_ALL_VIDEO_SETS:
            # 여러 세트를 처리할 때는 저장 경로가 파일명이어도 덮어쓰지 않도록 폴더처럼 사용합니다.
            save_root = Path(args.save)
            if save_root.suffix.lower() in VIDEO_EXTS:
                save_root = save_root.parent / save_root.stem
            save_root.mkdir(parents=True, exist_ok=True)
            stem = "_vs_".join(Path(v).stem for v in video_paths[:4])
            save_path = str(save_root / f"result_set{set_idx:02d}_{stem}.mp4")

        run(
            video_paths=video_paths,
            save_path=save_path,
            stride=args.stride,
            yolo_model=args.yolo_model,
            tracker_name=args.tracker,
            imgsz=args.imgsz,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_augment=args.yolo_augment,
            topk=args.topk,
            mean_weight=args.mean_weight,
            set_weight=args.set_weight,
            confirm_count=args.confirm,
            evidence_decay=args.evidence_decay,
            min_evidence=args.min_evidence,
            evidence_margin=args.evidence_margin,
            switch_margin=args.switch_margin,
            distance_margin=args.distance_margin,
            skip_occluded_update=not args.no_skip_occluded,
            occ_iou=args.occ_iou,
            edge_margin=args.edge_margin,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            memory_max_age=args.memory_max_age,
            reassoc_margin=args.reassoc_margin,
            min_det_area=args.min_det_area,
            initial_frames=args.initial_frames,
            initial_threshold=args.initial_threshold,
            initial_distance_margin=args.initial_distance_margin,
            initial_min_features=args.initial_min_features,
            show=args.show,
            only_gid=args.only_gid,
            avoid_current_frame_dup_gid=not args.allow_current_frame_duplicate_gid,
        )

    if RUN_ALL_VIDEO_SETS:
        print(f"\n[INFO] 전체 {len(selected_video_sets)}개 VIDEO_SETS 처리 완료.")
