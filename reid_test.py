"""
Cross-Camera Re-Identification Pipeline
========================================

YOLO11 detect-boost + BoT-SORT tracking + OSNet feature set matching + evidence reassignment + Global ID archive

핵심 변경점
- 기존 CLIP feature extractor를 OSNet(torchreid) feature extractor로 교체
- YOLO11/ByteTrack은 매 프레임 실행해서 bbox 표시를 부드럽게 유지
- OSNet feature extraction은 FRAME_STRIDE 간격으로만 수행
- 같은 camera/local track ID의 feature를 buffer에 누적
- 평균 feature cosine similarity와 feature-set top-k similarity를 함께 사용
- Camera B matching은 hard lock하지 않고 evidence score를 누적해 현재 best GID를 결정
- 초반 오판이 있어도 이후 더 강한 evidence가 쌓이면 GID 재배정 가능
- best/second candidate margin을 사용해 애매한 매칭의 조기 확정을 줄임
- 사람 bbox가 서로 겹치거나 crop 품질이 낮은 경우 feature/evidence 업데이트를 보류해 feature 오염을 줄임
- 새 local ID가 생겨도 먼저 Global ID archive와 비교해 기존 GID를 이어받을 수 있음
- 출력 형태는 기존과 동일하게 Camera A/B 나란히 표시 + LID/GID bbox 시각화
- bbox label에서 [A]/[B] 표기를 제거하고, 너무 작은 원거리 detection은 표시/매칭에서 제외 가능
- v10: v9 local stitching은 제거하고, 성능이 가장 안정적이었던 v7 archive 구조로 복귀
- v10: YOLO11 + BoT-SORT는 유지하되, 너무 작은 사람 과검출을 줄이기 위해 detection 설정을 균형형으로 조정

설치 예시
    pip install -U ultralytics opencv-python torch torchvision scipy numpy Pillow
    pip install torchreid

실행 예시
    python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4
    python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4 --save output.mp4
    python cross_camera_reid.py --video_a camA.mp4 --video_b camB.mp4 --yolo_model yolo11m.pt --stride 10 --threshold 0.50
"""

import argparse
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    import lpips
except ImportError:
    lpips = None
from scipy.spatial.distance import cosine
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────────────────
YOLO_MODEL = "yolo26n.pt"  # v10 안정형 기본값. 더 강하게는 yolo11l.pt / yolo11x.pt
OSNET_MODEL = "osnet_x1_0"
OSNET_WEIGHTS = "market1501"  # Re-ID pretrained weights
OSNET_HF_REPO = "MYerassyl/retail-heat-osnet"
OSNET_HF_FILENAME = "osnet_x1_0_market1501.pth"
PERSON_CLASS = 0

# OSNet feature는 cosine similarity가 높을수록 유사함.
# 기존 코드는 cosine distance threshold였지만, CLI 의미를 유지하기 위해 threshold는 distance 기준으로 사용.
# 예: distance 0.35 이하면 같은 사람 후보로 봄 = similarity 0.65 이상.
MATCH_THRESHOLD = 0.20  # hybrid distance 기준 시작값. 필요하면 --threshold로 조정
GALLERY_MAX_AGE = 300
DISPLAY_W = 640
DISPLAY_H = 480
FRAME_STRIDE = 20

# feature aggregation / matching 안정화 설정
MAX_FEATURES_PER_TRACK = 15       # local track별 최근 feature 최대 저장 개수
MIN_FEATURES_TO_MATCH = 1         # 짧은 track도 matching 후보로 사용
MIN_DET_CONF = 0.30               # OSNet feature에 사용할 crop confidence 필터
MIN_BBOX_AREA = 700               # OSNet feature update에 사용할 최소 bbox area
MIN_DET_BBOX_AREA = 800           # v8: 겹친 뒤 사람/부분 가림 bbox를 살리기 위해 완화
MATCH_CONFIRM_COUNT = 3           # 같은 후보가 N번 이상 관측되어야 current GID 후보로 인정
EVIDENCE_DECAY = 0.90             # feature frame마다 기존 evidence를 감쇠해 오래된 오판 영향 감소
MIN_EVIDENCE = 1.00               # current GID로 표시하기 위한 최소 evidence
EVIDENCE_MARGIN = 0.25            # 1등 evidence와 2등 evidence의 최소 차이
SWITCH_MARGIN = 0.50              # 이미 배정된 GID를 바꾸기 위해 필요한 추가 evidence 차이
DISTANCE_MARGIN = 0.03            # best distance와 second-best distance의 최소 차이
YOLO_IMGSZ = 1280                 # 작은 사람/겹침 상황 개선: 기본 640보다 크게 추론
YOLO_CONF = 0.18                  # v10: 너무 작은 과검출을 줄이면서 부분 가림도 일부 허용
YOLO_IOU = 0.85                   # v8: NMS IoU를 높여 겹친 사람 bbox가 제거되는 것을 줄임
YOLO_AUGMENT = False              # TTA/augment 추론. 켜면 느리지만 일부 검출이 늘 수 있음
CROP_PADDING = 0.15                # Re-ID crop 주변 context padding
TOPK_PAIRWISE = 5                  # feature-set 비교 시 가장 유사한 feature pair 상위 k개 평균
MEAN_SCORE_WEIGHT = 0.50           # 평균 feature similarity 반영 비율
SET_SCORE_WEIGHT = 0.50            # feature-set top-k similarity 반영 비율

# OSNet + LPIPS hybrid distance 설정
USE_LPIPS = True
LPIPS_NET = "alex"                 # "alex": 빠름, "vgg": 무거움, "squeeze": 가벼움
LPIPS_IMAGE_W = 128
LPIPS_IMAGE_H = 256
OSNET_DISTANCE_WEIGHT = 0.70       # ReID feature 중심
LPIPS_DISTANCE_WEIGHT = 0.30       # 이미지 질감/색상 유사도 보조
LPIPS_DISTANCE_SCALE = 1.0         # LPIPS 값 보정. 필요하면 0.5~2.0 조정
DEBUG_MATCH = False                # --debug일 때 bbox label에 score 표시
DEBUG_SCORE_MODE = "final"         # final: D만 표시, full: D/O/L 표시

# occlusion-aware / crop quality update 설정
SKIP_OCCLUDED_UPDATE = True              # bbox가 다른 사람 bbox와 많이 겹치면 feature/evidence update 보류
OCCLUSION_IOU_THRESHOLD = 0.10           # 같은 프레임 내 person bbox IoU가 이 값 이상이면 occlusion으로 간주
EDGE_MARGIN_RATIO = 0.02                 # 화면 가장자리 근처에서 잘린 crop update 보류 비율
MIN_ASPECT_RATIO = 0.20                  # bbox width/height 최소 비율. 너무 얇으면 품질 낮은 crop으로 간주
MAX_ASPECT_RATIO = 1.20                  # bbox width/height 최대 비율. 너무 넓으면 여러 사람이 섞였을 가능성
GLOBAL_MEMORY_MAX_AGE = 900             # 사라진 GID를 archive에 유지할 프레임 수
REASSOC_DISTANCE_MARGIN = 0.03          # 새 local track을 기존 GID와 재연결할 때 best/second distance 차이 조건

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
PALETTE = [
    (220, 80, 80), (80, 180, 80), (80, 120, 220),
    (200, 160, 50), (160, 80, 200), (50, 190, 190),
    (230, 120, 50), (100, 200, 150), (180, 80, 150),
    (80, 160, 230), (210, 210, 80), (130, 100, 230),
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
    """Torchreid OSNet으로 person crop → L2-normalized Re-ID feature."""

    def __init__(self, model_name: str = OSNET_MODEL, weights: str = OSNET_WEIGHTS, device: str = "cpu"):
        self.device = torch.device(device)

        try:
            import torchreid
            from torchreid.utils import FeatureExtractor
        except ImportError as e:
            raise ImportError(
                "torchreid가 설치되어 있지 않습니다.\n"
                "아래 명령을 먼저 실행해 주세요:\n"
                "  pip install torchreid\n"
                "만약 설치가 실패하면:\n"
                "  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git\n"
            ) from e

        # FeatureExtractor는 내부에서 resize / normalize / forward를 처리한다.
        # 중요: model_path를 비워두면 ImageNet pretrained만 로드될 수 있어서
        # person Re-ID에는 약하다. 가능하면 Market1501 Re-ID weight를 사용한다.
        model_path = ""
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id=OSNET_HF_REPO,
                filename=OSNET_HF_FILENAME,
            )
            print(f"[INFO] OSNet Re-ID weights : {model_path}")
        except Exception as e:
            print("[WARN] Could not download OSNet Market1501 weights.")
            print("[WARN] Falling back to torchreid default weights. Accuracy may be lower.")
            print(f"[WARN] Reason: {e}")

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

        # torchreid FeatureExtractor는 RGB ndarray/list 입력을 받을 수 있다.
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        feat = self.extractor([rgb])

        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat)
        if feat.ndim == 2:
            feat = feat[0]

        feat = feat.astype(np.float32)
        return l2_normalize(feat)



class LPIPSComparator:
    """person crop 이미지끼리 LPIPS distance 계산. 낮을수록 유사함."""

    def __init__(
        self,
        device: str = "cpu",
        net: str = "alex",
        image_size: Tuple[int, int] = (128, 256),  # (w, h)
        scale: float = 1.0,
    ):
        if lpips is None:
            raise ImportError(
                "LPIPS를 사용하려면 먼저 설치해야 합니다.\n"
                "  pip install lpips\n"
                "또는 --no_lpips 옵션으로 LPIPS를 끄세요."
            )
        self.device = torch.device(device)
        self.image_size = image_size
        self.scale = float(scale)
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.loss_fn.eval()

    def _preprocess(self, bgr_crop: np.ndarray) -> Optional[torch.Tensor]:
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        crop = cv2.resize(bgr_crop, self.image_size)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        x = x * 2.0 - 1.0                   # LPIPS input range: [-1, 1]
        return x.to(self.device)

    @torch.no_grad()
    def distance(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        xa = self._preprocess(crop_a)
        xb = self._preprocess(crop_b)
        if xa is None or xb is None:
            return float("inf")
        d = self.loss_fn(xa, xb)
        return float(d.item()) * self.scale

    @torch.no_grad()
    def set_distance(
        self,
        crops_a: List[np.ndarray],
        crops_b: List[np.ndarray],
        topk: int = 5,
    ) -> Optional[float]:
        """crop set 간 LPIPS distance. 작은 distance 상위 k개 평균."""
        if not crops_a or not crops_b:
            return None
        dists = []
        for ca in crops_a:
            for cb in crops_b:
                dists.append(self.distance(ca, cb))
        if not dists:
            return None
        dists = np.asarray(dists, dtype=np.float32)
        k = max(1, min(int(topk), len(dists)))
        best_vals = np.partition(dists, k - 1)[:k]
        return float(np.mean(best_vals))


# ── ByteTrack wrapper ─────────────────────────────────────────────────────────
class Tracker:
    """Ultralytics YOLO의 내장 tracker를 프레임 단위로 사용.

    기본은 BoT-SORT로 설정했다. ByteTrack보다 약간 느릴 수 있지만,
    occlusion/ID switch 상황에서 더 안정적인 경우가 있다.
    v8에서는 겹침/부분 가림 상황을 위해 YOLO NMS IoU와 augment 옵션도 받을 수 있게 했다.
    """

    def __init__(self, model_path: str, imgsz: int = YOLO_IMGSZ, conf: float = YOLO_CONF, iou: float = YOLO_IOU, augment: bool = YOLO_AUGMENT, tracker_name: str = "botsort.yaml"):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.augment = augment
        self.tracker_name = tracker_name

    def update(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, int, float]]:
        """
        Returns:
            list of (track_id, x1, y1, x2, y2, conf)
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_name,
            classes=[PERSON_CLASS],
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            augment=self.augment,
            verbose=False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if boxes.id is None:
            return detections

        ids = boxes.id.int().tolist()
        xyxy = boxes.xyxy.int().tolist()
        confs = boxes.conf.float().tolist() if boxes.conf is not None else [1.0] * len(ids)

        for tid, box, conf in zip(ids, xyxy, confs):
            x1, y1, x2, y2 = box
            detections.append((int(tid), int(x1), int(y1), int(x2), int(y2), float(conf)))
        return detections


# ── Feature buffer ────────────────────────────────────────────────────────────
class TrackFeatureStore:
    """camera/local_id별 feature buffer를 유지하고 평균 feature를 반환."""

    def __init__(self, max_features: int = MAX_FEATURES_PER_TRACK):
        self.max_features = max_features
        self.buffers: Dict[Union[int, str], Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=max_features))
        self.last_seen: Dict[Union[int, str], int] = {}

    def update(self, key: Union[int, str], feat: np.ndarray, frame_idx: int):
        self.buffers[key].append(l2_normalize(feat.astype(np.float32)))
        self.last_seen[key] = frame_idx

    def get_mean(self, key: Union[int, str]) -> Optional[np.ndarray]:
        buf = self.buffers.get(key)
        if not buf:
            return None
        mean_feat = np.mean(np.stack(list(buf), axis=0), axis=0)
        return l2_normalize(mean_feat.astype(np.float32))

    def get_features(self, key: Union[int, str]) -> List[np.ndarray]:
        buf = self.buffers.get(key)
        if not buf:
            return []
        return list(buf)

    def count(self, key: Union[int, str]) -> int:
        buf = self.buffers.get(key)
        return len(buf) if buf else 0

    def keys(self):
        return list(self.buffers.keys())

    def expire(self, frame_idx: int, max_age: int):
        expired = [k for k, last in self.last_seen.items() if frame_idx - last > max_age]
        for k in expired:
            self.last_seen.pop(k, None)
            self.buffers.pop(k, None)



class TrackCropStore:
    """camera/local_id 또는 GID별 crop image buffer를 유지."""

    def __init__(self, max_crops: int = MAX_FEATURES_PER_TRACK):
        self.max_crops = max_crops
        self.buffers: Dict[Union[int, str], Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=max_crops))
        self.last_seen: Dict[Union[int, str], int] = {}

    def update(self, key: Union[int, str], crop: np.ndarray, frame_idx: int):
        if crop is None or crop.size == 0:
            return
        self.buffers[key].append(crop.copy())
        self.last_seen[key] = frame_idx

    def get_crops(self, key: Union[int, str]) -> List[np.ndarray]:
        buf = self.buffers.get(key)
        if not buf:
            return []
        return list(buf)

    def count(self, key: Union[int, str]) -> int:
        buf = self.buffers.get(key)
        return len(buf) if buf else 0

    def keys(self):
        return list(self.buffers.keys())

    def expire(self, frame_idx: int, max_age: int):
        expired = [k for k, last in self.last_seen.items() if frame_idx - last > max_age]
        for k in expired:
            self.last_seen.pop(k, None)
            self.buffers.pop(k, None)


# ── Gallery / Global ID manager ───────────────────────────────────────────────
def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    """L2-normalized vector 기준 cosine similarity."""
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))


def pairwise_topk_similarity(feats_a: List[np.ndarray], feats_b: List[np.ndarray], topk: int = TOPK_PAIRWISE) -> Optional[float]:
    """
    Tracklet feature set 간 similarity.

    평균 feature 하나로만 비교하면 앞모습/뒷모습/옆모습 정보가 하나의 벡터로 뭉개질 수 있다.
    그래서 A tracklet의 개별 feature들과 B tracklet의 개별 feature들을 모두 비교한 뒤,
    가장 유사한 pair 상위 k개의 similarity 평균을 사용한다.
    """
    if not feats_a or not feats_b:
        return None

    A = np.stack([l2_normalize(f.astype(np.float32)) for f in feats_a], axis=0)
    B = np.stack([l2_normalize(f.astype(np.float32)) for f in feats_b], axis=0)
    sims = A @ B.T  # normalized feature이므로 dot product = cosine similarity
    flat = sims.reshape(-1)
    if flat.size == 0:
        return None

    k = max(1, min(int(topk), flat.size))
    top_vals = np.partition(flat, -k)[-k:]
    return float(np.mean(top_vals))


class Gallery:
    """
    Camera A/B local track을 Global ID memory에 연결한다.

    v7 핵심:
    - v6의 OSNet + mean/top-k matching + evidence 재평가 + occlusion-aware update 유지
    - local track이 끊겼다가 새 local ID로 다시 잡혀도 바로 새 GID를 만들지 않음
    - 모든 Global ID의 feature memory를 별도로 저장해 두고, 새 local track은 먼저 기존 GID memory와 비교
    - Camera A에서 사라진 GID도 archive에 남겨 Cam B에서 다시 등장했을 때 이어 받을 수 있음
    - B track은 더 이상 임시 상태에서 양수 새 GID를 남발하지 않고, 매칭 전에는 TMP(-)로 표시
    """

    def __init__(
        self,
        threshold: float,
        max_age: int,
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
        memory_max_age: int = 900,
        reassoc_margin: float = 0.02,
        use_lpips: bool = USE_LPIPS,
        lpips_net: str = LPIPS_NET,
        osnet_weight: float = OSNET_DISTANCE_WEIGHT,
        lpips_weight: float = LPIPS_DISTANCE_WEIGHT,
        lpips_scale: float = LPIPS_DISTANCE_SCALE,
        debug: bool = DEBUG_MATCH,
        debug_score_mode: str = DEBUG_SCORE_MODE,
    ):
        self.threshold = threshold
        self.max_age = max_age
        self.memory_max_age = int(memory_max_age)
        self.reassoc_margin = float(reassoc_margin)
        self.min_features_to_match = min_features_to_match
        self.confirm_count = confirm_count
        self.topk = topk
        weight_sum = max(1e-6, mean_weight + set_weight)
        self.mean_weight = mean_weight / weight_sum
        self.set_weight = set_weight / weight_sum

        self.evidence_decay = float(evidence_decay)
        self.min_evidence = float(min_evidence)
        self.evidence_margin = float(evidence_margin)
        self.switch_margin = float(switch_margin)
        self.distance_margin = float(distance_margin)

        # local track별 feature buffer
        self.store_a = TrackFeatureStore(max_features=max_features)
        self.store_b = TrackFeatureStore(max_features=max_features)

        # Global ID별 feature archive. local track이 사라져도 이 memory는 일정 시간 유지된다.
        self.store_gid = TrackFeatureStore(max_features=max_features * 2)

        # LPIPS용 crop memory. OSNet feature memory와 병행해서 유지한다.
        self.crop_store_a = TrackCropStore(max_crops=max_features)
        self.crop_store_b = TrackCropStore(max_crops=max_features)
        self.crop_store_gid = TrackCropStore(max_crops=max_features * 2)

        self.use_lpips = bool(use_lpips)
        hybrid_sum = max(1e-6, float(osnet_weight) + float(lpips_weight))
        self.osnet_distance_weight = float(osnet_weight) / hybrid_sum
        self.lpips_distance_weight = float(lpips_weight) / hybrid_sum
        self.debug = bool(debug)
        self.debug_score_mode = str(debug_score_mode)

        self.lpips_metric = None
        if self.use_lpips:
            lpips_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.lpips_metric = LPIPSComparator(
                device=lpips_device,
                net=lpips_net,
                image_size=(LPIPS_IMAGE_W, LPIPS_IMAGE_H),
                scale=lpips_scale,
            )
            print(
                f"[INFO] Hybrid distance : OSNet={self.osnet_distance_weight:.2f}, "
                f"LPIPS={self.lpips_distance_weight:.2f}, net={lpips_net}, scale={lpips_scale}"
            )
        else:
            print("[INFO] Hybrid distance : LPIPS OFF, OSNet only")

        self.gid_last_seen: Dict[int, int] = {}
        self.gid_last_cam: Dict[int, str] = {}

        self.a_gid: Dict[int, int] = {}
        self.b_current_gid: Dict[int, int] = {}   # 현재 evidence상 가장 그럴듯한 global GID 또는 TMP 음수 ID
        self.b_temp_gid: Dict[int, int] = {}      # 아직 global GID와 연결되지 않은 B track 표시용 TMP 음수 ID
        self.b_evidence: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.b_hits: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.b_last_best_dist: Dict[int, Optional[float]] = {}
        self.last_score_info_a: Dict[int, Optional[Dict[str, Optional[float]]]] = {}
        self.last_score_info_b: Dict[int, Optional[Dict[str, Optional[float]]]] = {}

        # 같은 카메라 안에서 서로 다른 local ID가 같은 Global ID를 동시에 사용하는 문제 방지
        self.gid_owner_a: Dict[int, int] = {}  # gid -> a_local_id
        self.gid_owner_b: Dict[int, int] = {}  # gid -> b_local_id

        self._next_gid = 0

    def _new_gid(self) -> int:
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _get_temp_gid(self, local_id: int) -> int:
        """B track이 아직 기존 GID와 연결되지 않았을 때 표시할 임시 ID. 양수 GID를 낭비하지 않는다."""
        if local_id not in self.b_temp_gid:
            self.b_temp_gid[local_id] = -(local_id + 1)
        return self.b_temp_gid[local_id]

    def _update_gid_memory(
        self,
        gid: int,
        feat: np.ndarray,
        crop: Optional[np.ndarray],
        frame_idx: int,
        cam: str,
    ):
        if gid is None or gid < 0:
            return
        self.store_gid.update(gid, feat, frame_idx)
        if crop is not None:
            self.crop_store_gid.update(gid, crop, frame_idx)
        self.gid_last_seen[gid] = frame_idx
        self.gid_last_cam[gid] = cam

    def _distance_feature_sets(
        self,
        mean_a: Optional[np.ndarray],
        feats_a: List[np.ndarray],
        mean_b: Optional[np.ndarray],
        feats_b: List[np.ndarray],
    ) -> Optional[float]:
        """두 feature set의 weighted cosine distance를 계산한다."""
        if mean_a is None or mean_b is None:
            return None
        mean_sim = cosine_similarity_np(mean_a, mean_b)
        set_sim = pairwise_topk_similarity(feats_a, feats_b, topk=self.topk)
        if set_sim is None:
            set_sim = mean_sim
        final_sim = self.mean_weight * mean_sim + self.set_weight * set_sim
        return float(1.0 - final_sim)

    def _osnet_track_to_gid_distance(self, store: TrackFeatureStore, lid: int, gid: int) -> Optional[float]:
        return self._distance_feature_sets(
            store.get_mean(lid),
            store.get_features(lid),
            self.store_gid.get_mean(gid),
            self.store_gid.get_features(gid),
        )

    def _lpips_track_to_gid_distance(self, cam: str, lid: int, gid: int) -> Optional[float]:
        if not self.use_lpips or self.lpips_metric is None:
            return None
        local_crops = self.crop_store_a.get_crops(lid) if cam == "A" else self.crop_store_b.get_crops(lid)
        gid_crops = self.crop_store_gid.get_crops(gid)
        if len(local_crops) < self.min_features_to_match or len(gid_crops) < self.min_features_to_match:
            return None
        return self.lpips_metric.set_distance(local_crops, gid_crops, topk=self.topk)

    def _track_to_gid_distance_detail(
        self,
        store: TrackFeatureStore,
        lid: int,
        gid: int,
        cam: str,
    ) -> Optional[Dict[str, Optional[float]]]:
        osnet_dist = self._osnet_track_to_gid_distance(store, lid, gid)
        if osnet_dist is None:
            return None
        lpips_dist = self._lpips_track_to_gid_distance(cam, lid, gid)
        if lpips_dist is None:
            final_dist = osnet_dist
        else:
            final_dist = (
                self.osnet_distance_weight * osnet_dist
                + self.lpips_distance_weight * lpips_dist
            )
        return {
            "gid": gid,
            "osnet_dist": float(osnet_dist),
            "lpips_dist": None if lpips_dist is None else float(lpips_dist),
            "final_dist": float(final_dist),
        }

    def _track_to_gid_distance(self, store: TrackFeatureStore, lid: int, gid: int, cam: str) -> Optional[float]:
        detail = self._track_to_gid_distance_detail(store, lid, gid, cam)
        return None if detail is None else float(detail["final_dist"])

    def _rank_gid_candidate_details(
        self,
        store: TrackFeatureStore,
        lid: int,
        cam: str,
        exclude_active_same_cam: bool = True,
    ) -> List[Dict[str, Optional[float]]]:
        """local track을 Global ID memory와 비교해 후보별 상세 score를 반환한다."""
        candidates: List[Dict[str, Optional[float]]] = []
        for gid in self.store_gid.keys():
            if gid < 0 or self.store_gid.count(gid) < self.min_features_to_match:
                continue
            if exclude_active_same_cam:
                if cam == "A" and self.gid_owner_a.get(gid) is not None and self.gid_owner_a.get(gid) != lid:
                    continue
                if cam == "B" and self.gid_owner_b.get(gid) is not None and self.gid_owner_b.get(gid) != lid:
                    continue
            detail = self._track_to_gid_distance_detail(store, lid, gid, cam)
            if detail is None:
                continue
            candidates.append(detail)
        candidates.sort(key=lambda x: float(x["final_dist"]))
        return candidates

    def _rank_gid_candidates(
        self,
        store: TrackFeatureStore,
        lid: int,
        cam: str,
        exclude_active_same_cam: bool = True,
    ) -> List[Tuple[int, float]]:
        """local track을 Global ID memory와 비교해 (gid, final_distance) 후보를 반환한다."""
        details = self._rank_gid_candidate_details(store, lid, cam, exclude_active_same_cam)
        return [(int(d["gid"]), float(d["final_dist"])) for d in details]

    def _decay_evidence(self, b_lid: int):
        if b_lid not in self.b_evidence:
            return
        remove = []
        for gid in list(self.b_evidence[b_lid].keys()):
            self.b_evidence[b_lid][gid] *= self.evidence_decay
            if self.b_evidence[b_lid][gid] < 1e-4:
                remove.append(gid)
        for gid in remove:
            self.b_evidence[b_lid].pop(gid, None)
            self.b_hits[b_lid].pop(gid, None)

    def _evidence_rank(self, b_lid: int) -> List[Tuple[int, float]]:
        items = [(gid, score) for gid, score in self.b_evidence.get(b_lid, {}).items() if gid >= 0]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def _owner_confidence(self, b_lid: int, gid: int) -> float:
        return float(self.b_evidence.get(b_lid, {}).get(gid, 0.0))

    def _release_gid_if_owned(self, b_lid: int, gid: Optional[int]):
        if gid is not None and gid >= 0 and self.gid_owner_b.get(gid) == b_lid:
            self.gid_owner_b.pop(gid, None)

    def _try_assign_gid(self, b_lid: int, candidate_gid: int, force_switch: bool = False) -> bool:
        """
        B camera one-to-one 제약을 고려해 candidate_gid를 b_lid의 current GID로 설정한다.
        이미 다른 B track이 같은 GID를 쓰고 있으면 evidence가 더 높은 쪽이 소유한다.
        """
        if candidate_gid is None or candidate_gid < 0:
            return False

        current = self.b_current_gid.get(b_lid)
        if current == candidate_gid:
            return True

        owner = self.gid_owner_b.get(candidate_gid)
        my_conf = self._owner_confidence(b_lid, candidate_gid)

        if owner is not None and owner != b_lid:
            owner_conf = self._owner_confidence(owner, candidate_gid)
            if my_conf <= owner_conf + self.switch_margin:
                return False
            self.b_current_gid[owner] = self._get_temp_gid(owner)
            self.gid_owner_b.pop(candidate_gid, None)

        if current is not None:
            self._release_gid_if_owned(b_lid, current)

        self.b_current_gid[b_lid] = candidate_gid
        self.gid_owner_b[candidate_gid] = b_lid
        return True

    def _select_existing_gid_for_a(self, local_id: int) -> Optional[int]:
        """A camera의 새 local track이 기존 GID memory와 이어질 수 있는지 확인한다."""
        candidates = self._rank_gid_candidates(self.store_a, local_id, cam="A", exclude_active_same_cam=True)
        if not candidates:
            return None
        best_gid, best_dist = candidates[0]
        second_dist = candidates[1][1] if len(candidates) > 1 else float("inf")
        if best_dist <= self.threshold and (second_dist - best_dist) >= self.reassoc_margin:
            return best_gid
        return None

    def update_cam_a(
        self,
        local_id: int,
        feat: np.ndarray,
        crop: Optional[np.ndarray],
        frame_idx: int,
    ) -> int:
        """
        Camera A local track update.
        OSNet feature와 crop을 함께 저장한다.
        """
        self.store_a.update(local_id, feat, frame_idx)
        if crop is not None:
            self.crop_store_a.update(local_id, crop, frame_idx)

        if local_id in self.a_gid:
            gid = self.a_gid[local_id]
        else:
            gid = self._select_existing_gid_for_a(local_id)
            if gid is None:
                gid = self._new_gid()
            self.a_gid[local_id] = gid
            self.gid_owner_a[gid] = local_id

        self._update_gid_memory(gid, feat, crop, frame_idx, "A")
        self.last_score_info_a[local_id] = None
        return gid

    def _update_current_assignment(self, b_lid: int, latest_best_gid: Optional[int] = None):
        """누적 evidence를 보고 B track의 current GID를 신규 설정하거나 재배정한다."""
        ranked = self._evidence_rank(b_lid)
        if not ranked:
            if b_lid not in self.b_current_gid:
                self.b_current_gid[b_lid] = self._get_temp_gid(b_lid)
            return

        best_gid, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        ev_margin = best_score - second_score
        hits = self.b_hits[b_lid].get(best_gid, 0)

        if best_score < self.min_evidence or hits < self.confirm_count or ev_margin < self.evidence_margin:
            if b_lid not in self.b_current_gid:
                self.b_current_gid[b_lid] = self._get_temp_gid(b_lid)
            return

        current = self.b_current_gid.get(b_lid)
        temp_gid = self.b_temp_gid.get(b_lid)

        if current is None or current == temp_gid or current < 0:
            self._try_assign_gid(b_lid, best_gid)
            return

        if current != best_gid:
            current_score = self._owner_confidence(b_lid, current)
            if latest_best_gid == best_gid and best_score >= current_score + self.switch_margin:
                self._try_assign_gid(b_lid, best_gid, force_switch=True)
            return

        self._try_assign_gid(b_lid, best_gid)

    def update_cam_b_and_match(
        self,
        local_id: int,
        feat: np.ndarray,
        crop: Optional[np.ndarray],
        frame_idx: int,
    ) -> Tuple[int, Optional[float]]:
        """
        Camera B local track update and match against Global ID memory.
        OSNet distance + LPIPS distance를 섞은 final distance를 사용한다.
        """
        self.store_b.update(local_id, feat, frame_idx)
        if crop is not None:
            self.crop_store_b.update(local_id, crop, frame_idx)

        self._decay_evidence(local_id)

        if self.store_b.count(local_id) < self.min_features_to_match:
            gid = self.b_current_gid.get(local_id, self._get_temp_gid(local_id))
            self.b_current_gid[local_id] = gid
            self.last_score_info_b[local_id] = None
            return gid, None

        candidate_details = self._rank_gid_candidate_details(
            self.store_b,
            local_id,
            cam="B",
            exclude_active_same_cam=True,
        )
        ranked_candidates = [(int(d["gid"]), float(d["final_dist"])) for d in candidate_details]

        if not ranked_candidates:
            gid = self.b_current_gid.get(local_id, self._get_temp_gid(local_id))
            self.b_current_gid[local_id] = gid
            self.last_score_info_b[local_id] = None
            return gid, None

        best_detail = candidate_details[0]
        best_gid, best_dist = ranked_candidates[0]
        second_dist = ranked_candidates[1][1] if len(ranked_candidates) > 1 else float("inf")
        self.b_last_best_dist[local_id] = best_dist

        self.last_score_info_b[local_id] = {
            "gid": best_detail.get("gid"),
            "final_dist": best_detail.get("final_dist"),
            "osnet_dist": best_detail.get("osnet_dist"),
            "lpips_dist": best_detail.get("lpips_dist"),
            "margin": None if second_dist == float("inf") else float(second_dist - best_dist),
            "threshold": float(self.threshold),
        }

        if best_dist <= self.threshold and (second_dist - best_dist) >= self.distance_margin:
            score = 1.0 / (1.0 + max(0.0, best_dist))
            self.b_evidence[local_id][best_gid] += score
            self.b_hits[local_id][best_gid] += 1

        self._update_current_assignment(local_id, latest_best_gid=best_gid)
        gid = self.b_current_gid.get(local_id, self._get_temp_gid(local_id))

        # 충분히 evidence가 있는 양수 GID로 연결된 경우에만 Global memory를 보강한다.
        # 너무 이른 오매칭 feature가 archive를 오염시키는 것을 막기 위한 조건이다.
        if gid is not None and gid >= 0:
            if self.b_hits[local_id].get(gid, 0) >= self.confirm_count and self.b_evidence[local_id].get(gid, 0.0) >= self.min_evidence:
                self._update_gid_memory(gid, feat, crop, frame_idx, "B")

        return gid, best_dist

    def get_gid(self, local_id: int, cam: str) -> Optional[int]:
        if cam == "A":
            return self.a_gid.get(local_id)
        return self.b_current_gid.get(local_id, self.b_temp_gid.get(local_id))

    def get_score_info(self, local_id: int, cam: str) -> Optional[Dict[str, Optional[float]]]:
        if cam == "A":
            return self.last_score_info_a.get(local_id)
        return self.last_score_info_b.get(local_id)

    def expire(self, frame_idx: int):
        # local track store는 짧게 expire한다. Global ID memory는 더 오래 유지한다.
        self.store_a.expire(frame_idx, self.max_age)
        self.store_b.expire(frame_idx, self.max_age)
        self.store_gid.expire(frame_idx, self.memory_max_age)
        self.crop_store_a.expire(frame_idx, self.max_age)
        self.crop_store_b.expire(frame_idx, self.max_age)
        self.crop_store_gid.expire(frame_idx, self.memory_max_age)

        valid_a = set(self.store_a.keys())
        valid_b = set(self.store_b.keys())
        valid_gid = set(self.store_gid.keys())

        for lid in list(self.a_gid.keys()):
            if lid not in valid_a:
                old_gid = self.a_gid.pop(lid, None)
                self.last_score_info_a.pop(lid, None)
                if old_gid is not None and self.gid_owner_a.get(old_gid) == lid:
                    self.gid_owner_a.pop(old_gid, None)

        for lid in list(self.b_current_gid.keys()):
            if lid not in valid_b:
                old_gid = self.b_current_gid.pop(lid, None)
                self.b_temp_gid.pop(lid, None)
                self.b_evidence.pop(lid, None)
                self.b_hits.pop(lid, None)
                self.b_last_best_dist.pop(lid, None)
                self.last_score_info_b.pop(lid, None)
                if old_gid is not None and old_gid >= 0 and self.gid_owner_b.get(old_gid) == lid:
                    self.gid_owner_b.pop(old_gid, None)

        # 오래된 global memory가 제거되면 owner도 정리한다.
        for gid in list(self.gid_owner_a.keys()):
            if gid not in valid_gid:
                self.gid_owner_a.pop(gid, None)
        for gid in list(self.gid_owner_b.keys()):
            if gid not in valid_gid:
                self.gid_owner_b.pop(gid, None)
        for gid in list(self.gid_last_seen.keys()):
            if gid not in valid_gid:
                self.gid_last_seen.pop(gid, None)
                self.gid_last_cam.pop(gid, None)


# ── crop quality / occlusion filtering ────────────────────────────────────────
def bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """두 bbox의 IoU 계산. box = (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def find_occluded_track_ids(
    detections: List[Tuple[int, int, int, int, int, float]],
    iou_threshold: float = OCCLUSION_IOU_THRESHOLD,
) -> set:
    """
    같은 카메라의 같은 프레임에서 person bbox끼리 많이 겹치는 track id를 찾는다.

    occlusion 상태의 crop은 다른 사람의 옷/신체 정보가 함께 들어갈 가능성이 높기 때문에
    Re-ID feature buffer와 evidence score 업데이트에서 제외한다.
    """
    occluded = set()
    n = len(detections)
    for i in range(n):
        lid_i, x1_i, y1_i, x2_i, y2_i, _ = detections[i]
        box_i = (x1_i, y1_i, x2_i, y2_i)
        for j in range(i + 1, n):
            lid_j, x1_j, y1_j, x2_j, y2_j, _ = detections[j]
            box_j = (x1_j, y1_j, x2_j, y2_j)
            if bbox_iou(box_i, box_j) >= iou_threshold:
                occluded.add(int(lid_i))
                occluded.add(int(lid_j))
    return occluded


def filter_small_detections(
    detections: List[Tuple[int, int, int, int, int, float]],
    min_area: int,
) -> List[Tuple[int, int, int, int, int, float]]:
    """너무 멀리 있는 작은 person bbox를 시각화/매칭 대상에서 제외한다.

    YOLO confidence를 낮추고 imgsz를 키우면 아주 멀리 있는 사람까지 잡히는 경우가 있다.
    이런 작은 bbox는 Re-ID feature 품질도 낮고 화면도 복잡하게 만들기 때문에
    tracker 결과에서 후처리로 제외한다. detection 자체를 막는 것은 아니지만,
    이후 GID 부여/표시/feature update 대상에서는 빠진다.
    """
    if min_area <= 0:
        return detections
    filtered = []
    for det in detections:
        lid, x1, y1, x2, y2, conf = det
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area >= min_area:
            filtered.append(det)
    return filtered


def valid_crop(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    conf: float,
    edge_margin_ratio: float = EDGE_MARGIN_RATIO,
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> Optional[np.ndarray]:
    if conf < MIN_DET_CONF:
        return None

    h, w = frame.shape[:2]
    bw_raw = max(1, x2 - x1)
    bh_raw = max(1, y2 - y1)
    raw_area = bw_raw * bh_raw
    if raw_area < MIN_BBOX_AREA:
        return None

    # 사람이 너무 얇거나 너무 넓게 잡힌 crop은 bbox 품질이 낮거나 여러 사람이 섞였을 가능성이 있다.
    aspect = bw_raw / max(1, bh_raw)
    if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
        return None

    # 화면 경계에 너무 붙어 있으면 사람 일부가 잘린 crop일 수 있으므로 feature update에서 제외한다.
    mx = int(w * edge_margin_ratio)
    my = int(h * edge_margin_ratio)
    if x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my:
        return None

    # Re-ID는 bbox가 너무 타이트하면 신발/가방/상하의 경계 정보가 잘릴 수 있으므로
    # crop 주변에 약간의 context padding을 추가한다.
    pad_x = int(bw_raw * CROP_PADDING)
    pad_y = int(bh_raw * CROP_PADDING)

    x1 = max(0, min(x1 - pad_x, w - 1))
    x2 = max(0, min(x2 + pad_x, w - 1))
    y1 = max(0, min(y1 - pad_y, h - 1))
    y2 = max(0, min(y2 + pad_y, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, gid, local_id, cam_label, score_info=None, debug_score_mode: str = "final"):
    color = get_color(gid)
    h, w = frame.shape[:2]

    # bbox 자체는 원래 위치에 그린다.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    gid_text = "?" if gid is None or int(gid) < 0 else str(gid)
    # 화면 상단에 이미 Camera A/B가 표시되므로 bbox label에는 카메라 표기를 생략한다.
    label = f"LID:{local_id} GID:{gid_text}"

    if score_info is not None:
        final_dist = score_info.get("final_dist")
        osnet_dist = score_info.get("osnet_dist")
        lpips_dist = score_info.get("lpips_dist")

        if final_dist is not None:
            label += f" D:{float(final_dist):.2f}"

        # --debug_score_mode full이면 OSNet/LPIPS도 같이 표시한다.
        if debug_score_mode == "full":
            if osnet_dist is not None:
                label += f" O:{float(osnet_dist):.2f}"
            if lpips_dist is not None:
                label += f" L:{float(lpips_dist):.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 if score_info is not None else 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # label 배경 박스가 현재 frame 내부에 완전히 들어오도록 x 좌표를 보정한다.
    label_w = tw + 6
    tx = int(max(0, min(x1, w - label_w - 1)))

    # bbox 위쪽에 공간이 부족하면 bbox 아래쪽에 label을 표시한다.
    ty = y1 - 6
    if ty - th - 4 < 0:
        ty = min(h - 4, y2 + th + 8)
    ty = int(max(th + 4, min(ty, h - 4)))

    cv2.rectangle(frame, (tx, ty - th - 4), (tx + label_w, ty + 2), color, -1)
    cv2.putText(
        frame,
        label,
        (tx + 3, ty - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_cached(frame, cached_dets, cam_label, orig_w, orig_h, debug_score_mode: str = "final"):
    sx = DISPLAY_W / max(orig_w, 1)
    sy = DISPLAY_H / max(orig_h, 1)

    for item in cached_dets:
        if len(item) == 6:
            lid, x1, y1, x2, y2, gid = item
            score_info = None
        else:
            lid, x1, y1, x2, y2, gid, score_info = item

        draw_box(
            frame,
            int(x1 * sx),
            int(y1 * sy),
            int(x2 * sx),
            int(y2 * sy),
            gid,
            lid,
            cam_label,
            score_info=score_info,
            debug_score_mode=debug_score_mode,
        )


def resize_frame(frame, w, h):
    return cv2.resize(frame, (w, h))


def make_display(frame_a, frame_b, frame_idx, stride, is_feature_frame, gid_map_a, gid_map_b):
    fa = resize_frame(frame_a, DISPLAY_W, DISPLAY_H)
    fb = resize_frame(frame_b, DISPLAY_W, DISPLAY_H)

    cv2.putText(fa, "Camera A", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(fb, "Camera B", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)

    tag = "OSNet" if is_feature_frame else f"track-only({stride})"
    cv2.putText(
        fa,
        f"frame {frame_idx} [{tag}]",
        (10, DISPLAY_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    matched = set(gid_map_a.values()) & set(gid_map_b.values())
    cv2.putText(
        fb,
        f"Matched IDs: {len(matched)}",
        (10, DISPLAY_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 220, 100),
        1,
    )


    divider = np.zeros((DISPLAY_H, 4, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)
    return np.hstack([fa, divider, fb])


# ── 디렉토리 유틸 ─────────────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def collect_videos(directory: str) -> list:
    d = Path(directory)
    if not d.is_dir():
        sys.exit(f"[ERROR] 디렉토리가 없습니다: {directory}")
    videos = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"[ERROR] 영상 파일이 없습니다: {directory}")
    return videos


def make_pairs(dir_a: str, dir_b: str):
    vids_a = collect_videos(dir_a)
    vids_b = collect_videos(dir_b)
    n = min(len(vids_a), len(vids_b))
    if len(vids_a) != len(vids_b):
        print(f"[WARN] 파일 수 불일치 (A:{len(vids_a)}, B:{len(vids_b)}) → 앞 {n}쌍만 처리")
    pairs = list(zip(vids_a[:n], vids_b[:n]))
    print(f"[INFO] 총 {n}쌍 처리 예정:")
    for i, (a, b) in enumerate(pairs, 1):
        print(f"  [{i:02d}] {a.name} ←→ {b.name}")
    return pairs


def resolve_save_path(save_path: Optional[str], video_a: str, video_b: str) -> Optional[str]:
    if not save_path:
        return None

    p = Path(save_path)
    # 확장자가 없거나 디렉토리로 보이면 자동 파일명 생성
    if p.exists() and p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        return str(p / f"result_{Path(video_a).stem}_vs_{Path(video_b).stem}.mp4")

    if p.suffix.lower() not in VIDEO_EXTS:
        p.mkdir(parents=True, exist_ok=True)
        return str(p / f"result_{Path(video_a).stem}_vs_{Path(video_b).stem}.mp4")

    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# ── 실행 환경 유틸 ─────────────────────────────────────────────────────────────
def is_colab_env() -> bool:
    """Colab/Jupyter처럼 cv2.imshow를 사용할 수 없는 headless 환경 감지."""
    return (
        "COLAB_GPU" in os.environ
        or "COLAB_RELEASE_TAG" in os.environ
        or "google.colab" in sys.modules
    )


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def run(
    video_a: str,
    video_b: str,
    save_path: str = None,
    stride: int = FRAME_STRIDE,
    show: Optional[bool] = None,
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
    use_lpips: bool = USE_LPIPS,
    lpips_net: str = LPIPS_NET,
    osnet_weight: float = OSNET_DISTANCE_WEIGHT,
    lpips_weight: float = LPIPS_DISTANCE_WEIGHT,
    lpips_scale: float = LPIPS_DISTANCE_SCALE,
    debug: bool = DEBUG_MATCH,
    debug_score_mode: str = DEBUG_SCORE_MODE,
):
    global MATCH_THRESHOLD

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if show is None:
        show = not is_colab_env()

    print(f"[INFO] Device : {device}")
    print(f"[INFO] Stride : {stride} (YOLO/ByteTrack은 매 프레임, OSNet은 {stride}프레임마다 실행)")
    print(f"[INFO] Match threshold : hybrid distance <= {MATCH_THRESHOLD}")
    print(f"[INFO] YOLO model : {yolo_model}, tracker={tracker_name}, imgsz={imgsz}, conf={yolo_conf}, iou={yolo_iou}, augment={yolo_augment}, min_det_area={min_det_area}")
    print("[INFO] v10 stable: v7 archive Re-ID + YOLO11/BoT-SORT balanced detection. Local stitching is disabled.")
    print(f"[INFO] Matching : mean_weight={mean_weight:.2f}, set_weight={set_weight:.2f}, topk={topk}")
    print(f"[INFO] Evidence : confirm={confirm_count}, decay={evidence_decay}, min={min_evidence}, ev_margin={evidence_margin}, switch_margin={switch_margin}, dist_margin={distance_margin}")
    print(f"[INFO] Occlusion-aware update : {'ON' if skip_occluded_update else 'OFF'}, occ_iou={occ_iou}, edge_margin={edge_margin}, aspect=[{min_aspect}, {max_aspect}]")
    print(f"[INFO] GID archive : memory_max_age={memory_max_age}, reassoc_margin={reassoc_margin}")
    print(f"[INFO] Display window : {'ON' if show else 'OFF'}")
    print(f"[INFO] BBox score display : {'ON' if debug else 'OFF'} ({debug_score_mode})")

    print("[INFO] Loading YOLO + ByteTrack ...")
    tracker_a = Tracker(yolo_model, imgsz=imgsz, conf=yolo_conf, iou=yolo_iou, augment=yolo_augment, tracker_name=tracker_name)
    tracker_b = Tracker(yolo_model, imgsz=imgsz, conf=yolo_conf, iou=yolo_iou, augment=yolo_augment, tracker_name=tracker_name)

    print("[INFO] Loading OSNet ...")
    extractor = OSNetExtractor(OSNET_MODEL, OSNET_WEIGHTS, device)

    gallery = Gallery(
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
        use_lpips=use_lpips,
        lpips_net=lpips_net,
        osnet_weight=osnet_weight,
        lpips_weight=lpips_weight,
        lpips_scale=lpips_scale,
        debug=debug,
        debug_score_mode=debug_score_mode,
    )

    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)
    if not cap_a.isOpened():
        sys.exit(f"[ERROR] Cannot open: {video_a}")
    if not cap_b.isOpened():
        sys.exit(f"[ERROR] Cannot open: {video_b}")

    save_path = resolve_save_path(save_path, video_a, video_b)
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap_a.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 20.0
        writer = cv2.VideoWriter(save_path, fourcc, float(fps), (DISPLAY_W * 2 + 4, DISPLAY_H))
        print(f"[INFO] Saving to : {save_path}")

    orig_h_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_idx = 0
    paused = False

    if show:
        print("[INFO] Starting — press Q to quit, P/Space to pause")
    else:
        print("[INFO] Starting in headless mode — result will be saved if --save is set")
    while True:
        if not paused:
            ret_a, frame_a = cap_a.read()
            ret_b, frame_b = cap_b.read()
            if not ret_a or not ret_b:
                print("[INFO] Video ended.")
                break

            vis_a = resize_frame(frame_a, DISPLAY_W, DISPLAY_H)
            vis_b = resize_frame(frame_b, DISPLAY_W, DISPLAY_H)

            is_feature_frame = (frame_idx % max(stride, 1) == 0)

            cache_a: List[Tuple[int, int, int, int, int, int]] = []
            cache_b: List[Tuple[int, int, int, int, int, int]] = []
            gid_map_a: Dict[int, int] = {}
            gid_map_b: Dict[int, int] = {}

            # ── Camera A: YOLO + ByteTrack 매 프레임 ─────────────────────────
            dets_a = tracker_a.update(frame_a)
            dets_a = filter_small_detections(dets_a, min_det_area)
            occluded_a = find_occluded_track_ids(dets_a, occ_iou) if skip_occluded_update else set()
            for (lid, x1, y1, x2, y2, conf) in dets_a:
                if is_feature_frame and lid not in occluded_a:
                    crop = valid_crop(frame_a, x1, y1, x2, y2, conf, edge_margin, min_aspect, max_aspect)
                    if crop is not None:
                        feat = extractor.extract(crop)
                        if feat is not None:
                            gid = gallery.update_cam_a(lid, feat, crop, frame_idx)
                        else:
                            gid = gallery.get_gid(lid, "A")
                    else:
                        gid = gallery.get_gid(lid, "A")
                else:
                    gid = gallery.get_gid(lid, "A")

                if gid is None:
                    continue
                gid_map_a[lid] = gid
                cache_a.append((lid, x1, y1, x2, y2, gid))

            # ── Camera B: YOLO + ByteTrack 매 프레임 ─────────────────────────
            dets_b = tracker_b.update(frame_b)
            dets_b = filter_small_detections(dets_b, min_det_area)
            occluded_b = find_occluded_track_ids(dets_b, occ_iou) if skip_occluded_update else set()
            for (lid, x1, y1, x2, y2, conf) in dets_b:
                if is_feature_frame and lid not in occluded_b:
                    crop = valid_crop(frame_b, x1, y1, x2, y2, conf, edge_margin, min_aspect, max_aspect)
                    if crop is not None:
                        feat = extractor.extract(crop)
                        if feat is not None:
                            gid, _best_dist = gallery.update_cam_b_and_match(lid, feat, crop, frame_idx)
                        else:
                            gid = gallery.get_gid(lid, "B")
                    else:
                        gid = gallery.get_gid(lid, "B")
                else:
                    gid = gallery.get_gid(lid, "B")

                if gid is None:
                    continue
                gid_map_b[lid] = gid
                score_info = gallery.get_score_info(lid, "B") if debug else None
                cache_b.append((lid, x1, y1, x2, y2, gid, score_info))

            if is_feature_frame:
                gallery.expire(frame_idx)

            draw_cached(vis_a, cache_a, "A", orig_w_a, orig_h_a, debug_score_mode=debug_score_mode)
            draw_cached(vis_b, cache_b, "B", orig_w_b, orig_h_b, debug_score_mode=debug_score_mode)

            display = make_display(vis_a, vis_b, frame_idx, stride, is_feature_frame, gid_map_a, gid_map_b)
            if writer:
                writer.write(display)

            frame_idx += 1

        if show:
            cv2.imshow("Cross-Camera ReID | Q: quit", display)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord("q"):
                print("[INFO] User quit.")
                break
            elif key == ord("p") or key == 32:
                paused = not paused
                print("[INFO] Paused" if paused else "[INFO] Resumed")

    cap_a.release()
    cap_b.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_idx} frames.")


# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Camera ReID Pipeline: YOLO + ByteTrack + OSNet + feature-set top-k + evidence reassignment + occlusion-aware update + GID archive",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 단일 파일 모드
    parser.add_argument("--video_a", default="/home/knuvi/Multi-scene-detection/dataset/MOT2024/Camera_0002.mp4", help="Camera A 영상 경로")
    parser.add_argument("--video_b", default="/home/knuvi/Multi-scene-detection/dataset/MOT2024/Camera_0003.mp4", help="Camera B 영상 경로")
    #parser.add_argument("--video_a", default="/home/knuvi/Multi-scene-detection/dataset/ours/recorded0001.mp4", help="Camera A 영상 경로")
    #parser.add_argument("--video_b", default="/home/knuvi/Multi-scene-detection/dataset/ours/recorded0002.mp4", help="Camera B 영상 경로")
    
    parser.add_argument("--save", default="/home/knuvi/Multi-scene-detection/results", help="결과 영상 저장 경로 또는 저장 폴더")

    # 디렉토리 모드
    parser.add_argument("--dir_a", default=None, help="Camera A 영상 폴더")
    parser.add_argument("--dir_b", default=None, help="Camera B 영상 폴더")
    parser.add_argument("--save_dir", default=None, help="결과 저장 폴더 (디렉토리 모드)")

    # 공통
    parser.add_argument(
        "--stride",
        type=int,
        default=FRAME_STRIDE,
        help=(
            f"N프레임마다 OSNet feature 추출 (기본: {FRAME_STRIDE})\n"
            "stride=1 : 매 프레임 feature 추출, 느리지만 feature가 많이 쌓임\n"
            "stride=10: 비교적 균형\n"
            "stride=20: 속도 우선"
        ),
    )
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help=f"매칭 hybrid distance 임계값 (기본: {MATCH_THRESHOLD})")
    parser.add_argument("--confirm", type=int, default=MATCH_CONFIRM_COUNT, help=f"Global ID 확정에 필요한 누적 매칭 횟수 (기본: {MATCH_CONFIRM_COUNT})")
    parser.add_argument("--show", default=True, action="store_true", help="OpenCV 화면 창을 표시합니다. Colab에서는 사용하지 마세요.")
    parser.add_argument("--yolo_model", default=YOLO_MODEL, help=f"YOLO 모델 파일명 (기본: {YOLO_MODEL}, 빠르게는 yolo11s.pt, 강하게는 yolo11l.pt/yolo11x.pt)")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker 설정 파일 (기본: botsort.yaml, 빠르게는 bytetrack.yaml)")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ, help=f"YOLO 추론 이미지 크기 (기본: {YOLO_IMGSZ}, 작을수록 빠름)")
    parser.add_argument("--yolo_conf", type=float, default=YOLO_CONF, help=f"YOLO detection confidence (기본: {YOLO_CONF})")
    parser.add_argument("--yolo_iou", type=float, default=YOLO_IOU, help=f"YOLO NMS IoU threshold (기본: {YOLO_IOU})")
    parser.add_argument("--yolo_augment", action="store_true", help="YOLO augment/TTA 추론을 사용합니다. 느리지만 일부 어려운 검출이 늘 수 있습니다.")
    parser.add_argument("--min_det_area", type=int, default=MIN_DET_BBOX_AREA, help=f"너무 작은 person bbox를 표시/매칭에서 제외하는 최소 area (기본: {MIN_DET_BBOX_AREA}, 0이면 비활성화)")
    parser.add_argument("--topk", type=int, default=TOPK_PAIRWISE, help=f"feature-set 비교에서 사용할 top-k pair 수 (기본: {TOPK_PAIRWISE})")
    parser.add_argument("--mean_weight", type=float, default=MEAN_SCORE_WEIGHT, help=f"mean feature similarity 가중치 (기본: {MEAN_SCORE_WEIGHT})")
    parser.add_argument("--set_weight", type=float, default=SET_SCORE_WEIGHT, help=f"feature-set top-k similarity 가중치 (기본: {SET_SCORE_WEIGHT})")
    parser.add_argument("--evidence_decay", type=float, default=EVIDENCE_DECAY, help=f"feature frame마다 기존 evidence를 감쇠하는 비율 (기본: {EVIDENCE_DECAY})")
    parser.add_argument("--min_evidence", type=float, default=MIN_EVIDENCE, help=f"A GID로 표시하기 위한 최소 누적 evidence (기본: {MIN_EVIDENCE})")
    parser.add_argument("--evidence_margin", type=float, default=EVIDENCE_MARGIN, help=f"1등/2등 evidence 차이 조건 (기본: {EVIDENCE_MARGIN})")
    parser.add_argument("--switch_margin", type=float, default=SWITCH_MARGIN, help=f"기존 GID를 새 GID로 바꾸기 위한 evidence 차이 조건 (기본: {SWITCH_MARGIN})")
    parser.add_argument("--distance_margin", type=float, default=DISTANCE_MARGIN, help=f"best/second distance 차이 조건 (기본: {DISTANCE_MARGIN})")
    parser.add_argument("--no_skip_occluded", action="store_true", help="겹침 bbox에 대해서도 feature/evidence update를 수행합니다. 기본은 skip.")
    parser.add_argument("--occ_iou", type=float, default=OCCLUSION_IOU_THRESHOLD, help=f"occlusion으로 간주할 bbox IoU threshold (기본: {OCCLUSION_IOU_THRESHOLD})")
    parser.add_argument("--edge_margin", type=float, default=EDGE_MARGIN_RATIO, help=f"프레임 경계 crop 제외 margin 비율 (기본: {EDGE_MARGIN_RATIO})")
    parser.add_argument("--min_aspect", type=float, default=MIN_ASPECT_RATIO, help=f"feature update에 사용할 bbox width/height 최소 비율 (기본: {MIN_ASPECT_RATIO})")
    parser.add_argument("--max_aspect", type=float, default=MAX_ASPECT_RATIO, help=f"feature update에 사용할 bbox width/height 최대 비율 (기본: {MAX_ASPECT_RATIO})")
    parser.add_argument("--memory_max_age", type=int, default=GLOBAL_MEMORY_MAX_AGE, help=f"사라진 GID를 archive에 유지할 프레임 수 (기본: {GLOBAL_MEMORY_MAX_AGE})")
    parser.add_argument("--reassoc_margin", type=float, default=REASSOC_DISTANCE_MARGIN, help=f"새 local track을 기존 GID와 재연결할 때 best/second distance 차이 조건 (기본: {REASSOC_DISTANCE_MARGIN})")
    parser.add_argument("--no_lpips", action="store_true", help="LPIPS distance를 끄고 기존 OSNet distance만 사용합니다.")
    parser.add_argument("--lpips_net", default=LPIPS_NET, choices=["alex", "vgg", "squeeze"], help=f"LPIPS backbone (기본: {LPIPS_NET})")
    parser.add_argument("--osnet_weight", type=float, default=OSNET_DISTANCE_WEIGHT, help=f"hybrid distance에서 OSNet distance 가중치 (기본: {OSNET_DISTANCE_WEIGHT})")
    parser.add_argument("--lpips_weight", type=float, default=LPIPS_DISTANCE_WEIGHT, help=f"hybrid distance에서 LPIPS distance 가중치 (기본: {LPIPS_DISTANCE_WEIGHT})")
    parser.add_argument("--lpips_scale", type=float, default=LPIPS_DISTANCE_SCALE, help=f"LPIPS distance scale 보정값 (기본: {LPIPS_DISTANCE_SCALE})")
    parser.add_argument("--debug", action="store_true", help="bbox label에 matching distance score를 표시합니다.")
    parser.add_argument("--debug_score_mode", default=DEBUG_SCORE_MODE, choices=["final", "full"], help="debug bbox score 표시 방식: final=D만, full=D/O/L")

    args = parser.parse_args()

    MATCH_THRESHOLD = args.threshold
    MATCH_CONFIRM_COUNT = args.confirm

    use_dir = args.dir_a is not None or args.dir_b is not None

    if use_dir:
        if not args.dir_a or not args.dir_b:
            sys.exit("[ERROR] --dir_a 와 --dir_b 를 모두 지정해야 합니다.")
        pairs = make_pairs(args.dir_a, args.dir_b)
        save_dir = Path(args.save_dir) if args.save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for idx, (path_a, path_b) in enumerate(pairs, 1):
            print(f"\n{'=' * 60}")
            print(f"[{idx:02d}/{len(pairs)}] {path_a.name} ←→ {path_b.name}")
            print(f"{'=' * 60}")
            save_path = str(save_dir / f"result_{idx:02d}_{path_a.stem}_vs_{path_b.stem}.mp4") if save_dir else None
            run(
                str(path_a),
                str(path_b),
                save_path=save_path,
                stride=args.stride,
                show=args.show,
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
                use_lpips=not args.no_lpips,
                lpips_net=args.lpips_net,
                osnet_weight=args.osnet_weight,
                lpips_weight=args.lpips_weight,
                lpips_scale=args.lpips_scale,
                debug=args.debug,
                debug_score_mode=args.debug_score_mode,
            )

        print(f"\n[INFO] 전체 {len(pairs)}쌍 처리 완료.")
    else:
        run(
            args.video_a,
            args.video_b,
            save_path=args.save,
            stride=args.stride,
            show=args.show,
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
            use_lpips=not args.no_lpips,
            lpips_net=args.lpips_net,
            osnet_weight=args.osnet_weight,
            lpips_weight=args.lpips_weight,
            lpips_scale=args.lpips_scale,
            debug=args.debug,
            debug_score_mode=args.debug_score_mode,
        )
