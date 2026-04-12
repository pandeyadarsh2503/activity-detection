"""
face_auth/face_engine.py
=========================
Lightweight wrapper around InsightFace FaceAnalysis.

Provides a clean API consumed by both the registration script
and the live inference pipeline.

Source: adapted from https://github.com/shikkoustic/face-recognition
"""

import numpy as np
try:
    from insightface.app import FaceAnalysis
except ImportError as _e:
    raise ImportError(
        "[face_auth] insightface is not installed.\n"
        "Run: pip install insightface onnxruntime"
    ) from _e

from face_auth import config


class FaceEngine:
    """
    Wraps InsightFace for face detection, embedding, and similarity matching.

    Thread-safety: a single instance should be owned by one thread.
    Create separate instances per thread if parallel access is needed.
    """

    def __init__(self):
        """Load the model pack and prepare the detector."""
        print(f"[FaceEngine] Loading model pack '{config.MODEL_NAME}' …")
        self.app = FaceAnalysis(name=config.MODEL_NAME)
        self.app.prepare(ctx_id=config.CTX_ID, det_size=config.DET_SIZE)
        print("[FaceEngine] Ready.")

    # ── Core operations ─────────────────────────────────────────────────────

    def detect_and_embed(self, bgr_frame: np.ndarray) -> list:
        """
        Detect all faces in *bgr_frame* and return bboxes + embeddings.

        Parameters
        ----------
        bgr_frame : np.ndarray
            OpenCV BGR image (uint8).

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            Each element: ``(bbox_xyxy [x1,y1,x2,y2], normed_embedding_512d)``.
        """
        faces = self.app.get(bgr_frame)
        results = []
        for face in faces:
            bbox      = face.bbox.astype(int)      # [x1, y1, x2, y2]
            embedding = face.normed_embedding      # 512-d, already L2-normed
            results.append((bbox, embedding))
        return results

    @staticmethod
    def similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normed embeddings.
        Because both vectors are unit-length, cosine == dot product.
        """
        return float(np.dot(emb_a, emb_b))

    @staticmethod
    def is_match(sim: float) -> bool:
        """Return True if *sim* meets the configured similarity threshold."""
        return sim >= config.SIMILARITY_THRESHOLD

    def best_match(self, faces: list, registered_emb: np.ndarray):
        """
        Find the face in *faces* with the highest similarity to *registered_emb*.

        Returns
        -------
        tuple (bbox, sim) if a match is found, else None.
        """
        best_bbox = None
        best_sim  = -1.0
        for bbox, emb in faces:
            sim = self.similarity(emb, registered_emb)
            if sim > best_sim:
                best_sim  = sim
                best_bbox = bbox
        if best_bbox is not None and self.is_match(best_sim):
            return best_bbox, best_sim
        return None

    # ── Tracking helper ──────────────────────────────────────────────────────

    @staticmethod
    def iou(box_a, box_b) -> float:
        """
        Intersection-over-Union between two ``[x1, y1, x2, y2]`` boxes.
        """
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)
