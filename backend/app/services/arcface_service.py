import json
from pathlib import Path
import numpy as np
from PIL import Image

class ArcFaceService:
    """Servicio simplificado de embeddings.

    Para producción puedes reemplazar `extract_embedding` por DeepFace/ArcFace real.
    Esta versión genera un embedding determinista de bajo costo para que el proyecto ejecute desde cero.
    """
    def __init__(self):
        self.store_path = Path(__file__).resolve().parents[1] / "database" / "embeddings.json"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.write_text("{}", encoding="utf-8")

    def extract_embedding(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img.resize((32, 32)).convert("L"), dtype="float32") / 255.0
        emb = arr.flatten()
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm

    def enroll(self, person_id: str, img: Image.Image) -> dict:
        db = self._load()
        emb = self.extract_embedding(img)
        db[person_id] = emb.tolist()
        self._save(db)
        return {"person_id": person_id, "enrolled": True}

    def verify(self, person_id: str, img: Image.Image, threshold: float = 0.65) -> dict:
        db = self._load()
        if person_id not in db:
            return {"verified": False, "person_id": person_id, "similarity": None, "threshold": threshold}
        emb = self.extract_embedding(img)
        ref = np.array(db[person_id], dtype="float32")
        sim = float(np.dot(emb, ref) / ((np.linalg.norm(emb) * np.linalg.norm(ref)) + 1e-8))
        return {"verified": sim >= threshold, "person_id": person_id, "similarity": round(sim, 6), "threshold": threshold}

    def _load(self):
        return json.loads(self.store_path.read_text(encoding="utf-8"))

    def _save(self, db):
        self.store_path.write_text(json.dumps(db, indent=2), encoding="utf-8")
