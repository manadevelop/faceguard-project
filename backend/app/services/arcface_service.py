import json
from pathlib import Path
import numpy as np
from PIL import Image


# Servicio simplificado para enrolamiento y verificación facial mediante embeddings.
class ArcFaceService:
    """Servicio simplificado de embeddings.

    Para producción puedes reemplazar `extract_embedding` por DeepFace/ArcFace real.
    Esta versión genera un embedding determinista de bajo costo para que el proyecto ejecute desde cero.
    """

    def __init__(self):
        # Define la ruta del archivo JSON donde se guardan los embeddings.
        self.store_path = Path(__file__).resolve().parents[1] / "database" / "embeddings.json"

        # Crea la carpeta de base de datos local si no existe.
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # Inicializa el archivo de embeddings si todavía no existe.
        if not self.store_path.exists():
            self.store_path.write_text("{}", encoding="utf-8")

    def extract_embedding(self, img: Image.Image) -> np.ndarray:
        # Convierte la imagen a escala de grises y la reduce a 32x32.
        arr = np.asarray(img.resize((32, 32)).convert("L"), dtype="float32") / 255.0

        # Aplana la imagen para formar un vector de características.
        emb = arr.flatten()

        # Calcula la norma del vector evitando división por cero.
        norm = np.linalg.norm(emb) + 1e-8

        # Retorna el embedding normalizado.
        return emb / norm

    def enroll(self, person_id: str, img: Image.Image) -> dict:
        # Carga la base local de embeddings.
        db = self._load()

        # Extrae el embedding de la imagen recibida.
        emb = self.extract_embedding(img)

        # Asocia el embedding al identificador de persona.
        db[person_id] = emb.tolist()

        # Guarda la base actualizada en disco.
        self._save(db)

        # Devuelve confirmación del enrolamiento.
        return {"person_id": person_id, "enrolled": True}

    def verify(self, person_id: str, img: Image.Image, threshold: float = 0.65) -> dict:
        # Carga la base local de embeddings.
        db = self._load()

        # Retorna no verificado si el person_id no existe.
        if person_id not in db:
            return {"verified": False, "person_id": person_id, "similarity": None, "threshold": threshold}

        # Extrae el embedding de la imagen enviada para verificación.
        emb = self.extract_embedding(img)

        # Recupera el embedding registrado para el person_id.
        ref = np.array(db[person_id], dtype="float32")

        # Calcula la similitud coseno entre el embedding nuevo y el registrado.
        sim = float(np.dot(emb, ref) / ((np.linalg.norm(emb) * np.linalg.norm(ref)) + 1e-8))

        # Devuelve el resultado de verificación según el threshold.
        return {"verified": sim >= threshold, "person_id": person_id, "similarity": round(sim, 6), "threshold": threshold}

    def _load(self):
        # Lee y convierte el archivo JSON de embeddings a diccionario.
        return json.loads(self.store_path.read_text(encoding="utf-8"))

    def _save(self, db):
        # Guarda el diccionario de embeddings en formato JSON.
        self.store_path.write_text(json.dumps(db, indent=2), encoding="utf-8")