import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib

from src.config import MODELS_DIR, MODELS_METADATA_PATH

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _read_metadata() -> List[Dict]:
    if not MODELS_METADATA_PATH.exists():
        return []
    try:
        with open(MODELS_METADATA_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def _write_metadata(data: List[Dict]):
    with open(MODELS_METADATA_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def save_model(model, model_name: str, metadata: Optional[Dict] = None) -> Path:
    """Save model artifact with a timestamped version and update metadata file.

    Returns the saved model path.
    """
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"model_{model_name}_{ts}.joblib"
    path = MODELS_DIR / filename
    joblib.dump(model, path)

    record = {
        'model_name': model_name,
        'path': str(path.as_posix()),
        'version': ts,
        'created_at_utc': datetime.utcnow().isoformat(),
    }
    if metadata:
        record.update(metadata)

    records = _read_metadata()
    records.append(record)
    _write_metadata(records)

    return path


def get_latest_model_path(model_name: str) -> Optional[Path]:
    records = _read_metadata()
    filtered = [r for r in records if r.get('model_name') == model_name]
    if not filtered:
        return None
    # Sort by version (timestamp string) desc
    filtered.sort(key=lambda r: r.get('version', ''), reverse=True)
    return Path(filtered[0]['path'])


def get_model_by_version(model_name: str, version: str) -> Optional[Path]:
    records = _read_metadata()
    for r in records:
        if r.get('model_name') == model_name and r.get('version') == version:
            return Path(r['path'])
    return None


def list_models() -> List[Dict]:
    return _read_metadata()
