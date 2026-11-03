import json
import os
import threading
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "experiment_logs")
LOG_FILE = os.path.join(LOG_DIR, "experiment_log.jsonl")

_lock = threading.Lock()


def log_event(event_type, **data):
    os.makedirs(LOG_DIR, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data
    }

    with _lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
