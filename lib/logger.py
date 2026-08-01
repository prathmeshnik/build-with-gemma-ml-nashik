import json
import os
from datetime import datetime


def save_transcript(config_title, history):
    os.makedirs("interviews", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"interviews/{ts}.json"
    with open(path, "w") as f:
        json.dump(
            {"config": config_title, "timestamp": ts, "turns": history}, f, indent=2
        )
    return path
