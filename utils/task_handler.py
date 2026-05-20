import os
import json
import logging
from pathlib import Path
from filelock import FileLock
from typing import Dict, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

TASK_DIR = os.getenv("TASK_STATUS_DIR", "./status_check_cache")
TASK_FILE = Path(TASK_DIR) / "tasks_status.json"
LOCK_FILE = Path(TASK_DIR) / "tasks_status.json.lock"

os.makedirs(TASK_DIR, exist_ok=True)

if not TASK_FILE.exists():
    with open(TASK_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)


def _load_data() -> Dict[str, Any]:
    lock = FileLock(LOCK_FILE)
    try:
        with lock.acquire(timeout=10):
            if not TASK_FILE.exists():
                return {}
            with open(TASK_FILE, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
    except Exception as e:
        logger.error(f"Error loading task data: {e}")
        return {}


def _save_data(data: Dict[str, Any]):
    lock = FileLock(LOCK_FILE)
    try:
        with lock.acquire(timeout=10):
            with open(TASK_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving task data: {e}")


# ================= Main Functions =================
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a specific task by ID."""
    all_tasks = _load_data()
    return all_tasks.get(task_id, {"status": "not_found", "message": "Task ID not found"})


def create_task_record() -> str:
    """Create a new task record and return its ID."""
    task_id = str(uuid.uuid4())
    new_record = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "message": "Task created and pending execution",
        "error_trace": "",
    }

    all_tasks = _load_data()
    all_tasks[task_id] = new_record
    _save_data(all_tasks)

    return task_id


def update_task_status(task_id: str, status: str, message: str = "", error_trace: str = ""):
    """Update task status."""
    all_tasks = _load_data()

    if task_id not in all_tasks:
        all_tasks[task_id] = {}

    all_tasks[task_id].update({
        "status": status,
        "updated_at": datetime.now().isoformat(),
        "message": message,
        "error_trace": error_trace,
    })

    _save_data(all_tasks)
