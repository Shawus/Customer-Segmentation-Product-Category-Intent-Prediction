"""
Classification module API endpoints.

Provides endpoints for:
- Model training (background task)
- Model inference
- Data updates
"""
import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from classification.data_preprocessing import ClassificationDataPreprocessor
from classification.classification_model import ClassificationModel
from utils.task_handler import create_task_record, update_task_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classification", tags=["classification"])


class APIResponse:
    @staticmethod
    def success(data: Any = None, message: str = "Success", meta: Optional[Dict] = None):
        response = {"success": True, "message": message, "timestamp": datetime.now().isoformat()}
        if data is not None:
            response["data"] = data
        if meta:
            response["meta"] = meta
        return response

    @staticmethod
    def error(message: str, error_code: str = "INTERNAL_ERROR", details: Optional[str] = None):
        response = {
            "success": False,
            "error": {"message": message, "code": error_code, "timestamp": datetime.now().isoformat()},
        }
        if details:
            response["error"]["details"] = details
        return response


@router.get("/health")
def health():
    return {"status": "classification service running"}


def _run_update_data(task_id: str):
    """
    Background task for data update.

    From my own case, this part calls the API to refresh product metadata.
    Replace with your own data source integration.
    """
    try:
        update_task_status(task_id, "IN PROGRESS", "Data update started")
        # ---------------------------------------------------------------
        # NOTE: API Calling Here
        # ---------------------------------------------------------------
        update_task_status(task_id, "FINISHED", "Data update completed")
    except Exception as e:
        logger.error(f"Error during data update: {e}")
        update_task_status(task_id, "FAILED", "Data update failed", str(e))


@router.get("/update_data")
async def update_data(background_tasks: BackgroundTasks):
    try:
        task_id = create_task_record()
        background_tasks.add_task(_run_update_data, task_id)
        return APIResponse.success(
            message="Data update task started.",
            data={"task_id": task_id},
        )
    except Exception as e:
        logger.error(f"Error starting data update task: {e}")
        return APIResponse.error(message="Error starting data update task.", details=str(e))


def _run_model_training(task_id: str, config: Optional[Dict]):
    """Background task for model training."""
    try:
        update_task_status(task_id, "IN PROGRESS", "Model training started")
        start_time = time.time()

        preprocessor = ClassificationDataPreprocessor()
        data = preprocessor.process_data_for_all_sectors()

        trainer = ClassificationModel()
        artifacts = trainer.train_all(data, config=config)

        evaluation_results = trainer.evaluate_all(data, artifacts=artifacts)

        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.1f}s")

        result = {
            "trained_models": list(artifacts.keys()),
            "training_time_seconds": elapsed_time,
            "evaluation_results": evaluation_results,
        }
        update_task_status(task_id, "FINISHED", "Model training completed", result)

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        logger.error(traceback.format_exc())
        update_task_status(task_id, "FAILED", "Model training failed", str(e))


class TrainConfig(BaseModel):
    config: Optional[Dict[str, Any]] = None


@router.post("/train")
async def model_training(background_tasks: BackgroundTasks, payload: TrainConfig):
    try:
        task_id = create_task_record()
        background_tasks.add_task(_run_model_training, task_id, payload.config)
        return APIResponse.success(
            data={"task_id": task_id},
            message="Model training task started.",
        )
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return APIResponse.error(message="Error starting training.", details=str(e))
