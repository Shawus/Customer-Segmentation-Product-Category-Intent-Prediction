"""
Clustering module API endpoints.

Provides endpoints for:
- Data extraction from enterprise sources (background task)
- Model training (background task)
"""
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from clustering.data_cleaning import ColumnConfig, ClusteringDataCleaning
from clustering.feature_engineering import ClusteringFeatureEngineering
from clustering.load_data import ClusteringLoadData
from clustering.clustering_model import ClusteringModel
from utils.task_handler import create_task_record, update_task_status

logger = logging.getLogger(__name__)

# Column configuration for shipment data
shipment_config = ColumnConfig(
    int_cols=["orderym"],
    float_cols=["us_amt", "sourcecost"],
    string_cols=[
        "endcustomer", "endcustomername", "customername", "orderno",
        "Tran_Type", "region", "sector", "s_sector", "site",
        "pd", "product_category_2", "product", "m_type",
    ],
)

# Initialize pipeline components
load_data = ClusteringLoadData()
shipment_cleaner = ClusteringDataCleaning(columns=shipment_config)
feature_engineering = ClusteringFeatureEngineering()
clustering_model = ClusteringModel()

router = APIRouter(prefix="/clustering", tags=["clustering"])


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
    return {"status": "clustering service running"}


def _run_update_data(task_id: str):
    """Background task: extract and clean raw data from enterprise sources."""
    try:
        update_task_status(task_id, "IN PROGRESS", "Data extraction started")
        origin_shipment = load_data.get_shipment_data()
        logger.info("Shipment data extraction complete.")
        update_task_status(task_id, "FINISHED", "Data extraction completed")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        update_task_status(task_id, "FAILED", "Data extraction failed", str(e))


@router.post("/update_data")
async def update_data(background_tasks: BackgroundTasks):
    try:
        task_id = create_task_record()
        background_tasks.add_task(_run_update_data, task_id)
        return APIResponse.success(
            message="Data extraction task started.",
            data={"task_id": task_id},
        )
    except Exception as e:
        logger.error(f"Error during data update: {e}")
        return APIResponse.error(message="Error during data update.", details=str(e))


def _run_training(task_id: str):
    """Background task: run full clustering training pipeline."""
    try:
        update_task_status(task_id, "IN PROGRESS", "Clustering training started")
        clustering_model.run_clustering_training()
        update_task_status(task_id, "FINISHED", "Clustering training completed")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
        update_task_status(task_id, "FAILED", "Training failed", str(e))


@router.post("/train")
async def model_training(background_tasks: BackgroundTasks):
    try:
        task_id = create_task_record()
        background_tasks.add_task(_run_training, task_id)
        return APIResponse.success(
            message="Clustering training task started.",
            data={"task_id": task_id},
        )
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return APIResponse.error(message="Error starting training.", details=str(e))


class InferenceRequest(BaseModel):
    erp_id: str
    region: str


@router.post("/inference")
async def model_inference(payload: InferenceRequest):
    """Run clustering inference for a given customer."""
    try:
        result = clustering_model.inference(
            erp_id=payload.erp_id,
            region=payload.region,
            top_K=3,
        )
        return APIResponse.success(data=result)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return APIResponse.error(message="Inference failed.", details=str(e))
