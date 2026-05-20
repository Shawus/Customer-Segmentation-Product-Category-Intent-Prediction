import logging
import traceback
import uvicorn
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from run_graph import run_agent
from classification.app import router as classification_router
from clustering.app import router as clustering_router
from utils.task_handler import get_task_status

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Segmentation Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classification_router)
app.include_router(clustering_router)


class APIResponse:
    @staticmethod
    def success(data: Any = None, message: str = "Success", meta: Optional[Dict] = None):
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if data is not None:
            response["data"] = data
        if meta:
            response["meta"] = meta
        return response

    @staticmethod
    def error(message: str, error_code: str = "INTERNAL_ERROR", details: Optional[str] = None):
        response = {
            "success": False,
            "error": {
                "message": message,
                "code": error_code,
                "timestamp": datetime.now().isoformat(),
            },
        }
        if details:
            response["error"]["details"] = details
        return response


@app.get("/")
def health():
    return {"status": "server is running"}


class StatusRequest(BaseModel):
    task_id: str


@app.get("/api/status_check")
async def status_check(request: StatusRequest):
    try:
        task_info = get_task_status(request.task_id)
        if not task_info:
            return APIResponse.error(
                message=f"Task {request.task_id} not found",
                error_code="NOT_FOUND",
            )
        return APIResponse.success(data={
            "task_id": request.task_id,
            "status": task_info.get("status"),
            "message": task_info.get("message"),
            "updated_at": task_info.get("updated_at"),
        })
    except Exception as e:
        logger.error(f"Error checking status for task {request.task_id}: {e}")
        return APIResponse.error(message="Error checking task status", details=str(e))


class AgentRequest(BaseModel):
    query: str


@app.post("/api/run_agent")
async def agent_endpoint(request: AgentRequest):
    try:
        query = request.query
        logger.info(f"API Service received user query: {query}")
        result = run_agent(query)
        return result
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        logger.error(traceback.format_exc())
        return APIResponse.error(
            message="An error occurred while running the agent.",
            details=str(e),
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)
