from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from bson.objectid import ObjectId
import logging

from app.config import FRONTEND_URL
from app.models import SubmissionRequest, SubmissionResponse, SubmissionListResponse, AdminSubmissionResponse
from app.database import Database
from app.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Feedback System API", version="1.0.0")

# Configure CORS
cors_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://fynd-assessment-dashboard-rfju-e84yijwfl.vercel.app",
]

# Add FRONTEND_URL from env if set
if FRONTEND_URL and FRONTEND_URL not in cors_origins:
    cors_origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_service = LLMService()

@app.on_event("startup")
async def startup():
    """Connect to database on startup"""
    try:
        Database.connect()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")

@app.on_event("shutdown")
async def shutdown():
    """Disconnect from database on shutdown"""
    Database.disconnect()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/api/submissions", response_model=SubmissionResponse, status_code=201)
async def create_submission(request: SubmissionRequest):
    """
    Create a new feedback submission.
    
    Request body:
    {
        "rating": 1-5,
        "review": "user review text"
    }
    
    Returns:
    {
        "id": "submission_id",
        "rating": int,
        "review": str,
        "ai_response": str,
        "ai_summary": str,
        "recommended_actions": str,
        "timestamp": datetime,
        "status": "success"
    }
    """
    try:
        # Validate input
        if not request.review or not request.review.strip():
            raise HTTPException(status_code=400, detail="Review cannot be empty")
        
        if len(request.review) > 5000:
            raise HTTPException(status_code=400, detail="Review is too long (max 5000 characters)")

        # Generate AI responses (server-side only)
        logger.info(f"Generating AI response for rating {request.rating}")
        ai_response = llm_service.generate_user_response(request.rating, request.review)
        
        logger.info("Generating review summary")
        ai_summary = llm_service.generate_summary(request.review)
        
        logger.info("Generating recommended actions")
        recommended_actions = llm_service.generate_recommended_actions(request.rating, request.review)

        # Prepare submission document
        submission_doc = {
            "rating": request.rating,
            "review": request.review.strip(),
            "ai_response": ai_response,
            "ai_summary": ai_summary,
            "recommended_actions": recommended_actions,
            "timestamp": datetime.utcnow(),
            "status": "success"
        }

        # Store in database
        submission_id = Database.insert_submission(submission_doc)
        logger.info(f"Submission created with ID: {submission_id}")

        # Retrieve and return the created submission
        submission = Database.get_submission(submission_id)
        
        return SubmissionResponse(
            id=str(submission["_id"]),
            rating=submission["rating"],
            review=submission["review"],
            ai_response=submission["ai_response"],
            ai_summary=submission["ai_summary"],
            recommended_actions=submission["recommended_actions"],
            timestamp=submission["timestamp"],
            status=submission["status"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating submission: {e}")
        raise HTTPException(status_code=500, detail="Failed to process submission. Please try again.")

@app.get("/api/submissions", response_model=SubmissionListResponse)
async def get_submissions(
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    rating: int = Query(None, ge=1, le=5)
):
    """
    Get all submissions for admin dashboard.
    
    Query parameters:
    - limit: number of submissions to return (default 50, max 100)
    - skip: number of submissions to skip for pagination (default 0)
    - rating: optional filter by rating (1-5)
    
    Returns:
    {
        "total": int,
        "submissions": [
            {
                "id": "submission_id",
                "rating": int,
                "review": str,
                "ai_summary": str,
                "recommended_actions": str,
                "timestamp": datetime,
                "status": str
            }
        ]
    }
    """
    try:
        submissions, total = Database.get_all_submissions(limit=limit, skip=skip, rating_filter=rating)
        
        admin_submissions = [
            AdminSubmissionResponse(
                id=str(sub["_id"]),
                rating=sub["rating"],
                review=sub["review"],
                ai_summary=sub.get("ai_summary", ""),
                recommended_actions=sub.get("recommended_actions", ""),
                timestamp=sub["timestamp"],
                status=sub.get("status", "success")
            )
            for sub in submissions
        ]
        
        return SubmissionListResponse(total=total, submissions=admin_submissions)
    
    except Exception as e:
        logger.error(f"Error retrieving submissions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve submissions")

@app.get("/api/submissions/{submission_id}", response_model=SubmissionResponse)
async def get_submission(submission_id: str):
    """
    Get a specific submission by ID.
    
    Returns full submission details including AI-generated response.
    """
    try:
        submission = Database.get_submission(submission_id)
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        return SubmissionResponse(
            id=str(submission["_id"]),
            rating=submission["rating"],
            review=submission["review"],
            ai_response=submission.get("ai_response", ""),
            ai_summary=submission.get("ai_summary", ""),
            recommended_actions=submission.get("recommended_actions", ""),
            timestamp=submission["timestamp"],
            status=submission.get("status", "success")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving submission: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve submission")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
