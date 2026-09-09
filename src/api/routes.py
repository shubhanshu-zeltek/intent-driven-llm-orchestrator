from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse
from orchestrator import run


router = APIRouter(
    prefix="/api/v1",
    tags=["LLM"]
)

@router.post("/query", response_model=QueryResponse,)
def process_query(request: QueryRequest) -> QueryResponse:
    try:
        response = run(
            user_query=request.query,
            validate_and_fix=True
        )

        return QueryResponse(
            response=response
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {e}"
        ) from e
