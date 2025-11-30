# FastAPI Lifespan Migration

This document explains the migration from deprecated `@app.on_event()` decorators to the modern lifespan context manager pattern in FastAPI.

## Changes Made

### 1. Updated Dependencies

Added FastAPI and Uvicorn to `pyproject.toml`:
```toml
[project]
dependencies = [
    # ... other deps ...
    "fastapi>=0.104.0",
    "uvicorn[standard]",
]
```

### 2. Replaced Basic HTTP Server with FastAPI

**Before:** The `src/trend_analysis/api_server/__init__.py` used Python's built-in `HTTPServer` with basic request handling.

**After:** Implemented a proper FastAPI application with:
- Modern lifespan context manager for startup/shutdown events
- Structured API endpoints with automatic OpenAPI documentation
- Proper async/await support
- Better error handling and response formatting

### 3. Lifespan Context Manager Implementation

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events.
    
    Modern FastAPI approach using context manager instead of deprecated
    @app.on_event() decorators.
    """
    # Startup logic
    logger.info("Starting up trend analysis API server")
    # Initialize any resources here (database connections, ML models, etc.)
    
    yield  # Application is running
    
    # Shutdown logic  
    logger.info("Shutting down trend analysis API server")
    # Clean up resources here

# Create FastAPI app with lifespan context manager
app = FastAPI(
    title="Trend Analysis API",
    description="REST API for volatility-adjusted trend portfolio analysis",
    version="1.0.0",
    lifespan=lifespan,
)
```

## Benefits of the Lifespan Pattern

1. **Modern API**: Uses the recommended approach for FastAPI 0.93+
2. **Future-proof**: The `@app.on_event()` decorators are deprecated and will be removed
3. **Better Resource Management**: Clear startup and shutdown phases for resource initialization/cleanup
4. **Async Context**: Proper async context management for startup/shutdown operations
5. **Testing**: Easier to test with proper lifecycle management

## API Endpoints

The new FastAPI server provides:

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint (compatible with existing Docker health checks)
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)
- `GET /openapi.json` - OpenAPI specification

## Docker Compatibility

The changes maintain full compatibility with the existing Docker setup:
- Still runs on port 8000 as configured in `docker-compose.yml`
- Health check endpoint remains at `/health`
- Same command structure: `python -m trend_analysis.api_server`

## Usage

### Development
```bash
cd /path/to/project
source .venv/bin/activate
python -m trend_analysis.api_server
```

### Docker
```bash
docker-compose up api
```

### Testing
```bash
pytest tests/test_api_server.py
```

## Migration Notes

- **No Breaking Changes**: The API maintains the same external interface
- **Enhanced Features**: Added automatic API documentation and better error handling
- **Performance**: Better async performance with proper FastAPI implementation
- **Extensibility**: Easy to add new endpoints and middleware

## Future Enhancements

With FastAPI in place, the API can be easily extended to support:
- Authentication and authorization
- Rate limiting
- Request validation with Pydantic models
- WebSocket support for real-time updates
- Integration with the trend analysis pipeline
- File upload endpoints for CSV data
- Background task processing