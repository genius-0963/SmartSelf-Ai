"""
SmartShelf AI - Enhanced Main Application
Production-ready FastAPI server with comprehensive features
"""

import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import time

# Import configuration and services
from core.config import settings, LOGGING_CONFIG, CORS_CONFIG
from services.cache_service import cache_service
from services.metrics_service import metrics_service
from services.analytics_service import analytics_service
from api.v1 import chat_router

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting SmartShelf AI Enhanced Backend...")
    
    try:
        # Start metrics monitoring
        await metrics_service.start_monitoring()
        logger.info("Metrics monitoring started")
        
        # Test cache connection
        cache_health = cache_service.health_check()
        if cache_health["status"] == "healthy":
            logger.info("Redis cache connection established")
        else:
            logger.warning(f"Redis cache unavailable: {cache_health.get('error', 'Unknown error')}")
        
        # Log startup info
        logger.info(f"SmartShelf AI Backend v{settings.app_version} started successfully")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"API listening on {settings.api_host}:{settings.api_port}")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down SmartShelf AI Backend...")
        
        try:
            # Stop metrics monitoring
            metrics_service.stop_monitoring()
            
            # Close cache connection
            cache_service.close()
            
            # Close Amazon scraper
            from integrations.amazon_scraper import amazon_scraper
            await amazon_scraper.close()
            
            logger.info("Shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Production-ready AI-powered retail analytics platform",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    **CORS_CONFIG
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware for request timing and metrics
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add request timing and metrics collection"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Record metrics (only for API endpoints, not for static files)
    if request.url.path.startswith("/api/"):
        metrics_service.record_request(
            endpoint=request.url.path,
            method=request.method,
            response_time=process_time,
            status_code=response.status_code
        )
    
    return response

# Include API routers
app.include_router(chat_router, prefix=settings.api_prefix)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with system information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
        "timestamp": time.time(),
        "docs": "/docs" if settings.debug else "Documentation disabled in production",
        "api_prefix": settings.api_prefix
    }

# Enhanced health check
@app.get("/health", tags=["health"])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Get system metrics
        system_metrics = metrics_service.get_metrics()
        
        # Check cache health
        cache_health = cache_service.health_check()
        
        # Check chat service health
        from services.chat_service import chat_service
        chat_health = chat_service.health_check()
        
        # Determine overall health
        overall_status = "healthy"
        if cache_health["status"] != "healthy":
            overall_status = "degraded"
        if chat_health["status"] != "healthy":
            overall_status = "degraded"
        
        # Check error rate
        if system_metrics["error_rate"] > 0.1:  # 10% error rate threshold
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "version": settings.app_version,
            "environment": settings.environment,
            "uptime_seconds": system_metrics["uptime_seconds"],
            "components": {
                "cache": cache_health,
                "chat_service": chat_health,
                "analytics": {"status": "operational"}
            },
            "metrics": {
                "total_requests": system_metrics["total_requests"],
                "error_rate": system_metrics["error_rate"],
                "active_connections": system_metrics["active_connections"],
                "avg_response_time_ms": system_metrics["avg_response_time_ms"]
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )

# Metrics endpoint
@app.get("/metrics", tags=["metrics"])
async def get_metrics():
    """Get comprehensive system metrics"""
    try:
        return metrics_service.get_metrics()
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

# Analytics endpoint
@app.get("/analytics", tags=["analytics"])
async def get_analytics(days: int = 7):
    """Get system analytics"""
    try:
        if days > 90:
            days = 90
        
        return await analytics_service.get_conversation_analytics(days)
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error" if not settings.debug else str(exc),
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

# Development-only endpoints
if settings.debug:
    @app.get("/debug/info", tags=["debug"])
    async def debug_info():
        """Debug information endpoint (development only)"""
        return {
            "settings": {
                "app_name": settings.app_name,
                "environment": settings.environment,
                "debug": settings.debug,
                "api_host": settings.api_host,
                "api_port": settings.api_port,
                "cache_ttl": settings.cache_ttl,
                "max_concurrent_requests": settings.max_concurrent_requests
            },
            "cache_health": cache_service.health_check(),
            "system_metrics": metrics_service.get_metrics()
        }
    
    @app.post("/debug/cache/clear", tags=["debug"])
    async def clear_cache():
        """Clear cache (development only)"""
        try:
            cleared_count = await cache_service.delete_pattern("*")
            return {"message": f"Cleared {cleared_count} cache entries"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "enhanced_main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=settings.debug
    )
