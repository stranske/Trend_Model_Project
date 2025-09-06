#!/usr/bin/env python3
"""
FastAPI wrapper for Streamlit app with dedicated /health endpoint.

This provides a bulletproof health endpoint that avoids experimental APIs
and returns plain text "OK" as requested in the issue.

The wrapper:
1. Serves /health endpoint directly (plain text "OK")
2. Proxies all other requests to the Streamlit app
3. Provides narrow exception handling for specific cases
"""

import asyncio
import os
import subprocess
import sys
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response

# Configuration - can be overridden by environment variables
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8502"))  # Internal Streamlit port
HEALTH_WRAPPER_PORT = int(os.environ.get("HEALTH_WRAPPER_PORT", "8501"))  # External port
STREAMLIT_STARTUP_TIMEOUT = int(os.environ.get("STREAMLIT_STARTUP_TIMEOUT", "30"))  # Seconds


class HealthWrapper:
    """FastAPI wrapper that provides /health endpoint and proxies to Streamlit."""
    
    def __init__(self):
        self.app = FastAPI(title="Trend Analysis Health Wrapper")
        self.streamlit_process: Optional[subprocess.Popen] = None
        self.http_client = httpx.AsyncClient()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health", response_class=PlainTextResponse)
        async def health_check():
            """Health endpoint that returns plain text OK."""
            return "OK"
        
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_to_streamlit(request: Request, path: str):
            """Proxy all other requests to Streamlit app."""
            try:
                # Build target URL
                target_url = f"http://localhost:{STREAMLIT_PORT}/{path}"
                if request.url.query:
                    target_url += f"?{request.url.query}"
                
                # Forward the request to Streamlit
                response = await self.http_client.request(
                    method=request.method,
                    url=target_url,
                    headers=dict(request.headers),
                    content=await request.body(),
                    timeout=30.0
                )
                
                # Return the response from Streamlit
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type")
                )
                
            except httpx.RequestError as e:
                # Narrow exception handling for connection issues
                return Response(
                    content=f"Service temporarily unavailable: {str(e)}",
                    status_code=503,
                    media_type="text/plain"
                )
            except Exception as e:
                # Narrow exception handling for unexpected errors
                return Response(
                    content=f"Internal server error: {str(e)}",
                    status_code=500,
                    media_type="text/plain"
                )

    async def start_streamlit(self):
        """Start the Streamlit app process."""
        try:
            # Determine the app path - prefer absolute path if available
            app_path = os.environ.get("STREAMLIT_APP_PATH", "src/trend_portfolio_app/app.py")
            
            # Start Streamlit on internal port
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                app_path,
                f"--server.port={STREAMLIT_PORT}",
                "--server.address=127.0.0.1",
                "--server.headless=true"
            ]
            
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for Streamlit to start
            await self._wait_for_streamlit_ready()
            
        except Exception as e:
            raise RuntimeError(f"Failed to start Streamlit: {e}")
    
    async def _wait_for_streamlit_ready(self):
        """Wait for Streamlit to be ready to serve requests."""
        start_time = time.time()
        
        while time.time() - start_time < STREAMLIT_STARTUP_TIMEOUT:
            try:
                response = await self.http_client.get(
                    f"http://localhost:{STREAMLIT_PORT}",
                    timeout=5.0
                )
                if response.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            
            await asyncio.sleep(0.5)
        
        raise RuntimeError(f"Streamlit failed to start within {STREAMLIT_STARTUP_TIMEOUT} seconds")
    
    async def shutdown(self):
        """Clean shutdown of Streamlit process and HTTP client."""
        await self.http_client.aclose()
        
        if self.streamlit_process:
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
                self.streamlit_process.wait()


# Global wrapper instance
wrapper = HealthWrapper()
app = wrapper.app


@app.on_event("startup")
async def startup_event():
    """Start Streamlit process when FastAPI starts."""
    await wrapper.start_streamlit()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown when FastAPI stops."""
    await wrapper.shutdown()


def main():
    """Run the health wrapper server."""
    uvicorn.run(
        "health_wrapper:app",
        host="0.0.0.0",
        port=HEALTH_WRAPPER_PORT,
        access_log=False,  # Reduce noise
        log_level="info"
    )


if __name__ == "__main__":
    main()