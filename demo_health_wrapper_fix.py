#!/usr/bin/env python3
"""Demo script showing the uvicorn module path fix.

This script demonstrates the difference between the incorrect and correct
module paths for uvicorn, showing that the fix resolves the ModuleNotFoundError.
"""

import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))


def demo_module_import_issue():
    """Demonstrate the module import issue and its resolution."""
    print("🔍 Demonstrating uvicorn module path fix")
    print("=" * 50)
    
    # Show that the old path would fail
    print("❌ OLD (incorrect) uvicorn call:")
    print('   uvicorn.run("health_wrapper:app", ...)')
    print("   → Would try to import 'health_wrapper' at top level")
    print("   → Results in: ModuleNotFoundError: No module named 'health_wrapper'")
    print()
    
    # Show that our fix works
    print("✅ NEW (fixed) uvicorn call:")
    print('   uvicorn.run("trend_portfolio_app.health_wrapper:app", ...)')
    print("   → Imports using fully qualified module name")
    print("   → Module can be found and imported successfully")
    print()
    
    # Demonstrate that our module can be imported
    print("📦 Testing module import:")
    try:
        from trend_portfolio_app import health_wrapper
        print(f"   ✅ Module imported: {health_wrapper.__name__}")
        print(f"   ✅ Module path: {health_wrapper.__file__}")
        
        # Show the uvicorn string that would be used
        module_string = "trend_portfolio_app.health_wrapper:app"
        print(f"   ✅ Correct uvicorn module string: '{module_string}'")
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
    
    print()
    print("🚀 Testing module execution:")
    print("   Command: python -m trend_portfolio_app.health_wrapper")
    print("   → This now works (fails gracefully if dependencies missing)")
    
    # Test the execution path
    import subprocess
    import os
    
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(src_path)
        result = subprocess.run([
            sys.executable, "-m", "trend_portfolio_app.health_wrapper"
        ], capture_output=True, text=True, timeout=5, env=env)
        
        if "uvicorn is required" in result.stderr:
            print("   ✅ Module executes correctly (missing deps detected)")
        elif result.returncode == 0:
            print("   ✅ Module executed successfully")
        else:
            print(f"   ⚠️  Module execution result: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  Module started (stopped due to timeout)")
    except Exception as e:
        print(f"   ❌ Execution test failed: {e}")
    
    print()
    print("🎯 Summary:")
    print("   The fix ensures uvicorn uses the fully qualified module name")
    print("   'trend_portfolio_app.health_wrapper:app' instead of 'health_wrapper:app'")
    print("   This resolves the ModuleNotFoundError when starting the service")


if __name__ == "__main__":
    demo_module_import_issue()