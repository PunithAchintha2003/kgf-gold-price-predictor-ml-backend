#!/usr/bin/env python3
"""
Diagnostic script to check API issues
"""
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 80)
print("API DIAGNOSTIC TOOL")
print("=" * 80)

# Test 1: Import main app
print("\n1. Testing app import...")
try:
    from backend.app.main import app
    print("✅ App imported successfully")
except Exception as e:
    print(f"❌ Failed to import app: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check routes
print("\n2. Testing route registration...")
try:
    routes = [route.path for route in app.routes]
    print(f"✅ Found {len(routes)} routes")
    print(f"   Sample routes: {routes[:5]}")
except Exception as e:
    print(f"❌ Failed to get routes: {e}")
    traceback.print_exc()

# Test 3: Check dependencies
print("\n3. Testing dependency injection...")
try:
    from backend.app.core.dependencies import (
        get_market_data_service,
        get_prediction_service,
        get_prediction_repo,
        get_exchange_service
    )
    print("✅ Dependency functions imported")

    # Check if services are set
    import asyncio

    async def test_deps():
        try:
            market_service = await get_market_data_service()
            print(f"✅ Market data service: {type(market_service).__name__}")
        except RuntimeError as e:
            print(f"⚠️  Market data service not initialized: {e}")

        try:
            prediction_service = await get_prediction_service()
            print(f"✅ Prediction service: {type(prediction_service).__name__}")
        except RuntimeError as e:
            print(f"⚠️  Prediction service not initialized: {e}")

        try:
            prediction_repo = await get_prediction_repo()
            print(f"✅ Prediction repo: {type(prediction_repo).__name__}")
        except RuntimeError as e:
            print(f"⚠️  Prediction repo not initialized: {e}")

        try:
            exchange_service = await get_exchange_service()
            print(f"✅ Exchange service: {type(exchange_service).__name__}")
        except RuntimeError as e:
            print(f"⚠️  Exchange service not initialized: {e}")

    asyncio.run(test_deps())
except Exception as e:
    print(f"❌ Failed to test dependencies: {e}")
    traceback.print_exc()

# Test 4: Check specific routes
print("\n4. Testing route handlers...")
try:
    from backend.app.api.v1.routes import xauusd, health, exchange
    print("✅ Route modules imported")

    # Check if routes have handlers
    xauusd_routes = [r.path for r in xauusd.router.routes]
    print(f"✅ XAUUSD routes: {len(xauusd_routes)} routes")
    print(f"   Routes: {xauusd_routes}")

except Exception as e:
    print(f"❌ Failed to test routes: {e}")
    traceback.print_exc()

# Test 5: Check for common issues
print("\n5. Checking for common issues...")
issues = []

# Check if all dependencies are properly imported
try:
    from backend.app.api.v1.routes.xauusd import router
    import inspect
    import asyncio
    for route in router.routes:
        if hasattr(route, 'endpoint'):
            sig = inspect.signature(route.endpoint)
            params = list(sig.parameters.values())
            for param in params:
                if param.default != inspect.Parameter.empty:
                    if hasattr(param.default, 'dependency'):
                        dep = param.default.dependency
                        if dep:
                            try:
                                # Try to get the dependency function
                                if asyncio.iscoroutinefunction(dep):
                                    print(
                                        f"   ✅ Route {route.path}: async dependency {dep.__name__}")
                                else:
                                    print(
                                        f"   ✅ Route {route.path}: sync dependency {dep.__name__}")
                            except Exception as e:
                                issues.append(
                                    f"Route {route.path}: dependency issue - {e}")
except Exception as e:
    issues.append(f"Route inspection failed: {e}")

if issues:
    print("⚠️  Issues found:")
    for issue in issues:
        print(f"   - {issue}")
else:
    print("✅ No obvious issues found")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)


