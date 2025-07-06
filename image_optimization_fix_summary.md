# Image Optimization Fix for Unauthenticated Users

## Problem
The image optimization route was failing when accessed without authentication, causing a `TypeError: 'NoneType' object is not subscriptable` error. This happened because the code was trying to access `req.state.current_user['user_id']` when `current_user` was `None`.

## Root Cause
The issue was in the `key_builder` function in `src/api/routes/utils.py`:
```python
key_builder=lambda req: f"user_settings:{req.state.current_user['user_id']}:{req.state.current_user.get('org_id')}"
```

When there's no authentication, `req.state.current_user` is `None`, causing the subscript access `['user_id']` to fail.

## Solution
Made the following changes to support unauthenticated users with fallback to default bucket settings:

### 1. Fixed the key_builder function
**File**: `src/api/routes/utils.py`
```python
key_builder=lambda req: f"user_settings:{req.state.current_user['user_id'] if req.state.current_user else 'default'}:{req.state.current_user.get('org_id') if req.state.current_user else 'none'}"
```

### 2. Added get_default_user_settings() function
**File**: `src/api/routes/utils.py`
```python
def get_default_user_settings():
    """
    Return default user settings for unauthenticated users.
    These settings will use the default public bucket.
    """
    return UserSettings(
        user_id=None,
        org_id=None,
        api_version="v2",
        spend_limit=5,
        max_spend_limit=5,
        output_visibility="public",
        custom_output_bucket=False,
        enable_custom_output_bucket=False,
        # These will be None, indicating to use default public bucket
        s3_access_key_id=None,
        s3_secret_access_key=None,
        s3_bucket_name=None,
        s3_region=None,
        assumed_role_arn=None,
        hugging_face_token=None,
        workflow_limit=None,
        machine_limit=None,
        always_on_machine_limit=None,
        max_gpu=0,
        credit=0
    )
```

### 3. Updated get_user_settings() function
**File**: `src/api/routes/utils.py`
```python
async def get_user_settings(request: Request, db: AsyncSession):
    # Check if user is authenticated
    current_user = getattr(request.state, 'current_user', None)
    
    if current_user is None:
        # Return default settings for unauthenticated users
        return get_default_user_settings()
    
    # ... rest of the function remains the same
```

## How It Works
1. **Unauthenticated requests**: When there's no authentication, the system will:
   - Use a default cache key ("user_settings:default:none")
   - Return default user settings that use the global public bucket
   - Allow access to public images in the default bucket

2. **Authenticated requests**: Continue to work as before with user-specific settings

3. **Security**: The system still enforces authentication for:
   - Private bucket images 
   - Custom bucket images
   - Any images that require user-specific credentials

## Benefits
- ✅ Fixes the `TypeError` for unauthenticated users
- ✅ Enables image optimization for public images without authentication
- ✅ Maintains security for private/custom bucket images
- ✅ Preserves existing functionality for authenticated users
- ✅ Uses efficient caching with fallback to default settings

## Testing
The fix allows unauthenticated users to access image optimization for public images in the default bucket, while still requiring authentication for private or custom bucket images.