# Machine Export/Import API Endpoints - COM-1322

## Overview
This PR implements machine export/import functionality for the ComfyDeploy platform, allowing users to backup and restore machine configurations as JSON files. **Updated to align with workflow export environment format and include automatic machine build triggering after import.**

## Changes Made

### Backend API Endpoints
- **GET `/machine/{machine_id}/export`**: Export machine configurations as JSON
  - **Refactored to use workflow export environment format structure**
  - Creates standardized `environment` object with fields like `max_containers`, `scaledown_window`, `min_containers`
  - Supports optional version parameter for specific version exports
  - Includes comprehensive machine data: snapshot, models, dependencies, environment settings
  - Handles both classic and serverless machine types
  - Respects existing permission model for machine operations

- **POST `/machine/import`**: Import machines from JSON files
  - **Updated to handle both new environment format and legacy format for backward compatibility**
  - Validates required fields and JSON structure
  - Handles name conflicts with timestamp-based renaming
  - Creates machine versions for serverless machines
  - **Automatically triggers machine build for serverless machines after successful import**
  - Comprehensive error handling and validation

### Environment Format Alignment
- **Aligned machine export format with workflow export environment structure**
- Field mappings: `concurrency_limit` → `max_containers`, `idle_timeout` → `scaledown_window`, `keep_warm` → `min_containers`
- Maintains JSON compatibility and UUID string conversion
- Ensures consistency across workflow and machine exports

### Machine Build Integration
- **Added build trigger logic to import function following workflow import patterns**
- Constructs `BuildMachineItem` with all required parameters after successful machine creation
- Uses background tasks to trigger `build_logic` without blocking import response
- Only triggers builds for serverless machines (`comfy-deploy-serverless` type)
- Follows existing patterns from machine creation and update functions

### Machine Environment Documentation
- **Updated `MACHINE_DEACTIVATION.md` to focus on machine environment configuration**
- Documents the standardized environment object structure
- Explains field mappings and export/import format
- Covers backward compatibility and benefits of the new format

## Implementation Details
- **Export format now uses environment object matching workflow export pattern**
- **Import function supports both new environment format and legacy format**
- Export format includes metadata like export version and timestamp for future compatibility
- Import handles both classic and serverless machine types with proper version creation
- **Build logic ensures imported serverless machines are automatically deployed and ready for use**
- File naming follows pattern: `machine-{name}-{date}.json`
- UUID fields are properly serialized to strings for JSON compatibility
- Comprehensive error handling with clear error messages

## Testing
- Tested export functionality with existing machines
- Verified import creates machines with correct configurations
- **Confirmed build trigger works for imported serverless machines**
- **Verified new environment format structure matches workflow exports**
- **Tested backward compatibility with legacy import format**
- Tested permission enforcement and error handling
- Validated JSON format compatibility

## Related
- Ticket: COM-1322
- Link to Devin run: https://app.devin.ai/sessions/f493883f86c543acbbc3d0781214b047
- Requested by: Benny (benny@comfydeploy.com)
