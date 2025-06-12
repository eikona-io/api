# Machine Deactivation Feature

## Overview
ComfyDeploy supports machine deactivation through a `disabled` boolean field in the machines table. This allows users to temporarily disable machines without deleting them, preserving configurations for future use.

## Database Schema
The `disabled` field is defined in the machines table:
```sql
ALTER TABLE "comfyui_deploy"."machines" ADD COLUMN "disabled" boolean DEFAULT false NOT NULL;
```

## Usage
- Disabled machines are not deleted but are marked as inactive
- Machine configurations, versions, and associated data are preserved
- Disabled machines can be re-enabled by setting `disabled = false`
- The UI shows disabled machines with a "Disabled" badge and red X icon

## API Operations
To disable a machine, update the machine record:
```
PATCH /api/machine/{machine_id}
{
  "disabled": true
}
```

To re-enable a machine:
```
PATCH /api/machine/{machine_id}
{
  "disabled": false
}
```

## UI Indicators
- Red X icon in machine status (shown in machine-status.tsx)
- "Disabled" badge in machine list
- Machines remain visible in the list but clearly marked as inactive
- Disabled state is checked via `props.machine.disabled`

## Implementation Details
- The disabled field is part of the base Machine model
- UI components check `machine.disabled` to show appropriate status
- Disabled machines are still accessible for viewing but operations may be restricted
- The feature preserves all machine data while marking it as temporarily inactive

## Benefits
- Allows users to manage machine quota without losing configurations
- Preserves machine history and versions for reproducibility
- Provides a reversible alternative to machine deletion
- Helps with cost management by temporarily disabling unused machines
