# Machine Environment Configuration

## Overview
ComfyDeploy machine export/import functionality uses a standardized environment format that aligns with workflow exports. This ensures consistency across the platform and enables seamless machine configuration management.

## Environment Object Structure
The machine environment contains all runtime and deployment configuration:

```json
{
  "environment": {
    "comfyui_version": "string",
    "gpu": "string",
    "docker_command_steps": [],
    "max_containers": 2,
    "install_custom_node_with_gpu": false,
    "run_timeout": 300,
    "scaledown_window": 60,
    "extra_docker_commands": [],
    "base_docker_image": "string",
    "python_version": "3.11",
    "extra_args": "string",
    "prestart_command": "string",
    "min_containers": 0,
    "machine_hash": "string",
    "disable_metadata": true,
    "allow_concurrent_inputs": 1,
    "machine_builder_version": "4",
    "version": 1
  }
}
```

## Field Mappings
The environment format uses standardized field names:
- `max_containers` - Maximum concurrent containers (from `concurrency_limit`)
- `scaledown_window` - Idle timeout before scaling down (from `idle_timeout`)
- `min_containers` - Minimum warm containers (from `keep_warm`)
- `comfyui_version` - ComfyUI version for the machine
- `docker_command_steps` - Build steps and custom nodes
- `gpu` - GPU configuration and type

## Export Format
Machine exports include both base machine data and environment configuration:
```json
{
  "name": "machine-name",
  "type": "comfy-deploy-serverless",
  "environment": {
    // Environment object as defined above
  },
  "export_version": "1.0",
  "exported_at": "2024-01-01T00:00:00.000Z"
}
```

## Import Compatibility
The import function supports both:
- **New format**: Environment object with standardized field names
- **Legacy format**: Direct fields for backward compatibility
- Runtime fields like endpoint are ignored and replaced with a default value based on the machine type

## Benefits
- Consistent format across workflow and machine exports
- Standardized field naming for better maintainability
- Backward compatibility with existing exports
- Complete environment preservation for reproducibility
