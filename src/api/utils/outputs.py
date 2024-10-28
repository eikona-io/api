from typing import Any, Dict, List, Optional

custom_output_nodes: Dict[str, Dict[str, str]] = {
    "ComfyDeployStdOutputImage": {"output_id": "string"},
    "ComfyDeployStdOutputAny": {"output_id": "string"},
}


def get_outputs_from_workflow(
    workflow: Optional[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    if not workflow:
        return None

    outputs = []

    # Iterate through nodes in the workflow
    for node in workflow.get("nodes", []):
        node_type = node.get("type")

        # Check if node type exists in custom_output_nodes
        if node_type in custom_output_nodes:
            # Get output_id from widgets_values[0]
            output_id = node.get("widgets_values", [""])[0]

            # Add to outputs array
            outputs.append({"class_type": node_type, "output_id": output_id})

    return outputs