import folder_paths

original_get_filename_list = folder_paths.get_filename_list

# Redefine get_filename_list
def get_filename_list(filename):
    # some nodes break when we try to replace the filename list so instead let them do their output
    if filename == "VHS_video_formats":
        return original_get_filename_list(filename)
    return ["__"+filename+"__"]

# Override the original get_filename_list with the new one
folder_paths.get_filename_list = get_filename_list

import nodes
import asyncio
import execution
from nodes import init_extra_nodes
import server

# Creating a new event loop and setting it as the default loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Creating an instance of PromptServer with the loop
server_instance = server.PromptServer(loop)
execution.PromptQueue(server_instance)

# Initializing custom nodes
init_extra_nodes()

def node_info(node_class):
    obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
    info = {
        'input': obj_class.INPUT_TYPES(),
        'output': obj_class.RETURN_TYPES,
        'output_is_list': getattr(obj_class, 'OUTPUT_IS_LIST', [False] * len(obj_class.RETURN_TYPES)),
        'output_name': getattr(obj_class, 'RETURN_NAMES', obj_class.RETURN_TYPES),
        'name': node_class,
        'display_name': nodes.NODE_DISPLAY_NAME_MAPPINGS.get(node_class, node_class),
        'description': getattr(obj_class, 'DESCRIPTION', ''),
        'category': getattr(obj_class, 'CATEGORY', 'sd'),
        'output_node': getattr(obj_class, 'OUTPUT_NODE', False)
    }
    return info

async def get_object_info():
    out = {}
    for x in nodes.NODE_CLASS_MAPPINGS:
        try:
            out[x] = node_info(x)
        except Exception as _:
            print("oops")
    return out

if __name__ == "__main__":
    import asyncio
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: script.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]
    object_info = asyncio.run(get_object_info())

    with open(output_file, 'w') as f:
        json.dump(object_info, f)