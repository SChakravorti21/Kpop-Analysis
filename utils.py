import os
import errno
import json

def write_json(result, output_path):
    try:
        os.makedirs(os.path.dirname(output_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)