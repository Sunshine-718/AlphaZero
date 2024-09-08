import importlib

def load(env_name):
    try:
        return importlib.import_module(f"environments.{env_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Environment '{env_name}' not found.")
    except AttributeError:
        raise ValueError(f"Environment '{env_name}' does not implement the required function.")
    