import importlib

def load(env_name):
    return importlib.import_module(f"environments.{env_name}")
    