import json
from pathlib import Path

class Params:
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        '''Loads parameters from json file'''
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__


def resolve_path(path: str, base_dir: str | Path) -> str:
    """Resolve a path relative to ``base_dir`` when it is not absolute."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(base_dir) / resolved
    return str(resolved.resolve())
