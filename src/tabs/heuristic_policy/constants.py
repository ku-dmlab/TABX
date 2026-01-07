from pathlib import Path

ASSET_PATH = Path(__file__).resolve().parent.joinpath("parameters")
HEURISTIC_PARAMS = [f.stem for f in ASSET_PATH.iterdir() if f.is_file()]
