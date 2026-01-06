from pathlib import Path

ASSET_PATH = Path(__file__).resolve().parent.joinpath("units")
UNIT_SCENARIOS = [f.stem for f in ASSET_PATH.iterdir() if f.is_file()]
ASSET_PATH = Path(__file__).resolve().parent.joinpath("zones")
ZONE_SCENARIOS = [f.stem for f in ASSET_PATH.iterdir() if f.is_file()]
ASSET_PATH = Path(__file__).resolve().parent.joinpath("challenges")
CHALLENGES = [f.stem for f in ASSET_PATH.iterdir() if f.is_file()]
