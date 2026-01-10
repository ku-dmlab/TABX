from pathlib import Path

_asset_dir = Path(__file__).resolve().parent

UNIT_SCENARIOS = [f.stem for f in _asset_dir.joinpath("units").iterdir() if f.is_file()]
ZONE_SCENARIOS = [f.stem for f in _asset_dir.joinpath("zones").iterdir() if f.is_file()]
CHALLENGES = [f.stem for f in _asset_dir.joinpath("challenges").iterdir() if f.is_file()]

EVAL_UNIT_SCENARIOS = [
    f.stem for f in _asset_dir.joinpath("eval_scenarios", "units").iterdir() if f.is_file()
]
EVAL_ZONE_SCENARIOS = [
    f.stem for f in _asset_dir.joinpath("eval_scenarios", "zones").iterdir() if f.is_file()
]
