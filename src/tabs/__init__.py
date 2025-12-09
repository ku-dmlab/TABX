from src.tabs.tabs import TABS
from src.tabs.constants import ALL_UNIT_NAMES, UnitID, UNITID2CHAR


# Check if the length of UnitID and get_all_unit_names are the same for
unit_id_length = len([v for k, v in UnitID.__dict__.items() if not k.startswith("__")])

unit_id_to_letter_length = len(UNITID2CHAR) - 1  # -1 for empty space
all_unit_names_length = len(ALL_UNIT_NAMES)

if unit_id_to_letter_length != all_unit_names_length:
    raise ValueError(
        f"unit_id_to_letter and get_all_unit_names must have the same length: {unit_id_to_letter_length} != {all_unit_names_length}"
    )

if unit_id_to_letter_length != unit_id_length:
    raise ValueError(
        f"unit_id_to_letter and UnitID must have the same length: {unit_id_to_letter_length} != {unit_id_length}"
    )

if unit_id_length != all_unit_names_length:
    raise ValueError(
        f"UnitID and get_all_unit_names must have the same length: {unit_id_length} != {all_unit_names_length}"
    )
