import os

UNIT_SCENARIOS = [f.replace(".json", "") for f in os.listdir("src/scenarios/units")]
ZONE_SCENARIOS = [f.replace(".json", "") for f in os.listdir("src/scenarios/zones")]
CHALLENGES = [f.replace(".json", "") for f in os.listdir("src/scenarios/challenges")]
