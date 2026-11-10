# Central configuration: location of the data folder.
# Set the environment variable UAE_SBT_DATA_ROOT to the folder that holds
# hotsat/, predictors/, lst/, buildings_gba/, ancillary/ ... (see README),
# or leave it unset to use ./data next to this repository.
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get(
    "UAE_SBT_DATA_ROOT",
    Path(__file__).resolve().parent.parent / "data")).expanduser()
