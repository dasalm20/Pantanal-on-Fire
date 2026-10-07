#----------------------------------------------------------------------------------------
# This script is used to set up the configuration for the project.
# It defines various parameters and settings that will be used throughout the codebase.
# Import libraries
from pathlib import Path
#----------------------------------------------------------------------------------------

# Repo root 
MAIN = Path(__file__).resolve().parents[1]

#Define paths for raw data, code, processed data, and output
RAW       = MAIN / "03-Data"
CODE      = MAIN / "02-Code"
PROCESSED = MAIN / "04-Processed"
OUTPUT    = MAIN / "05-Output"

# Define the Coordinate Reference System (CRS) to be used for spatial data
CRS = "EPSG:31981"   # SIRGAS 2000 / UTM 21S

# ---- Raw inputs ----
INPUTS = {
    "ibge":      RAW / "IBGE Biomes" / "lml_bioma_e250k_v20250911_A.shp",
    "firms":     RAW / "DL_FIRE_J1V-C2_818288" / "fire_archive_J1V-C2_818288.csv",
    "chirps":    RAW / "CHIRPS_monthly_2018_2025",
    "tmax":      RAW / "CHIRTS-ERA5_Tmax_monthly_2018_2025",
    "landcover": RAW / "MapBiomas Collection 11",
    "water":     RAW / "Pantanal_MapBiomas_Water",
    "roads":     RAW / "OpenStreetMap_Geofabrik" / "centro-oeste-200101-free.shp" / "gis_osm_roads_free_1.shp",
    "waterways": RAW / "OpenStreetMap_Geofabrik" / "centro-oeste-200101-free.shp" / "gis_osm_waterways_free_1.shp",
}

# ---- Outputs ----
OUTPUTS = {
    "grid": PROCESSED / "grid.gpkg",
    "fire": PROCESSED / "fire_month.parquet",
    "figures": OUTPUT / "figures",
}

# ---- Model settings ----
GRID_M        = 5000
MIN_AREA_KM2  = 12.5
TRAIN_YEARS   = list(range(2019, 2025))
VAL_YEAR      = 2025
TARGET_MONTHS = [8, 9, 10]
LAG_START     = 2018
RANDOM_SEED   = 654


