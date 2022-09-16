# file_pathnames.py
# define pathnames for model code, input and output
# this single file contains all pathnames to avoid repeated hardcoding in model scripts

from pathlib import Path

ROOT_PATH = Path('C:\\Users\\JOHNSMITH\\CRRELRTM\\')              # Project root path

CODE_PATH = ROOT_PATH / 'Code/CRREL-GOSRT'                        # Path where code resides

MCT_IMAGE_PATH = ROOT_PATH / 'MicroCT_Data/SnowPitA/'                 # Path to MicroCT data for specific sample of interest
MATERIAL_PATH = ROOT_PATH / 'Code/CRREL-GOSRT/Materials/'         # Path to material property CSV files
VTK_DATA_OUTPATH = ROOT_PATH / 'MicroCT_Data/SnowPitA/VTK'           # Output path for mesh VTK data

OPT_PROP_OUTPATH = ROOT_PATH / 'MicroCT_Data/OptProps'          # Output path for optical properties file

SLAB_MODEL_MASTER_PATH = ROOT_PATH / 'Model_Runs/'      # Output path for slab model run results