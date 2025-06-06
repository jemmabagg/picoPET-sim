import opengate as gate
from opengate.tests import utility
from scipy.spatial.transform import Rotation
import numpy as np
from opengate.sources.base import get_rad_yield
import picopet as pet_vereos
from pet_helpers import add_vereos_digitizer_v1
from opengate.geometry.utility import get_circular_repetition


# create the simulation
sim = gate.Simulation()

# main options
sim.progress_bar = True
sim.visu = False # visualisation is extremely slow with complex phantoms
sim.visu_type = "qt"

sim.number_of_threads = 1
sim.random_seed = 123456
sim.output_dir = "./output"

print(sim)

# add a material database
#had to hard link to a local directory 
sim.volume_manager.add_material_database("/Users/jameskeaveney/.virtualenvs/py310env-DSTut/lib/python3.10/site-packages/opengate/contrib/GateMaterials.db")

# - there would be a better way if one can automatically find the local opengate install path as below (but this currently doesn't work)
#sim.volume_manager.add_material_database(paths.data / "GateMaterials.db")

# units
m = gate.g4_units.m
mm = gate.g4_units.mm
cm = gate.g4_units.cm
keV = gate.g4_units.keV
MeV = gate.g4_units.MeV
Bq = gate.g4_units.Bq
kBq = 1000 * Bq
sec = gate.g4_units.s

#  change world size
sim.world.size = [1.5 * m, 1.5 * m, 1.5 * m]
pet = pet_vereos.add_pet(sim, "pet", create_housing=False)

# colors
red = [1, 0.7, 0.7, 0.8]
blue = [0.5, 0.5, 1, 0.8]
gray = [0.5, 0.5, 0.5, 1]
transparent = [0, 0, 0, 0]

# ---------------------------------------------------
# CT image volume
ct = sim.add_volume("Image", "ct")
ct.image = 'template_with_skull_2mm.mhd' # https://pmc.ncbi.nlm.nih.gov/articles/PMC7274757/
# adding big volumes may cause geometry clashes and you need to enlarge the pet ring to avoid these
# In picopet.py, setting pet.rmin = 185 and putting the modules at r = 195 works for the above phantom

#assigning G4 materials to voxel intensities, this can be improved by
#understanding the above paper a little more
ct.material = "G4_AIR"  # material used by default
ct.voxel_materials = [
[-2000, 20, "G4_AIR"],
[20, 100, "G4_WATER"],
[100, 800, "G4_TISSUE_SOFT_ICRP"],
[800, 1600, "G4_B-100_BONE"]
]

ct_info = gate.image.read_image_info(ct.image)
print(f"CT image origin and size: ", ct_info.origin, ct_info.size, ct_info.spacing)

#source
#source from image
#also using the CT image to define the source
#I understand that gate assign activity according to voxel intensity
source = sim.add_source('VoxelSource', 'vox')
source.image = 'template_with_skull_2mm.mhd'

total_yield = get_rad_yield("F18")
source.attached_to = "ct"
source.particle = "e+"
source.energy.type = "F18"
source.activity = 370e6 * Bq * total_yield

# cuts
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
sim.physics_manager.enable_decay = False
sim.physics_manager.set_production_cut("world", "all", 1 * mm)

# add the PET digitizer
add_vereos_digitizer_v1(sim, pet, f"output_picopet.root")

# add stat actor
stats = sim.add_actor("SimulationStatisticsActor", "Stats")
stats.track_types_flag = True
stats.output_filename = "stats_picopet.txt"

# timing
sim.run_timing_intervals = [[0, 0.00001 * sec]]

# start simulation
sim.run()