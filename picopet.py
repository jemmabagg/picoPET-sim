from scipy.spatial.transform import Rotation
from opengate.utility import g4_units
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition
import json
import math

# colors (similar to the ones of Gate)
red = [1, 0, 0, 1]
blue = [0, 0, 1, 1]
green = [0, 1, 0, 1]
yellow = [0.9, 0.9, 0.3, 1]
gray = [0.5, 0.5, 0.5, 1]
white = [1, 1, 1, 0.8]

def create_material(sim):
    g_cm3 = g4_units.g_cm3
    sim.volume_manager.material_database.add_material_nb_atoms(
        "ABS", ["C", "H", "N"], [15, 17, 1], 1.04 * g_cm3
    )
    sim.volume_manager.material_database.add_material_weights(
        "Copper", ["Cu"], [1], 8.920 * g_cm3
    )
    sim.volume_manager.material_database.add_material_nb_atoms(
        "LYSO", ["Lu", "Y", "Si", "O"], [18, 2, 10, 50], 7.1 * g_cm3
    )
    sim.volume_manager.material_database.add_material_nb_atoms(
        "LYSO_debug", ["Lu", "O"], [1, 100], 7.1 * g_cm3
    )
    sim.volume_manager.material_database.add_material_weights(
        "Lead", ["Pb", "Sb"], [0.95, 0.05], 11.16 * g_cm3
    )
    sim.volume_manager.material_database.add_material_nb_atoms(
        "Lexan", ["C", "H", "O"], [15, 16, 2], 1.2 * g_cm3
    )
    sim.volume_manager.material_database.add_material_weights(
        "CarbonFiber", ["C"], [1], 1.78 * g_cm3
    )

def add_pet(sim, name="pet", create_housing=True, create_mat=True, debug=False):
    """
    Geometry of a PET Philips VEREOS
    Salvadori J, Labour J, Odille F, Marie PY, Badel JN, Imbert L, Sarrut D.
    Monte Carlo simulation of digital photon counting PET.
    EJNMMI Phys. 2020 Apr 25;7(1):23.
    doi: 10.1186/s40658-020-00288-w
    """

    # unit
    mm = g4_units.mm

    # define the materials (if needed)
    if create_mat:
        create_material(sim)

    # ring volume
    pet = sim.add_volume("Tubs", name)
    #pet.rmax = 500 * mm
    #pet.rmin = 254 * mm
    pet.rmax = 180 * mm #if you are changing the ring size you need to change the ring volume
    pet.rmin = 140 * mm
    pet.dz = 60 * mm / 2.0
    pet.color = gray
    pet.material = "G4_AIR"


    # Module 
    #The modules are the detectors all around the ring
    module = sim.add_volume("Box", f"{name}_module")
    module.mother = pet.name
    # x is the length of the scintillator, y is tangential to that, z is axis around which detectors are placed
    module.size = [15 * mm, 24 * mm, 24 * mm] #Matching size with Simon
    module.material = "ABS"
    module.color = red
    translations_ring, rotations_ring = get_circular_repetition(
        # 18 -> 2
        16, [140*mm + module.size[0]/2 * mm, 0, 0], start_angle_deg=0, axis=[0, 0, 1] #changing the number of modules that are around the ring, also changes the size of the ring (175 default)
    )
    module.translation = translations_ring
    module.rotation = rotations_ring

    # Stack (each stack has 4x4 diedie)
    stack = sim.add_volume("Box", f"{name}_stack")
    stack.mother = module.name
    stack.size = [module.size[0], module.size[1], module.size[2]] #Matching size with Simon
    stack.material = "G4_AIR"
    stack.translation = get_grid_repetition([1, 1, 1], [0 * mm, 0 * mm, 0 * mm])
    stack.color = green

    # Die (each die has 2x2 crystal)
    die = sim.add_volume("Box", f"{name}_die")
    die.mother = stack.name
    die.size = [stack.size[0], stack.size[1]/4, stack.size[2]/4] #Matching size with Simon
    die.material = "G4_AIR"
    die.translation = get_grid_repetition([1, 4, 4], [0, stack.size[1]/4, stack.size[2]/4])
    die.color = white

    # Crystal
    crystal = sim.add_volume("Box", f"{name}_crystal")
    crystal.mother = die.name
    crystal.size = [die.size[0], die.size[1]/2, die.size[2]/2] #Matching size with Simon
    #crystal.size = [module.size[0], 20 * mm, 4 * mm]
    crystal.material = "LYSO"
    crystal.translation = get_grid_repetition([1, 2, 2], [0 * mm, die.size[1]/2, die.size[2]/2])

    # with debug mode, only very few crystal to decrease the number of created
    # volumes, speed up the visualization
    if debug:
        crystal.size = [module.size[0], 8 * mm, 8 * mm]
        crystal.translation = get_grid_repetition([1, 1, 1], [0, 4 * mm, 4 * mm])

    # ------------------------------------------
    # Housing
    # ------------------------------------------

    if not create_housing:
        return pet

    # SiPMs HOUSING
    housing = sim.add_volume("Box", f"{name}_housing")
    housing.mother = pet.name
    housing.size = [1 * mm, 131 * mm, 164 * mm]
    housing.material = "G4_AIR"
    housing.color = yellow
    translations_ring, rotations_ring = get_circular_repetition(
        18, [408 * mm, 0, 0], start_angle_deg=190, axis=[0, 0, 1]
    )
    housing.translation = translations_ring
    housing.rotation = rotations_ring

    # SiPMs UNITS
    sipms = sim.add_volume("Box", f"{name}_sipms")
    sipms.mother = housing.name

    sipms.size = [1 * mm, 32.6 * mm, 32.6 * mm]
    spacing = 32.8 * mm
    sipms.translation = get_grid_repetition([1, 4, 5], [0, spacing, spacing])
    sipms.rotation = None
    sipms.material = "G4_AIR"
    sipms.color = green

    # cooling plate
    coolingplate = sim.add_volume("Box", f"{name}_coolingplate")
    coolingplate.mother = pet.name
    coolingplate.size = [30 * mm, 130.2 * mm, 164 * mm]
    coolingplate.material = "Copper"
    coolingplate.color = blue
    translations_ring, rotations_ring = get_circular_repetition(
        18, [430 * mm, 0, 0], start_angle_deg=190, axis=[0, 0, 1]
    )
    coolingplate.translation = translations_ring
    coolingplate.rotation = rotations_ring

    # ------------------------------------------
    # Shielding
    # ------------------------------------------
    # end shielding 1
    endshielding1 = sim.add_volume("Tubs", f"{name}_endshielding1")
    endshielding1.mother = pet.name
    endshielding1.translation = [0, 0, 95 * mm]
    endshielding1.rmax = 410 * mm
    endshielding1.rmin = 362.5 * mm
    endshielding1.dz = 25 * mm / 2.0
    endshielding1.material = "Lead"
    endshielding1.color = yellow

    # end shielding 2
    endshielding2 = sim.add_volume("Tubs", f"{name}_endshielding2")
    endshielding2.mother = pet.name
    endshielding2.translation = [0, 0, -95 * mm]
    endshielding2.rmax = 410 * mm
    endshielding2.rmin = 362.5 * mm
    endshielding2.dz = 25 * mm / 2.0
    endshielding2.material = "Lead"
    endshielding2.color = yellow

    cover = sim.add_volume("Tubs", f"{name}_cover")
    cover.mother = pet.name
    cover.translation = [0, 0, 0]
    cover.rmax = 355.5 * mm
    cover.rmin = 354 * mm
    cover.dz = 392 * mm / 2.0 * mm
    cover.material = "Lexan"
    cover.color = white
    cover.color = red

    return pet

def add_scanner(sim, scanner_file, name="pet", create_mat=True):
    # Read in scanner dimensions
    with open(scanner_file, "r") as file:
        config = json.load(file)
    ring_config = config["ring"]
    module__config = config["module"]
    crystal_config = config["crystal"]
    multi_ring_config = config["multi_ring"]

    # Units
    mm = g4_units.mm

    # Calculate dimensions and translations
    if not module__config["rotation"]:
        # dimensions
        module_dim = [crystal_config["length"], module__config["no_sipms"][0] * module__config["sipm_size"], module__config["no_sipms"][1] * module__config["sipm_size"]]
        sipm_chan_dim = [module_dim[0], module__config["sipm_size"], module__config["sipm_size"]]
        crystal_dim = [crystal_config["length"], module__config["sipm_size"] / math.sqrt(crystal_config["crystals_per_sipm"]), module__config["sipm_size"] / math.sqrt(crystal_config["crystals_per_sipm"])]

        # translations
        sipm_chan_trans = [[1, module__config["no_sipms"][0], module__config["no_sipms"][1]], [0, module__config["sipm_size"] * mm, module__config["sipm_size"] * mm]]
        crystal_trans = [[1, int(math.sqrt(crystal_config["crystals_per_sipm"])), int(math.sqrt(crystal_config["crystals_per_sipm"]))], [0, crystal_dim[1] * mm, crystal_dim[2] * mm]]

    else:  
        # dimensions
        module_dim = [module__config["no_sipms"][0] * module__config["sipm_size"], crystal_config["length"], module__config["no_sipms"][1] * module__config["sipm_size"]]
        sipm_chan_dim = [module__config["sipm_size"], crystal_config["length"], module__config["sipm_size"]]
        crystal_dim = [module__config["sipm_size"] / math.sqrt(crystal_config["crystals_per_sipm"]), crystal_config["length"], module__config["sipm_size"] / math.sqrt(crystal_config["crystals_per_sipm"])]

        # translations
        sipm_chan_trans = [[module__config["no_sipms"][0], 1, module__config["no_sipms"][1]], [module__config["sipm_size"] * mm, 0, module__config["sipm_size"] * mm]]
        crystal_trans = [[int(math.sqrt(crystal_config["crystals_per_sipm"])), 1, int(math.sqrt(crystal_config["crystals_per_sipm"]))], [crystal_dim[0] * mm, 0, crystal_dim[2] * mm]]

    # DOI
    if crystal_config["doi_resolution"] > 0:
        # Get possible DOI
        length_factors = [i for i in range(1, crystal_config["length"] + 1) if crystal_config["length"] % i == 0]
        doi = min(length_factors, key=lambda num: abs(num - crystal_config["doi_resolution"]))
        no_divisions = int(crystal_config["length"] / doi)

        # Cut crystals
        if not module__config["rotation"]:
            # change sipm channel and crystal dimensions
            sipm_chan_dim[0] = sipm_chan_dim[0] / no_divisions
            crystal_dim[0] = crystal_dim[0] / no_divisions

            # sipm channel translations
            sipm_chan_trans = [[no_divisions, module__config["no_sipms"][0], module__config["no_sipms"][1]], [doi, module__config["sipm_size"] * mm, module__config["sipm_size"] * mm]]
        else:
            # change sipm channel and crystal dimensions
            sipm_chan_dim[1] = sipm_chan_dim[1] / no_divisions
            crystal_dim[1] = crystal_dim[1] / no_divisions

            # sipm channel translations
            sipm_chan_trans = [[module__config["no_sipms"][0], no_divisions, module__config["no_sipms"][1]], [module__config["sipm_size"] * mm, doi, module__config["sipm_size"] * mm]]

    # Define Materials
    if create_mat:
        create_material(sim)

    # Add Scanner Volume
    pet = sim.add_volume("Tubs", name)
    pet.rmax = (ring_config["inner_radius"] + ring_config["module_buffer"] + module_dim[0]) * mm
    pet.rmin = (ring_config["inner_radius"] - ring_config["module_buffer"]) * mm
    scanner_depth = (module_dim[2] * multi_ring_config["no_rings"] + multi_ring_config["space"] * (multi_ring_config["no_rings"] - 1))
    pet.dz = (scanner_depth / 2) * mm
    pet.color = white
    pet.color = white
    pet.material = "G4_AIR"

    # Add Modules
    module = sim.add_volume("Box", f"{name}_module")
    module.mother = pet.name
    module.size = [module_dim[0] * mm, module_dim[1] * mm, module_dim[2] * mm]
    module.material = "ABS"
    module.color = red
    translations = []
    rotations = []
    for ring_no in range(multi_ring_config["no_rings"]):
        # calculate the z_offset
        z_offset = (-0.5 * scanner_depth  + 0.5 * module_dim[2] + module_dim[2] * ring_no + multi_ring_config["space"] * ring_no) * mm
        trans, rot = get_circular_repetition(
            ring_config["no_modules"], [(ring_config["inner_radius"] + module_dim[0] / 2) * mm, 0, z_offset], start_angle_deg=90, axis=[0, 0, 1]
        )
        translations.extend(trans)
        rotations.extend(rot)
    module.translation = translations
    module.rotation = rotations

    # Add SiPM Channels
    sipm_channel = sim.add_volume("Box", f"{name}_sipm_channel")
    sipm_channel.mother = module.name
    sipm_channel.size = [sipm_chan_dim[0] * mm, sipm_chan_dim[1] * mm, sipm_chan_dim[2] * mm]
    sipm_channel.material = "G4_AIR"
    sipm_channel.translation = get_grid_repetition(sipm_chan_trans[0], sipm_chan_trans[1])
    sipm_channel.color = white

    # Add Crystals
    crystal = sim.add_volume("Box", f"{name}_crystal")
    crystal.mother = sipm_channel.name
    crystal.size = [crystal_dim[0] * mm, crystal_dim[1] * mm, crystal_dim[2] * mm]
    crystal.material = "LYSO"
    crystal.translation = get_grid_repetition(crystal_trans[0], crystal_trans[1])

    return pet