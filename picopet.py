from scipy.spatial.transform import Rotation
from opengate.utility import g4_units
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition

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

    # ------------------------------------------
    # 18 modules
    #   of 4x5 stack
    #       of 4x4 die
    #           of 2x2 crystal
    # ------------------------------------------
    #the hierarchy from the the bottom up is crystal->die (2x2 crystals) -> stack (NxM) dies? -> module (AxB stacks?)

    # Module (each module has 4x1 stacks)
    #The modules are the detectors all around the ring
    module = sim.add_volume("Box", f"{name}_module")
    module.mother = pet.name
    module.size = [24 * mm, 24 * mm, 15 * mm] #Matching size with Simon
    #module.size = [19 * mm, 32.85 * mm, 33.6 * mm] #ORIGINAL
    #module.size = [19 * mm, 80 * mm, 33.6 * mm]
    module.material = "ABS"
    module.color = red
    translations_ring, rotations_ring = get_circular_repetition(
        # 18 -> 2
        16, [160 * mm, 0, 0], start_angle_deg=0, axis=[0, 0, 1] #changing the number of modules that are around the ring, also changes the size of the ring (175 default)
    )
    module.translation = translations_ring
    module.rotation = rotations_ring

    # Stack (each stack has 4x4 die)
    stack = sim.add_volume("Box", f"{name}_stack")
    stack.mother = module.name
    stack.size = [module.size[0], 24 * mm, 15 * mm] #Matching size with Simon
    #stack.size = [module.size[0], 80 * mm, 32.6 * mm]
    stack.material = "G4_AIR"
    #stack.translation = get_grid_repetition([1, 1, 1], [0, 32.85 * mm, 32.85 * mm]) ORIGINAL
    stack.translation = get_grid_repetition([1, 1, 1], [0, 24 * mm, 15 * mm])
    stack.color = green

    # Die (each die has 2x2 crystal)
    die = sim.add_volume("Box", f"{name}_die")
    die.mother = stack.name
    die.size = [module.size[0], 6 * mm, 15 * mm] #Matching size with Simon
    #die.size = [module.size[0], 40 * mm, 8 * mm]
    die.material = "G4_AIR"
    die.translation = get_grid_repetition([1, 4, 1], [0, 6 * mm, 0 * mm])
    die.color = white

    # Crystal
    crystal = sim.add_volume("Box", f"{name}_crystal")
    crystal.mother = die.name
    crystal.size = [module.size[0], 3 * mm, 3 * mm] #Matching size with Simon
    #crystal.size = [module.size[0], 20 * mm, 4 * mm]
    crystal.material = "LYSO"
    crystal.translation = get_grid_repetition([1, 2, 5], [0, 3 * mm, 3 * mm])

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