import h5py
import numpy as np
import os



#####################
# Parsing Functions #
#####################

def split_name(name):
    """
    Parses the name of a molecule in the format 'mass_element_num' into its components.
    """
    mass = ''
    element = ''
    num = ''
    # get all the characters until we hit a non-digit
    for char in name:
        if char.isdigit():
            mass += char
        else: break
    # get all the characters until we hit a digit
    for char in name[len(mass):]:
        if char.isalpha():
            element += char
        else: break
    # the rest is the mass
    num = name[len(mass) + len(element):]
    # If num is empty, set it to 1
    num = int(num) if num else 1
    # Convert mass to integer
    mass = int(mass)
    return mass, element, num

def get_molecular_mass_component(molecule_name):
    """
    Returns the molecular mass of a molecule given its name.
    """
    mass, element, num = split_name(molecule_name)
    return mass * num

def get_molecular_mass(molecule_name):
    """
    Returns the molecular mass of a molecule given its name.
    """
    comp_a, _, comp_b = molecule_name.partition('__')[0].partition('-')
    mass_a = get_molecular_mass_component(comp_a)
    mass_b = get_molecular_mass_component(comp_b)
    return mass_a + mass_b


def read(filename):
    with h5py.File(filename, 'r') as f:
        molecule_name = f['mol_name'][0].decode('utf-8')
        # print(f"Reading in {molecule_name}")
        key_iso_ll = f['key_iso_ll'][0].decode('utf-8')
        pressures = np.array(f['p'])
        temperatures = np.array(f['t'])
        bin_edges = np.array(f['bin_edges'])
        cross_sections = np.array(f['xsecarr'])
    wavelengths = 1e4 / bin_edges # in µm
    try:
        molecular_mass = get_molecular_mass(key_iso_ll)
    except:
        print(f'Could not determine molecular mass for {key_iso_ll}! Please check the file format.')
        molecular_mass = None

    return {
        'molecule_name': molecule_name,
        'molecular_mass': molecular_mass,
        'key_iso_ll': key_iso_ll,
        'pressures': pressures,
        'temperatures': temperatures,
        'bin_edges': bin_edges,
        'cross_sections': cross_sections,
        'wavelengths': wavelengths
    }


##################
# LineList Class #
##################

class LineList:
    def __init__(self, filepath):
        with h5py.File(filepath, 'r') as f:
            self.molecule_name = f['mol_name'][0].decode('utf-8')
            # print(f"Reading in {self.molecule_name}")
            self.key_iso_ll = f['key_iso_ll'][0].decode('utf-8')
            self.pressures = np.array(f['p'])
            self.temperatures = np.array(f['t'])
            self.bin_edges = np.array(f['bin_edges'])
            self.cross_sections = np.array(f['xsecarr'])
        self.wavelengths = 1e4 / self.bin_edges # in µm

        # Attempt to determine the molecular mass from the file name
        try:
            self.molecular_mass = get_molecular_mass(self.key_iso_ll)
        except:
            print(f'Could not determine molecular mass for {self.key_iso_ll}! Please check the file format.')
            self.molecular_mass = None

    def __repr__(self):
        return f'LineList(molecule_name={self.molecule_name}, molecular_mass={self.molecular_mass}, cross_sections_shape={self.cross_sections.shape})'