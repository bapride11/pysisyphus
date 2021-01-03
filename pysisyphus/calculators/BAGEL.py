import re
import textwrap

import numpy as np

from pysisyphus.calculators.Calculator import Calculator


class BAGEL(Calculator):

    conf_key = "bagel"

    def __init__(self, json_file, **kwargs):
        super().__init__(**kwargs)

        self.inp_fn = "bagelin.json"
        self.out_fn = "bagelout.out"
        self.to_keep = ("inp","bagelout.out")
        self.parser_funcs = {
            "energy": self.parse_energy,
            "grad": self.parse_grad
        }
        self.input_json_file = json_file

        self.base_cmd = self.get_cmd("cmd")

    def prepare_input(self, atoms, coords, calc_type):
        xyz = self.bagel_coordstring(atoms, coords)

        with open(self.input_json_file, 'r') as handle:
            input_json = handle.read()

        if not (re.search('export', input_json) and re.search('export_single', input_json)):
            raise Exception("Both the export and export_single options must be set in your BAGEL input!")
        inp = re.sub('INSERT_XYZ',xyz,input_json)

        return inp

    def bagel_coordstring(self, atoms, coords):
        coordstring = ""
        for iatom, atom in enumerate(atoms):
            coordstring += '{ "atom" : "' + atom + '", "xyz" : [ ' + str(coords[3*iatom]) + ', ' + \
                           str(coords[3*iatom+1]) + ', ' + str(coords[3*iatom+2]) + ' ] },\n'
        return coordstring[:-2]

    def get_energy(self, atoms, coords):
        calc_type = "energy"
        inp = self.prepare_input(atoms, coords, calc_type)
        results = self.run(inp, calc="energy")
        return results

    def get_forces(self, atoms, coords):
        calc_type = "grad"
        inp = self.prepare_input(atoms, coords, calc_type)
        results = self.run(inp, calc="grad")
        return results

    def run_calculation(self, atoms, coords):
        return self.get_energy(atoms, coords)

    def parse_energy(self, path):
        with open(path / "ENERGY.out") as handle:
            text = handle.read()
            energy = float(text)
        result = {
            "energy": energy
        }
        return result

    def parse_grad(self, path):
        with open(path / "FORCE.out") as handle:
            lines = handle.read().splitlines()
        gradient = np.empty((len(lines)-3)*3)
        for iline, line in enumerate(lines):
            if iline == 0:
                energy = float(line)
            if iline == 1:
                continue
            if len(line.split()) < 2:
                continue
            iatom = iline - 2
            gradient[iatom*3] = line.split()[1]
            gradient[iatom*3+1] = line.split()[2]
            gradient[iatom*3+2] = line.split()[3]
        
        forces = -1*gradient
        result = {
            "energy": energy,
            "forces": forces,
        }
        return result

    def __str__(self):
        return f"BAGEL({self.name})"
