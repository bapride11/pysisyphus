import re
import textwrap

import numpy as np
import os.path

from pysisyphus.calculators.Calculator import OverlapCalculator


class BAGELEXPT(OverlapCalculator):
    conf_key = "bagel_experimental"

    def __init__(self, method, basis=None, df_basis=None,
                 nstate=1, nact=None, nclosed=None,
                 guess='hf',guess_fn=None,keep_ref=False,
                 casscf={}, smith={},
                 memory=None, pal=None, **kwargs):
        super().__init__(**kwargs)

        if memory != None:
            raise Exception("BAGEL does not respect memory preferences!")
        if pal != None:
            raise Exception("Please set BAGEL parallelism in your run script.")
        self.method = method.lower()
        self.basis = basis
        self.df_basis = df_basis
        self.charge = charge
        self.mult = mult
        self.nstate = int(nstate)
        self.nact = int(nact)
        self.nclosed = int(nclosed)



        self.casscf = casscf # Dict of optional user-provided CASSCF parameters
        self.smith = smith # Dict of optional user-provided SMITH parameters

        if self.method in ['casscf', 'caspt2']:
            self.multireference = True
        else:
            self.multireference = False

        if self.multireference and None in [nact, nclosed]:
            raise Exception("Must set nact and nclosed for multireference calculations.")


        if os.path.exists('bagelref.archive'):
            self.guess = 'load_ref'
            self.guess_fn = 'bagelref.archive'
        elif guess.lower() == 'load_ref':
            if guess_fn == None:
                raise Exception("load_ref guess requires guess_fn")
            elif os.path.exists(guess_fn):
                self.guess = 'load_ref'
                self.guess_fn = guess_fn
            else:
                raise Exception("guess_fn not found.")
        elif guess.lower() in ['hf', 'rhf', 'uhf', 'rohf']:
            self.guess = guess.lower()
            self.guess_fn = None
        else:
            raise Exception(f"Invalid guess type {guess}")



        self.inp_fn = "bagel.json"
        self.out_fn = "bagel.out"
        self.to_keep = ("bagel.json", "bagel.out")
        self.parser_funcs = {
            "energy": self.parse_energy,
            "grad": self.parse_grad
        }

        self.base_cmd = self.get_cmd("cmd")


    def prepare_input(self, atoms, coords, calc_type):
        inputstring = '{ "bagel" : [\n\n'
        inputstring += get_molecule_block(atoms, coords)
        if self.multireference:
            inputstring += get_guess_block()
            inputstring += get_casscf_block()
            inputstring += get_saveref_block()
            inputstring += get_molden_block()
        if calc_type == 'grad':
            inputstring += get_forces_block()
        inputstring += ']}'
        return inputstring

    def get_molecule_block(self, atoms, coords):
        xyz = bagel_coordstring(self, atoms, coords)
        molecule_string = textwrap.dedent(f'''\
        {{
          "title" : "molecule",
          "basis" : "{self.basis}",
        ''')

        if self.df_basis != None:
            molecule_string += f'  "df_basis" : "{self.df_basis}",\n'

        molecule_string += textwrap.dedent(f'''\
          "geometry" : [
          {xyz}
          ]
        }},
        ''')

        return molecule_string

    def get_guess_block(self):
        guess_string = str()
        if self.guess == 'load_ref':
            guess_string = textwrap.dedent(f'''\
            {{
              "title" : "load_ref",
              "file" : "{self.guess_fn}",
              "continue_geom" : false
            }},
            ''')
        elif self.guess in ['hf', 'uhf', 'rhf', 'rohf']:
            guess_string = textwrap.dedent(f'''\
            {{
              "title" : {self.guess},
              "charge" : {self.charge},
              "nopen" : {self.mult - 1}
            }},
            ''')
        else:
            raise Exception(f"Guess type {self.guess} not valid!")
        return guess_string

    def get_casscf_block(self):
        casscf_string = textwrap.dedent(f'''\
        {{
          "title" : "casscf",
          "nstate" : {self.nstate},
          "nact" : {self.nact},
          "nclosed" : {self.nclosed}
        ''')
        # Add user-provided arguments to CASSCF block
        if bool(self.casscf):
            for key in self.casscf:
                casscf_string += f',\n  "{key}" : {casscf[key]},\n'
        casscf_string += '},\n'

    def get_saveref_block(self):
        saveref_string = textwrap.dedent(f'''\
        {{
          "title" : "save_ref",
          "file" : "bagelref"
        }},
        ''')

    def get_molden_block(self):
        return

    def get_forces_block(self):
        if int(self.nroots) > 1:
            assert self.multireference
            forces_string = textwrap.dedent(f'''\
            {{
              "title" : "forces",
              "export" : true,
              "grads" : [
            ''')
            for istate in range(0, self.nroots):
                forces_string += f'    {{ "title" : "force", "target" : {istate} }}'
                if istate == self.nroots - 1:
                    forces_string += '\n  ],\n'
                else:
                    forces_string += ',\n'
            if self.method == 'casscf':
                method_string = get_casscf_block()
            elif self.method == 'caspt2':
                method_string = textwrap.dedent(f'''\
                {{
                  "title" : "caspt2",
                  "nstate" : "{self.nstate}",
                  "nact" : "{self.nact}",
                  "nclosed" : "{self.nclosed}",
                  "smith" : {{
                    "method" : "caspt2"''')
                if bool(self.smith):
                    for key in smith:
                        method_string += ',\n    "{key}" : "{smith[key]}"'
                method_string += '\n  }\n},'
            else:
                raise Exception(f"{self.method} not implemented for multiroot forces.")
            forces_string += textwrap.dedent(f'''\
              ],
              "method" : [
            {textwrap.indent(method_string, "  ")} ]
            }}
            ''')
        elif self.method in ['hf', 'rhf', 'uhf', 'rohf']:
            forces_string = textwrap.dedent(f'''\
            {{
              "title" : force,
              "export" : true,
              "method" : [ {{
                "title" : "{self.method}",
                "charge" : {self.charge},
                "nopen" : {self.mult - 1}
              ]
            }}
            ''')
        else:
            raise Exception(f"Forces not implemented for {self.method}!")
        return forces_string

    def bagel_coordstring(self, atoms, coords):
        coordstring = ""
        for iatom, atom in enumerate(atoms):
            coordstring += '{ "atom" : "' + atom + '", "xyz" : [ ' + str(coords[3 * iatom]) + ', ' + \
                           str(coords[3 * iatom + 1]) + ', ' + str(coords[3 * iatom + 2]) + ' ] },\n'
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
        gradient = np.empty((len(lines) - 3) * 3)
        for iline, line in enumerate(lines):
            if iline == 0:
                energy = float(line)
            if iline == 1:
                continue
            if len(line.split()) < 2:
                continue
            iatom = iline - 2
            gradient[iatom * 3] = line.split()[1]
            gradient[iatom * 3 + 1] = line.split()[2]
            gradient[iatom * 3 + 2] = line.split()[3]

        forces = -1 * gradient
        result = {
            "energy": energy,
            "forces": forces,
        }
        return result

    def __str__(self):
        return f"BAGEL({self.name})"
