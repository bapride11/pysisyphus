#!/usr/bin/env python3

import numpy as np
import pytest

from pysisyphus.calculators.ONIOMv2 import ONIOM
from pysisyphus.helpers import geom_from_library
from pysisyphus.testing import using_gaussian16, using_pyscf

from pysisyphus.init_logging import init_logging


init_logging()


@using_gaussian16
def test_energy():
    geom = geom_from_library("alkyl17_sto3g_opt.xyz")

    real = set(range(len(geom.atoms)))
    medmin = set((0,1,2,3,4,5,6, 46,47,48,49,50,51,52))
    med = list(real - medmin)
    h1 = list(range(13, 22))
    h2 = list(range(31, 40))

    calcs = {
        "real": {
            "route": "HF/STO-3G",
        },
        "medium": {
            "route": "HF/3-21G",
        },
        "high1": {
            "route": "HF/6-31G",
        },
        "high2": {
            "route": "HF/6-311G",
        },
    }
    for key, calc in calcs.items():
        calc["type"] = "g16"
        calc["pal"] = 2
        calc["mult"] = 1
        calc["charge"] = 0

    models = {
        "med" : {
            # "inds": (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
            "inds": med,
            "calc": "medium",
        },
        "h1": {
            # "inds": (4, 5, 6),
            "inds": h1,
            "calc": "high1",
        },
        "h2": {
            # "inds": (10, 11, 12),
            "inds": h2,
            "calc": "high2",
        }
    }

    layers = ["med", ["h1", "h2"]]

    oniom = ONIOM(calcs, models, geom, layers)

    assert oniom.layer_num == 3

    geom.set_calculator(oniom)

    assert geom.energy == pytest.approx(-661.3512410069466)


@pytest.mark.parametrize(
    "calcs, ref_energy, ref_force_norm",
    [
        # From https://doi.org/10.1016/S0166-1280(98)00475-8
        pytest.param(
            {"real": {"type": "g16", "route": "hf sto-3g"},
             "high": {"type": "g16", "route": "b3lyp d95v"}},
            -153.07432042299052 , 0.03768246934785125,
            marks=using_gaussian16,
        ),
        # The following two tests should yield identical results
        pytest.param(
            {"real": {"type": "g16", "route": "hf sto-3g"},
             "high": {"type": "g16", "route": "b3lyp 3-21g"}},
            -152.4529060634755 , 0.018462670668992546,
            marks=using_gaussian16,
        ),
        pytest.param(
            {"real": {"type": "pyscf", "basis": "sto3g"},
             "high": {"type": "pyscf", "xc": "b3lypg", "basis": "321g"}},
            -152.4529060634755, 0.01839279960703439,
            marks=using_pyscf,
        ),
])
def test_gradient(calcs, ref_energy, ref_force_norm):
    geom = geom_from_library("acetaldehyd_oniom.xyz", coord_type="redund")

    real = set(range(len(geom.atoms)))
    high = [4, 5, 6]

    for key, calc in calcs.items():
        calc["pal"] = 2
        calc["mult"] = 1
        calc["charge"] = 0

    models = {
        "high": {
            "inds": high,
            "calc": "high",
        },
    }

    # layers = ["high"]
    # No layers specified
    layers = None

    oniom = ONIOM(calcs, models, geom, layers)

    assert oniom.layer_num == 2

    geom.set_calculator(oniom)

    # Calculate forces
    forces = geom.forces
    energy = geom.energy

    assert np.linalg.norm(forces) == pytest.approx(ref_force_norm)
    assert energy == pytest.approx(ref_energy)


@pytest.mark.parametrize(
    "embedding, ref_energy, ref_force_norm",
    [
        # pytest.param(None, -582.3920349478807, 0.09403378140930853,
                     # marks=using_gaussian16),
        pytest.param("electronic", -582.3997769406087, 0.0942232377699925,
                     marks=using_gaussian16),
])
def test_electronic_embedding(embedding, ref_energy, ref_force_norm):
    geom = geom_from_library("oniom_ee_model_system.xyz", coord_type="redund")

    all_ = set(range(len(geom.atoms)))
    high = list(sorted(all_ - set((21, 20, 19, 15, 14, 13))))

    calcs = {
        "real": {
            "route": "hf sto-3g",
        },
        "highC": {
            "route": "hf 3-21g",
        },
    }
    for key, calc in calcs.items():
        calc["type"] = "g16"
        calc["pal"] = 2
        calc["mult"] = 1
        calc["charge"] = 0

    models = {
        "high": {
            "inds": high,
            "calc": "highC",
        },
    }

    oniom_kwargs = {
        "embedding": embedding,
    }
    oniom = ONIOM(calcs, models, geom, **oniom_kwargs)
    geom.set_calculator(oniom)

    # Calculate forces and energy
    forces = geom.forces
    energy = geom.energy

    assert energy == pytest.approx(ref_energy)
    assert np.linalg.norm(forces) == pytest.approx(ref_force_norm)