#!/usr/bin/env python3


from pysisyphus.calculators.XTB import XTB
from pysisyphus.optimizers.RFOptimizer import RFOptimizer
from pysisyphus.helpers import geom_from_library


def run():
    # geom = geom_from_library("split.image_021.xyz")
    # geom.set_calculator(XTB())
    # opt = RFOptimizer(geom)
    # opt.run()

    # geom = geom_from_library("split.image_021.xyz", coord_type="redund")
    # geom.set_calculator(XTB())
    # opt = RFOptimizer(geom)
    # opt.run()

    xyz_fn = "fluorethylene.xyz"
    # xyz_fn = "split.image_021.xyz"
    xyz_fn = "h2o2_hf_321g_opt.xyz"
    geom = geom_from_library(xyz_fn, coord_type="dlc")
    # geom = geom_from_library(xyz_fn, coord_type="redund")
    # int_ = geom.internal
    # int_.B
    # int_.set_delocalized_vectors()
    geom.set_calculator(XTB())
    opt_kwargs = {
        # "max_cycles": 1,
        "thresh": "gau_tight",
    }
    opt = RFOptimizer(geom, **opt_kwargs)
    opt.run()


if __name__ == "__main__":
    run()
