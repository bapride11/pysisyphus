# [1] https://dx.doi.org/10.1063/1.3587240
#     Adaptive integration grids in instanton theory
#     improve the numerical accuracy at low temperature
#     Rommel, Kästner, 2011
# [2] https://doi.org/10.1021/ct100658y
#     Locating Instantons in Many Degrees of Freedom
#     Rommel, Goumans, Kästner, 2011

from typing import Optional
import logging
from copy import deepcopy

import numpy as np
import scipy as sp
import scipy.interpolate
from scipy.optimize import minimize

from pysisyphus import logger as pysis_logger
from pysisyphus.irc.Instanton import Instanton

logger = pysis_logger.getChild("instanton")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("instanton.log", mode="w", delay=True)
logger.addHandler(file_handler)


def log_progress(val, key, i):
    logger.debug(f"Calculated {key} at image {i}")
    return val


class AdaptiveInstanton(Instanton):
    def __init__(
        self,
        calc_getter,
        T: float,
        starting_instanton: Instanton,
        Vb: Optional[float] = None,
        interpolate_path: bool = True,
        dk_strategy: str = "adaptive",
        **kwargs,
    ):
        super().__init__(starting_instanton.images, calc_getter, T, **kwargs)
        if Vb:
            self.Vb = Vb
        else:
            self.Vb = starting_instanton.tunneling_energy()
        if dk_strategy == "adaptive":
            self.dks, self.images = self.adaptive_grid(
                starting_instanton, interpolate_path
            )
        elif dk_strategy == "fit":
            self.dks = self.fit_grid(starting_instanton.images)
        elif dk_strategy == "interp_and_fit":
            _, self.images = self.adaptive_grid(
                starting_instanton, interpolate_path=True
            )
            self.dks = self.fit_grid(self.images)

    def adaptive_grid(self, source: Instanton, interpolate_path: bool = False):
        if not source.T == self.T:
            logger.warning(
                f"""Source instanton is of different temperature ({source.T} K) \
                            than new adaptive instanton ({self.T} K)."""
            )
        beta_source = 1 / (sp.constants.value("kelvin-hartree relationship") * source.T)
        V_source = [image.energy for image in source.images]
        assert np.isclose(np.sum(source.dks[:-1]) * 2, source.beta_hbar)
        dists = self.get_image_distances()
        x_source = np.cumsum(dists)
        f = scipy.interpolate.PchipInterpolator(x_source, V_source)

        dists_target = [0.0] + [self.path_length / (self.P - 1)] * (self.P - 1)
        x_target = np.cumsum(dists_target)
        # "midpoint" for d_1 and d_P+1 is chosen as y_1, y_P as in [1]
        midpoints = np.concatenate(
            (
                (x_target[self.ks] - x_target[self.ksm1]) / 2 + x_target[self.ksm1],
                [x_target[-1]],
            )
        )
        midpoint_Vs = f(midpoints)
        dks = 1.0 / np.sqrt(2 * (midpoint_Vs - self.Vb))
        dks *= self.beta_hbar / (2 * np.sum(dks[1:-1]) + dks[0] + dks[-1])
        assert np.isclose(2 * np.sum(dks[1:-1]) + dks[0] + dks[-1], self.beta_hbar)

        if interpolate_path:
            images = []
            logger.debug("Interpolating images along path")
            source_coords = np.array([image.coords for image in source.images])
            g = scipy.interpolate.interp1d(x_source, np.swapaxes(source_coords, 0, 1))
            for i in range(self.P):
                new_coords = g(x_target[i])
                closest_source_image = np.argmin(
                    [np.linalg.norm(c - new_coords) for c in source_coords]
                )
                new_image = deepcopy(source.images[closest_source_image])
                new_image.coords = new_coords
                images.append(new_image)
            return dks, images
        else:
            return dks, self.images

    def fit_grid(self, images):
        def obj_fun(dks):
            return np.linalg.norm(self.action_gradient(dks)["gradient"])

        guess_dks = np.array([1 / (2 * self.P_bh)] * (self.P + 1))
        logger.debug("Optimizing dks to minimize gradient at first point.")
        logger.debug(f"Initial action gradient norm: {obj_fun(guess_dks)}")
        backup = self.scheduler_file
        self.scheduler_file = None
        res = minimize(obj_fun, guess_dks)
        if res.success:
            logger.debug("Success!")
            logger.debug(f"Optimized gradient norm at initial geometry: {res.fun}")
            logger.debug(f"Fitted dks: {res.x}")
        else:
            logger.debug("Failed to optimize dks")
            raise RuntimeError
        self.scheduler_file = backup
        return res.x

    def action(self):
        dists = self.get_image_distances()
        S_0 = 2 * np.sum((dists ** 2) / self.dks[self.ks]) * self.amu_to_me
        energies = np.array([image.energy for image in self.images])
        S_pot = np.sum(energies * (self.dks[self.ks] + self.dks[self.ksp1]))
        results = {"action": S_0 / 2 + S_pot}
        return results

    def action_gradient(self, dks=None):
        if dks is None:
            dks = self.dks
        image_coords = np.swapaxes(
            np.array([image.coords for image in self.images]), 0, 1
        )
        # gradient relies on error cancellation for first and last image
        # e.g. last image cancels contribution with prefactor d_(k+1)
        # because y_P = y_(P+1) (1-based index)
        kin_grad = 4 * (
            image_coords[:, self.ks] * (1 / dks[self.ks] + 1 / dks[self.ksp1])  # y_k
            - image_coords[:, self.ksm1] / dks[self.ks]  # y_(k-1)
            - image_coords[:, self.ksp1] / dks[self.ksp1]  # y_(k+1)
        )
        kin_grad = np.swapaxes(kin_grad, 0, 1).flatten()
        kin_grad *= self.amu_to_me

        if self.scheduler_file:
            pot_grads = self.parallel_image_gradients()
        else:
            pot_grads = np.array(
                [
                    log_progress(image.gradient, "gradient", i)
                    for i, image in enumerate(self.images)
                ]
            )
        pot_grad = np.swapaxes(
            np.swapaxes(pot_grads, 0, 1) * (dks[self.ks] + dks[self.ksp1]), 0, 1
        ).flatten()
        gradient = kin_grad / 2 + pot_grad
        results = {"gradient": gradient}
        results.update(self.action())
        return results

    def action_hessian(self):
        if self.scheduler_file:
            image_hessians = self.parallel_image_hessians()
        else:
            image_hessians = np.array(
                [
                    log_progress(image.hessian, "hessian", i)
                    for i, image in enumerate(self.images)
                ]
            )
        # we actually want d_{P+1} instead of d_{P}, so we avoid using ksp1
        scaled_image_hessians = np.array(
            [h * (self.dks[k] + self.dks[k + 1]) for k, h in enumerate(image_hessians)]
        )
        pot_hess = sp.linalg.block_diag(*scaled_image_hessians)

        coord_num = pot_hess.shape[0]
        zeroes = np.zeros((coord_num, coord_num))
        image_coord_num = self.images[0].coords.size

        diag = np.zeros((self.P, image_coord_num, self.P, image_coord_num))
        km = np.zeros((self.P, image_coord_num, self.P, image_coord_num))
        lm = np.zeros((self.P, image_coord_num, self.P, image_coord_num))

        for k in range(1, self.P):
            km[k, :, k - 1, :] = np.eye(image_coord_num) / self.dks[k]
            lm[k - 1, :, k, :] = np.eye(image_coord_num) / self.dks[k]

        for k in range(1, self.P - 1):
            diag[k, :, k, :] = np.eye(image_coord_num) * (
                1 / self.dks[k] + 1 / self.dks[k + 1]
            )
        # First and last k only appear once
        diag[0, :, 0, :] = (
            np.eye(image_coord_num) / self.dks[1]
        )  # k = 2 in 1-based indexing
        diag[self.P - 1, :, self.P - 1, :] = (
            np.eye(image_coord_num) / self.dks[self.P - 1]
        )  # k = P in 1-based indexing

        kin_hess = 4 * (diag - km - lm)
        np.savetxt("kinhess", kin_hess.reshape((coord_num, coord_num)))
        kin_hess *= self.amu_to_me
        hessian = kin_hess.reshape((coord_num, coord_num)) / 2 + pot_hess
        results = {
            "hessian": hessian,
            "kin": kin_hess.reshape((coord_num, coord_num)) / 2,
            "pot": pot_hess,
        }
        return results
