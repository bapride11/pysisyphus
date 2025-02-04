# [1] https://doi.org/10.1021/acs.jctc.8b00068
#     Dual-Level Approach to Instanton Theory
#     Meisner, Kästner 2018
# [2] https://doi.org/10.1021/ct100658y
#     Locating Instantons in Many Degrees of Freedom
#     Rommel, Goumans, Kästner, 2011
# [3] https://aip.scitation.org/doi/full/10.1063/1.4932362
#     Ring-polymer instanton theory of electron transofer in the
#     nonadiabatic limit
#     Richardson, 2015
# [4] https://dx.doi.org/10.1063/1.3587240
#     Adaptive integration grids in instanton theory
#     improve the numerical accuracy at low temperature
#     Rommel, Kästner, 2011

import logging
from math import fsum

import numpy as np
from numpy.typing import NDArray
import scipy as sp
from distributed import Client

from pysisyphus import logger as pysis_logger
from pysisyphus.Geometry import Geometry
from pysisyphus.helpers import align_coords
from pysisyphus.helpers_pure import eigval_to_wavenumber
from pysisyphus.optimizers.hessian_updates import bofill_update


def T_crossover_from_eigval(eigval):
    nu = eigval_to_wavenumber(eigval)  # in cm⁻¹
    nu_m = 100 * abs(nu)  # in m⁻¹
    freq = nu_m * sp.constants.speed_of_light
    # In papers the crossover temperature is usually defined with the angular
    # frequency nu*2*pi. When using the "normal" frequency we don't have to
    # divide by 2*pi. Both approaches obviously yield the same T_c.
    # ang_freq = freq * 2 * np.pi
    # T_c_ = sp.constants.hbar * ang_freq / (sp.constants.Boltzmann * 2 * np.pi)
    T_c = sp.constants.hbar * freq / (sp.constants.Boltzmann)
    return T_c


def T_crossover_from_ts(ts_geom):
    mw_hessian = ts_geom.mw_hessian
    proj_hessian = ts_geom.eckart_projection(mw_hessian, full=True)
    eigvals, eigvecs = np.linalg.eigh(proj_hessian)
    T_c = T_crossover_from_eigval(eigvals[0])
    return T_c


logger = pysis_logger.getChild("instanton")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("instanton.log", mode="w", delay=True)
logger.addHandler(file_handler)


def log_progress(val, key, i):
    logger.debug(f"Calculated {key} at image {i}")
    return val


class Instanton:
    def __init__(self, images, calc_getter, T, scheduler_file=None):
        self.images = images
        for image in self.images:
            image.set_calculator(calc_getter())
        self.T = T
        self.scheduler_file = scheduler_file

        # Pre-calculate action prefactors for the given temperature
        self.beta_hbar = 1 / (
            sp.constants.value("kelvin-hartree relationship") * self.T
        )  # in hbar * Eh^-1
        self.P_over_beta_hbar = self.P / self.beta_hbar
        self.P_bh = self.P_over_beta_hbar
        self.dks = np.array([1 / (2 * self.P_bh)] * (self.P + 1))
        self.amu_to_me = (
            sp.constants.value("atomic mass constant") / sp.constants.electron_mass
        )
        """The Instanton is periodic, but we only optimize the unique half of the path.
        At k=0 the index k-1 will be 0, which points to the first image.
        At k=P, the index k+1 will be P, which points to the last image.

        Below we pre-calculate some indices (assuming N images).
            unshifted indices ks: k = {0, 1, .. , P-1}
            shifted indices ksm1: k-1 = {0, 0, 1, .. , P-2}
            shifted indices ksp1: k+1 = {1, 2, .. , P-1, P-1}
        """
        self.ks = np.arange(self.P)
        self.ksm1 = self.ks - 1
        self.ksp1 = self.ks + 1
        self.ksp1[-1] = -1  # k+1 for the last image points to the last image
        self.ksm1[0] = 0  # k-1 for first image points to the first image

        self.coord_type = "mwcartesian"
        self.internal = None

    @property
    def P(self):
        return len(self.images)

    @classmethod
    def from_ts(cls, ts_geom, P, dr=0.4, delta_T=25, cart_hessian=None, **kwargs):
        assert ts_geom.coord_type == "mwcartesian"
        atoms = ts_geom.atoms
        ts_coords = ts_geom.coords

        if cart_hessian is None:
            mw_hessian = ts_geom.mw_hessian
        else:
            mw_hessian = ts_geom.mass_weigh_hessian(cart_hessian)

        proj_hessian = ts_geom.eckart_projection(mw_hessian, full=True)
        eigvals, eigvecs = np.linalg.eigh(proj_hessian)
        # Use crossover temperature with a little offset (delta_T) if no T is given.
        try:
            kwargs["T"]
        except KeyError:
            T_c = T_crossover_from_eigval(eigvals[0])
            kwargs["T"] = T_c - delta_T
        imag_mode = eigvecs[:, 0]
        cosines = np.cos((np.arange(P) + 1 - 0.5) / P * np.pi)
        image_mw_coords = ts_coords + dr * cosines[:, None] * imag_mode
        image_coords = image_mw_coords / np.sqrt(ts_geom.masses_rep)
        if not ts_geom.is_analytical_2d:
            align_coords(image_coords)
        images = [
            Geometry(atoms, coords, coord_type="mwcartesian") for coords in image_coords
        ]
        instanton = Instanton(images, **kwargs)
        return instanton

    @classmethod
    def from_instanton(cls, other, **kwargs):
        images = other.images
        instanton = Instanton(images, **kwargs)
        return instanton

    def as_xyz(self):
        return "\n".join([geom.as_xyz() for geom in self.images])

    @property
    def coords(self):
        return np.ravel([image.coords for image in self.images])

    @coords.setter
    def coords(self, coords):
        coords = coords.reshape(len(self.images), -1)
        for img_coords, image in zip(coords, self.images):
            image.coords = img_coords

    def action(self, reactant_E=0.0):
        """Returns action in units of hbar (atomic units)."""
        dists = self.get_image_distances()
        S_kin = 2 * self.P_bh * (dists ** 2).sum() * self.amu_to_me

        energies = [image.energy - reactant_E for image in self.images]
        S_pot = sum(energies) / self.P_bh
        # Sum is only over only half the path
        S_E = S_kin + S_pot
        results = {"action": S_E, "S0": S_kin * 2}
        return results

    def action_gradient(self):
        """
        kin_grad corresponds to the gradient of S_0 in (Eq. 1 in [1], or
        first term in Eq. 6 in [2].) It boils down to the derivative of a sum
        of vector norms

            d     sum_k (||y_k - y_(k-1)||_2)²
            ---
            d y_k

        The derivative of a norm of a vector difference is quite simple, but
        care has to be taken to recognize, that y_k appears two times in the sum.
        It appears in the first summand for k and in the second summand for k+1.

            sum_k (||y_k - y_(k-1)||_2)²
                        1. term                 2. term
                = (||y_k - y_(k-1)||_2)² + (||y_(k+1) - y_k||_2)² + ... and so on

        The derivative of the first term is

            2 * (y_k - y_(k-1))

        and the derivative of the second term is

            -2 * (y_(k+1) - y_k))

        which is equal to

            2 * (y_k - y_(k+1)) .

        To summarize:

            d     sum_k(||y_k - y_(k-1)||_2)²
            ---
            d y_k

            =   2 * (2 * y_k - y_(k-1) - y_(k+1)) .
        
        Distances beyond endpoints of half-instanton are always zero due to periodicity,
        with derivative zero, which requires no special handling if boundary conditions
        are set appropriately. The "overcounting" here adds zero, which is no problem for
        the gradient, but leads to issues in the hessian.
        """
        image_coords = np.array([image.coords for image in self.images])
        kin_grad = (
            2
            * (
                2 * image_coords  # y_k
                - image_coords[self.ksm1]  # y_(k-1)
                - image_coords[self.ksp1]  # y_(k+1)
            ).flatten()
        )
        kin_grad *= 2 * self.P_bh * self.amu_to_me
        if self.scheduler_file:
            pot_grad = self.parallel_image_gradients().flatten()
        else:
            pot_grad = np.array(
                [
                    log_progress(image.gradient, "gradient", i)
                    for i, image in enumerate(self.images)
                ]
            ).flatten()
        pot_grad /= self.P_bh
        gradient = kin_grad + pot_grad
        # gradient = pot_grad
        # gradient = kin_grad
        # gradient = kin_grad / 2
        results = {"gradient": gradient}
        results.update(self.action())
        return results

    def action_hessian(self):
        if self.scheduler_file:
            image_hessians = self.parallel_image_hessians()
        else:
            image_hessians = [
                log_progress(image.hessian, "hessian", i)
                for i, image in enumerate(self.images)
            ]
        pot_hess = sp.linalg.block_diag(*image_hessians)
        pot_hess /= self.P_bh
        coord_num = pot_hess.shape[0]
        zeroes = np.zeros((coord_num, coord_num))
        image_coord_num = self.images[0].coords.size
        kronecker_delta = np.arange(image_coord_num)
        km = np.zeros((self.P, image_coord_num, self.P, image_coord_num))
        lm = np.zeros((self.P, image_coord_num, self.P, image_coord_num))
        for k, l in zip(self.ksm1, self.ks):
            km[k, :, l, :] = np.eye(image_coord_num)
            lm[l, :, k, :] = np.eye(image_coord_num)
        # Dirty hack - First k doesn't "have" k-1 in the optimization functional
        km[0, :, 0, :] = 0.0
        lm[0, :, 0, :] = 0.0
        kin_hess = (
            4 * np.eye(coord_num)
            - 2 * km.reshape((coord_num, coord_num))
            - 2 * lm.reshape((coord_num, coord_num))
        )
        # First and last k only appear once
        kin_hess[:image_coord_num, :image_coord_num] = 2 * np.eye(image_coord_num)
        kin_hess[
            (self.P - 1) * image_coord_num :, (self.P - 1) * image_coord_num :
        ] = 2 * np.eye(image_coord_num)
        kin_hess *= 2 * self.P_bh * self.amu_to_me
        hessian = kin_hess + pot_hess
        results = {"hessian": hessian, "kin": kin_hess, "pot": pot_hess}
        return results

    def full_action_hessian(self):
        """Returns action hessian of the full, redundant path in atu^-1."""
        if self.scheduler_file:
            image_hessians = self.parallel_image_hessians()
        else:
            image_hessians = [
                log_progress(image.hessian, "hessian", i)
                for i, image in enumerate(self.images)
            ]
        image_hessians = image_hessians + list(reversed(image_hessians))
        pot_hess = sp.linalg.block_diag(*image_hessians)
        pot_hess /= self.P_bh * 2 * self.amu_to_me

        coord_num = pot_hess.shape[0]

        image_coord_num = self.images[0].coords.size
        kin_hess = np.zeros((2 * self.P, image_coord_num, 2 * self.P, image_coord_num))
        for k in range(1, 2 * self.P):
            kin_hess[k, :, k, :] = 4 * np.eye(image_coord_num)
            kin_hess[k, :, k - 1, :] = -2 * np.eye(image_coord_num)
            kin_hess[k - 1, :, k, :] = -2 * np.eye(image_coord_num)
        kin_hess[0, :, 0, :] = 4 * np.eye(image_coord_num)
        kin_hess[0, :, 2 * self.P - 1, :] = -2 * np.eye(image_coord_num)
        kin_hess[2 * self.P - 1, :, 0, :] = -2 * np.eye(image_coord_num)
        kin_hess *= 2 * self.P_bh

        kin_hess = kin_hess.reshape((coord_num, coord_num))

        h = kin_hess / 2 + pot_hess
        return h

    def get_dask_client(self):
        return Client(scheduler_file=self.scheduler_file)

    def parallel_image_gradients(self):
        client = self.get_dask_client()
        image_futures = client.map(
            self.get_image_gradient, self.images, list(range(self.P))
        )
        self.set_images(list(range(self.P)), client.gather(image_futures))
        return np.array([image.gradient for image in self.images])

    def parallel_image_hessians(self):
        client = self.get_dask_client()
        image_futures = client.map(
            self.get_image_hessian, self.images, list(range(self.P))
        )
        self.set_images(list(range(self.P)), client.gather(image_futures))
        return np.array([image.hessian for image in self.images])

    def get_image_gradient(self, image, i):
        g = image.gradient
        return image

    def get_image_hessian(self, image, i):
        h = image.hessian
        return image

    def set_images(self, indices, images):
        for ind, image in zip(indices, images):
            self.images[ind] = image

    @property
    def energy(self):
        """Scales action to energy in hartree."""
        return self.action()["action"] / self.beta_hbar

    @property
    def gradient(self):
        """In hartree / (bohr * amu^0.5)"""
        return self.action_gradient()["gradient"] / self.beta_hbar

    @property
    def forces(self):
        return -self.gradient

    @property
    def hessian(self):
        """In hartree / (bohr^2 amu)"""
        return self.action_hessian()["hessian"] / self.beta_hbar

    @property
    def cart_hessian(self):
        return sp.linalg.block_diag(*[image.cart_hessian for image in self.images])

    @property
    def cart_coords(self):
        return np.ravel([image.cart_coords for image in self.images])

    @property
    def cart_forces(self):
        return np.ravel([image.cart_forces for image in self.images])

    @property
    def cart_hessian(self):
        return sp.linalg.block_diag(*[image.cart_forces for image in self.images])

    def is_analytical_2d(self):
        return self.images[0].is_analytical_2d

    def get_image_distances(self) -> NDArray:
        all_coords = np.array([image.coords for image in self.images])
        diffs = all_coords[self.ks] - all_coords[self.ksm1]
        dists = np.linalg.norm(diffs, axis=1)
        return dists

    def calc_instanton_rate(self, reactant_E, reactant_freqs):
        """Calculates the thermal instanton rate in s^-1, given
           the reactant vibrational frequencies in cm^-1."""
        # in Hartree, which is numerically equal in atomic units to "per atomic time",
        # differing only in factors of hbar - i.e. sqrt(lambda^V) in [4]
        rfreqs = (
            np.array(reactant_freqs)
            * 100
            / sp.constants.value("hartree-inverse meter relationship")
        )
        instanton_action_hessian = self.full_action_hessian()
        w, _ = np.linalg.eigh(instanton_action_hessian)
        w2, _ = np.linalg.eigh(self.action_hessian()["hessian"])
        print(f"w2: {w2[:6+2]}")
        w = np.sort(w)
        d = 1 / (2 * self.P_bh)
        sinfactors = (4 / d) * np.sin(
            np.pi * np.arange(1, 2 * self.P + 1) / (2 * self.P)
        ) ** 2
        dlv = d * rfreqs ** 2
        N0 = 6
        reactant_vals = np.sort(
            np.concatenate(
                (
                    np.add.outer(sinfactors, dlv).flatten(),
                    [1 / d] * (N0 * (2 * self.P - 1)),
                )
            )
        )
        print(len(reactant_vals))
        w = np.sort(np.abs(w))
        print(f"Dropping eigenvalues: {w[:N0+1]}")
        print(f"Next eigenvalue greater: {w[N0+1]}")
        path_vals = w[N0:]
        path_vals[0] = d
        path_vals = np.sort(path_vals)
        N = len(reactant_freqs) + N0
        assert len(path_vals) == len(reactant_vals)
        sumvals = fsum((np.log(reactant_vals) - np.log(path_vals))) / 2
        sumvals += N0 * np.log(self.P * 2)

        actioncalc = self.action(reactant_E)
        S0 = actioncalc["S0"]
        SE = actioncalc["action"]
        k_inst = (
            np.sqrt(S0 / (2 * np.pi))
            * np.exp(sumvals - SE)
            / sp.constants.value("atomic unit of time")
        )
        print(f"Action: {SE}")
        print(f"Action contrib: {np.exp(-SE)}")
        print(f"S0: {S0}")
        return k_inst

    @property
    def path_length(self):
        """Yields length of the discretized instanton path,
           which is only approximately half the instanton length."""
        return self.get_image_distances().sum()

    def tunneling_energy(self):
        dists = self.get_image_distances()
        Vbs = [
            self.images[i].energy
            - (dists[i] * dists[i + 1])
            / (2 * self.dks[i] * self.dks[i + 1])
            * self.amu_to_me
            for i in range(1, self.P - 1)
        ]
        return np.median(Vbs)

    def get_additional_print(self):
        length = self.path_length
        e = self.tunneling_energy()
        result = (
            f"\t\tInstanton length={length:.2f} √a̅m̅u̅·au"
        )  # \n\t\tTunneling energy={e:.6e} E_h"
        return result


class InstantonImage:
    def __init__(self, geometry: Geometry, cart_hessian, update_minstepsquared=1e-5):
        self._valid = {"energy": False, "gradient": False, "hessian": False}
        self._geometry = geometry
        self._prev_coords = geometry.coords.copy()
        self.set_cart_hessian(cart_hessian)
        self.update_minstepsquared = update_minstepsquared

    @property
    def coords(self):
        return self._geometry.coords

    @coords.setter
    def coords(self, new_coords):
        self._prev_coords = self._geometry.coords.copy()
        self._geometry.coords = new_coords
        for key in self._valid:
            self._valid[key] = False

    @property
    def coords3d(self):
        return self._geometry.coords3d

    @property
    def cart_coords(self):
        return self._geometry.cart_coords

    @property
    def atoms(self):
        return self._geometry.atoms

    @property
    def energy(self):
        if self._valid["energy"]:
            return self._energy
        else:
            self._energy = self._geometry.energy
            self._valid["energy"] = True
            return self._energy

    @property
    def gradient(self):
        if self._valid["gradient"]:
            return self._gradient.copy()
        else:
            try:
                self._prev_gradient = self._gradient.copy()
                self._gradient = self._geometry.gradient.copy()
            except:
                self._prev_gradient = self._geometry.gradient.copy()
                self._gradient = self._prev_gradient.copy()
            self._valid["gradient"] = True
            return self._gradient.copy()

    @property
    def forces(self):
        return -self.gradient

    @property
    def cart_forces(self):
        return self._geometry.cart_forces

    @property
    def hessian(self):
        if self._valid["hessian"]:
            return self._hessian
        else:
            dx = self.coords - self._prev_coords
            # Update hessian only with sufficient change in geometry,
            # as in [2].
            if dx @ dx > self.update_minstepsquared:
                dg = self.gradient - self._prev_gradient
                dH, _ = bofill_update(self._hessian, dx, dg)
                self._hessian += dH
            self._valid["hessian"] = True
            return self._hessian

    def set_calculator(self, value):
        self._geometry.set_calculator(value)

    def set_cart_hessian(self, value):
        self._geometry.cart_hessian = value
        self._hessian = self._geometry.internal.transform_hessian(value)
        self._valid["hessian"] = True

    def as_xyz(self):
        return self._geometry.as_xyz()
