# GRID is a numerical integration module for quantum chemistry.
#
# Copyright (C) 2011-2020 The GRID Development Team.
#
# This file is part of GRID.
#
# GRID is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GRID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Hirshfeld Weights Module."""


from importlib_resources import path

import numpy as np

from scipy.interpolate import CubicSpline


class HirshfeldWeights:
    """Hirshfeld weights functions holder class."""

    def __init__(self, expansion=None):
        """Initialize class.

        Parameters
        ----------
        expansion : dict, optional
            Dictionary specifying the linear expansion of proatomic densities for
            each atom in the molecule.The organization should be as it follows:

            {atomic position_1 : {charge_proatom_1 : contribution, charge_proatom_2 : contribution,
            …}, atomic position_2 : {charge_proatom_1 : contribution,
            charge_proatom_2 : contribution, …}, …}

            Example for methane:
            atom nums variable is [6 1 1 1 1]

            {'1' : {0.0: 0.8634485108888619}, '0': {-0.0: 0.45386513052363586, -1.0: 0.5461348694763641},
            '3': {0.0: 0.8634648321179402}, '2': {0.0: 0.8634611330830618}, '4': {0.0: 0.8634979080458045}}

            If none standard Hirshfeld weights will be generated
        """

        self._expansion = expansion

    @staticmethod
    def _load_npz_proatom(num):
        """Return radial grid points and neutral density for a given atomic number."""
        with path("grid.data.proatoms", f"a{num:03d}.npz") as fname:
            data = np.load(fname)
        return data["r"], data["dn"]

    @staticmethod
    def _get_proatom_density(num, coords_radial):
        """Evaluate density of pro-atom on the given points.

        Parameters
        ----------
        num: int
            Atomic number of pro-atom.
        coords_radial: np.ndarray(K,)
            Radial grid points.

        Returns
        -------
        np.ndarray(K,)
            Pro-atom densities evaluated on :math:`K` radial grid points.
        """
        # get pre-computed radial grid points & density of pro-atom
        rad, rho = HirshfeldWeights._load_npz_proatom(num)
        # interpolate pro-atom density
        cspline = CubicSpline(rad, rho, bc_type="natural", extrapolate=True)
        # evaluate pro-atom density on the given points
        out = cspline(coords_radial).flatten()
        return out

    @staticmethod
    def generate_proatom(points, coord, num):
        """Evaluate pro-atom densities on the given grid points.

        Parameters
        ----------
        points: np.ndarray(N, 3)
            Cartesian coordinates of :math:`N` grid points.
        coord: np.ndarray(3,)
            Cartesian coordinates of the atom.
        num: int
            Atomic number of the atom.

        Returns
        -------
        np.ndarray(N,)
            Pro-atom densities evaluated on :math:`N` grid points.
        """
        # compute distance of grid points from atom
        dist = np.linalg.norm(points[:, None] - coord, axis=-1)
        # compute pro-atom density
        return HirshfeldWeights._get_proatom_density(num, dist)

    def __call__(self, points, atom_coords, atom_nums, indices):
        """Evaluate integration weights on the given grid points.

        Parameters
        ----------
        points: np.ndarray(N,3)
            Cartesian coordinates of :math:`N` grid points.
        atom_coords: np.ndarray(M,3)
            Cartesian coordinates of the :math:`M` atoms in molecule.
        atom_nums: np.ndarray(M,)
            Atomic number of :math:`M` atoms in molecule.
        indices : np.ndarray(M+1,)
            Indices of atomic grid points for each :math:`M` atoms in molecule.

        Return
        ------
        np.ndarray(N,)
            Hirshfeld integration weights evaluated on :math:`N` grid points.
        """
        expansion = self._expansion
        aim_weights = np.zeros(len(points))
        promolecule = np.zeros(len(points))
        # evaluate pro-atom densities & pro-molecule density
        for index, atom_num in enumerate(atom_nums):
            if expansion is None:
                proatom = self.generate_proatom(
                    points, atom_coords[index], atom_num
                )
            else:
                proatom = self.generate_proatom(
                    points, atom_coords[index], atom_num, expansion[str(index)]
                )
            promolecule += proatom
            start, end = indices[index], indices[index + 1]
            aim_weights[start:end] = proatom[start:end]
        # evaluate weights
        aim_weights /= promolecule
        # check weights are all positive
        # assert np.all(aim_weights >= 0), np.min(aim_weights)
        return aim_weights
