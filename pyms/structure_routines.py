"""
The structures module.

A collection of functions and classes for reading in and manipulating structures
and creating potential arrays for multislice simulation.
"""

import itertools
import ase
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
from re import split, match
from os.path import splitext
from .atomic_scattering_params import (
    e_scattering_factors,
    atomic_symbol,
    e_scattering_factors_WK,
    EELS_EDX_1s_data,
    EELS_EDX_2s_data,
    EELS_EDX_2p_data,
)
from .Probe import wavev, relativistic_mass_correction
from .utils.numpy_utils import q_space_array, ensure_array
from .utils.torch_utils import (
    sinc,
    get_device,
    bandwidth_limit_array_torch,
    complex_to_real_dtype_torch,
)
from . import _float, _int, _uint
import scipy.special as sc


def remove_common_factors(nums):
    """Remove common divisible factors from a set of numbers."""
    nums = np.asarray(nums, dtype=_int)
    g_ = np.gcd.reduce(nums)
    while g_ > 1:
        nums //= g_
        g_ = np.gcd.reduce(nums)
    return nums


def psuedo_rational_tiling(dim1, dim2, EPS):
    """
    Calculate the pseudo-rational tiling for matching objects of different dimensions.

    For two dimensions, dim1 and dim2, work out the multiplicative
    tiling so that those dimensions might be matched to within error EPS.
    """
    if np.any([dim1 < EPS, dim2 < EPS]):
        return 1, 1
    tile1 = int(np.round(np.abs(dim2 / dim1) / EPS))
    tile2 = int(np.round(1 / EPS))
    return remove_common_factors([tile1, tile2])


def Xray_scattering_factor(Z, gsq, units="A"):
    """
    Calculate the X-ray scattering factor for atom with atomic number Z.

    Parameters
    ----------
    Z : int
        Atomic number of atom of interest.
    gsq : float or array_like
        Reciprocal space value(s) in Angstrom squared at which to evaluate the
        X-ray scattering factor.
    units : string, optional
        Units in which to calculate X-ray scattering factor, can be 'A' for
        Angstrom, or 'VA' for volt-Angstrom.
    """
    # Bohr radius in Angstrom
    a0 = 0.529177
    # gsq = g**2
    return Z - 2 * np.pi**2 * a0 * gsq * electron_scattering_factor(
        Z, gsq, units=units
    )


def electron_scattering_factor(Z, gsq, units="VA"):
    """
    Calculate the electron scattering factor for atom with atomic number Z.

    Parameters
    ----------
    Z : int
        Atomic number of atom of interest.
    gsq : float or array_like
        Reciprocal space value(s) in Angstrom squared at which to evaluate the
        electron scattering factor.
    units : string, optional
        Units in which to calculate electron scattering factor, can be 'A' for
        Angstrom, or 'VA' for volt-Angstrom.
    """
    ai = e_scattering_factors[Z - 1, 0:10:2]
    bi = e_scattering_factors[Z - 1, 1:10:2]

    # Planck's constant in kg Angstrom/s
    h = 6.62607004e-24
    # Electron rest mass in kg
    me = 9.10938356e-31
    # Electron charge in Coulomb
    qe = 1.60217662e-19

    fe = np.zeros_like(gsq)

    for i in range(5):
        fe += ai[i] * (2 + bi[i] * gsq) / (1 + bi[i] * gsq) ** 2

    # Result can be returned in units of Volt Angstrom ('VA') or Angstrom ('A')
    if units == "VA":
        return h**2 / (2 * np.pi * me * qe) * fe
    elif units == "A":
        return fe


def calculate_scattering_factors(gridshape, gridsize, elements):
    """Calculate the electron scattering factors on a reciprocal space grid.

    Parameters
    ----------
    gridshape : (2,) array_like
        pixel size of the grid
    gridsize : (2,) array_like
        Lateral real space sizing of the grid in Angstrom
    elements : (M,) array_like
        List of elements for which electron scattering factors are required

    Returns
    -------
    fe : (M, *gridshape)
        Array of electron scattering factors in reciprocal space for each
        element
    """
    # Get reciprocal space array
    g = q_space_array(gridshape, gridsize)
    gsq = np.square(g[0]) + np.square(g[1])

    # Initialise scattering factor array
    fe = np.zeros((len(elements), *gridshape), dtype=_float)

    # Loop over unique elements
    for ielement, element in enumerate(elements):
        fe[ielement, :, :] = electron_scattering_factor(element, gsq)

    return fe


def find_equivalent_sites(positions, EPS=1e-3):
    """Find equivalent atomic sites in a list of atomic positions object.

    This function is used to detect two atoms sharing the same positions (are
    with EPS of each other) with fractional occupancy, and return an index of
    these equivalent sites.
    """
    # Import  the pair-wise distance function from scipy
    from scipy.spatial.distance import pdist

    natoms = positions.shape[0]
    # Calculate pairwise distance between each atomic site
    distance_matrix = pdist(positions)

    # Initialize index of equivalent sites (initially assume all sites are
    # independent)
    equivalent_sites = np.arange(natoms, dtype=_int)

    # Find equivalent sites
    equiv = distance_matrix < EPS

    # If there are equivalent sites correct the index of equivalent sites
    if np.any(equiv):
        # Masking function to get indices from distance_matrix
        iu = np.mask_indices(natoms, np.triu, 1)

        # Get a list of equivalent sites
        sites = np.nonzero(equiv)[0]
        for site in sites:
            # Use the masking function to
            equivalent_sites[iu[1][site]] = iu[0][site]
    return equivalent_sites


def interaction_constant(E, units="rad/VA"):
    """
    Calculate the electron interaction constant, sigma.

    The electron interaction constant converts electrostatic potential (in V
    Angstrom) to radians. Units of this constant are rad/(V Angstrom).  See
    Eq. (2.5) in Kirkland's Advanced Computing in electron microscopy.
    """
    # Planck's constant in kg Angstrom /s
    h = 6.62607004e-24
    # Electron rest mass in kg
    me = 9.10938356e-31
    # Electron charge in Coulomb
    qe = 1.60217662e-19
    # Electron wave number (reciprocal of wavelength) in Angstrom
    k0 = wavev(E)
    # Relativistic electron mass correction
    gamma = relativistic_mass_correction(E)
    if units == "rad/VA":
        return 2 * np.pi * gamma * me * qe / k0 / h / h
    elif units == "rad/A":
        return gamma / k0


def change_of_basis(coords, newuc, olduc):
    """Change of basis for structure unit cell."""
    return np.mod(coords[:, :3] @ olduc @ np.linalg.inv(newuc), 1.0)


def rot_matrix(theta, u=np.asarray([0, 0, 1], dtype=_float)):
    """
    Generate a 3D rotational matrix.

    Parameters
    ----------
    theta : float
        Angle of rotation in radians
    u : (3,) array_like
        Axis of rotation
    """
    from numpy import sin, cos

    c = cos(theta)
    s = sin(theta)
    ux, uy, uz = u / np.linalg.norm(u)
    R = np.zeros((3, 3))
    R[0, :] = [
        c + ux * ux * (1 - c),
        ux * uy * (1 - c) - uz * s,
        ux * uz * (1 - c) + uy * s,
    ]
    R[1, :] = [
        uy * uz * (1 - c) + uz * s,
        c + uy * uy * (1 - c),
        uy * uz * (1 - c) - ux * s,
    ]
    R[2, :] = [
        uz * ux * (1 - c) - uy * s,
        uz * uy * (1 - c) + ux * s,
        c + uz * uz * (1 - c),
    ]
    return R


class structure:
    """
    Class for simulation objects.

    Elements in a structure object:
    unitcell :
        An array containing the side lengths of the orthorhombic unit cell
    atoms :
        An array of dimensions total number of atoms by 6 which for each atom
        contains the fractional coordinates within the unit cell for each atom
        in the first three entries, the atomic number in the fourth entry,
        the atomic occupancy (not yet implemented in the multislice) in the
        fifth entry and mean squared atomic displacement in the sixth entry
    Title :
        Short description of the object of output purposes
    """

    def __init__(self, unitcell, atoms, dwf, occ=None, Title="", EPS=1e-2):
        """
        Initialize a structure object with necessary variables.

        This method sets up the structures object with the provided unit cell, atomic positions, Debye-Waller factors,
        occupancies, and other related parameters. It ensures that the unit cell is orthorhombic and calculates an orthorhombic
        tiling if it is not.

        Parameters:
        ----------
        unitcell : array-like (3,) or (3,3) array_like
            The unit cell dimensions or a 3x3 matrix with rows describing the unit cell edges.
        atoms : (natoms,3) array-like
            Atomic positions in the unit cell.
        dwf : (natoms,) array-like or None
            Debye-Waller factors for the atoms. If None assume root-mean-square vibration of 1 Angstrom
        occ : array-like, optional
            Occupancies of the atoms. If None, all atoms are assumed to have full occupancy. Default is None.
        Title : str, optional
            A title for the simulation. Default is an empty string.
        EPS : float, optional
            A small value used for numerical tolerance in determining if the unit cell is orthorhombic. Default is 1e-2.

        Attributes:
        ----------
        unitcell : numpy.ndarray
            The unit cell dimensions.
        atoms : numpy.ndarray
            Atomic positions, occupancies, and Debye-Waller factors combined in one array.
        Title : str
            The title of the simulation.
        fractional_occupancy : bool
            Indicates if there is any fractional occupancy of atom sites in the sample.

        Notes:
        -----
        - If the unit cell is given as a 3x3 matrix, the method checks if it is orthorhombic.
        - If the unit cell is not orthorhombic, the method attempts a pseudo-rational tiling to convert it to an orthorhombic cell.
        - The `atoms` array is augmented to include occupancies and Debye-Waller factors for each atom.

        Example:
        -------
        >>> unitcell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        >>> atoms = [[0, 0, 0], [0.5, 0.5, 0.5]]
        >>> dwf = [0.05, 0.05]
        >>> occ = [1.0, 0.5]
        >>> struc = pyms.structure(unitcell, atoms, dwf, occ, Title="Example structure")
        """
        self.unitcell = np.asarray(unitcell)
        natoms = np.asarray(atoms).shape[0]

        if occ is None:
            occ = np.ones(natoms)

        if dwf is None:
            dwf = np.ones(natoms) * 0.01

        self.atoms = np.concatenate(
            [atoms, occ.reshape(natoms, 1), np.asarray(dwf).reshape(natoms, 1)], axis=1
        )
        self.Title = Title

        # Up till now unitcell can be a 3 x 3 matrix with rows describing the
        # unit cell edges. If this is the case we need to make sure that the
        # unit cell is orthorhombic and find an orthorhombic tiling if this it
        # is not orthorhombic
        if self.unitcell.ndim > 1:
            # Check to see if unit cell is orthorhombic
            ortho = np.abs(np.sum(self.unitcell) - np.trace(self.unitcell)) < EPS

            if ortho:
                # If unit cell is orthorhombic then extract unit cell
                # dimension
                self.unitcell = np.diag(self.unitcell)
            else:
                # If not orthorhombic attempt psuedo rational tiling
                self.orthorhombic_supercell(EPS=EPS)

        # Check if there is any fractional occupancy of atom sites in
        # the sample
        self.fractional_occupancy = np.any(np.abs(self.atoms[:, 4] - 1.0) > 1e-3)

    @classmethod
    def from_materials_project_api(cls, MPid, MPIkey):
        from mp_api.client import MPRester

        # Pull Materials project ID from online database
        with MPRester(api_key=MPIkey) as mpr:
            data = mpr.materials.search(material_ids=[MPid])[
                0
            ].structure.to_conventional()

        # Extract, the coordinates and atomic weights (Z)
        coord = data.frac_coords
        natoms = coord.shape[0]
        Z = np.asarray(data.atomic_numbers).reshape((natoms, 1))

        # Join coordinate and atomic weight (Z) arrays
        atoms = np.concatenate((coord, Z), axis=1)

        # Extract unit cell vectors
        unitcell = np.asarray(data.lattice.matrix)

        # Set title to be chemical formula
        Title = data.alphabetical_formula

        return cls(unitcell, atoms, None, None, Title)

    @classmethod
    def fromfile(
        cls,
        fnam,
        temperature_factor_units="ums",
        atomic_coordinates="fractional",
        EPS=1e-2,
        T=None,
    ):
        """
        Read in a simulation object from a structure file.

        Appropriate structure files include *.p1 files, which is outputted by
        the vesta software:

        K. Momma and F. Izumi, "VESTA 3 for three-dimensional visualization of
        crystal, volumetric and morphology data," J. Appl. Crystallogr., 44,
        1272-1276 (2011).

        or a *.xyz file in the standard of the prismatic software

        or a *.xtl file in the standard of the muSTEM software NOT VESTA

        Parameters
        ----------
        fnam : string
            Filepath of the structure file
        temperature_factor_units : string,optional
            Units of the Debye-Waller temperature factors in the structure file
            appropriate inputs are B (crystallographic temperature factor),
            urms (root mean squared displacement) and ums (mean squared
            displacement, the default)
        atomic_coordinates : string, optional
            Units of the atomic coordinates can be "fractional" or "cartesian"
        EPS : float,optional
            Tolerance for procedures such as pseudo-rational tiling for
            non-orthorhombic crystal unit cells
        T : (3,3) array_like or None
            An optional transformation matrix to be applied to the unit cell
            and the atomic coordinates
        """
        f = open(fnam, "r")

        ext = splitext(fnam)[1].lower()

        # Read title
        Title = f.readline().strip()

        if ext == ".p1":

            # I have no idea what the second line in the p1 file format means
            # so ignore it
            f.readline()

            # Get unit cell vector - WARNING assume an orthorhombic unit cell
            unitcell = np.loadtxt(f, max_rows=3, dtype=_float)

            # Get the atomic symbol of each element
            atomtypes = np.loadtxt(f, max_rows=1, dtype=str, ndmin=1)  # noqa

            # Get the number of atoms of each type
            natoms = np.loadtxt(f, max_rows=1, dtype=int, ndmin=1)

            # Skip empty line
            f.readline()

            # Total number of atoms
            totnatoms = np.sum(natoms)
            # Initialize array containing atomic information
            atoms = np.zeros((totnatoms, 6))
            dwf = np.zeros((totnatoms,))
            occ = np.zeros((totnatoms,))

            for i in range(totnatoms):
                atominfo = split(r"\s+", f.readline().strip())[:6]
                # First three entries are the atomic coordinates
                atoms[i, :3] = np.asarray(atominfo[:3], dtype=_float)
                # Fourth entry is the atomic symbol
                atoms[i, 3] = atomic_symbol.index(
                    match("([A-Za-z]+)", atominfo[3]).group(0)
                )
                # Final entries are the fractional occupancy and the temperature
                # (Debye-Waller) factor
                occ[i] = atominfo[4]
                dwf[i] = atominfo[5]

        elif ext == ".xyz":
            # Read in unit cell dimensions
            unitcell = np.asarray(
                [float(x) for x in split(r"\s+", f.readline().strip())[:3]]
            )

            atoms = []
            for line in f:

                # Look for end of file marker
                if line.strip() == "-1":
                    break
                # Otherwise parse line
                atoms.append(
                    np.array([float(x) for x in split(r"\s+", line.strip())[:6]])
                )

            # Now stack all atoms into numpy array
            atoms_ = np.stack(atoms, axis=0)

            # Rearrange columns of numpy array to match standard
            totnatoms = atoms_.shape[0]
            atoms = np.zeros((totnatoms, 4))
            # Atomic coordinates
            atoms[:, :3] = atoms_[:, 1:4]
            # Atomic numbers (Z)
            atoms[:, 3] = atoms_[:, 0]
            # Fractional occupancy and Debye-Waller (temperature) factor
            dwf = atoms_[:, 5]
            occ = atoms_[:, 4]
        elif ext == ".xtl":
            # Read in unit cell dimensions
            unitcell = np.asarray(
                [float(x) for x in split(r"\s+", f.readline().strip())[:3]]
            )

            keV = float(f.readline().split("\n")[0])

            # Read in number of elements in structure file
            ntypes = int(f.readline().strip())

            atoms = np.zeros((1, 6))
            for i in range(ntypes):
                symb = f.readline().strip()
                line = f.readline().split()
                no_atoms = int(line[0])
                Z, occ, dwf = [float(x) for x in line[1:]]
                posnlist = []
                for j in range(no_atoms):
                    posnlist.append(
                        [float(x) for x in f.readline().split()] + [Z, occ, dwf]
                    )
                atoms = np.vstack([atoms, posnlist])

            # Fractional occupancy and Debye-Waller (temperature) factor
            occ = atoms[1:, 4]
            dwf = atoms[1:, 5]
            # Reduce array to (x, y, z, Z) for fractional coordinates (x,y,z) and atomic number Z
            atoms = atoms[1:, :4]
        else:
            print("File extension: {0} not recognized".format(ext))
            return None

        # Close file
        f.close()

        # If temperature factors are given in any other format than mean square
        # (ums) convert to mean square. Acceptable formats are crystallographic
        # temperature factor B and root mean square (urms) displacements
        if temperature_factor_units == "B":
            dwf *= 1 / (8 * np.pi**2)
        elif temperature_factor_units == "urms":
            dwf = dwf**2
        elif temperature_factor_units == "ums":
            pass
        else:
            raise ValueError("Unrecognized temperature factor units")

        # If necessary, Convert atomic positions to fractional coordinates
        if atomic_coordinates == "cartesian":
            atoms[:, :3] /= unitcell[:3][np.newaxis, :]
            atoms[:, :3] = atoms[:, :3] % 1.0

        if T is not None:
            # Transform atoms to cartesian basis and then apply transformation
            # matrix
            atoms[:, :3] = (T @ unitcell @ atoms[:, :3].T).T

            # Apply transformation matrix to unit-cell
            unitcell = unitcell @ T.T

            # Apply inverse of unit cell
            atoms[:, :3] = (np.linalg.inv(unitcell) @ atoms[:, :3].T).T

        return cls(unitcell, atoms[:, :4], dwf, occ, Title, EPS=EPS)

    @classmethod
    def from_ase_cluster(cls, asecell, occupancy=None, Title="", dwf=None):
        """Initialize from Atomic Simulation Environment (ASE) cluster object."""
        unitcell = asecell.cell[:]
        # Sometimes there is no unit cell provided so we have to estimate it
        nounitcell = np.isclose(np.prod(unitcell), 0.0)
        if nounitcell:
            unitcell = np.ptp(asecell.positions, axis=0)
            atompositions = (
                np.mod(asecell.positions - np.amin(asecell.positions, axis=0), unitcell)
                / unitcell
            )
        else:
            atompositions = asecell.cell.scaled_positions(asecell.positions)

        natoms = asecell.numbers.shape[0]
        atoms = np.concatenate(
            [
                atompositions,
                asecell.numbers.reshape(natoms, 1),
            ],
            axis=1,
        )
        if occupancy is None:
            occ = np.ones(natoms)
        if dwf is None:
            dwf = np.ones(natoms) * 3 / np.pi**2 / 8
        return cls(unitcell, atoms, dwf, occ, Title)

    def to_ase_atoms(self):
        """Convert structure to Atomic Simulation Environment (ASE) atoms object."""
        scaled_positions = self.atoms[:, :3]
        numbers = self.atoms[:, 3].astype(_int)
        cell = self.unitcell
        pbc = [True, True, True]
        return ase.Atoms(
            scaled_positions=scaled_positions, numbers=numbers, cell=cell, pbc=pbc
        )

    def orthorhombic_supercell(self, EPS=1e-2):
        """
        Create an orthorhombic supercell from a monoclinic crystal unit cell.

        If not orthorhombic attempt pseudo rational tiling of general
        monoclinic structure. Assumes that the self.unitcell matrix is lower
        triangular.
        """
        # if not np.abs(np.dot(self.unitcell[0], self.unitcell[1])) < EPS:

        # Rotate a to align with the [1,0,0] axis
        xhat = np.asarray([1, 0, 0])
        theta = np.arccos(self.unitcell[0, 0] / np.linalg.norm(self.unitcell[0]))
        u = np.cross(self.unitcell[0], xhat)

        # Only rotate if a larger angle is required (otherwise u goes to 0)
        if theta > EPS:
            R = rot_matrix(theta, u)
            self.unitcell = copy.deepcopy(self.unitcell) @ R.T

        # Rotate b to ensure it is in the [1,0,0] x [0,1,0] plane
        theta = np.arctan2(self.unitcell[1, 2], self.unitcell[1, 1])
        R = rot_matrix(theta, np.asarray([1, 0, 0]))
        self.unitcell = copy.deepcopy(self.unitcell) @ R.T

        # Do a psuedo rational tiling to make the structure orthorhombic
        # in the [1,0,0] x [0,1,0] plane
        self.psuedo_rational_tiling(0, 1, EPS)

        # Now do psuedo-rational tilings of z with respect to x and y
        # to fit whole 3d structure into an orthorhombic supercell
        self.psuedo_rational_tiling(0, 2, EPS)
        self.psuedo_rational_tiling(1, 2, EPS)

        # Now we can take the diagnoal values of the
        self.unitcell = np.diag(self.unitcell)

    def psuedo_rational_tiling(self, dim1, dim2, EPS=1e-2):
        """
        Calculate the pseudo-rational tiling for matching objects of different dimensions.

        For two dimensions, dim1 and dim2, work out the multiplicative
        tiling so that those dimensions might be matched to within error EPS.
        """
        # Catch case where the unit cell vectors are already orthogonal
        # and nothing needs to be done
        mag1 = np.linalg.norm(self.unitcell[dim1])
        mag2 = np.linalg.norm(self.unitcell[dim1])

        if np.abs(np.dot(self.unitcell[dim1], self.unitcell[dim2]) / mag1 / mag2) < EPS:
            return

        # d2 is projection of second unit cell vector onto first
        d2 = np.dot(self.unitcell[dim1], self.unitcell[dim2]) / mag2

        tile1 = int(np.round(np.abs(d2 / mag1) / EPS))
        tile2 = int(np.round(1 / EPS))
        tile1, tile2 = remove_common_factors([tile1, tile2])

        # Make deepcopy of old unit cell
        tiling = [1, 1, 1]
        tiling[dim1] = tile1
        tiling[dim2] = tile2

        # Tile out atoms, save a copy of the "old" unit cell
        self.tile(*tiling)
        olduc = copy.deepcopy(self.unitcell)

        # Now remove the projection vector 2 on vector 1 from vector 2
        self.unitcell[dim2] -= (
            np.dot(self.unitcell[dim1], self.unitcell[dim2])
            / np.dot(self.unitcell[dim1], self.unitcell[dim1])
            * self.unitcell[dim1]
        )

        # # Now calculate fractional coordinates in new orthorhombic cell
        self.atoms[:, :3] = np.mod(
            self.atoms[:, :3] @ olduc @ np.linalg.inv(self.unitcell), 1.0
        )

    def quickplot(
        self,
        atomscale=None,
        cmap=plt.get_cmap("Dark2"),
        tiling=[1, 1, 1],
        block=True,
        colors=None,
        aspect=True,
    ):
        """
        Make a quick 3D scatter plot of the atomic sites within the structure.

        For more detailed visualization output the structure file to a file format
        readable by the Vesta software using output_vesta_xtl
        """
        from mpl_toolkits.mplot3d import Axes3D  # NOQA

        tiledcopy = copy.deepcopy(self).tile(*tiling)

        if atomscale is None:
            atomscale = 1e-3 * np.amax(tiledcopy.unitcell)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if colors is None:
            colors = cmap(tiledcopy.atoms[:, 3] / np.amax(tiledcopy.atoms[:, 3]))
        sizes = tiledcopy.atoms[:, 3] * atomscale

        xs, ys, zs = [tiledcopy.atoms[:, i] * tiledcopy.unitcell[i] for i in [1, 0, 2]]
        ax.scatter(xs, ys, zs, c=colors, s=sizes)
        for ele, iele in zip(*np.unique(self.atoms[:, 3], return_index=True)):
            ax.scatter(
                [],
                [],
                [],
                color=colors[iele],
                label=atomic_symbol[int(ele)],
                s=sizes[iele],
            )

        ax.set_xlim3d(0.0, tiledcopy.unitcell[1])
        ax.set_ylim3d(top=0.0, bottom=tiledcopy.unitcell[0])
        ax.set_zlim3d(top=0.0, bottom=tiledcopy.unitcell[2])
        if aspect:
            ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.set_xlabel("x (\\A)")
        ax.set_ylabel("y (\\A)")
        ax.set_zlabel("z (\\A)")
        ax.legend()

        plt.show(block=block)
        return fig

    def output_vesta_xtl(self, fnam):
        """Output an .xtl file which is viewable by the vesta software.

        See K. Momma and F. Izumi, "VESTA 3 for three-dimensional visualization
        of crystal, volumetric and morphology data," J. Appl. Crystallogr., 44,
        1272-1276 (2011).

        Warning: Vesta xtl files do not contain fractional occupancy information
        """
        f = open(splitext(fnam)[0] + ".xtl", "w")
        f.write("TITLE " + self.Title + "\n CELL \n")
        f.write("  {0:.5f} {1:.5f} {2:.5f} 90 90 90\n".format(*self.unitcell))
        f.write("SYMMETRY NUMBER 1\n SYMMETRY LABEL  P1\n ATOMS \n")
        f.write("NAME         X           Y           Z" + "\n")
        for i in range(self.atoms.shape[0]):
            f.write(
                "{0} {1:.4f} {2:.4f} {3:.4f}\n".format(
                    atomic_symbol[int(self.atoms[i, 3])], *self.atoms[i, :3]
                )
            )
        f.write("EOF")
        f.close()

    def output_xyz(
        self, fnam, atomic_coordinates="cartesian", temperature_factor_units="sqrturms"
    ):
        """
        Output an .xyz structure file.

        This is the input format used by Kirkland's EM codes and the prismatic
        software.
        """
        f = open(splitext(fnam)[0] + ".xyz", "w")
        f.write(self.Title + "\n {0:.4f} {1:.4f} {2:.4f}\n".format(*self.unitcell))

        if atomic_coordinates == "cartesian":
            coords = self.atoms[:, :3] * self.unitcell
        else:
            coords = self.atoms[:, :3]

        # If temperature factors are given as B then convert to urms
        if temperature_factor_units == "B":
            DWFs = self.atoms[:, 5] * 8 * np.pi**2
        elif temperature_factor_units == "sqrturms":
            DWFs = np.sqrt(self.atoms[:, 5])

        for coord, atom, DWF in zip(coords, self.atoms, DWFs):
            f.write(
                "{0:d} {1:.4f} {2:.4f} {3:.4f} {4:.2f}  {5:.3f}\n".format(
                    int(atom[3]), *coord, atom[4], DWF
                )
            )
        f.write("-1")
        f.close()

    def make_potential(
        self,
        pixels,
        subslices=[1.0],
        tiling=[1, 1],
        displacements=True,
        fractional_occupancy=True,
        sinc_deconvolution=True,
        bandwidthlimit=0.2,
        fe=None,
        device=None,
        dtype=torch.float32,
        seed=None,
    ):
        """
        Generate the projected potential of the structure.

        Calculate the projected electrostatic potential for a structure on a
        pixel grid with dimensions specified by pixels. Subslicing the unit
        cell is achieved by passing an array subslices that contains as its
        entries the depths at which each subslice should be terminated in units
        of fractional coordinates. Tiling of the unit cell (often necessary to
        make a sufficiently large simulation grid to fit the probe) is achieved
        by passing the tiling factors in the array tiling.

        Parameters
        ----------
        pixels: int, (2,) array_like
            The pixel size of the grid on which to calculate the projected
            potentials
        subslices: float, array_like, optional
            An array containing the depths at which each slice ends as a fraction
            of the simulation unit-cell
        tiling: int, (2,) array_like, optional
            Tiling of the simulation object (often necessary to  make a
            sufficiently large simulation grid to fit the probe)
        displacements: bool, optional
            Pass displacements = False to turn off random displacements of the
            atoms due to thermal motion
        fractional_occupancy: bool, optional
            Pass fractional_occupancy = False to turn off fractional occupancy
            of atomic sites
        fe: float, array_like
            An array containing the electron scattering factors for the elements
            in the structure as calculated by the function
            calculate_scattering_factors, can be passed to save recalculating
            each time new potentials are generated
        device: torch.device
            Allows the user to control which device the calculations will occur
            on
        dtype: torch.dtype
            Controls the data-type of the output
        seed: int
            Seed for random number generator for atomic displacements.
        """
        # Get equivalent precison real datatype
        realdtype = complex_to_real_dtype_torch(dtype)

        # Force the subslices to be an array
        sblce = ensure_array(subslices)

        # Initialize device cuda if available, CPU if no cuda is available
        device = get_device(device)

        # Ensure pixels is integer
        pixels_ = [int(x) for x in pixels]
        # Seed random number generator for displacements
        if seed is not None:
            torch.manual_seed(seed)

        tiling_ = np.asarray(tiling[:2])
        gsize = np.asarray(self.unitcell[:2]) * tiling_
        psize = np.asarray(pixels_)

        pixperA = np.asarray(pixels_) / np.asarray(self.unitcell[:2]) / tiling_

        # Get a list of unique atomic elements
        elements = list(set(np.asarray(self.atoms[:, 3], dtype=_int)))

        # Get number of unique atomic elements
        nelements = len(elements)
        nsubslices = len(sblce)
        # Build list of equivalent sites if Fractional occupancy is to be
        # taken into account
        if fractional_occupancy and self.fractional_occupancy:
            equivalent_sites = find_equivalent_sites(self.atoms[:, :3], EPS=1e-3)

        # FDES method
        # Initialize potential array
        P = torch.zeros(
            np.prod([nelements, nsubslices, *pixels_]), device=device, dtype=realdtype
        )

        # Construct a map of which atom corresponds to which slice
        islice = np.zeros((self.atoms.shape[0]), dtype=_int)
        slice_stride = np.prod(pixels_)
        # if nsubslices > 1:
        # Finds which slice atom can be found in
        # WARNING Assumes that the slices list ends with 1.0 and is in
        # ascending order
        for i in range(nsubslices):
            zmin = 0 if i == 0 else sblce[i - 1]
            atoms_in_slice = (self.atoms[:, 2] % 1.0 >= zmin) & (
                self.atoms[:, 2] % 1.0 < sblce[i]
            )
            islice[atoms_in_slice] = i * slice_stride
        islice = torch.from_numpy(islice).type(torch.long).to(device)
        # else:
        #     islice = 0
        # Make map a pytorch Tensor

        # Construct a map of which atom corresponds to which element
        element_stride = nsubslices * slice_stride
        ielement = torch.tensor(
            [
                element_stride * elements.index(int(self.atoms[iatom, 3]))
                for iatom in range(self.atoms.shape[0])
            ],
            dtype=torch.long,
            device=device,
        )

        if displacements:
            # Generate thermal displacements
            urms = torch.tensor(
                np.sqrt(self.atoms[:, 5])[:, np.newaxis] * pixperA[np.newaxis, :],
                dtype=realdtype,
                device=device,
            ).view(self.atoms.shape[0], 2)

        # FDES algorithm implemented using the pytorch scatter_add function,
        # which takes a list of numbers and adds them to a corresponding list
        # of coordinates
        for tile in range(tiling[0] * tiling[1]):
            # For these atomic coordinates (in fractional coordinates) convert
            # to pixel coordinates
            posn = (
                (
                    self.atoms[:, :2]
                    + np.asarray([tile % tiling[0], tile // tiling[0]])[np.newaxis, :]
                )
                / tiling_
                * psize
            )
            posn = torch.from_numpy(posn).to(device).type(realdtype)

            if displacements:

                # Add displacement sampled from normal distribution to account
                # for atomic thermal motion
                disp = (
                    torch.randn(self.atoms.shape[0], 2, dtype=realdtype, device=device)
                    * urms
                )

                # If using fractional occupancy force atoms occupying equivalent
                # sites to have the same displacement
                if fractional_occupancy and self.fractional_occupancy:
                    disp = disp[equivalent_sites, :]

                posn[:, :2] += disp

            yc = (
                torch.remainder(torch.ceil(posn[:, 0]).type(torch.long), pixels_[0])
                * pixels_[1]
            )
            yf = (
                torch.remainder(torch.floor(posn[:, 0]).type(torch.long), pixels_[0])
                * pixels_[1]
            )
            xc = torch.remainder(torch.ceil(posn[:, 1]).type(torch.long), pixels_[1])
            xf = torch.remainder(torch.floor(posn[:, 1]).type(torch.long), pixels_[1])

            yh = torch.remainder(posn[:, 0], 1.0)
            yl = 1.0 - yh
            xh = torch.remainder(posn[:, 1], 1.0)
            xl = 1.0 - xh

            # Account for fractional occupancy of atomic sites if requested
            if fractional_occupancy and self.fractional_occupancy:
                xh *= torch.from_numpy(self.atoms[:, 4]).type(realdtype).to(device)
                xl *= torch.from_numpy(self.atoms[:, 4]).type(realdtype).to(device)

            # Each pixel is set to the overlap of a shifted rectangle in that pixel
            P.scatter_add_(0, ielement + islice + yc + xc, yh * xh)
            P.scatter_add_(0, ielement + islice + yc + xf, yh * xl)
            P.scatter_add_(0, ielement + islice + yf + xc, yl * xh)
            P.scatter_add_(0, ielement + islice + yf + xf, yl * xl)

        # Now view potential as a 4D array for next bit
        P = P.view(nelements, nsubslices, *pixels_)

        # Use real fast fourier transforms to save memory and time
        P = torch.fft.rfft2(P, s=pixels_)

        # pp is number of pixels in x
        pp = pixels_[1] // 2 + 1

        if sinc_deconvolution:
            # Make sinc functions with appropriate singleton dimensions for pytorch
            # broadcasting /gridsize[0]*pixels_[0] /gridsize[1]*pixels_[1]
            sincy = (
                sinc(torch.fft.fftfreq(pixels_[0]))
                .view([1, 1, pixels_[0], 1])
                .to(device)
                .type(realdtype)
            )

            # The real FFT is only half the cell in the x direction, pp stores
            # this size

            sincx = sinc(torch.fft.rfftfreq(pixels_[1])).to(device).type(realdtype)

            sincx = sincx.view([1, 1, 1, pp])

            # #Divide by sinc functions
            P /= sincy
            P /= sincx

        # Option to precalculate scattering factors and pass to program which
        # saves computation for
        if fe is None:
            fe_ = calculate_scattering_factors(psize, gsize, elements)
        else:
            fe_ = fe

        # Convolve with electron scattering factors using Fourier convolution theorem
        P *= (
            torch.from_numpy(fe_[..., :pp])
            .view(nelements, 1, pixels_[0], pp)
            .to(device)
        )

        norm = np.prod(pixels_) / np.prod(self.unitcell[:2]) / np.prod(tiling)

        # Add different elements atoms together
        P = norm * torch.sum(P, dim=0)

        # Apply bandwidth limit to remove high frequency numerical artefacts
        if bandwidthlimit is not None:
            P = bandwidth_limit_array_torch(P, limit=1, soft=bandwidthlimit, rfft=True)

        # Only return real part, shape of array needs to be given for
        # correct array shape to be return in the case of an odd numbered
        # array size
        return torch.fft.irfft2(P, s=pixels_)
        # return np.fft.irfft2(P.cpu().numpy(),s=pixels_)

    def make_potential_absorptive(
        self,
        pixels,
        eV,
        subslices=[1.0],
        tiling=[1, 1],
        displacements=False,
        fractional_occupancy=True,
        sinc_deconvolution=True,
        bandwidthlimit=0.2,
        fe_DWF=None,
        fe_TDS=None,
        device=None,
        dtype=torch.float32,
        showProgress=True,
    ):
        """
        Generate the projected potential of the structure, assuming an absorptive
        model whereby the elastic potential is thermally-smeared and the inelastic
        potential describes absorption for TDS electrons.

        Calculate the projected electrostatic potential for a structure on a
        pixel grid with dimensions specified by pixels. Subslicing the unit
        cell is achieved by passing an array subslices that contains as its
        entries the depths at which each subslice should be terminated in units
        of fractional coordinates. Tiling of the unit cell (often necessary to
        make a sufficiently large simulation grid to fit the probe) is achieved
        by passing the tiling factors in the array tiling.

        Parameters
        ----------
        pixels: int, (2,) array_like
            The pixel size of the grid on which to calculate the projected
            potentials
        subslices: float, array_like, optional
            An array containing the depths at which each slice ends as a fraction
            of the simulation unit-cell
        tiling: int, (2,) array_like, optional
            Tiling of the simulation object (often necessary to  make a
            sufficiently large simulation grid to fit the probe)
        displacements: bool, optional
            Pass displacements = False to turn off random displacements of the
            atoms due to thermal motion
        fractional_occupancy: bool, optional
            Pass fractional_occupancy = False to turn off fractional occupancy
            of atomic sites
        fe_DWF: float, array_like
            An array containing the electron scattering factors for the elements
            in the structure as calculated by the function
            calculate_scattering_factors_DWF, can be passed to save recalculating
            each time new potentials are generated
        fe_TDS: float, array_like
            An array containing the inelastic (TDS) electron scattering factors for
            the elements in the structure as calculated by the function
            calculate_scattering_factors_DWF(TDS=True), can be passed to save
            recalculating each time new potentials are generated
        device: torch.device
            Allows the user to control which device the calculations will occur
            on
        dtype: torch.dtype
            Controls the data-type of the output
        """
        # Get equivalent precison real datatype
        realdtype = complex_to_real_dtype_torch(dtype)

        # Force the subslices to be an array
        sblce = ensure_array(subslices)

        # Initialize device cuda if available, CPU if no cuda is available
        device = get_device(device)

        # Ensure pixels is integer
        pixels_ = [int(x) for x in pixels]

        tiling_ = np.asarray(tiling[:2])
        gsize = np.asarray(self.unitcell[:2]) * tiling_
        psize = np.asarray(pixels_)

        pixperA = np.asarray(pixels_) / np.asarray(self.unitcell[:2]) / tiling_

        # Get a list of unique atomic elements
        # elements = list(set(np.asarray(self.atoms[:, 3], dtype=_int)))
        # Note: now unique is defined by having same atomic number AND DWF
        elements_dwf, ielement_np = np.unique(
            np.stack(
                (np.asarray(self.atoms[:, 3]), np.asarray(self.atoms[:, 5])), axis=-1
            ),
            axis=0,
            return_inverse=True,
        )

        # Get number of unique atomic elements
        nelements = len(elements_dwf)
        nsubslices = len(sblce)
        # Build list of equivalent sites if Fractional occupancy is to be
        # taken into account
        if fractional_occupancy and self.fractional_occupancy:
            equivalent_sites = find_equivalent_sites(self.atoms[:, :3], EPS=1e-3)

        # FDES method
        # Intialize potential array
        P = torch.zeros(
            np.prod([nelements, nsubslices, *pixels_]), device=device, dtype=realdtype
        )

        # Construct a map of which atom corresponds to which slice
        islice = np.zeros((self.atoms.shape[0]), dtype=_int)
        slice_stride = np.prod(pixels_)
        # if nsubslices > 1:
        # Finds which slice atom can be found in
        # WARNING Assumes that the slices list ends with 1.0 and is in
        # ascending order
        for i in range(nsubslices):
            zmin = 0 if i == 0 else sblce[i - 1]
            atoms_in_slice = (self.atoms[:, 2] % 1.0 >= zmin) & (
                self.atoms[:, 2] % 1.0 < sblce[i]
            )
            islice[atoms_in_slice] = i * slice_stride
        islice = torch.from_numpy(islice).type(torch.long).to(device)
        # else:
        #     islice = 0
        # Make map a pytorch Tensor

        # Construct a map of which atom corresponds to which element
        element_stride = nsubslices * slice_stride
        ielement = torch.tensor(
            element_stride * ielement_np, dtype=torch.long, device=device
        )

        # FDES algorithm implemented using the pytorch scatter_add function,
        # which takes a list of numbers and adds them to a corresponding list
        # of coordinates
        for tile in range(tiling[0] * tiling[1]):
            # For these atomic coordinates (in fractional coordinates) convert
            # to pixel coordinates
            posn = (
                (
                    self.atoms[:, :2]
                    + np.asarray([tile % tiling[0], tile // tiling[0]])[np.newaxis, :]
                )
                / tiling_
                * psize
            )
            posn = torch.from_numpy(posn).to(device).type(realdtype)

            yc = (
                torch.remainder(torch.ceil(posn[:, 0]).type(torch.long), pixels_[0])
                * pixels_[1]
            )
            yf = (
                torch.remainder(torch.floor(posn[:, 0]).type(torch.long), pixels_[0])
                * pixels_[1]
            )
            xc = torch.remainder(torch.ceil(posn[:, 1]).type(torch.long), pixels_[1])
            xf = torch.remainder(torch.floor(posn[:, 1]).type(torch.long), pixels_[1])

            yh = torch.remainder(posn[:, 0], 1.0)
            yl = 1.0 - yh
            xh = torch.remainder(posn[:, 1], 1.0)
            xl = 1.0 - xh

            # Account for fractional occupancy of atomic sites if requested
            if fractional_occupancy and self.fractional_occupancy:
                xh *= torch.from_numpy(self.atoms[:, 4]).type(realdtype).to(device)
                xl *= torch.from_numpy(self.atoms[:, 4]).type(realdtype).to(device)

            # Each pixel is set to the overlap of a shifted rectangle in that pixel
            P.scatter_add_(0, ielement + islice + yc + xc, yh * xh)
            P.scatter_add_(0, ielement + islice + yc + xf, yh * xl)
            P.scatter_add_(0, ielement + islice + yf + xc, yl * xh)
            P.scatter_add_(0, ielement + islice + yf + xf, yl * xl)

        # Now view potential as a 4D array for next bit
        P = P.view(nelements, nsubslices, *pixels_)

        # Use real fast fourier transforms to save memory and time
        P = torch.fft.rfft2(P, s=pixels_)
        # fig,ax = plt.subplots(ncols=2)
        # ax[0].imshow(np.real(P[0,0].cpu()))
        # ax[1].imshow(np.imag(P[0,0].cpu()))
        # plt.show(block=True)

        # Copy P tensor to P_abs - has position info but will use different scattering factors to store (real) absorptive potential
        P_abs = P.clone().detach()

        # pp is number of pixels in x
        pp = pixels_[1] // 2 + 1

        if sinc_deconvolution:
            # Make sinc functions with appropriate singleton dimensions for pytorch
            # broadcasting /gridsize[0]*pixels_[0] /gridsize[1]*pixels_[1]
            sincy = (
                sinc(torch.fft.fftfreq(pixels_[0]))
                .view([1, 1, pixels_[0], 1])
                .to(device)
                .type(realdtype)
            )

            # The real FFT is only half the cell in the x direction, pp stores
            # this size

            sincx = sinc(torch.fft.rfftfreq(pixels_[1])).to(device).type(realdtype)

            sincx = sincx.view([1, 1, 1, pp])

            # #Divide by sinc functions
            P /= sincy
            P /= sincx

        # Option to precalculate scattering factors (including DWFs) and pass to program which
        # saves computation for
        if fe_DWF is None:
            fe_DWF_ = calculate_scattering_factors_dwf(
                psize, gsize, elements_dwf, showProgress=showProgress
            )
        else:
            fe_DWF_ = fe_DWF

        # Option to precalculate inelastic (TDS) scattering factors and pass to program which
        # saves computation for
        if fe_TDS is None:
            fe_TDS_ = calculate_scattering_factors_dwf(
                psize, gsize, elements_dwf, TDS=True, eV=eV, showProgress=showProgress
            )
        else:
            fe_TDS_ = fe_TDS

        # Convolve with electron scattering factors using Fourier convolution theorem
        P *= (
            torch.from_numpy(fe_DWF_[..., :pp])
            .view(nelements, 1, pixels_[0], pp)
            .to(device)
        )
        P_abs *= (
            torch.from_numpy(fe_TDS_[..., :pp])
            .view(nelements, 1, pixels_[0], pp)
            .to(device)
        )

        norm = np.prod(pixels_) / np.prod(self.unitcell[:2]) / np.prod(tiling)

        # Add different elements atoms together
        P = norm * torch.sum(P, dim=0)
        P_abs = norm * torch.sum(P_abs, dim=0)

        # Apply bandwidth limit to remove high frequency numerical artefacts
        if bandwidthlimit is not None:
            P = bandwidth_limit_array_torch(P, limit=1, soft=bandwidthlimit, rfft=True)
            P_abs = bandwidth_limit_array_torch(
                P_abs, limit=1, soft=bandwidthlimit, rfft=True
            )

        # Only return real part, shape of array needs to be given for
        # correct array shape to be return in the case of an odd numbered
        # array size
        return torch.fft.irfft2(P, s=pixels_), torch.fft.irfft2(P_abs, s=pixels_)
        # return np.fft.irfft2(P.cpu().numpy(),s=pixels_)

    def make_effective_scattering_potential(
        self,
        signal_list,
        pixels,
        eV,
        subslices=[1.0],
        tiling=[1, 1],
        fractional_occupancy=True,
        sinc_deconvolution=True,
        bandwidthlimit=0.2,
        device=None,
        dtype=torch.float32,
        qspace_out=True,
    ):
        """
            Generate effective scattering potentials.

            Calculate on a pixel grid with dimensions specified by pixels. Subslicing
            the unit cell is achieved by passing an array subslices that contains
            as its entries the depths at which each subslice should be terminated
            in units of fractional coordinates. Tiling of the unit cell (often
            necessary to make a sufficiently large simulation grid to fit the probe)
            is achieved by passing the tiling factors in the array tiling.

            Parameters
            ----------
            signal_list: dict, (N,) array_like
                List of dictionaries containing the information on the transitions
                for which effective scattering potentials are to be calculated
            pixels: int, (2,) array_like
                The pixel size of the grid on which to calculate the projected
                potentials
            subslices: float, array_like, optional
                An array containing the depths at which each slice ends as a fraction
                of the simulation unit-cell
            tiling: int, (2,) array_like, optional
                Tiling of the simulation object (often necessary to  make a
                sufficiently large simulation grid to fit the probe)
            fractional_occupancy: bool, optional
                Pass fractional_occupancy = False to turn off fractional occupancy
                of atomic sites
            device: torch.device
                Allows the user to control which device the calculations will occur
                on
            dtype: torch.dtype
                Controls the data-type of the output
            qspace_out : bool, optional
                Can set to True if intended for plotting (in real space)
                but default is False becaues reciprocal space form needed for
                cross-section expression
        Returns
        -------
            Veff: (nelements,nsubslices,Y,X) complex torch.tensor
                Effective scattering potential (\mu's) for EELS and/or EDX
        """
        # Get equivalent precison real datatype
        realdtype = complex_to_real_dtype_torch(dtype)

        # Force the subslices to be an array
        sblce = ensure_array(subslices)

        # Initialize device cuda if available, CPU if no cuda is available
        device = get_device(device)

        # Ensure pixels is integer
        pixels_ = [int(x) for x in pixels]

        tiling_ = np.asarray(tiling[:2])
        gsize = np.asarray(self.unitcell[:2]) * tiling_
        psize = np.asarray(pixels_)

        pixperA = np.asarray(pixels_) / np.asarray(self.unitcell[:2]) / tiling_

        # Get a list of unique atomic elements
        # elements = list(set(np.asarray(self.atoms[:, 3], dtype=_int)))
        # Note: now unique is defined by having same atomic number AND DWF
        elements_dwf, ielement_np = np.unique(
            np.stack(
                (np.asarray(self.atoms[:, 3]), np.asarray(self.atoms[:, 5])), axis=-1
            ),
            axis=0,
            return_inverse=True,
        )

        # Get number of unique atomic elements
        nelements = len(elements_dwf)
        nsubslices = len(sblce)
        # Build list of equivalent sites if Fractional occupancy is to be
        # taken into account
        if fractional_occupancy and self.fractional_occupancy:
            equivalent_sites = find_equivalent_sites(self.atoms[:, :3], EPS=1e-3)

        # FDES method
        # Intialize atom location array
        P = torch.zeros(
            np.prod([nelements, nsubslices, *pixels_]), device=device, dtype=realdtype
        )

        # Construct a map of which atom corresponds to which slice
        islice = np.zeros((self.atoms.shape[0]), dtype=_int)
        slice_stride = np.prod(pixels_)
        # if nsubslices > 1:
        # Finds which slice atom can be found in
        # WARNING Assumes that the slices list ends with 1.0 and is in
        # ascending order
        for i in range(nsubslices):
            zmin = 0 if i == 0 else sblce[i - 1]
            atoms_in_slice = (self.atoms[:, 2] % 1.0 >= zmin) & (
                self.atoms[:, 2] % 1.0 < sblce[i]
            )
            islice[atoms_in_slice] = i * slice_stride
        islice = torch.from_numpy(islice).type(torch.long).to(device)
        # else:
        #     islice = 0
        # Make map a pytorch Tensor

        # Construct a map of which atom corresponds to which element
        element_stride = nsubslices * slice_stride
        ielement = torch.tensor(
            element_stride * ielement_np, dtype=torch.long, device=device
        )

        # FDES algorithm implemented using the pytorch scatter_add function,
        # which takes a list of numbers and adds them to a corresponding list
        # of coordinates
        for tile in range(tiling[0] * tiling[1]):
            # For these atomic coordinates (in fractional coordinates) convert
            # to pixel coordinates
            posn = (
                (
                    self.atoms[:, :2]
                    + np.asarray([tile % tiling[0], tile // tiling[0]])[np.newaxis, :]
                )
                / tiling_
                * psize
            )
            posn = torch.from_numpy(posn).to(device).type(realdtype)

            yc = (
                torch.remainder(torch.ceil(posn[:, 0]).type(torch.long), pixels_[0])
                * pixels_[1]
            )
            yf = (
                torch.remainder(torch.floor(posn[:, 0]).type(torch.long), pixels_[0])
                * pixels_[1]
            )
            xc = torch.remainder(torch.ceil(posn[:, 1]).type(torch.long), pixels_[1])
            xf = torch.remainder(torch.floor(posn[:, 1]).type(torch.long), pixels_[1])

            yh = torch.remainder(posn[:, 0], 1.0)
            yl = 1.0 - yh
            xh = torch.remainder(posn[:, 1], 1.0)
            xl = 1.0 - xh

            # Account for fractional occupancy of atomic sites if requested
            if fractional_occupancy and self.fractional_occupancy:
                xh *= torch.from_numpy(self.atoms[:, 4]).type(realdtype).to(device)
                xl *= torch.from_numpy(self.atoms[:, 4]).type(realdtype).to(device)

            # Each pixel is set to the overlap of a shifted rectangle in that pixel
            P.scatter_add_(0, ielement + islice + yc + xc, yh * xh)
            P.scatter_add_(0, ielement + islice + yc + xf, yh * xl)
            P.scatter_add_(0, ielement + islice + yf + xc, yl * xh)
            P.scatter_add_(0, ielement + islice + yf + xf, yl * xl)

        # Now view potential as a 4D array for next bit
        P = P.view(nelements, nsubslices, *pixels_)

        # Use real fast fourier transforms to save memory and time
        P = torch.fft.rfft2(P, s=pixels_)

        # pp is number of pixels in x
        pp = pixels_[1] // 2 + 1

        if sinc_deconvolution:
            # Make sinc functions with appropriate singleton dimensions for pytorch
            # broadcasting /gridsize[0]*pixels_[0] /gridsize[1]*pixels_[1]
            sincy = (
                sinc(torch.fft.fftfreq(pixels_[0]))
                .view([1, 1, pixels_[0], 1])
                .to(device)
                .type(realdtype)
            )

            # The real FFT is only half the cell in the x direction, pp stores
            # this size

            sincx = sinc(torch.fft.rfftfreq(pixels_[1])).to(device).type(realdtype)

            sincx = sincx.view([1, 1, 1, pp])

            # #Divide by sinc functions
            P /= sincy
            P /= sincx

        # Intialize effective scattering potential array
        Veff = torch.zeros(
            (len(signal_list), nsubslices, *pixels_), device=device, dtype=realdtype
        )

        for idx, signal in enumerate(signal_list):
            if signal["signal"] != "ADF":
                EELS_EDX_params = get_EELS_EDX_params(signal)

                feff = np.zeros((nelements, nsubslices, *pixels_))
                for ielement, element in enumerate(elements_dwf):
                    if int(element[0]) != signal["Z"]:
                        continue

                    feff_temp = calculate_EELS_EDX_scattering_factors(
                        psize, gsize, EELS_EDX_params, element[1], eV
                    )
                    feff[ielement, :, :, :] = np.repeat(
                        feff_temp[np.newaxis, :, :], nsubslices, axis=0
                    )

                # Convolve with electron scattering factors using Fourier convolution theorem
                Veff_set = P * (
                    torch.from_numpy(feff[..., :pp])
                    .view(nelements, nsubslices, pixels_[0], pp)
                    .to(device)
                )

                norm = (
                    np.prod(pixels_) / np.prod(self.unitcell[:2]) / np.prod(tiling)
                )  # Note: factor of no. pixels here compensates the division by same in irfft2

                # Add different elements atoms together
                Veff_single = norm * torch.sum(
                    Veff_set, dim=0
                )  # Note torch.sum squeezes, so Veff_single is now (nsublices,npy,npx_reduced)

                # Apply bandwidth limit to remove high frequency numerical artefacts
                if bandwidthlimit is not None:
                    Veff_single = bandwidth_limit_array_torch(
                        Veff_single, limit=1, soft=bandwidthlimit, rfft=True
                    )

                Veff[idx, :, :, :] = torch.fft.irfft2(Veff_single, s=pixels_)

            else:
                print("ADF currently unavailable.")

        # # Convolve with electron scattering factors using Fourier convolution theorem
        #         Veff_set = P * (
        #             torch.from_numpy(feff[..., :pp])
        #             .view(nelements, 1, pixels_[0], pp)
        #             .to(device)
        #         )

        #         norm = np.prod(pixels_) / np.prod(self.unitcell[:2]) / np.prod(tiling)

        # # Add different elements atoms together
        #         Veff_single = norm * torch.sum(Veff_set, dim=0) # Note torch.sum squeezes, so Veff_single is now (nsublices,npy,npx_reduced)

        # # Apply bandwidth limit to remove high frequency numerical artefacts
        #         if bandwidthlimit is not None:
        #             Veff_single = bandwidth_limit_array_torch(Veff_single, limit=1, soft=bandwidthlimit, rfft=True)

        #         Veff[idx,:,:,:] = torch.fft.irfft2(Veff_single, s=pixels_)

        # Dimensions for FFT
        # d_ = (-2, -1)
        # # Inverse Fourier transform back to real space for next iteration
        # Veff = torch.fft.ifftn(Veff, dim=d_)

        # return Veff
        # Return correct array data type
        if qspace_out:
            return torch.fft.fftn(Veff, dim=(-2, -1)) / np.prod(
                pixels_
            )  # The combination of irfft2 and fftn is normalised, and so
            # the earlier multiplication by no. pixels needs to be undone
        else:
            return Veff

    def make_transmission_functions(
        self,
        pixels,
        eV,
        subslices=[1.0],
        tiling=[1, 1],
        fe=None,
        displacements=True,
        fftout=False,
        dtype=None,
        device=None,
        fractional_occupancy=True,
        seed=None,
        bandwidth_limit=2 / 3,
    ):
        """
        Make the transmission functions for the simulation object.

        Transmission functions are the exponential of the specimen electrostatic
        potential scaled by the interaction constant for electrons, sigma. These
        are used to model scattering by a thin slice of the object in the
        multislice algorithm

        Parameters:
        -----------
        pixels : array_like
            Output pixel grid
        eV : float
            Probe accelerating voltage in electron-volts
        subslices : array_like, optional
            An array containing the depths at which each slice ends as a fraction
            of the simulation unit-cell, used for simulation objects thicker
            than typical multislice slicing (about 2 Angstrom)
        tiling : array_like,optional
            Repeat tiling of the simulation object
        fe: array_like,optional
            An array containing the electron scattering factors for the elements
            in the simulation object as calculated by the function
            calculate_scattering_factors
        """
        # Make the specimen electrostatic potential
        T = self.make_potential(
            pixels,
            subslices,
            tiling,
            fe=fe,
            displacements=displacements,
            device=device,
            dtype=dtype,
            fractional_occupancy=fractional_occupancy,
            seed=seed,
        )
        # Now take the complex exponential of the electrostatic potential
        # scaled by the electron interaction constant
        T = torch.fft.fftn(torch.exp(1j * interaction_constant(eV) * T), dim=[-2, -1])

        # Band-width limit the transmission function, see Earl Kirkland's book
        # for an discussion of why this is necessary
        return bandwidth_limit_array_torch(
            T, limit=bandwidth_limit, qspace_in=True, qspace_out=fftout
        )

    def make_transmission_functions_absorptive(
        self,
        pixels,
        eV,
        subslices=[1.0],
        tiling=[1, 1],
        fe=None,
        displacements=False,
        fftout=False,
        dtype=None,
        device=None,
        fractional_occupancy=True,
        seed=None,
        bandwidth_limit=2 / 3,
        showProgress=True,
    ):
        """
        Make the transmission functions for the simulation object, assuming an
        absorptive model whereby the elastic potential is thermally-smeared and
        the inelastic potential describes absorption for TDS electrons.

        Transmission functions are the exponential of the specimen electrostatic
        potential scaled by the interaction constant for electrons, sigma. These
        are used to model scattering by a thin slice of the object in the
        multislice algorithm.

        Parameters:
        -----------
        pixels : array_like
            Output pixel grid
        eV : float
            Probe accelerating voltage in electron-volts
        subslices : array_like, optional
            An array containing the depths at which each slice ends as a fraction
            of the simulation unit-cell, used for simulation objects thicker
            than typical multislice slicing (about 2 Angstrom)
        tiling : array_like,optional
            Repeat tiling of the simulation object
        fe: array_like,optional
            An array containing the electron scattering factors for the elements
            in the simulation object as calculated by the function
            calculate_scattering_factors
        """
        # Make the specimen electrostatic potential
        Tel, Tabs = self.make_potential_absorptive(
            pixels,
            eV,
            subslices,
            tiling,
            fe_DWF=fe,
            displacements=displacements,
            device=device,
            dtype=dtype,
            fractional_occupancy=fractional_occupancy,
            showProgress=showProgress,
        )
        # Now take the complex exponential of the electrostatic potential
        # scaled by the electron interaction constant
        # Note that an absorptive component is included
        T = torch.fft.fftn(
            torch.exp(1j * interaction_constant(eV) * (Tel + 1j * Tabs)), dim=[-2, -1]
        )

        # Band-width limit the transmission function, see Earl Kirkland's book
        # for an discussion of why this is necessary
        return bandwidth_limit_array_torch(
            T, limit=bandwidth_limit, qspace_in=True, qspace_out=fftout
        )

    def autoslice(self, maxproperror=1.5):
        """
        Auto-generate optimal sub-slices using Jenks Natural Breaks clustering.

        This method uses the Jenks Natural Breaks algorithm to determine the optimal way
        to divide atomic positions (depths) into clusters (sub-slices) such that the
        maximum propagation error from the propagation plane within each cluster is minimized.

        Parameters:
        ----------
        maxproperror : float, optional
            The maximum allowable propagation error in Angstroms within each sub-slice.
            Default is 1.5.

        Returns:
        -------
        planes : list
            A list of positions where each sub-slice begins, including the end position 1.
            If no sub-slicing is required or a satisfactory slicing cannot be found,
            it returns maximal slicing based on unique atom depths.

        Notes:
        -----
        - If the algorithm cannot find a sub-slicing that meets the `maxproperror` criterion,
        it returns the maximal slicing based on unique atomic depths.
        - The `self.atoms` array is assumed to contain atomic positions with the third
        column representing depth, and `self.unitcell` contains the unit cell dimensions.
        """
        from jenkspy import JenksNaturalBreaks

        # Unique atom depths
        uniquepositions = np.unique(self.atoms[:, 2])

        for n_clusters in range(1, len(uniquepositions)):

            # JenksNaturalBreaks clustering only works with > 1 clusters
            if n_clusters > 1:
                # JenksNaturalBreaks clustering of atom depths
                jnb = JenksNaturalBreaks(n_clusters)
                jnb.fit(self.atoms[:, 2])

                # Set subslices so that they begin at the start of each cluster
                planes = [min(group) for group in jnb.groups_] + [1]

                # Find maximum distance from propagation plane within group
                properror = (
                    max([max(g - p) for g, p in zip(jnb.groups_, planes)])
                    * self.unitcell[2]
                )
                # If the first slice is tolerably close to the top of the unit cell
                # anyway, remove this slice
                if planes[0] + max(jnb.groups_[0]) < maxproperror:
                    planes.pop(0)
            else:
                # No sub-slicing
                planes = [1]
                properror = max(self.atoms[:, 2]) * self.unitcell[2]

            # Terminate sequence upon convergence
            if properror <= maxproperror:
                return planes

        # If a satisfactory slicing cannot be found just return maximal slicing,
        # removing first slice if not required
        if uniquepositions[0] < maxproperror:
            return list(uniquepositions)[1:] + [1]
        else:
            return list(uniquepositions) + [1]

    def generate_slicing_figure(self, slices, show=True):
        """
        Generate slicing figure.

        Generate a slicing figure that to aid in setting up the slicing
        of the sample for multislice algorithm. This will show where each of the
        slices end for a chosen slicing relative to the atoms. To minimize
        errors, the atoms should sit as close to the top of the slice as possible.

        Parameters
        ----------
        slices: array_like, float
            An array containing the depths at which each slice ends as a fraction
            of the simulation unit-cell
        """
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

        coords = self.atoms[:, :3] * self.unitcell[None, :]
        # Projection down the x-axis
        for i in range(2):
            ax[i].plot(coords[:, i], coords[:, 2], "bo", label="Atoms")
            for j, slice_ in enumerate(slices):
                if j == 0:
                    label = "Slices"
                else:
                    label = "_"
                ax[i].plot(
                    [0, self.unitcell[i]],
                    [slice_ * self.unitcell[2], slice_ * self.unitcell[2]],
                    "r--",
                    label=label,
                )
            ax[i].set_xlim([0, self.unitcell[i]])
            ax[i].set_xlabel(["y", "x"][i])
            ax[i].set_ylim([self.unitcell[2], 0])
            ax[i].set_ylabel("z")
            ax[i].set_title("View down {0} axis".format(["x", "y"][i]))
        ax[0].legend()
        if show:
            plt.show(block=True)
        return fig

    def rotate(self, theta, axis, origin=[0.5, 0.5, 0.5]):
        """
        Rotate simulation object an amount an angle theta (in radians) about axis.

        Parameters
        ----------
        theta: float
            Angle to rotate simulation object by in radians
        axis: array_like
            Axis about which to rotate simulation object eg [0,0,1]

        Keyword arguments
        ------------------
        origin : array_like, optional
            Origin (in fractional coordinates) about which to rotate simulation
            object eg [0.5, 0.5, 0.5]
        """
        new = copy.deepcopy(self)

        # Make rotation matrix, R, and  the point about which we rotate, O
        R = rot_matrix(theta, axis)
        origin_ = np.asarray(origin) * self.unitcell

        # Get atomic coordinates in cartesian (not fractional coordinates)
        new.atoms[:, :3] = self.atoms[:, :3] * self.unitcell[np.newaxis, :]

        # Apply rotation matrix to each atom coordinate
        new.atoms[:, :3] = (new.atoms[:, :3] - origin_) @ R + origin_

        # Apply rotation matrix to cell vertices
        vertices = (
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            * self.unitcell
            - origin_
        ) @ R + origin_

        # Get new unit cell from maximum range of unit cell vertices
        origin_ = np.amin(vertices, axis=0)
        new.unitcell = np.ptp(vertices, axis=0)

        # Convert atoms back into fractional coordinates in new unit cell
        new.atoms[:, :3] = ((new.atoms[:, :3] - origin_) / new.unitcell) % 1.0

        # Return rotated structure
        return new

    def rot90(self, k=1, axes=(0, 1)):
        """
        Rotates a structure by 90 degrees in the plane specified by axes.

        Rotation direction is from the first towards the second axis.

        Parameters
        ----------
        k : integer, optional
            Number of times the structure is rotated by 90 degrees.
        axes: (2,) array_like
            The array is rotated in the plane defined by the axes.
            Axes must be different.
        """
        # Much of the following is adapted from the numpy.rot90 function
        axes = tuple(axes)
        if len(axes) != 2:
            raise ValueError("len(axes) must be 2.")

        k %= 4

        if k == 0:
            # Do nothing
            return
        if k == 2:
            # Reflect in both axes
            self.reflect(axes)
            return

        axes_list = np.arange(0, 3)
        (axes_list[axes[0]], axes_list[axes[1]]) = (
            axes_list[axes[1]],
            axes_list[axes[0]],
        )

        if k == 1:
            self.reflect([axes[1]])
            self.transpose(axes_list)
        else:
            # k == 3
            self.transpose(axes_list)
            self.reflect([axes[1]])

        return self

    def transpose(self, axes):
        """Transpose the axes of a simulation object."""
        self.atoms[:, :3] = self.atoms[:, axes]
        self.unitcell = self.unitcell[axes]
        return self

    def tile(self, x=1, y=1, z=1):
        """Make a repeat unit tiling of the simulation object."""
        # Make copy of original structure
        # new = copy.deepcopy(self)

        tiling = np.asarray([x, y, z], dtype=_int)

        # Get atoms in unit cell
        natoms = self.atoms.shape[0]

        # Initialize new atom list
        newatoms = np.zeros((natoms * x * y * z, 6))

        # Calculate new unit cell size the transpose esnures correct
        # tiling of non-orthorhombic cells (ie. where self.unitcell is
        # a matrix)
        self.unitcell = (self.unitcell.T * np.asarray([x, y, z])).T

        # tile out the integer amounts
        from itertools import product

        for j, k, l in product(*[np.arange(int(i)) for i in [x, y, z]]):

            # Calculate origin of this particular tile
            origin = np.asarray([j, k, l])

            # Calculate index of this particular tile
            indx = j * int(y) * int(z) + k * int(z) + l

            # Add new atoms to unit cell
            newatoms[indx * natoms : (indx + 1) * natoms, :3] = (
                self.atoms[:, :3] + origin[np.newaxis, :]
            ) / tiling[np.newaxis, :]

            # Copy other information about atoms
            newatoms[indx * natoms : (indx + 1) * natoms, 3:] = self.atoms[:, 3:]
        self.atoms = newatoms
        return self

    def concatenate(self, other, axis=2, side=1, eps=1e-2):
        """
        Concatenate two simulation objects.

        Adds other simulation object to the current object. other is added to
        the bottom (top being z =0) routine will attempt to tile objects to
        match dimensions.

        Parameters:
        other : structure class
            Object that will be concatenated onto the other object.
        axis : int, optional
            Axis along which the two structures will be joined.
        side : int, optional
            Determines which side the other structure will be added onto the
            first structure. If side == 0 the structures will be added onto each
            other at the origin, if side == 1 the structures will be added onto
            each other at the end.
        eps : float, optional
            Fractional tolerance of the pseudo rational tiling to make the
            structure dimensions perpendicular to the beam direction match.
        """
        # Make deep copies of the structure object and the slice this is so
        # that these objects remain untouched by the operation of this function
        new = copy.deepcopy(self)
        other_ = copy.deepcopy(other)

        tile1, tile2 = [np.ones(3, dtype=_int) for i in range(2)]

        # Check if the two slices are the same size and
        # tile accordingly
        for ax in range(3):
            # If this axis is the concatenation axis, then it's not necessary
            # that the structures are the same size
            if ax == axis:
                continue
            # Calculate the pseudo-rational tiling
            if self.unitcell[ax] < other.unitcell[ax]:
                tile1[ax], tile2[ax] = psuedo_rational_tiling(
                    self.unitcell[ax], other.unitcell[ax], eps
                )
            else:
                tile2[ax], tile1[ax] = psuedo_rational_tiling(
                    other.unitcell[ax], self.unitcell[ax], eps
                )

            tile1[ax], tile2[ax] = psuedo_rational_tiling(
                self.unitcell[ax], other.unitcell[ax], eps
            )

        new = new.tile(*tile1)
        tiled_zdim = new.unitcell[axis]
        other_ = other_.tile(*tile2)

        # Update the thickness of the resulting structure object.
        new.unitcell[axis] = tiled_zdim + other_.unitcell[axis]

        # Adjust fractional coordinates of atoms, multiply by old unitcell
        # size to transform into cartesian coordinates and then divide by
        # the old unitcell size to transform into fractional coordinates
        # in the new basis
        new.atoms[:, axis] *= tiled_zdim / new.unitcell[axis]
        other_.atoms[:, axis] *= other_.unitcell[axis] / new.unitcell[axis]

        # Adjust the coordinates of the new or old atoms depending on which
        # side the new structure is to be added.
        if side == 0:
            new.atoms[:, axis] += other_.unitcell[axis] / new.unitcell[axis]
        else:
            other_.atoms[:, axis] += self.unitcell[axis] / new.unitcell[axis]

        # Concatenate adjusted atomic coordinates
        new.atoms = np.concatenate([new.atoms, other_.atoms], axis=0)

        # Concatenate titles
        new.Title = self.Title + " and " + other.Title

        return new

    def reflect(self, axes):
        """Reflect structure in each of the axes enumerated in list axes."""
        for ax in axes:
            self.atoms[:, ax] = (1 - self.atoms[:, ax]) % 1.0
        return self

    def resize(self, fraction, axis):
        """
        Resize (either crop or pad with vacuum) the simulation object.

        Resize the simulation object ranging such that the new axis runs from
        fraction[iax,0] to fraction[iax,1] on specified axis iax, slice_frac is
        in units of fractional coordinates. If fraction[iax,0] is < 0 then
        additional vacuum will be added, if > 0 then parts of the sample will
        be removed for axis[iax]. Likewise if fraction[iax,1] is > 1 then
        additional vacuum will be added, if < 1 then parts of the sample will
        be removed for axis[iax].

        Parameters
        ----------
        fraction : (nax,2) array_like
            Describes the size of the new simulation object as a fraction of
            old simulation object dimensions.
        axis : int or (nax,) array_like
            The axes of the simulation object that will be resized

        Returns
        -------
        New structure : pyms.structure object
            The resized structure object
        """
        ax = ensure_array(axis)
        frac = ensure_array(fraction)
        if np.asarray(frac).ndim < 2:
            frac = [frac]

        # Work out which atoms will stay in the sliced structure
        mask = np.ones((self.atoms.shape[0],), dtype=bool)
        for a, f in zip(ax, frac):
            atomsin = np.logical_and(self.atoms[:, a] >= f[0], self.atoms[:, a] <= f[1])
            mask = np.logical_and(atomsin, mask)

        # Make a copy of the structure
        new = copy.deepcopy(self)

        # Put remaining atoms back in
        new.atoms = self.atoms[mask, :]

        # Origin for atomic coordinates
        origin = np.zeros((3))

        for a, f in zip(ax, frac):
            # Adjust unit cell dimensions
            new.unitcell[a] = (f[1] - f[0]) * self.unitcell[a]

            # Adjust origin of atomic coordinates
            origin[a] = f[0]

        new.atoms[:, :3] = (new.atoms[:, :3] - origin) * self.unitcell / new.unitcell

        # Return modified structure
        return new

    def cshift(self, shift, axis):
        """
        Circular shift routine.

        Shift the atoms within the simulation cell an amount shift in fractional
        coordinates along specified axis (or axes if both shift and axis are
        array_like).

        Parameters
        ----------
        shift : array_like or int
            Amount in fractional coordinates to shift (each) axis.
        axis : array_like or int
            Axis or list of axes to apply shift(s) to.
        """

        def _cshift(atoms, x, ax):
            atoms[:, ax % 3] = np.mod(atoms[:, ax % 3] + x, 1.0)
            return atoms

        if hasattr(axis, "__len__"):
            for x, ax in zip(shift, axis):
                self.atoms = _cshift(self.atoms, x, ax)
        else:
            self.atoms = _cshift(self.atoms, shift, axis)

        return self


class layered_structure_transmission_function:
    """
    A class that mimics multislice transmission functions for a layered object.

    Useful for performing multislice calculations of heterostructures (epitaxially
    layered crystalline structures).
    """

    def __init__(
        self,
        gridshape,
        eV,
        structures,
        nslices,
        subslices,
        tilings=None,
        kwargs={},
        nT=5,
        dtype=torch.cfloat,
        device=None,
        specimen_tilt=[0, 0],
    ):
        """
        Generate a layered structure transmission function object.

        This function assumes that the lateral (x and y) cell sizes of the
        structures are identical,

        Input
        -----
        structures : (N,) array_like of pyms.Structure objects
            The input structures for which the transmission functions for a
            layered structure will be calculated.
        nslices : int (N,) array_like
            The number of units of each structure in the multilayer
        subslices : array_like (N,) of array_like
            Multislice subslicing for each object in the multilayer structure
        kwargs : Dict, optional
            Keyword arguments that are passed to the make_transmission_functions

        Returns
        -----
        self : layered_structure_transmission_function object
            This will behave like a normal transmission function array, if
            T = layered_structure_transmission_function(...,[structure1,structure2 etc])
            then T[0,islice,...] will return a transmission function from whichever
            structure islice happens to be in. T.Propagator[islice] returns the
            relevant multislice propagator
        """
        self.dtype = dtype
        self.device = get_device(device)
        self.nslicestot = np.sum(nslices)
        self.structures = structures

        if tilings is None:
            tilings = len(structures) * [[1, 1]]

        self.Ts = []
        self.nT = nT
        self.gridshape = gridshape
        self.tilings = tilings
        self.eV = eV
        self.specimen_tilt = specimen_tilt
        self.unitcell = np.zeros(3)
        self.unitcell[:2] = structures[0].unitcell[:2]  # * np.asarray(tilings[0])
        self.unitcell[2] = np.sum(
            [struc.unitcell[2] * nslice for struc, nslice in zip(structures, nslices)]
        )
        args = [gridshape, eV]

        # Like every Melbourne restaurant, within the slab structure we do things
        # a little differently: since the number of subslices can be different
        # for each structure we have to store the transmission functions for
        #  each structure in a list so the indexing of self.Ts is
        # self.Ts[istructure][iT][isubslice], the __get_item__ method
        # makes indexing this synthetic object consistent with standard practice
        for structure, subslices_, tiling in zip(structures, subslices, tilings):
            self.Ts.append(
                torch.stack(
                    [
                        structure.make_transmission_functions(
                            *args,
                            subslices=subslices_,
                            tiling=tiling,
                            **kwargs,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        for i in range(nT)
                    ]
                )
            )

        self.slicemap = list(
            itertools.chain(
                *[len(subslices[i]) * n * [i] for i, n in enumerate(nslices)]
            )
        )
        nsubslices = [len(subslice) for subslice in subslices]
        self.subslicemap = list(
            itertools.chain(
                *[
                    (np.arange(nsubslices[i] * n) % nsubslices[i]).tolist()
                    for i, n in enumerate(nslices)
                ]
            )
        )
        self.N = len(self.slicemap)
        self.subslices = []
        T = 0
        # Calculate the fractional depth of every subslice in the new synthetic
        # structure.
        for subslices_, slices, struct in zip(subslices, nslices, structures):
            for i in range(slices):
                self.subslices += (
                    (np.asarray(subslices_) * struct.unitcell[2] + T) / self.unitcell[2]
                ).tolist()
                T = T + struct.unitcell[2]

        # Mimics the shape property of a numpy array
        self.shape = (self.nT, self.N, *self.gridshape)
        self.Propagator = layered_structure_propagators(
            self, subslices, propkwargs={"tilt": specimen_tilt}
        )

    def dim(self):
        """Return the array dimension of the synthetic array."""
        return 4

    def __getitem__(self, ind):
        """
        __getitem__ method for the transmission function synthetic array.

        This enables the transmission function object to mimic a standard
        transmission function numpy or torch.Tensor array
        """
        it, islice = ind[:2]

        # First get the proper slice and subslice, self.Ts is a list object
        # with each entry containing the transmission functions for that
        # structure
        if isinstance(islice, int) or np.issubdtype(np.asarray(islice).dtype, int):
            T = self.Ts[self.slicemap[islice]][:, self.subslicemap[islice]]
        elif isinstance(islice, slice):
            islice_ = np.arange(*islice.indices(self.N))
            T = torch.stack(
                [self.Ts[self.slicemap[j]][:, self.subslicemap[j]] for j in islice_],
                axis=1,
            )
        else:
            raise TypeError("Invalid argument type.")

        if isinstance(it, int) or np.issubdtype(np.asarray(it).dtype, int):
            return T[it]
        elif isinstance(it, slice):
            it_ = np.arange(*it.indices(self.nT))
            return T[it_]
        else:
            raise TypeError("Invalid argument type.")


class layered_structure_propagators:
    """
    A class that mimics multislice propagators for a layered object.

    Complements layered_transmission_function
    """

    def __init__(self, layered_T, subslices, propkwargs={}):
        """
        Generate a layered structure multislice propagator function object.

        This function assumes that the lateral (x and y) cell sizes of the
        structures are identical,

        Input
        -----
        T : layered_structure_transmission_function object
            This should contain all the necessary information about the layered
            object to generate the propagators

        Keyword arguments:
        -------------------
        propkwargs : dict
            Keyword arguments to pass onto make_propagator function

        Returns
        -----
        self : layered_structure_propagators object
            This will behave like a normal propagator array, if
            P = layered_structure_propagators(T)
            then P[islice,...] will return a transmission function from whichever
            structure islice happens to be in.
        """
        from .py_multislice import make_propagators

        self.rsizes = [
            copy.deepcopy(struc.unitcell * np.asarray(t + [1]))
            for struc, t in zip(layered_T.structures, layered_T.tilings)
        ]

        self.Ps = [
            torch.from_numpy(
                make_propagators(layered_T.gridshape, r, layered_T.eV, s, **propkwargs)
            )
            .to(layered_T.device)
            .type(layered_T.dtype)
            for s, r in zip(subslices, self.rsizes)
        ]
        self.slicemap = layered_T.slicemap
        self.subslicemap = layered_T.subslicemap
        # Mimics the shape property of a numpy array
        self.shape = (layered_T.nslicestot, *layered_T.gridshape, 2)
        self.ndim = 4

    def dim(self):
        """Return the array dimension of the synthetic array."""
        return 4

    def __getitem__(self, islice):
        """
        __getitem__ method for the propagator synthetic array.

        This enables the propagator object to mimic a standard propagator numpy
        or torch.Tensor array
        """
        if isinstance(islice, int) or np.issubdtype(np.asarray(islice).dtype, int):
            return self.Ps[self.slicemap[islice]][self.subslicemap[islice]]
        elif isinstance(islice, slice):
            islice_ = np.arange(*islice.indices(len(self.slicemap)))
            return torch.stack(
                [self.Ps[self.slicemap[j]][self.subslicemap[j]] for j in islice_]
            )
        else:
            raise TypeError("Invalid argument type.")


def calculate_scattering_factors_dwf(
    gridshape,
    gridsize,
    elements_dwf,
    TDS=False,
    eV=None,
    showProgress=True,
):
    """Calculate the electron scattering factors on a reciprocal space grid, include Debye-Waller factors.
       If TDS=True, calculate inelastic (TDS) electron scattering factors using the parameterisation of
       Weikenmeier and Kohl.

    Parameters
    ----------
    gridshape : (2,) array_like
        pixel size of the grid
    gridsize : (2,) array_like
        Lateral real space sizing of the grid in Angstrom
    elements_dwf: (M,2) array_like
        List of elements (atomic number and ums values) for which electron scattering factors are required
    TDS: bool, optional
        If False, calculates electron scattering factors for elastic scattering (including Debye-Waller factors)
        If True, calculates electron scattering factors for inelastic (TDS) scattering
    eV : float, optional (required for TDS calculation only)
        Probe energy in electron volts


    Returns
    -------
    fe : (M, *gridshape)
        Array of electron scattering factors in reciprocal space for each
        element
    """

    if eV is None:
        if TDS:
            print(
                "ERROR: an eV value must be provided to calculate TDS scattering factors."
            )
            print("Will assume 300 keV.")
            eV = 3e5

    # Get reciprocal space array
    g = q_space_array(gridshape, gridsize)
    gsq = np.square(g[0]) + np.square(g[1])

    # Initialise scattering factor array
    fe = np.zeros((len(elements_dwf), *gridshape), dtype=_float)

    # Loop over unique elements
    for ielement, element in enumerate(elements_dwf):
        if TDS:
            fe[ielement, :, :] = electron_tds_wk_scattering_factor(
                int(element[0]), gsq, element[1], eV, showProgress
            )
            # fe[ielement, :, :] = electron_tds_scattering_factor(int(element[0]),gsq,element[1],eV)
        else:
            fe[ielement, :, :] = electron_scattering_factor(
                int(element[0]), gsq
            ) * np.exp(-2 * np.pi**2 * gsq * element[1])

    return fe


def electron_tds_wk_scattering_factor(Z, gsq, ums, eV, showProgress=True):
    """
    Calculate the electron TDS scattering factor for atom with atomic number Z and mean
    square vibrational amplitude ums using Weikenmeier & Kohl parameterisation
    Adapted from FSCATT code included in https://github.com/EMsoft-org/EMsoft/blob/develop/Source/EMsoftLib/others.f90

    Parameters
    ----------
    Z : int
        Atomic number of atom of interest.
    gsq : float or array_like
        Reciprocal space value(s) in Angstrom squared at which to evaluate the
        electron scattering factor.
    ums : mean square vibrational amplitude (Å^2 units)
    eV : float
        Probe accelerating voltage in electron-volts
    """
    from .py_multislice import tqdm_handler

    # Thesis: Since f_TDS(g) is rotationally symmetric, it's faster to calculate in 1D
    #        and interpolate to 2D than it is to calculate in 2D
    from scipy import interpolate

    showProgress = showProgress
    tdisable, tqdm = tqdm_handler(showProgress)

    # Planck's constant in kg Angstrom/s
    h = 6.62607004e-24
    # Electron rest mass in kg
    me = 9.10938356e-31
    # Electron charge in Coulomb
    qe = 1.60217662e-19
    # Electron wave number (reciprocal of wavelength) in Angstrom
    k0 = wavev(eV)
    # Relativistic electron mass correction
    gamma = relativistic_mass_correction(eV)

    V = e_scattering_factors_WK[
        Z - 2, 0
    ]  # Note: parameterisation can't handle hydrogen
    B_wk = e_scattering_factors_WK[Z - 2, 1:]
    A_wk = np.zeros((6,))
    A_wk[0:3] = 0.02395 * Z / (3 * (1 + V))
    A_wk[3:6] = A_wk[0] * V

    Mx4 = 8 * np.pi**2 * ums  # Mx4*s2 = M*g2

    modg = np.sqrt(gsq)
    gmax = np.max(modg)
    g1d = np.arange(0, gmax, 0.1)
    N1d = np.size(g1d)
    # Initialise scattering factor array
    fe_TDS_1D = np.zeros(N1d, dtype=_float)

    for ig in tqdm(
        range(N1d), desc="Calculating TDS scattering factors", disable=tdisable
    ):
        ssq = g1d[ig] * g1d[ig] / 4
        dwf = np.exp(-Mx4 * ssq)

        for j in range(6):
            fe_TDS_1D[ig] += (
                A_wk[j]
                * A_wk[j]
                * (dwf * RI1(B_wk[j], B_wk[j], ssq) - RI2(B_wk[j], B_wk[j], ssq, Mx4))
            )
            for i in range(j):
                fe_TDS_1D[ig] += (
                    2
                    * A_wk[i]
                    * A_wk[j]
                    * (
                        dwf * RI1(B_wk[i], B_wk[j], ssq)
                        - RI2(B_wk[i], B_wk[j], ssq, Mx4)
                    )
                )

    fe_TDS_1D *= (
        4 * np.pi / (2 * np.pi * k0)
    )  # 2\pi in denominator is WK convention. 4\pi in numerator I'm unclear about, but
    # is in the FSCATT code and does yield agreement with other approaches to calculate TDS scattering factor.

    fe_TDS_1D *= gamma * (h**2 / (2 * np.pi * me * qe))  # Converts f'_g to V'_g

    # Interpolate onto 2D grid
    tck = interpolate.splrep(g1d, fe_TDS_1D, s=0)
    fe_TDS = interpolate.splev(modg, tck, der=0)

    return fe_TDS


def RI1(Bi, Bj, s2):
    """
    Supporting function for evaluating electron scattering factor for TDS using Weikenmeier & Kohl parameterisation
    Adapted from FSCATT code included in https://github.com/EMsoft-org/EMsoft/blob/develop/Source/EMsoftLib/others.f90

    Parameters
    ----------
    Bi, Bj : Entries in Weikenmeier & Kohl parameterisation of electron scattering factors
    s2 : square of scattering vector magnitude. Note s=g/2
    """

    C = 0.577215664901532  # Euler constant

    eps = max(Bi, Bj)
    eps = eps * s2

    if eps <= 0.05:
        RI1 = Bi * np.log((Bi + Bj) / Bi) + Bj * np.log((Bi + Bj) / Bj)
        RI1 = RI1 * np.pi
        if s2 == 0:
            return RI1

        Bi2 = Bi * Bi
        Bj2 = Bj * Bj
        temp = 0.5 * Bi2 * np.log(Bi / (Bi + Bj)) + 0.5 * Bj2 * np.log(Bj / (Bi + Bj))
        temp += 0.75 * (Bi2 + Bj2) - 0.25 * (Bi + Bj) * (Bi + Bj)
        temp += -0.5 * (Bi - Bj) * (Bi - Bj)
        RI1 += np.pi * s2 * temp
        return RI1

    Bis2 = Bi * s2
    Bjs2 = Bj * s2
    RI1 = 2.0 * C + np.log(Bis2) + np.log(Bjs2) - 2 * sc.expi(-Bi * Bj * s2 / (Bi + Bj))

    X1 = Bis2
    X2 = Bis2 * Bi / (Bi + Bj)
    X3 = Bis2
    RI1 += RIH1(X1, X2, X3)
    # RI1 += np.exp(-X1) * ( sc.expi(X2)-sc.expi(X3) )

    X1 = Bjs2
    X2 = Bjs2 * Bj / (Bi + Bj)
    X3 = Bjs2
    RI1 += RIH1(X1, X2, X3)
    # RI1 += np.exp(-X1) * ( sc.expi(X2)-sc.expi(X3) )

    RI1 *= np.pi / s2

    return RI1


def RI2(Bi, Bj, s2, Mx4):
    """
    Supporting function for evaluating electron scattering factor for TDS using Weikenmeier & Kohl parameterisation
    Adapted from FSCATT code included in https://github.com/EMsoft-org/EMsoft/blob/develop/Source/EMsoftLib/others.f90

    Parameters
    ----------
    Bi, Bj : Entries in Weikenmeier & Kohl parameterisation of electron scattering factors
    s2 : square of scattering vector magnitude. Note s=g/2
    Mx4 : 8 * pi^2 * ums. Note that Mx4*s2 = M*g2
    """

    Mx8 = 2 * Mx4

    BiplusMx4 = Bi + Mx4
    BjplusMx4 = Bj + Mx4
    BiplusMx8 = Bi + Mx8
    BjplusMx8 = Bj + Mx8

    eps = max(Bi, Bi, s2)
    eps = eps * s2

    if eps <= 0.05:
        RI2 = BiplusMx8 * np.log((Bi + Bj + Mx8) / BiplusMx8)
        RI2 += Bj * np.log((Bi + Bj + Mx8) / (BjplusMx8))
        RI2 += Mx8 * np.log(Mx8 / (BjplusMx8))
        RI2 *= np.pi
        if s2 == 0:
            return RI2

        temp = 0.5 * Mx4 * Mx4 * np.log(BiplusMx8 * BjplusMx8 / (Mx8 * Mx8))
        temp += (
            0.5 * BiplusMx4 * BiplusMx4 * np.log(BiplusMx8 / (BiplusMx4 + BjplusMx4))
        )
        temp += (
            0.5 * BjplusMx4 * BjplusMx4 * np.log(BjplusMx8 / (BiplusMx4 + BjplusMx4))
        )
        temp += 0.25 * BiplusMx8 * BiplusMx8 + 0.5 * Bi * Bi
        temp += 0.25 * BjplusMx8 * BjplusMx8 + 0.5 * Bj * Bj
        temp += -0.25 * (BiplusMx4 + BjplusMx4) * (BiplusMx4 + BjplusMx4)
        temp += (
            -0.5 * ((Bi * BiplusMx8 - Bj * BjplusMx8) / (BiplusMx4 + BjplusMx4)) ** 2
        )
        temp += -Mx4 * Mx4
        RI2 += np.pi * s2 * temp
        return RI2

    RI2 = sc.expi(-0.5 * Mx8 * s2 * BiplusMx4 / BiplusMx8) + sc.expi(
        -0.5 * Mx8 * s2 * BjplusMx4 / BjplusMx8
    )
    RI2 += -sc.expi(-BiplusMx4 * BjplusMx4 * s2 / (BiplusMx4 + BjplusMx4)) - sc.expi(
        -0.25 * Mx8 * s2
    )
    RI2 *= 2

    X1 = 0.5 * Mx8 * s2
    X2 = 0.25 * Mx8 * s2
    X3 = 0.25 * Mx8 * Mx8 * s2 / BiplusMx8
    RI2 += RIH1(X1, X2, X3)
    # RI2 += np.exp(-X1) * ( sc.expi(X2)-sc.expi(X3) )

    X1 = 0.5 * Mx8 * s2
    X2 = 0.25 * Mx8 * s2
    X3 = 0.25 * Mx8 * Mx8 * s2 / BjplusMx8
    RI2 += RIH1(X1, X2, X3)
    # RI2 += np.exp(-X1) * ( sc.expi(X2)-sc.expi(X3) )

    X1 = BiplusMx4 * s2
    X2 = BiplusMx4 * BiplusMx4 * s2 / (BiplusMx4 + BjplusMx4)
    X3 = BiplusMx4 * BiplusMx4 * s2 / BiplusMx8
    RI2 += RIH1(X1, X2, X3)
    # RI2 += np.exp(-X1) * ( sc.expi(X2)-sc.expi(X3) )

    X1 = BjplusMx4 * s2
    X2 = BjplusMx4 * BjplusMx4 * s2 / (BiplusMx4 + BjplusMx4)
    X3 = BjplusMx4 * BjplusMx4 * s2 / BjplusMx8
    RI2 += RIH1(X1, X2, X3)
    # RI2 += np.exp(-X1) * ( sc.expi(X2)-sc.expi(X3) )

    RI2 *= np.pi / s2

    return RI2


def RIH1(X1, X2, X3):
    """
    Supporting function for evaluating electron scattering factor for TDS using Weikenmeier & Kohl parameterisation
    Adapted from FSCATT code included in https://github.com/EMsoft-org/EMsoft/blob/develop/Source/EMsoftLib/others.f90
    """
    if X2 <= 20.0 and X3 <= 20.0:
        RIH1 = np.exp(-X1) * (sc.expi(X2) - sc.expi(X3))
        return RIH1

    if X2 > 20.0:
        RIH1 = np.exp(X2 - X1) * RIH2(X2) / X2
    else:
        RIH1 = np.exp(-X1) * sc.expi(X2)

    if X3 > 20.0:
        RIH1 -= np.exp(X3 - X1) * RIH2(X3) / X3
    else:
        RIH1 -= np.exp(-X1) * sc.expi(X3)

    return RIH1


def RIH2(X):
    """
    Supporting function for evaluating electron scattering factor for TDS using Weikenmeier & Kohl parameterisation
    Adapted from FSCATT code included in https://github.com/EMsoft-org/EMsoft/blob/develop/Source/EMsoftLib/others.f90
    """
    F = np.asarray(
        [
            1.000000,
            1.005051,
            1.010206,
            1.015472,
            1.020852,
            1.026355,
            1.031985,
            1.037751,
            1.043662,
            1.049726,
            1.055956,
            1.062364,
            1.068965,
            1.075780,
            1.082830,
            1.090140,
            1.097737,
            1.105647,
            1.113894,
            1.122497,
            1.131470,
        ]
    )

    X1 = 1.0 / X
    I = np.round(200.0 * X1).astype(int)
    I1 = I + 1

    RIH2 = F[I] + 200.0 * (F[I1] - F[I]) * (X1 - 0.5e-3 * I)

    return RIH2


def get_EELS_EDX_params(itransition_map):

    import sys
    from scipy import interpolate

    E0list = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    DElist = np.array([1.0, 10.0, 25.0, 50.0, 100.0])

    Z = itransition_map["Z"]

    if itransition_map["shell"] == "1s":
        if Z < 6 or Z > 50:
            sys.exit("Requested atomic number not handled")
        param_set = np.squeeze(EELS_EDX_1s_data[Z - 6, :, :, :])
    elif itransition_map["shell"] == "2s":
        if Z < 20 or Z > 86:
            sys.exit("Requested atomic number not handled")
        param_set = np.squeeze(EELS_EDX_2s_data[Z - 20, :, :, :])
    elif itransition_map["shell"] == "2p":
        if Z < 20 or Z > 86:
            sys.exit("Requested atomic number not handled")
        param_set = np.squeeze(EELS_EDX_2p_data[Z - 20, :, :, :])
    else:
        sys.exit("Requested shell not handled")

    if (itransition_map["E0"] / 1000 < 50.0) or (itransition_map["E0"] / 1000 > 400.0):
        sys.exit("Requested acceelrating voltage not handled")

    param_set2 = np.zeros(param_set.shape[1:])
    # Interpolate for accelerating voltage
    for ii in range(6):
        for i in range(29):
            tck = interpolate.splrep(E0list, param_set[:, ii, i], s=0)
            param_set2[ii, i] = interpolate.splev(
                itransition_map["E0"] / 1000, tck, der=0
            )

    params = np.zeros(param_set2.shape[1])
    if itransition_map["signal"] == "EDX":
        params = param_set2[5, :]
    elif itransition_map["signal"] == "EELS":
        if (itransition_map["DeltaE"] < 1.0) or (itransition_map["DeltaE"] > 100.0):
            sys.exit("Requested energy window not handled")

        for i in range(29):
            tck = interpolate.splrep(
                DElist, param_set2[:5, i] / DElist, s=0
            )  # [:4] excises the EDX data
            params[i] = (
                interpolate.splev(itransition_map["DeltaE"], tck, der=0)
                * itransition_map["DeltaE"]
            )

    else:
        sys.exit("How did you end up here if not doing EDX or EELS?")

    return params


def calculate_EELS_EDX_scattering_factors(
    gridshape,
    gridsize,
    EELS_EDX_params,
    ums,
    eV,
):
    """Calculate the electron scattering factors on a reciprocal space grid, include Debye-Waller factors.
       If TDS=True, calculate inelastic (TDS) electron scattering factors using the parameterisation of
       Weikenmeier and Kohl.

    Parameters
    ----------
    gridshape : (2,) array_like
        pixel size of the grid
    gridsize : (2,) array_like
        Lateral real space sizing of the grid in Angstrom
    EELS_EDX_params: (29,) array_like
        Effective scattering potential scattering factors at the sampling defined by svalList
        in this function
    ums: float
        Mean square vibrational amplitude of atom of interest
    eV : float
        Probe energy in electron volts


    Returns
    -------
    fe : (M, *gridshape)
        Array of electron scattering factors in reciprocal space for each
        element
    """

    from scipy import interpolate

    # Electron wave number (reciprocal of wavelength) in Angstrom
    k0 = wavev(eV)

    svalList = [
        0.0,
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
    ]
    tck = interpolate.splrep(svalList, EELS_EDX_params, s=0)

    # Initialise scattering factor array
    feff = np.zeros(gridshape, dtype=_float)

    # Get reciprocal space array
    g = q_space_array(gridshape, gridsize)
    gsq = np.square(g[0]) + np.square(g[1])
    svals = np.sqrt(gsq / 4.0)

    # Notes:
    # 1. Dividing by unit cell volume is replaced by the area division implicit in the Dkx.Dky
    #    product in make_effective_scattering_potential(), and no need to multiply by unit
    #    cell thickness in the cross-section evaluation.
    # 2. Fractional occupancy is included in defining the delta-function-like site array
    #    that will be convolved with this in make_effective_scattering_potential()
    # 3. The 2\pi K factor makes the results \mu's as defined by the Allen group
    feff = (
        interpolate.splev(svals, tck, der=0)
        * np.exp(-2 * np.pi**2 * gsq * ums)
        / (2.0 * np.pi * k0)
    )

    return feff
