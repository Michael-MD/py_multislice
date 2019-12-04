import numpy as np
import torch
import tqdm
from .py_multislice import make_propagators, scattering_matrix, generate_STEM_raster
from .utils.torch_utils import (
    cx_from_numpy,
    cx_to_numpy,
    amplitude,
    complex_matmul,
    complex_mul,
    get_device
)
from .utils.numpy_utils import fourier_shift,crop
import matplotlib.pyplot as plt
from .Ionization import make_transition_potentials


def window_indices(center, windowsize, gridshape):
    """Makes indices for a cropped window centered at center with size given
    by windowsize on a grid"""
    window = []
    for i, wind in enumerate(windowsize):
        indices = np.arange(-wind // 2, wind // 2, dtype=np.int) + wind % 2
        indices += int(round(center[i] * gridshape[i]))
        indices = np.mod(indices, gridshape[i])
        window.append(indices)

    return (window[0][:, None] * gridshape[0] + window[1][None, :]).ravel()

def STEM_EELS_multislice(
    probe,
    crystal,
    ionization_potentials,
    ionization_sites,
    thicknesses,
    eV,
    alpha,
    subslices= [1.0],
    batch_size=1,
    detectors=None,
    FourD_STEM=False,
    datacube=None,
    scan_posn=None,
    dtype=None,
    device_type=None,
    tiling=[1, 1],
    seed=None,
    showProgress=True,
    threshhold=1e-4,
    nT=5
):  
    """Perform a STEM-EELS simulation using only the multislice algorithm"""
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = crystal.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)
    
    if dtype is None:
        dtype = transmission_functions.dtype
    
    nslices = np.ceil(thicknesses / crystal.unitcell[2]).astype(np.int)
    P = make_propagators(gridshape, rsize, eV, subslices)
    T = torch.zeros(nT, len(subslices), *gridshape, 2, device=device)

    from . import transition_potential_multislice
    method = transition_potential_multislice
    args = (nslices,subslices, propagators, transmission_functions, ionization_potentials,ionization_sites,tiling, device, seed)
    kwargs = {
        "return_numpy": False,
        "qspace_in": True,
        "qspace_out": True,
        "threshhold": threshhold,
        "showProgress": showProgress,
    }

    EELS_image = STEM(
        rsize,
        probe,
        method,
        nslices,
        eV,
        alpha,
        batch_size=batch_size,
        detectors=detectors,
        FourD_STEM=FourD_STEM,
        datacube=datacube,
        scan_posn=scan_posn,
        device=device,
        tiling=tiling,
        seed=seed,
        showProgress=showProgress,
        method_args=args,
        method_kwargs=kwargs,
    )

    return tile_out_ionization_image(EELS_image,tiling)

def STEM_EELS_PRISM(
    crystal,
    gridshape,
    eV,
    app,
    det,
    thicknesses,
    Ztarget,
    boundConfiguration,
    boundQuantumNumbers,
    freeConfiguration,
    freeQuantumNumbers,
    epsilon,
    Hn0_crop = None,
    subslices=[1.0],
    device_type=None,
    tiling=[1, 1],
    nT=5,
    PRISM_factor = [1,1]
):

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = crystal.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)

    # TODO enable a thickness series
    nslices = np.ceil(thicknesses / crystal.unitcell[2]).astype(np.int)
    P = make_propagators(gridshape, rsize, eV, subslices)
    T = torch.zeros(nT, len(subslices), *gridshape, 2, device=device)

    # Make the transmission functions for multislice
    for i in range(nT):
        T[i] = crystal.make_transmission_functions(gridshape, eV, subslices, tiling)

    # Get the coordinates of the target atoms in a unit cell
    mask = crystal.atoms[:, 3] == Ztarget
    natoms = np.sum(mask)

    # Adjust fractional coordinates for tiling of unit cell
    coords = crystal.atoms[mask][:, :3] / np.asarray(tiling + [1])

    # Scattering matrix 1 propagates probe from surface of specimen to slice of
    # interest
    S1 = scattering_matrix(rsize, P, T, 0, eV, app, batch_size=5, subslicing=True,PRISM_factor=PRISM_factor)

    


    # Scattering matrix 2 propagates probe from slice of interest to exit surface
    S2 = scattering_matrix(
        rsize,
        P,
        T,
        nslices * len(subslices),
        eV,
        app,
        batch_size=5,
        subslicing=True,
        transposed=True,
        PRISM_factor = PRISM_factor,
    )
    # Link the slices and seeds of both scattering matrices
    S1.seed = S2.seed

    from .Ionization import orbital, transition_potential, tile_out_ionization_image
    
    nstates = len(freeQuantumNumbers)
    Hn0 = make_transition_potentials(gridshape,rsize,eV,Ztarget,epsilon,boundQuantumNumbers,boundConfiguration,freeQuantumNumbers,freeConfiguration)
    print(gridshape,rsize,eV,Ztarget,epsilon,boundQuantumNumbers,boundConfiguration,freeQuantumNumbers,freeConfiguration)
    fig,ax = plt.subplots(nrows = Hn0.shape[0])
    for i,iHn0 in enumerate(Hn0):
        ax[i].imshow(np.abs(np.fft.ifft2(iHn0)))
    plt.show(block=True)

    if Hn0_crop is None:
        Hn0_crop = [ S1.stored_gridshape[i] for i in range(2)]
    else:
        Hn0_crop = [min(Hn0_crop[i], S1.stored_gridshape[i]) for i in range(2)]
        Hn0 = np.fft.fft2(np.fft.ifftshift(crop(np.fft.fftshift(Hn0,axes=[-2,-1]),Hn0_crop),axes=[-2,-1]))

    fig,ax = plt.subplots(nrows = Hn0.shape[0])
    for i,iHn0 in enumerate(Hn0):
        ax[i].imshow(np.abs(np.fft.ifft2(iHn0)))
    plt.show(block=True)
    # Make probe wavefunction vectors for scan
    # Get kspace grid in units of inverse pixels
    ky, kx = [
        np.fft.fftfreq(gridshape[-2 + i], d=1 / gridshape[-2 + i]) for i in range(2)
    ]

    # Generate scan positions in pixels
    scan = generate_STEM_raster(S1.S.shape[-2:], rsize[:2], eV, app)
    nprobe_posn = len(scan[0]) * len(scan[1])

    scan_array = np.zeros((nprobe_posn, S1.S.shape[0]), dtype=np.complex)

    # TODO implement aberrations and defocii
    for i in range(S1.nbeams):
        scan_array[:, i] = (
            np.exp(-2 * np.pi * 1j * ky[S1.beams[i, 0]] * scan[0])[:, None]
            * np.exp(-2 * np.pi * 1j * kx[S1.beams[i, 1]] * scan[1])[None, :]
        ).ravel()

    scan_array = cx_from_numpy(scan_array, dtype=S1.dtype, device=device)

    # Initialize Image
    EELS_image = torch.zeros(len(scan[0]) * len(scan[1]), dtype=S1.dtype, device=device)

    total_slices = nslices * len(subslices)
    for islice in tqdm.tqdm(range(total_slices), desc="Slice"):

        # Propagate scattering matrices to this slice
        if islice > 0:
            S1.Propagate(
                islice, P, T, subslicing=True, batch_size=5, showProgress=False
            )
            S2.Propagate(
                total_slices - islice,
                P,
                T,
                subslicing=True,
                batch_size=5,
                showProgress=False,
            )


        S2.S = S2.S.reshape(S2.S.shape[0], np.product(S2.stored_gridshape), 2)
        
        #Work out which subslice of the crystal unit cell we are in
        subslice = islice % S1.nsubslices
        
        # Get list of atoms within this slice
        atomsinslice = coords[
            np.logical_and(
                coords[:, 2] >= subslice / S1.nsubslices,
                coords[:, 2] < (subslice + 1) / S1.nsubslices,
            ),
            :2,
        ]

        # Iterate over atoms in this slice
        for atom in tqdm.tqdm(atomsinslice, "Transitions in slice"):

            windex = torch.from_numpy(
                window_indices(atom, Hn0_crop, S1.stored_gridshape)
            )

            for i in range(nstates):
                # Initialize matrix describing this transition event
                SHn0 = torch.zeros(
                    S1.S.shape[0], S2.S.shape[0], 2, dtype=S1.S.dtype, device=device
                )

                # Sub-pixel shift of Hn0
                posn = atom * np.asarray(gridshape)
                destination = np.remainder(posn, 1.0)
                Hn0_ = np.fft.fftshift(
                    fourier_shift(Hn0[i], destination, qspacein=True)
                ).ravel()
                fig,ax = plt.subplots(nrows=1)
                ax.imshow(np.abs(Hn0_).reshape(Hn0_crop))
                plt.show(block = True)

                # Convert Hn0 to pytorch Tensor
                Hn0_ = cx_from_numpy(Hn0_, dtype=S1.S.dtype, device=device)

                for i, S1component in enumerate(S1.S):
                    # Multiplication of component of first scattering matrix
                    # (takes probe it depth of ionization) with the transition 
                    # potential
                    Hn0S1 = complex_mul(Hn0_, S1component.flatten(end_dim=-2)[windex, :])

                    fig,ax = plt.subplots(nrows=3)
                    ax[0].imshow(np.abs(cx_to_numpy(Hn0S1)).reshape(Hn0_crop))
                    ax[1].imshow(np.abs(cx_to_numpy(Hn0_)).reshape(Hn0_crop))
                    ax[2].imshow(np.abs(cx_to_numpy( S1component.flatten(end_dim=-2)[windex, :])).reshape(Hn0_crop))
                    plt.show(block = True)

                    # Matrix multiplication with second scattering matrix (takes
                    # scattered electrons to EELS detector)
                    SHn0[i] = complex_matmul(S2.S[:, windex], Hn0S1)

                # Build a mask such that only probe positions within a PRISM 
                # cropping region about the probe are evaluated
                scan_mask = np.logical_and((np.abs((atom[0] - scan[0]+0.5) % 1.0 -0.5)<=1/PRISM_factor[0]/2)[:,None],
                                           (np.abs((atom[1] - scan[1]+0.5) % 1.0 -0.5)<=1/PRISM_factor[1]/2)[None,:]).ravel()
                
                EELS_image += torch.sum(amplitude(complex_matmul(scan_array, SHn0)), axis=1)

        # Reshape scattering matrix S2 for propagation
        S2.S = S2.S.reshape((S2.S.shape[0], *S2.stored_gridshape, 2))
    
    # Move EELS_image to cpu and numpy and then reshape to rectangular grid
    EELS_image = EELS_image.cpu().numpy().reshape(len(scan[0]), len(scan[1]))

    return tile_out_ionization_image(EELS_image,tiling)
    