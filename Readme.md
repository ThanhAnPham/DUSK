# Deep-prior ODEs Augment Fluorescence Imaging with Chemical Sensors

This code contains the method Deep spatiotemporal priors for Uncoupling Sensors Kinetics (DUSK). The run file estimates the spatiotemporal distribution of a chemical species of interest (CSI) in a simulated case and on one real data.

Paper: https://doi.org/10.1038/s41467-024-53232-2

Note: if the model is already optimized for the same configuration, the script does not re-run the optimization.

<p align="center">
<img src="Illustration.gif" alt="animated" width="400"/>
</p>
<p align="center">
  From left to right: raw measurements, predicted measurements, and fitted concentration
</p>

## Dependencies

These packages can be installed via conda and/or pip. Adapt the versions according to your hardware (notably the CUDA version for the GPU).

- kornia==0.7.0
- matplotlib==3.7.2
- pytorch==2.0.1
- pytorch-cuda==11.7
- scikit-image==0.22.0
- scipy==1.11.4
- tifffile==2023.8.12

## Structure of the repositery

DUSK is mainly ran via the file `main_recon.py`.
The parameters are stored in the file `GCAMPparam.py`, where the role of each parameter is explained therein.
Since it is using a parser of arguments, parameters can be changed during the call (see next Section).

## How to run DUSK on the provided examples

On Code Ocean, the reproducible run will call the file ``run" which includes the same commands shown in this section. Note that the run may take some times because of the number of epochs (Fewer iterations without much loss of quality are possible).

*Reconstruction of simulated data with DUSK (Fig. 2)*

```sh
python -u main_recon.py --sensor 'simGCAMP' --Toi 2. 130. --qe0 100 --dt 0.005 0.005 --paramZ 64
```

*Reconstruct real data with DUSK (Fig. 5, jGCaMP8s). Default values in GCAMPparam.py reproduce Fig. 3A*

```sh
python -u main_recon.py
```

## How to run DUSK on your own dataset

To run on your own dataset, please adapt few parameters.
- Set the acquisition frame rate of your acquired data via `--fs` (e.g., [Hz])
- Provide the data file (Stack tiff file) `--fileoi`
- Set the `--sensor customXXX` with XXX being your whatever you would like.
- Set the kinetics parameters of your sensor via `--kf` and `--kb`. Units should agree with ``fs``
- Set the acquisition (1/fs) and forward time step via `--dt a b` (see GCAMPparam.py). Units should agree with fs (e.g., inverse of fs unit [sec])

You can process a spatiotemporal region of interest of your data with DUSK.
- For time
  - Set the exp_start your acquired data via `--exp_start` (0 if not relevant).
  - Set the time window of interest via `--Toi ta tb` with `ta, tb` the beginning and ending time, respectively. The units should agree with ``fs``.
- For space, set the `--roim a b w h` with `a,b` the top left indices of the window of interest and `w,h` the width and length, respectively, of the window of interest.

## License

Distributed under the MIT License. See LICENSE for more information.
