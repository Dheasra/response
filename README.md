[![License](https://img.shields.io/badge/license-%20GPLv3-blue.svg)](../master/LICENSE)

# SCF solver implementation using the VAMPyR library for MultiWavelets computations
This is a Python implementation for a SCF solver afor zeroth and first order perturbed orbitals using the VAMPyR package. 
This implementation serves as a playground for a future linear response implementation in MultiWavelets in the 
[MRChem](https://github.com/MRChemSoft/mrchem) software.

## How to run the code 
# Dependancies 
Required python packages are listed in the SCFsolv.yml file. 
These can be imported into a Conda environment by running the following command:

`conda env create -n scfsolv --file scfsolv.yml`

# Run the code
The code itself in concentrated in the `scfsolv.py` file and can be called as a class. 
An exemple of use is provided in the `main.py` file.
