## Installation

The easiest way to install is to use `pip`,

    $ pip install .
    
Note that Ramos requires Python 3, so you need to use a pip command that is tied
to Python 3. On some systems, this may be called `pip3`.

    $ pip3 install .
    
Pip is often not installed by default, and may be found in a distribution
package named something like `python-pip` or `python3-pip`.

The above commands install gmesh system-wide and may require sudo. To install
locally, use

    $ pip3 install −−user .
    
The `ramos` script will then be installed to a directory such as `~/.local/bin`.
To run gmesh, make sure that this is in your `PATH`, e.g. put

    export PATH=$HOME/.local/bin:$PATH
    
in your shell startup file.

### Dependencies

`pip` automatically installs the hard dependencies (such as numpy), however some
of the `ramos` commands require optional dependencies to be installed. If you need
any of them, please install them manually, and please do ensure that you get the
Python 3 versions.

#### File types

- The Python HDF5 bindings are required for reading HDF5 files. E.g. in Ubuntu,
  try the `python3-h5py` package.
- The Python `vtk` bindings are required for reading and writing VTK files.
    
## Usage

### Structure

To interpolate an unstructured vtk file to a structured one, use the `structure`
command.

    gmesh structure --out out.vtk in.vtk
    
It accepts arguments `--nx`, `--ny`, and `--nz` for the number of *cells* (not
points) in each axial direction. Non-axially aligned grids are not supported.
The default values are 10.

Optionally, you may choose to slice at a given point in any direction. For that,
use the `--xval`, `--yval` and `--zval` options. These are turned off by
default. You should not specify both `--xval` and `--nx`, and similarly for the
other directions.
