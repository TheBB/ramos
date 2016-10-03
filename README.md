## Installation

The easiest way to install is to use `pip`,

    $ pip install .
    
Note that gmesh requires Python 3, so you need to use a pip command that is tied
to Python 3. On some systems, this may be called `pip3`.

    $ pip3 install .
    
Pip is often not installed by default, and may be found in a distribution
package named something like `python-pip` or `python3-pip`.

The above commands install gmesh system-wide and may require sudo. To install
locally, use

    $ pip3 install −−user .
    
The `gmesh` script will then be installed to a directory such as `~/.local/bin`.
To run gmesh, make sure that this is in your `PATH`, e.g. put

    export PATH=$HOME/.local/bin:$PATH
    
in your shell startup file.

### Dependencies

`pip` automatically installs the hard dependencies (such as numpy), however most
of the gmesh commands require optional dependencies to be installed. If you need
any of them, please install them manually, and please do ensure that you get the
Python 3 versions.

#### File types

- The Python NetCDF4 bindings are required for reading NetCDF4 files. E.g. in
  Ubuntu, try the `python3-netcdf4` package.
- The Python HDF5 bindings are required for reading HDF5 files. E.g. in Ubuntu,
  try the `python3-h5py` package.
- The Python `vtk` bindings are required for reading and writing VTK files. You
  may have to compile this yourself, since Python 3 support in VTK is rather
  new. The instructions
  [here](http://www.vtk.org/Wiki/VTK/Configure_and_Build#Configure_VTK_with_CMake) are
  quite complete and easy to follow. When configuring with `cmake` or `ccmake`,
  make sure that you enable the Python bindings and that the Python version is
  Python 3. After building and installing VTK, it is possible that it installs
  to `/usr/local`, in which case many of the libraries can’t be found. Add the
  path of the python modules to `PYTHONPATH` and the path of the libraries to
  `LD_LIBRARY_PATH` before starting, for example
  
      export PYTHONPATH=/usr/local/lib/python3.5/site-packages:$PYTHONPATH
      export LD_LIBRARY_PATH=/usr/local/lib:$PYTHONPATH
    
#### GUI

Showing the map and the GUI requires matplotlib (`python3-matplotlib`) and PyQt5
(`python3-pyqt5`).

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
