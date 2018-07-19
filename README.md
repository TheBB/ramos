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

### Data sources

Ramos can handle a variety of data sets, but the most mature support is for VTK files, where the
data is in two dimensions (triangular or quadrilateral meshes) or three dimensions (tetrahedral or
hexahedral meshes). Mixed meshes are fine.

VTK data sets are commonly found in a structure such as this:

    folder/filename-0.vtk
    folder/filename-1.vtk
    folder/filename-2.vtk
    ...

Or this:

    folder/<time1>/<fieldname1>.vtk
    folder/<time1>/<fieldname2>.vtk
    ...
    folder/<time2>/<fieldname1>.vtk
    folder/<time2>/<fieldname2>.vtk
    ...

To ensure that Ramos can understand your data source, execute the command

    ramos summary <folder>

and inspect the output, especially make sure that all steps are accounted for and that all fields
you expect to be found are found.

Note: By and large, all Ramos operations will write new data in the same format as the source data.

### Mesh interpolation

For reduction to work, all source data must coexist on the same mesh. This is not necessarily
guaranteed across multiple steps and data sources. For this, Ramos supplies the `interpolate` command.

    ramos interpolate -o <output> <input>

This interpolates all data sets from the input source on the same mesh and writes the data to
output. The mesh chosen is the first one in the input source (from the first timestep). You can
optionally specify which mesh to use with

    ramos interpolate -o <output> -t <target> <input>

After this, it's probably a good idea to do

    ramos summary <output>

to ensure that everything worked as planned.

### Reduction

To perform reduction on a data set, use the `reduce` command.

    ramos reduce -f <fieldname> -e <error> --min-modes <nmodes> -o <output> <inputs...>

Specify the field name you want to reduce over, and optionally the error threshold you are
interested in. In some cases it may be useful to specify a minimum number of modes to output as
well.

The modes will be written to the output data target as separate timesteps.
