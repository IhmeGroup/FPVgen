[![DOI](https://zenodo.org/badge/880419274.svg)](https://zenodo.org/badge/latestdoi/880419274)

# FPVgen
FPVgen is a software tool for generating flamelet progress variable (FPV) tables for combustion simulations. It handles the generation of flamelet solutions for counterflow diffusion flames, including the computation of complete S-curves with stable, unstable, and extinction branches.

## Installation

**Note:** This repository requires Cantera >= 3.1.

To install FPVgen, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/IhmeGroup/FPVgen.git
    cd FPVgen
    ```

2. Create a virtual environment and activate it (using `venv` or `conda`):
    ```sh
    # Using venv
    python -m venv fpvgen
    source fpvgen/bin/activate  # On Windows use `fpvgen\Scripts\activate`

    # Using conda
    conda create --name fpvgen python==3.12 pip
    conda activate fpvgen
    ```

3. Install FPVgen and its dependencies using `pip`:
    ```sh
    pip install .
    ```

## Basic Structure of an Input File

The input file for FPVgen uses TOML format. Below is an example of the structure of a minimal input file:

```toml
# input.toml
# Configuration for flamelet table generation

[mechanism]
file = "gri30.yaml"                                      # Path to mechanism file
transport_model = "mixture-averaged"                     # Transport model: "unity-Lewis-number", "mixture-averaged", or "multicomponent"
prog_def = { CO = 1.0, H2 = 1.0, CO2 = 1.0, H2O = 1.0 }  # Progress variable definition

[conditions]
pressure = 101325.0  # Operating pressure in Pa

[fuel_inlet]
composition = { CH4 = 1.0 }  # Pure methane
temperature = 300.0          # Temperature in K

[oxidizer_inlet]
composition = { O2 = 0.21, N2 = 0.79 }  # Air
temperature = 300.0                     # Temperature in K
```

A comprehensive input file with all available options may be found at [sample/input.toml](sample/input.toml)

## Usage

### Generating Flamelet Tables

To run the full process, including computation of flamelets, plotting of results, and assembly of the table, run the following command:

```sh
generate_table <input> [--verbose]
```

- `<input>`: Path to the TOML input file.
- `--verbose`: Enable verbose logging (optional).

### Plotting Flamelet Solutions

To plot flamelet solutions from an HDF5 file, run the following command:

```sh
plot_flamelets <solutions_file> [-o <output_dir>] [-v]
```

- `<solutions_file>`: Path to the HDF5 solutions file.
- `-o <output_dir>`: Output directory for plots (optional, default: same as solutions file).
- `-v`: Enable verbose logging (optional).

### Assembling FPV Table

To assemble the FPV table from existing solutions, run the following command:

```sh
assemble_table <solutions_file> <output_dir> [--verbose]
```

- `<solutions_file>`: Path to the HDF5 solutions file.
- `<output_dir>`: Directory to save the assembled FPV table.
- `--verbose`: Enable verbose logging (optional).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
