# generate_table.py
"""
Command-line interface for generating flamelet tables using TOML configuration files.

This script provides a command-line interface for generating flamelet tables using
configuration files in TOML format. It handles the complete workflow of:
- Loading and validating configuration
- Setting up the flamelet generator
- Computing flamelet solutions including extinction points
- Saving solutions and generating visualization plots

Usage:
    python generate_table.py <config> [--verbose]

Configuration File Format:
    The TOML configuration file must contain the following sections:
    - mechanism: Chemical mechanism specification
        - file: Path to mechanism file
    - fuel_inlet: Fuel stream properties
        - composition: Dict of species and their mole fractions
        - temperature: Temperature in Kelvin
    - oxidizer_inlet: Oxidizer stream properties
        - composition: Dict of species and their mole fractions
        - temperature: Temperature in Kelvin
    - solver: Solver configuration
        - output_dir: Output directory (optional)
        - width_ratio: Ratio of the domain width to the flame thickness (optional)
        - n_extinction_points: Number of extinction branch points (optional)
        - create_plots: Whether to generate plots (optional)
        - options: Additional solver options (optional)
            - loglevel: Cantera solver log level (optional, default: 0)
                0: No solver output
                1: Basic solver output
                2: Detailed convergence information
                3: Very detailed output including Jacobian information
            ... other solver options ...
    - conditions: Operating conditions
        - pressure: Operating pressure in Pa (optional, default: 101325)
        - initial_chi_st: Initial scalar dissipation rate (optional)
    - plotting: Plot customization (optional)
        - s_curve: Options for S-curve plot
        - profiles: Options for temperature profile plots
    - restart: Optional restart configuration
        - solutions_file: Path to previous solutions file
        - solution_index: Index of solution to restart from (optional, defaults to last solution)
"""
import argparse
import logging
import sys
from pathlib import Path
import tomli

from fpvgen.flamelet_table_generator import FlameletTableGenerator, InletCondition


def load_config(config_file: Path) -> dict:
    """Load and validate TOML configuration file.

    Args:
        config_file: Path to the TOML configuration file

    Returns:
        dict: Validated configuration dictionary

    Raises:
        ValueError: If the file cannot be read or required sections are missing
    """
    try:
        with open(config_file, "rb") as f:
            config = tomli.load(f)
    except Exception as e:
        raise ValueError(f"Error reading config file: {e}")

    # Validate required sections
    required_sections = ["mechanism", "fuel_inlet", "oxidizer_inlet", "solver"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    # Validate restart configuration if present
    if "restart" in config:
        restart = config["restart"]
        if "solutions_file" not in restart:
            raise ValueError("Restart configuration must specify 'solutions_file'")
        if not Path(restart["solutions_file"]).exists():
            raise ValueError(f"Restart solutions file not found: {restart['solutions_file']}")

    return config


def create_generator(config: dict) -> FlameletTableGenerator:
    """Create FlameletTableGenerator instance from configuration.

    Creates and initializes a FlameletTableGenerator with the specified mechanism
    and inlet conditions from the configuration.

    Args:
        config: Configuration dictionary containing mechanism and inlet specifications

    Returns:
        FlameletTableGenerator: Initialized generator instance

    Note:
        Mass flux values are set to placeholder values as they are automatically
        adjusted by the solver based on the target scalar dissipation rate.
    """
    fuel_inlet = InletCondition(
        composition=config["fuel_inlet"]["composition"],
        temperature=config["fuel_inlet"]["temperature"],
    )

    oxidizer_inlet = InletCondition(
        composition=config["oxidizer_inlet"]["composition"],
        temperature=config["oxidizer_inlet"]["temperature"],
    )

    # Create generator
    return FlameletTableGenerator(
        mechanism_file=config["mechanism"]["file"],
        transport_model=config["mechanism"].get("transport_model"),
        fuel_inlet=fuel_inlet,
        oxidizer_inlet=oxidizer_inlet,
        pressure=config["conditions"].get("pressure"),
        prog_def=config["mechanism"].get("prog_def"),
        width_ratio=config["solver"].get("width_ratio"),
        width_change_enable=config["solver"].get("width_change_enable"),
        width_change_max=config["solver"].get("width_change_max"),
        width_change_min=config["solver"].get("width_change_min"),
        initial_chi_st=config["conditions"].get("initial_chi_st"),
        solver_loglevel=config["solver"].get("loglevel"),
        strain_chi_st_model_param_file=config["solver"].get("strain_chi_st_model_param_file"),
    )


def main():
    """Main entry point for the flamelet table generator.

    Handles command-line argument parsing, configuration loading, and orchestrates
    the flamelet generation process. Creates output directory, generates solutions,
    saves results, and optionally creates visualization plots.

    Command-line Arguments:
        config: Path to TOML configuration file
        --verbose, -v: Enable verbose logging

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate flamelet tables from TOML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="Path to TOML configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Set output directory
        output_dir = Path(config["solver"].get("output_dir", "flamelet_results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Handle restart case
        if "restart" in config:
            logger.info("Restarting from previous solution")
            restart_config = config["restart"]
            solutions_file = Path(restart_config["solutions_file"])
            generator = FlameletTableGenerator.load_solutions(solutions_file)
            restart_from = restart_config.get("solution_index", -1)  # Default to last solution

            # If output_dir is different from solutions_file location, copy existing solutions
            if output_dir != solutions_file.parent:
                logger.info(f"Copying existing solutions to new output directory: {output_dir}")
                generator.save_all_solutions(output_dir)
        else:
            # Create new generator
            logger.info("Initializing new flamelet generator")
            generator = create_generator(config)
            restart_from = None

        # Pop some solver options that are not passed to compute_s_curve
        solver_options = config["solver"]
        create_plots = solver_options.pop("create_plots", True)
        solver_options.pop("width_ratio", None)
        solver_options.pop("width_change_enable", None)
        solver_options.pop("width_change_max", None)
        solver_options.pop("width_change_min", None)
        solver_options.pop("loglevel", None)

        # Compute flamelets
        logger.info("Computing flamelet solutions")
        solver_options = config.get("solver", {})
        generator.compute_s_curve(**solver_options)

        # Learn the strain rate and chi_st relationship
        logger.info("Learning the strain rate and chi_st relationship")
        generator.learn_strain_chi_st_mapping(output_file=output_dir / "strain_chi_st.json")

        # Assemble the FPV table
        logger.info("Assembling the FPV table")
        generator.assemble_FPV_table_CharlesX(
            output_dir=output_dir,
            **config.get("tabulation", {})
        )

        # Create plots if requested
        if create_plots:
            logger.info("Creating visualization plots")
            generator.plot_s_curve(
                output_file=output_dir / "s_curve.png",
                **config.get("plotting", {}).get("s_curve", {}),
            )
            generator.plot_temperature_profiles(
                output_file=output_dir / "temperature_profiles.png",
                **config.get("plotting", {}).get("profiles", {}),
            )
            generator.plot_strain_chi_st(strain_rate_type="max", output_file=output_dir / "strain_max_chi_st.png")
            generator.plot_strain_chi_st(strain_rate_type="nom", output_file=output_dir / "strain_nom_chi_st.png")

        logger.info("Flamelet generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error generating flamelets: {e}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
