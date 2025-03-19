# assemble_table.py
"""
Script to assemble the FPV table from precomputed solutions. Uses the same input file as generate_table.py.
Usage:
    python assemble_table.py <config> <solutions_file> [<output_dir>] [-v]
"""
import argparse
import logging
from pathlib import Path
import tomli

from fpvgen.flamelet_table_generator import FlameletTableGenerator


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
    required_sections = ["tabulation"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    return config


def main():
    """Main entry point for assembling the FPV table.

    Handles command-line argument parsing, solution loading, and calls the
    assemble_FPV_table_CharlesX method to generate the FPV table.

    Command-line Arguments:
        config: Path to TOML configuration file
        solutions_file: Path to the HDF5 file containing precomputed solutions
        output_dir: Directory to save the assembled FPV table (optional, defaults to current directory)
        --verbose, -v: Enable verbose logging
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Assemble FPV table from precomputed solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="Path to TOML configuration file")
    parser.add_argument("solutions_file", type=Path, help="Path to HDF5 file containing precomputed solutions")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory to save the assembled FPV table (default: current directory)",
    )
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

        # Load solutions
        logger.info(f"Loading solutions from {args.solutions_file}")
        generator = FlameletTableGenerator.load_solutions(args.solutions_file)

        # Assemble FPV table
        logger.info(f"Assembling FPV table in {args.output_dir}")
        generator.assemble_FPV_table_CharlesX(output_dir=args.output_dir, **config["tabulation"])

        # Create plots if requested
        if config["plotting"].get("create_plots", True):
            logger.info("Creating visualization plots")
            generator.plot_table(
                output_prefix=output_dir / "table",
                **config["plotting"]["table"],
            )

        logger.info("FPV table assembly completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error assembling FPV table: {e}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
