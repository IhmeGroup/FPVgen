# plot_flamelets.py
"""
Script to load flamelet solutions and generate plots.
Usage:
    python plot_flamelets.py <solutions_file> [-o <output_dir>] [-v]
"""
import argparse
import logging
from pathlib import Path

from fpvgen.flamelet_table_generator import FlameletTableGenerator


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot flamelet solutions from HDF5 file")
    parser.add_argument("solutions_file", type=Path, help="Path to HDF5 solutions file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for plots (default: same as solutions file)",
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
        # Load solutions
        logger.info(f"Loading solutions from {args.solutions_file}")
        generator = FlameletTableGenerator.load_solutions(args.solutions_file)

        # Set output directory
        output_dir = args.output or args.solutions_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Generate S-curve plot
        logger.info("Generating S-curve plot")
        generator.plot_s_curve(output_dir=output_dir)

        # Generate temperature profiles plot
        logger.info("Generating temperature profiles plot")
        generator.plot_temperature_profiles(output_dir=output_dir / "temperature_profiles.png")

        # Generate strain vs chi_st plot
        logger.info("Generating strain vs chi_st plot")
        generator.plot_strain_chi_st(strain_rate_type="max", output_dir=output_dir / "strain_max_chi_st.png")
        generator.plot_strain_chi_st(strain_rate_type="nom", output_dir=output_dir / "strain_nom_chi_st.png")

        logger.info("Plot generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    main()
