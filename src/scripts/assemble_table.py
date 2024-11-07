import argparse
import logging
from pathlib import Path

from fpvgen.flamelet_table_generator import FlameletTableGenerator


def main():
    """Main entry point for assembling the FPV table.

    Handles command-line argument parsing, solution loading, and calls the
    assemble_FPV_table_CharlesX method to generate the FPV table.

    Command-line Arguments:
        solutions_file: Path to the HDF5 file containing precomputed solutions
        output_dir: Directory to save the assembled FPV table (optional, defaults to current directory)
        --verbose, -v: Enable verbose logging
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Assemble FPV table from precomputed solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        # Load solutions
        logger.info(f"Loading solutions from {args.solutions_file}")
        generator = FlameletTableGenerator.load_solutions(args.solutions_file)

        # Assemble FPV table
        logger.info(f"Assembling FPV table in {args.output_dir}")
        generator.assemble_FPV_table_CharlesX(output_dir=args.output_dir)

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
