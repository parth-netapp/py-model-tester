#!/usr/bin/env python3
"""
PDF to PNG Converter using poppler-utils (pdftocairo)

This script can be used both as a standalone command-line tool
and as a module to be imported in other Python scripts.

python pdf_processor.py document.pdf output/
python pdf_processor.py document.pdf output/ --dpi 600
python pdf_processor.py --check  # Check if poppler-utils is installed

Example: uv run ./pdf_processor.py ./page13.pdf ./pdf_images_dir/
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
from typing import Union

from logging_config import get_logger


logger = get_logger(__name__)
def check_poppler_installed() -> bool:
    """Check if pdftocairo is available on the system."""
    try:
        result = subprocess.run(
            ["pdftocairo", "-v"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the total number of pages in a PDF using pdfinfo.

    Raises RuntimeError if pdfinfo is not available or page count
    cannot be determined.
    """
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        logger.error(f"pdfinfo not found when inspecting {pdf_path}")
        raise RuntimeError(
            "pdfinfo not found. Please install poppler-utils (pdfinfo) alongside pdftocairo."
        ) from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to read PDF info for {pdf_path}: {e.stderr.strip()}")
        raise RuntimeError(f"Failed to read PDF info: {e.stderr.strip()}") from e

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                total_pages = int(line.split(":", 1)[1].strip().split()[0])
                logger.info(f"PDF page count | path={pdf_path} | pages={total_pages}")
                return total_pages
            except (IndexError, ValueError):
                break

    raise RuntimeError("Could not determine page count from pdfinfo output.")


def pdf_to_png(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    dpi: int = 140,
) -> dict:
    """
    Convert a PDF file to PNG images using pdftocairo.
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory where PNG images will be saved
        dpi: Resolution for output images (default: 140)
    
    Returns:
        dict: Contains 'success' (bool), 'message' (str), and 'output_files' (list)
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        RuntimeError: If pdftocairo is not installed
    """
    # Convert to Path objects
    pdf_path = Path(pdf_path)
    output_root = Path(output_dir)
    
    # Validate PDF file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    
    # Check if poppler-utils is installed
    if not check_poppler_installed():
        raise RuntimeError(
            "pdftocairo not found. Please install poppler-utils:\n"
            "  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
            "  macOS: brew install poppler\n"
            "  Fedora: sudo dnf install poppler-utils"
        )

    # Determine total number of pages so we can process in batches
    total_pages = get_pdf_page_count(pdf_path)
    
    # Determine output directory and prefix using a structured layout with
    # pdftocairo's own page numbering:
    #   {output_root}/{pdf_name}/{pdf_name}_page-<page>.png
    pdf_name = pdf_path.stem

    # Structured output under a subdirectory named after the PDF
    images_dir = output_root / pdf_name
    images_dir.mkdir(parents=True, exist_ok=True)

    # Use a prefix that encodes both the PDF name and "page"; pdftocairo
    # will append "-<page>" to this prefix, e.g. mydoc_page-1.png.
    internal_prefix = f"{pdf_name}_page"
    output_prefix = images_dir / internal_prefix

    # Process the PDF in batches of 10 pages to limit memory usage.
    batch_size = 15
    combined_stdout = ""
    combined_stderr = ""

    logger.info(
        f"Starting PDF to PNG conversion | path={pdf_path} | dpi={dpi} | batch_size={batch_size}"
    )

    # Track total command execution time across all batches
    conversion_time = 0.0

    try:
        for start_page in range(1, total_pages + 1, batch_size):
            end_page = min(start_page + batch_size - 1, total_pages)

            command = [
                "pdftocairo",
                "-png",
                "-r",
                str(dpi),
                "-f",
                str(start_page),
                "-l",
                str(end_page),
                str(pdf_path),
                str(output_prefix),
            ]

            logger.info(
                f"Running pdftocairo | path={pdf_path} | pages={start_page}-{end_page}"
            )

            # Time only the subprocess execution
            batch_start = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
            batch_time = time.time() - batch_start
            conversion_time += batch_time

            combined_stdout += result.stdout
            combined_stderr += result.stderr

        # After processing all batches, discover all generated PNG files.
        # pdftocairo may use zero-padded page numbers (e.g. -01, -02), so
        # rely on a glob pattern instead of constructing filenames by hand.
        final_files: list[Path] = sorted(images_dir.glob(f"{internal_prefix}-*.png"))
        if not final_files:
            final_files = sorted(images_dir.glob(f"{internal_prefix}.png"))

        logger.info(
            f"Completed PDF to PNG conversion | path={pdf_path} | images={len(final_files)} | time={conversion_time:.3f}s"
        )

        return {
            "success": True,
            "message": f"Successfully converted {pdf_path.name} to {len(final_files)} PNG image(s)",
            "output_files": [str(f) for f in final_files],
            "stdout": combined_stdout,
            "stderr": combined_stderr,
            "conversion_time_seconds": conversion_time,
            "total_pages": total_pages,
        }

    except subprocess.CalledProcessError as e:
        logger.error(
            f"pdftocairo failed | path={pdf_path} | returncode={e.returncode} | stderr={e.stderr}"
        )
        return {
            "success": False,
            "message": f"Error converting PDF: {e.stderr}",
            "output_files": [],
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to PNG images using poppler-utils",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s document.pdf output/
    %(prog)s document.pdf output/ --dpi 600
        """
    )
    
    parser.add_argument(
        "pdf_path",
        help="Path to the input PDF file"
    )
    
    parser.add_argument(
        "output_dir",
        help="Directory where PNG images will be saved"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for output images (default: 150)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if poppler-utils is installed and exit"
    )
    
    args = parser.parse_args()
    
    logger.info(f"pdf_processor CLI invoked with args: {args}")

    # Handle --check flag
    if args.check:
        if check_poppler_installed():
            logger.info("pdftocairo is installed and available")
            print("✓ pdftocairo is installed and available")
            sys.exit(0)
        else:
            logger.error("pdftocairo is not installed")
            print("✗ pdftocairo is not installed")
            print("\nInstallation instructions:")
            print("  Ubuntu/Debian: sudo apt-get install poppler-utils")
            print("  macOS: brew install poppler")
            print("  Fedora: sudo dnf install poppler-utils")
            sys.exit(1)
    
    # Convert PDF to PNG
    try:
        result = pdf_to_png(
            pdf_path=args.pdf_path,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
        
        if result["success"]:
            logger.info(
                f"CLI conversion succeeded | path={args.pdf_path} | images={len(result['output_files'])}"
            )
            print(f"✓ {result['message']}")
            print(f"\nOutput files:")
            for file in result["output_files"]:
                print(f"  - {file}")
            sys.exit(0)
        else:
            logger.error(
                f"CLI conversion failed | path={args.pdf_path} | message={result['message']}"
            )
            if result["stderr"]:
                logger.error(f"CLI conversion stderr: {result['stderr']}")
            print(f"✗ {result['message']}", file=sys.stderr)
            if result["stderr"]:
                print(f"\nError details:\n{result['stderr']}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        logger.exception("Unhandled exception in pdf_processor CLI")
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
