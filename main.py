#!/usr/bin/env python3
"""
PDF Processing Pipeline

This script processes PDF files through a multi-stage pipeline:
1. Convert PDF pages to PNG images using poppler-utils
2. Call Nemotron-parse model to extract text and identify image/table/figure regions
3. Crop identified regions from the page images based on bounding boxes

Usage:
	uv run python3 main.py ./page13.pdf 
	uv run main.py ./inPDFs/
    python main.py <pdf-file-or-directory>
    make run ARGS=<pdf-file-or-directory>

Examples:
    python main.py document.pdf
    python main.py ./pdfs_folder/
    make run ARGS=document.pdf
"""

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from pdf_processor import pdf_to_png
from image_cropper import crop_from_model_response
from nemotron_parse_client import call_nemotron_parse, MAX_CONCURRENT_API_CALLS
from logging_config import get_logger
from metrics_collector import MetricsCollector


logger = get_logger(__name__)


def get_pdf_paths(input_path: str) -> list[Path]:
	"""Return a list of PDF files from a file or directory path."""
	p = Path(input_path)
	if p.is_dir():
		return sorted(
			[f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
		)
	if p.is_file() and p.suffix.lower() == ".pdf":
		return [p]
	raise FileNotFoundError(f"No PDF file(s) found at: {input_path}")



async def process_page_image(
	image_path: str,
	semaphore: asyncio.Semaphore,
	page_number: int,
	pdf_name: str,
) -> tuple[list[str], int, float, str, float, float]:
	"""Process a single page image: call API.

	Args:
		image_path: Path to the PNG page image
		semaphore: Semaphore to limit concurrent API calls
		page_number: Page number for metrics tracking
		pdf_name: Name of the PDF this page belongs to

	Returns:
		Tuple of (empty list, page number, API latency in seconds, pdf_name, start_time, end_time)
	"""
	start_time = time.time()
	# Use semaphore to limit concurrent API calls
	async with semaphore:
		logger.info(f"Calling Nemotron-parse for image: {image_path}")
		response_json, latency = await call_nemotron_parse(image_path=image_path)
		logger.info(
			f"Received Nemotron-parse response for image {image_path} "
			f"(choices={len(response_json.get('choices', []))}) | latency={latency:.3f}s"
		)

		# Save response_json to file for later cropping in Phase 3
		# Create subdirectory structure: model_responses/{pdf_name}/
		response_dir = Path("model_responses") / pdf_name
		response_dir.mkdir(parents=True, exist_ok=True)
		image_path_obj = Path(image_path)
		response_file = response_dir / f"{image_path_obj.stem}.json"
		with open(response_file, "w", encoding="utf-8") as f:
			json.dump(response_json, f, indent=2)
		logger.debug(f"Saved model response to: {response_file}")

	end_time = time.time()
	return [], page_number, latency, pdf_name, start_time, end_time


async def main() -> None:
	if len(sys.argv) < 2:
		logger.error("No input path provided. Usage: python main.py <pdf-file-or-directory>")
		sys.exit(1)

	input_path = sys.argv[1]
	logger.info(f"Starting workload processing for input: {input_path}")
	try:
		pdf_paths = get_pdf_paths(input_path)
	except FileNotFoundError as e:
		logger.error(f"{e}")
		sys.exit(1)

	if not pdf_paths:
		logger.error(f"No PDF files found in: {input_path}")
		sys.exit(1)

	# Cleanup directories from previous runs
	logger.info("Cleaning up previous run directories...")
	for dir_path in ["pdf_cropped_components", "pdf_images_dir", "model_responses"]:
		if Path(dir_path).exists():
			shutil.rmtree(dir_path)
			logger.info(f"Removed directory: {dir_path}")

	# Initialize metrics collector
	metrics_collector = MetricsCollector()
	
	# PHASE 1: Convert all PDFs to images
	logger.info("=" * 80)
	logger.info("PHASE 1: Converting all PDFs to images")
	logger.info("=" * 80)
	
	for pdf_path in pdf_paths:
		logger.info(f"Converting PDF to images: {pdf_path}")
		
		# pdf_processor will log its own metrics
		pdf_images = pdf_to_png(str(pdf_path), "pdf_images_dir/", metrics_collector=metrics_collector)
		if pdf_images["success"]:
			logger.info(
				f"{pdf_images['message']} | pages={len(pdf_images['output_files'])}"
			)
		else:
			logger.error(f"Error converting PDF to images: {pdf_images['message']}")
			if pdf_images["stderr"]:
				logger.error(f"Conversion stderr: {pdf_images['stderr']}")
			# Skip to next PDF instead of exiting entire workload
			continue
	
	logger.info("Phase 1 complete: All PDFs converted to images")

	# PHASE 2: Process all images with model API
	logger.info("=" * 80)
	logger.info("PHASE 2: Processing all images with Nemotron-parse API")
	logger.info("=" * 80)

	# Create semaphore to limit concurrent API calls
	semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
	logger.info(f"Using max concurrent API calls: {MAX_CONCURRENT_API_CALLS}")

	# Discover all PNG images by walking the pdf_images_dir directory
	images_dir = Path("pdf_images_dir")
	if not images_dir.exists():
		logger.error("No pdf_images_dir found - no images to process")
		sys.exit(1)

	# Collect all PNG files with their metadata (pdf_name, page_number, image_path)
	all_page_info = []
	for png_file in sorted(images_dir.rglob("*.png")):
		# Extract pdf_name from parent directory
		pdf_name = png_file.parent.name

		# Extract page number from filename (e.g., "doc_page-05.png" -> 5)
		# Pattern: {pdf_name}_page-{N}.png
		filename = png_file.stem  # e.g., "doc_page-05"
		try:
			page_str = filename.split("_page-")[-1]
			page_num = int(page_str)
			all_page_info.append((pdf_name, page_num, str(png_file)))
		except (IndexError, ValueError) as e:
			logger.warning(f"Could not extract page number from {png_file}: {e}")
			continue

	logger.info(f"Discovered {len(all_page_info)} page images to process")

	# Process all pages concurrently with semaphore controlling concurrency
	tasks = [
		process_page_image(image_path, semaphore, page_num, pdf_name)
		for pdf_name, page_num, image_path in all_page_info
	]

	# Gather all results
	page_results = await asyncio.gather(*tasks, return_exceptions=True)

	# Organize results by PDF
	pdf_to_results = {}
	all_start_times = []
	all_end_times = []

	for idx, result in enumerate(page_results):
		if isinstance(result, tuple):
			_, page_num, latency, pdf_name, start_time, end_time = result
			
			if pdf_name not in pdf_to_results:
				pdf_to_results[pdf_name] = []
			
			pdf_to_results[pdf_name].append((page_num, latency, start_time, end_time))
			all_start_times.append(start_time)
			all_end_times.append(end_time)

	# Calculate overall Phase 2 wall-clock time across ALL PDFs
	# Since all pages run concurrently, this is earliest start to latest end
	overall_phase2_wall_clock = max(all_end_times) - min(all_start_times) if all_start_times else 0.0

	# Now process metrics sequentially by PDF (not concurrently)
	for pdf_name, results in pdf_to_results.items():
		# Calculate actual wall-clock time: earliest start to latest end
		if results:
			start_times = [start_time for _, _, start_time, _ in results]
			end_times = [end_time for _, _, _, end_time in results]
			wall_clock_duration = max(end_times) - min(start_times)
			total_api_time = sum(latency for _, latency, _, _ in results)
		else:
			wall_clock_duration = 0.0
			total_api_time = 0.0

		metrics_collector.start_pdf(pdf_name, wall_clock_time=wall_clock_duration, total_api_time=total_api_time)

		for page_num, latency, _, _ in results:
			metrics_collector.add_page_metric(
				page_number=page_num,
				latency_seconds=latency,
			)

		metrics_collector.finish_pdf()

	# Set overall Phase 2 wall-clock time
	metrics_collector.set_overall_phase2_time(overall_phase2_wall_clock)

	# Save performance metrics to JSON file
	metrics_file = metrics_collector.save_to_file("performance_metrics.json")
	logger.info(f"Performance metrics saved to: {metrics_file}")
	print(f"\n✓ Performance metrics saved to: {metrics_file}")

	# PHASE 3: Crop images based on saved JSON responses
	logger.info("=" * 80)
	logger.info("PHASE 3: Cropping images based on model responses")
	logger.info("=" * 80)

	response_dir = Path("model_responses")
	if not response_dir.exists() or not list(response_dir.rglob("*.json")):
		logger.warning("No model response JSON files found - skipping cropping phase")
	else:
		# Process each JSON file and crop corresponding image
		# JSON files are organized as: model_responses/{pdf_name}/{pdf_name}_page-{N}.json
		for json_file in sorted(response_dir.rglob("*.json")):
			# Match JSON file to corresponding image
			# JSON file pattern: model_responses/{pdf_name}/{pdf_name}_page-{N}.json
			# Image file pattern: pdf_images_dir/{pdf_name}/{pdf_name}_page-{N}.png

			# Find matching image file
			matching_images = list(images_dir.rglob(f"{json_file.stem}.png"))

			if not matching_images:
				logger.warning(f"No matching image found for {json_file.name}")
				continue

			image_path = str(matching_images[0])

			# Load response JSON
			with open(json_file, "r", encoding="utf-8") as f:
				response_json = json.load(f)

			logger.info(f"Cropping image: {image_path} using response: {json_file.name}")

			# Call image_cropper to crop regions
			try:
				cropped_paths, num_crops, crops_size = crop_from_model_response(
					payload=response_json,
					image_path=image_path,
					out_dir="pdf_cropped_components",
					prefix="cropped_image",
				)
				logger.info(
					f"Generated {num_crops} cropped components for {json_file.name} "
					f"(total size: {crops_size / 1024:.2f} KB)"
				)
			except Exception as e:
				logger.error(f"Error cropping {image_path}: {e}", exc_info=True)
				continue
	
	logger.info("Phase 3 complete: All images cropped")
	print("\n✓ Phase 3 complete: Image cropping finished")


if __name__ == "__main__":
	asyncio.run(main())
