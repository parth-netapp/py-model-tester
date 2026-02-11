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
) -> tuple[list[str], int, float]:
	"""Process a single page image: call API and crop components.

	Args:
		image_path: Path to the PNG page image
		semaphore: Semaphore to limit concurrent API calls
		page_number: Page number for metrics tracking

	Returns:
		Tuple of (list of cropped image paths, page number, API latency in seconds)
	"""
	# Use semaphore to limit concurrent API calls
	async with semaphore:
		logger.info(f"Calling Nemotron-parse for image: {image_path}")
		response_json, latency = await call_nemotron_parse(image_path=image_path)
		logger.info(
			f"Received Nemotron-parse response for image {image_path} "
			f"(choices={len(response_json.get('choices', []))}) | latency={latency:.3f}s"
		)
	
	# Crop images based on bounding boxes from the model response.
	# image_cropper will place them under:
	#   pdf_cropped_components/{pdf_name}/{pdf_name}_page{N}/
	cropped_paths = crop_from_model_response(
		payload=response_json,
		image_path=image_path,
		out_dir="pdf_cropped_components",
		prefix="cropped_image",
	)
	logger.info(
		f"Generated {len(cropped_paths)} cropped components for image {image_path}"
	)

	return cropped_paths, page_number, latency


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
	for dir_path in ["pdf_cropped_components", "pdf_images_dir"]:
		if Path(dir_path).exists():
			shutil.rmtree(dir_path)
			logger.info(f"Removed directory: {dir_path}")

	all_crops: list[str] = []
	
	# Create semaphore to limit concurrent API calls
	semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
	logger.info(f"Using max concurrent API calls: {MAX_CONCURRENT_API_CALLS}")
	
	# Initialize metrics collector
	metrics_collector = MetricsCollector()

	for pdf_path in pdf_paths:
		logger.info(f"Processing PDF: {pdf_path}")
		
		# Start tracking metrics for this PDF
		metrics_collector.start_pdf(pdf_path.name)
		
		pdf_images = pdf_to_png(str(pdf_path), "pdf_images_dir/")
		if pdf_images["success"]:
			logger.info(
				f"{pdf_images['message']} | pages={len(pdf_images['output_files'])}"
			)
			# Collect pdftocairo metrics
			metrics_collector.add_pdftocairo_metric(
				pdf_name=pdf_path.name,
				total_pages=pdf_images["total_pages"],
				conversion_time=pdf_images["conversion_time_seconds"]
			)
		else:
			logger.error(f"Error converting PDF to images: {pdf_images['message']}")
			if pdf_images["stderr"]:
				logger.error(f"Conversion stderr: {pdf_images['stderr']}")
			# Skip to next PDF instead of exiting entire workload
			continue

		# Use nemotron-parse model to:
		# 1. Extract text
		# 2. Extract text from images (OCR equivalent)
		# 3. Get images/pictures/graphs bounding boxes
		
		# Process all pages concurrently with semaphore controlling concurrency
		logger.info(f"Processing {len(pdf_images['output_files'])} pages concurrently")
		tasks = [
			process_page_image(image_path, semaphore, idx + 1)
			for idx, image_path in enumerate(pdf_images["output_files"])
		]
		
		# Gather all results
		page_results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Collect successful crops, metrics, and log errors
		for idx, result in enumerate(page_results):
			if isinstance(result, Exception):
				logger.error(
					f"Error processing page {pdf_images['output_files'][idx]}: {result}"
				)
			elif isinstance(result, tuple):
				cropped_paths, page_num, latency = result
				all_crops.extend(cropped_paths)
				metrics_collector.add_page_metric(
					page_number=page_num,
					image_path=pdf_images["output_files"][idx],
					latency_seconds=latency,
				)
		
		# Finish metrics for this PDF
		metrics_collector.finish_pdf()

	logger.info(
		f"Total cropped components across all PDFs and pages: {len(all_crops)}"
	)
	
	# Save performance metrics to JSON file
	metrics_file = metrics_collector.save_to_file("performance_metrics.json")
	logger.info(f"Performance metrics saved to: {metrics_file}")
	print(f"\n✓ Performance metrics saved to: {metrics_file}")

	


if __name__ == "__main__":
	asyncio.run(main())
