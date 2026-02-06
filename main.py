import json

import sys
from pathlib import Path

from pdf_processor import pdf_to_png
from image_cropper import crop_from_model_response
from nemotron_parse_client import call_nemotron_parse
from logging_config import get_logger


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


def main() -> None:
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

	all_crops: list[str] = []

	for pdf_path in pdf_paths:
		logger.info(f"Processing PDF: {pdf_path}")
		pdf_images = pdf_to_png(str(pdf_path), "pdf_images_dir/")
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

		# Use nemotron-parse model to:
		# 1. Extract text
		# 2. Extract text from images (OCR equivalent)
		# 3. Get images/pictures/graphs bounding boxes
		for image_path in pdf_images["output_files"]:
			# Prepare payload with base64 images
			logger.info(f"Calling Nemotron-parse for image: {image_path}")
			response_json = call_nemotron_parse(image_path=image_path)
			logger.info(
				f"Received Nemotron-parse response for image {image_path} "
				f"(choices={len(response_json.get('choices', []))})"
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
			all_crops.extend(cropped_paths)

	logger.info(
		f"Total cropped components across all PDFs and pages: {len(all_crops)}"
	)

	


if __name__ == "__main__":
	main()
