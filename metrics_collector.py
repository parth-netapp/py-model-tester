#!/usr/bin/env python3
"""
Performance Metrics Collection Module

Tracks and aggregates performance metrics for the PDF processing pipeline.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from logging_config import get_logger


logger = get_logger(__name__)


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self):
        self.pdf_metrics: List[Dict[str, Any]] = []
        self.pdftocairo_metrics: List[Dict[str, Any]] = []
    
    def start_pdf(self, pdf_name: str):
        """Start tracking metrics for a new PDF."""
        self.current_pdf = {
            "pdf_name": pdf_name,
            "total_pages": 0,
            "total_time_seconds": 0.0,
            "per_page_metrics": []
        }
    
    def add_page_metric(self, page_number: int, image_path: str, latency_seconds: float):
        """Add a single page's API call metric."""
        self.current_pdf["per_page_metrics"].append({
            "page": page_number,
            "image_path": image_path,
            "latency_seconds": round(latency_seconds, 3)
        })
        self.current_pdf["total_pages"] += 1
        self.current_pdf["total_time_seconds"] += latency_seconds
    
    def add_pdftocairo_metric(self, pdf_name: str, total_pages: int, conversion_time: float):
        """Add pdftocairo conversion metrics for a PDF."""
        avg_latency = conversion_time / total_pages if total_pages > 0 else 0.0
        throughput = total_pages / conversion_time if conversion_time > 0 else 0.0
        
        metric = {
            "pdf_name": pdf_name,
            "total_pages": total_pages,
            "total_time_seconds": round(conversion_time, 3),
            "average_latency_per_page": round(avg_latency, 3),
            "throughput_pages_per_second": round(throughput, 3)
        }
        self.pdftocairo_metrics.append(metric)
        
        logger.info(
            f"pdftocairo metrics | pdf={pdf_name} | "
            f"pages={total_pages} | "
            f"total_time={metric['total_time_seconds']}s | "
            f"avg_latency={metric['average_latency_per_page']}s | "
            f"throughput={metric['throughput_pages_per_second']} pages/s"
        )
    
    def finish_pdf(self):
        """Complete the current PDF and calculate aggregates."""
        if self.current_pdf["total_pages"] > 0:
            self.current_pdf["average_latency_per_page"] = round(
                self.current_pdf["total_time_seconds"] / self.current_pdf["total_pages"], 3
            )
            self.current_pdf["throughput_pages_per_second"] = round(
                self.current_pdf["total_pages"] / self.current_pdf["total_time_seconds"], 3
            ) if self.current_pdf["total_time_seconds"] > 0 else 0.0
        else:
            self.current_pdf["average_latency_per_page"] = 0.0
            self.current_pdf["throughput_pages_per_second"] = 0.0
        
        self.current_pdf["total_time_seconds"] = round(self.current_pdf["total_time_seconds"], 3)
        self.pdf_metrics.append(self.current_pdf)
        
        logger.info(
            f"PDF metrics complete | pdf={self.current_pdf['pdf_name']} | "
            f"pages={self.current_pdf['total_pages']} | "
            f"total_time={self.current_pdf['total_time_seconds']}s | "
            f"avg_latency={self.current_pdf['average_latency_per_page']}s | "
            f"throughput={self.current_pdf['throughput_pages_per_second']} pages/s"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics in JSON-serializable format."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_pdfs_processed": len(self.pdf_metrics),
            "pdftocairo_metrics": self.pdftocairo_metrics,
            "nemotron_parse_metrics": self.pdf_metrics
        }
    
    def save_to_file(self, output_path: str = "performance_metrics.json"):
        """Save metrics to a JSON file with timestamp suffix."""
        metrics = self.get_metrics()
        output_file = Path(output_path)
        
        # Create performance_metrics directory
        metrics_dir = Path("performance_metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp suffix to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = metrics_dir / f"{output_file.stem}_{timestamp}{output_file.suffix}"
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Performance metrics saved to {output_file}")
        return str(output_file)
