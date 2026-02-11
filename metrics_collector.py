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
        self.overall_phase2_wall_clock: float = 0.0
    
    def start_pdf(self, pdf_name: str, wall_clock_time: float = 0.0, total_api_time: float = 0.0):
        """Start tracking metrics for a new PDF.
        
        Args:
            pdf_name: Name of the PDF
            wall_clock_time: Actual elapsed time from first page start to last page end
            total_api_time: Sum of all individual API call latencies
        """
        self.current_pdf = {
            "pdf_name": pdf_name,
            "total_pages": 0,
            "wall_clock_time_seconds": wall_clock_time,
            "total_api_time_seconds": total_api_time,
            "per_page_metrics": [],
        }
    
    def add_page_metric(
        self,
        page_number: int,
        latency_seconds: float,
    ):
        """Add a single page's API call metric.

        Args:
            page_number: 1-based page index in the PDF
            latency_seconds: Nemotron API latency
        """
        page_entry: Dict[str, Any] = {
            "page": page_number,
            "latency_seconds": round(latency_seconds, 3),
        }

        self.current_pdf["per_page_metrics"].append(page_entry)
        self.current_pdf["total_pages"] += 1
    
    def add_pdftocairo_metric(self, pdf_name: str, total_pages: int, conversion_time: float, total_size_bytes: int = 0, pdf_size_bytes: int = 0):
        """Add pdftocairo conversion metrics for a PDF."""
        avg_latency = conversion_time / total_pages if total_pages > 0 else 0.0
        throughput = total_pages / conversion_time if conversion_time > 0 else 0.0
        avg_size_mb = (total_size_bytes / (1024 * 1024) / total_pages) if total_pages > 0 else 0.0
        total_size_mb = total_size_bytes / (1024 * 1024)
        pdf_size_mb = pdf_size_bytes / (1024 * 1024)
        
        metric = {
            "pdf_name": pdf_name,
            "pdf_size_mb": round(pdf_size_mb, 3),
            "total_pages": total_pages,
            "total_time_seconds": round(conversion_time, 3),
            "average_latency_per_page": round(avg_latency, 3),
            "throughput_pages_per_second": round(throughput, 3),
            "total_size_mb": round(total_size_mb, 3),
            "average_size_per_page_mb": round(avg_size_mb, 3)
        }
        self.pdftocairo_metrics.append(metric)
        
        logger.info(
            f"pdftocairo metrics | pdf={pdf_name} | "
            f"pdf_size={metric['pdf_size_mb']:.2f}MB | "
            f"pages={total_pages} | "
            f"total_time={metric['total_time_seconds']}s | "
            f"avg_latency={metric['average_latency_per_page']}s | "
            f"throughput={metric['throughput_pages_per_second']} pages/s | "
            f"total_size={metric['total_size_mb']:.2f}MB | "
            f"avg_size={metric['average_size_per_page_mb']:.2f}MB"
        )
    
    def finish_pdf(self):
        """Complete the current PDF and calculate aggregates."""
        if self.current_pdf["total_pages"] > 0:
            self.current_pdf["average_latency_per_page"] = round(
                self.current_pdf["total_api_time_seconds"] / self.current_pdf["total_pages"], 3
            )
            self.current_pdf["throughput_pages_per_second"] = round(
                self.current_pdf["total_pages"] / self.current_pdf["wall_clock_time_seconds"], 3
            ) if self.current_pdf["wall_clock_time_seconds"] > 0 else 0.0
        else:
            self.current_pdf["average_latency_per_page"] = 0.0
            self.current_pdf["throughput_pages_per_second"] = 0.0
        
        self.current_pdf["wall_clock_time_seconds"] = round(self.current_pdf["wall_clock_time_seconds"], 3)
        self.current_pdf["total_api_time_seconds"] = round(self.current_pdf["total_api_time_seconds"], 3)
        
        self.pdf_metrics.append(self.current_pdf)
        
        logger.info(
            f"PDF metrics complete | pdf={self.current_pdf['pdf_name']} | "
            f"pages={self.current_pdf['total_pages']} | "
            f"wall_clock_time={self.current_pdf['wall_clock_time_seconds']}s | "
            f"total_api_time={self.current_pdf['total_api_time_seconds']}s | "
            f"avg_latency={self.current_pdf['average_latency_per_page']}s | "
            f"throughput={self.current_pdf['throughput_pages_per_second']} pages/s"
        )
    
    def set_overall_phase2_time(self, wall_clock_seconds: float):
        """Set the overall Phase 2 wall-clock time across all PDFs.
        
        Args:
            wall_clock_seconds: Total elapsed time from first page start to last page end across all PDFs
        """
        self.overall_phase2_wall_clock = wall_clock_seconds
        logger.info(f"Overall Phase 2 wall-clock time: {wall_clock_seconds:.3f}s")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics in JSON-serializable format."""
        # Calculate overall averages across all PDFs for pdftocairo
        total_pdfs = len(self.pdftocairo_metrics)
        total_pdf_size_mb = sum(m["pdf_size_mb"] for m in self.pdftocairo_metrics)
        avg_pdf_size_mb = (total_pdf_size_mb / total_pdfs) if total_pdfs > 0 else 0.0
        
        total_pages_all = sum(m["total_pages"] for m in self.pdftocairo_metrics)
        total_size_all_mb = sum(m["total_size_mb"] for m in self.pdftocairo_metrics)
        overall_avg_page_size_mb = (total_size_all_mb / total_pages_all) if total_pages_all > 0 else 0.0
        
        # Calculate overall statistics for Nemotron API metrics
        total_api_pdfs = len(self.pdf_metrics)
        if total_api_pdfs > 0:
            # Use the actual overall Phase 2 wall-clock time (not sum of individual PDFs)
            total_wall_clock_time = self.overall_phase2_wall_clock
            avg_wall_clock_time = sum(m["wall_clock_time_seconds"] for m in self.pdf_metrics) / total_api_pdfs
            
            total_api_time = sum(m["total_api_time_seconds"] for m in self.pdf_metrics)
            avg_total_api_time = total_api_time / total_api_pdfs
            
            # Average latency per page across all PDFs
            total_pages_api = sum(m["total_pages"] for m in self.pdf_metrics)
            avg_latency_per_page = total_api_time / total_pages_api if total_pages_api > 0 else 0.0
            
            # Overall throughput: total pages / overall wall-clock time
            overall_throughput = total_pages_api / total_wall_clock_time if total_wall_clock_time > 0 else 0.0
            
            # Average per-PDF throughput
            avg_throughput = sum(m["throughput_pages_per_second"] for m in self.pdf_metrics) / total_api_pdfs
            
        else:
            total_wall_clock_time = 0.0
            avg_wall_clock_time = 0.0
            total_api_time = 0.0
            avg_total_api_time = 0.0
            avg_latency_per_page = 0.0
            overall_throughput = 0.0
            avg_throughput = 0.0
            
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_pdfs_processed": len(self.pdf_metrics),
            "overall_statistics": {
                "pdftocairo": {
                    "total_pdf_size_mb": round(total_pdf_size_mb, 3),
                    "average_pdf_size_mb": round(avg_pdf_size_mb, 3),
                    "total_pages": total_pages_all,
                    "total_page_images_size_mb": round(total_size_all_mb, 3),
                    "average_page_image_size_mb": round(overall_avg_page_size_mb, 3)
                },
                "nemotron_parse": {
                    "total_wall_clock_time_seconds": round(total_wall_clock_time, 3),
                    "average_wall_clock_time_per_pdf_seconds": round(avg_wall_clock_time, 3),
                    "total_api_time_seconds": round(total_api_time, 3),
                    "average_total_api_time_per_pdf_seconds": round(avg_total_api_time, 3),
                    "average_latency_per_page_seconds": round(avg_latency_per_page, 3),
                    "overall_throughput_pages_per_second": round(overall_throughput, 3),
                    "average_per_pdf_throughput_pages_per_second": round(avg_throughput, 3)
                }
            },
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
