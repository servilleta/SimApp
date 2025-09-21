"""
JSON Export Service - Exact same logic as frontend
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import tempfile

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

logger = logging.getLogger(__name__)

class JSONExportService:
    """Service for generating JSON exports using exact same logic as frontend"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "monte_carlo_exports"
        self.temp_dir.mkdir(exist_ok=True)
    
    def generate_json_export(
        self, 
        simulation_id: str, 
        results_data: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Generate JSON export using exact same logic as frontend.
        Returns path to the generated JSON file.
        """
        try:
            logger.info(f"Starting JSON export for simulation {simulation_id}")
            
            # Use the exact same structure as frontend
            export_data = {
                "simulation_metadata": {
                    "simulation_id": simulation_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "version": "2.0",
                    "engine_type": metadata.get("engine_type", "Ultra") if metadata else "Ultra",
                    "total_iterations": metadata.get("iterations", 10000) if metadata else 10000,
                    "export_source": "B2B_API"
                },
                "results": {}
            }
            
            # Add summary statistics if available
            if metadata:
                # Convert datetime objects to ISO strings
                created_at = metadata.get("created_at", datetime.now())
                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                
                export_data["simulation_metadata"].update({
                    "execution_time": metadata.get("execution_time", "N/A"),
                    "created_at": created_at,
                    "status": metadata.get("status", "completed")
                })
            
            # Process each result (same logic as frontend)
            if "results" in results_data:
                for cell_name, cell_result in results_data["results"].items():
                    target_name = cell_result.get("cell_name", cell_name)
                    stats = cell_result.get("statistics", {})
                    
                    export_data["results"][target_name] = {
                        "cell_coordinate": cell_name,
                        "target_name": target_name,
                        "decimal_precision": 4,  # Default precision
                        "statistics": {
                            "mean": stats.get("mean"),
                            "median": stats.get("median"),
                            "standard_deviation": stats.get("std"),
                            "minimum": stats.get("min"),
                            "maximum": stats.get("max"),
                            "percentiles": {
                                "5th": stats.get("percentiles", {}).get("5"),
                                "25th": stats.get("percentiles", {}).get("25"),
                                "50th": stats.get("percentiles", {}).get("50"),
                                "75th": stats.get("percentiles", {}).get("75"),
                                "95th": stats.get("percentiles", {}).get("95")
                            },
                            "value_at_risk": {
                                "var_95": stats.get("var_95"),
                                "var_99": stats.get("var_99")
                            },
                            "skewness": stats.get("skewness"),
                            "kurtosis": stats.get("kurtosis")
                        },
                        "histogram_data": cell_result.get("distribution_data", {}),
                        "raw_values": cell_result.get("values", []),
                        "sensitivity_analysis": {},
                        "simulation_metadata": {
                            "iterations_run": metadata.get("iterations", 10000) if metadata else 10000,
                            "engine_type": metadata.get("engine_type", "Ultra") if metadata else "Ultra",
                            "created_at": created_at,  # Already converted above
                            "updated_at": datetime.now().isoformat(),
                            "status": metadata.get("status", "completed") if metadata else "completed"
                        }
                    }
            
            # Generate file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = self.temp_dir / f"simulation_results_{simulation_id}_{timestamp}.json"
            
            # Write JSON file (exact same format as frontend)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
            
            logger.info(f"JSON export completed: {json_path}")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"JSON export failed for simulation {simulation_id}: {e}")
            raise

# Global instance
json_export_service = JSONExportService()
