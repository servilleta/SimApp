#!/usr/bin/env python3
"""
Simple Big File Limits Test

Quick test of our file size limits and system behavior without complex dependencies.
"""

import os
import time
import tempfile
import psutil
from pathlib import Path
from openpyxl import Workbook

def get_memory_mb():
    """Get current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024

def create_test_file(size_mb_target: int) -> str:
    """Create a test Excel file of approximately the target size."""
    print(f"🏗️ Creating test file targeting {size_mb_target} MB...")
    
    wb = Workbook()
    ws = wb.active
    ws.title = "BigFileTest"
    
    # Estimate cells needed (rough calculation)
    # Each cell with data is ~50-100 bytes in Excel format
    bytes_per_cell = 80
    target_bytes = size_mb_target * 1024 * 1024
    estimated_cells = target_bytes // bytes_per_cell
    
    # Calculate grid dimensions
    grid_size = int(estimated_cells ** 0.5)
    rows = min(grid_size, 1048576)  # Excel max rows
    cols = min(estimated_cells // rows, 16384)  # Excel max cols
    
    print(f"   Creating {rows} x {cols} = {rows * cols} cells")
    
    # Fill the sheet
    formula_count = 0
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cell = ws.cell(row=row, column=col)
            
            # Every 10th cell gets a formula
            if (row * col) % 10 == 0:
                if col > 1:
                    cell.value = f"=A{row}*RAND()+{col}"
                else:
                    cell.value = "=RAND()*100"
                formula_count += 1
            else:
                # Data cells
                cell.value = f"Data_{row}_{col}_{row*col}"
    
    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, f"test_{size_mb_target}mb.xlsx")
    
    start_time = time.time()
    wb.save(file_path)
    save_time = time.time() - start_time
    
    # Get actual file size
    actual_size = os.path.getsize(file_path)
    actual_mb = actual_size / (1024 * 1024)
    
    print(f"✅ Created file: {actual_mb:.1f} MB ({formula_count} formulas) in {save_time:.1f}s")
    
    return file_path

def test_file_creation_limits():
    """Test our ability to create files of various sizes."""
    print("🚀 TESTING FILE CREATION LIMITS")
    print("=" * 50)
    
    initial_memory = get_memory_mb()
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    results = {}
    test_sizes = [5, 25, 50, 100]  # MB targets
    
    for size_mb in test_sizes:
        print(f"\n📁 Testing {size_mb} MB file creation...")
        
        try:
            start_memory = get_memory_mb()
            start_time = time.time()
            
            file_path = create_test_file(size_mb)
            
            end_time = time.time()
            end_memory = get_memory_mb()
            
            # Get file stats
            file_size = os.path.getsize(file_path)
            file_mb = file_size / (1024 * 1024)
            
            results[f"{size_mb}MB"] = {
                "success": True,
                "actual_size_mb": file_mb,
                "creation_time": end_time - start_time,
                "memory_used_mb": end_memory - start_memory,
                "peak_memory_mb": end_memory
            }
            
            print(f"   ✅ Success: {file_mb:.1f} MB in {end_time - start_time:.1f}s")
            print(f"   💾 Memory used: {end_memory - start_memory:.1f} MB")
            
            # Cleanup
            os.remove(file_path)
            os.rmdir(os.path.dirname(file_path))
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[f"{size_mb}MB"] = {
                "success": False,
                "error": str(e),
                "memory_used_mb": get_memory_mb() - start_memory
            }
    
    return results

def test_system_limits():
    """Test system configuration and limits."""
    print("\n🔧 TESTING SYSTEM CONFIGURATION")
    print("=" * 50)
    
    # Import our settings
    try:
        from config import settings
        print(f"✅ Config loaded successfully")
        print(f"   MAX_UPLOAD_SIZE: {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f} MB")
        print(f"   MAX_EXCEL_CELLS: {settings.MAX_EXCEL_CELLS:,}")
        print(f"   STREAMING_THRESHOLD: {settings.STREAMING_THRESHOLD:,}")
        print(f"   USE_GPU: {settings.USE_GPU}")
        
        config_ok = True
        
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        config_ok = False
    
    # Test upload validator import
    try:
        from shared.upload_middleware import upload_validator
        print(f"✅ Upload validator imported")
        
        # Test disk space check
        disk_stats = upload_validator.check_disk_space()
        print(f"   Disk space: {disk_stats['free_mb']:.0f} MB free ({disk_stats['used_percent']:.1f}% used)")
        
        validator_ok = True
        
    except Exception as e:
        print(f"❌ Upload validator import failed: {e}")
        validator_ok = False
    
    # Test file cleanup service
    try:
        from shared.file_cleanup import file_cleanup_service
        usage_stats = file_cleanup_service.get_disk_usage()
        print(f"✅ File cleanup service available")
        
        cleanup_ok = True
        
    except Exception as e:
        print(f"❌ File cleanup service failed: {e}")
        cleanup_ok = False
    
    # System memory info
    memory = psutil.virtual_memory()
    print(f"\n💾 SYSTEM MEMORY:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Available: {memory.available / (1024**3):.1f} GB")
    print(f"   Used: {memory.percent:.1f}%")
    
    # CPU info
    print(f"\n🖥️ SYSTEM CPU:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
    
    return {
        "config_ok": config_ok,
        "validator_ok": validator_ok,
        "cleanup_ok": cleanup_ok,
        "memory_gb": memory.total / (1024**3),
        "available_memory_gb": memory.available / (1024**3),
        "cpu_cores": psutil.cpu_count()
    }

def test_bigfiles_config():
    """Test bigfiles configuration endpoints."""
    print("\n📊 TESTING BIGFILES CONFIGURATION")
    print("=" * 50)
    
    try:
        # Test local import
        from main import BIGFILES_CONFIG
        print("✅ BIGFILES_CONFIG imported")
        
        print(f"   Version: {BIGFILES_CONFIG['version']}")
        print(f"   Features enabled: {sum(BIGFILES_CONFIG['features_enabled'].values())} of {len(BIGFILES_CONFIG['features_enabled'])}")
        
        thresholds = BIGFILES_CONFIG['file_size_thresholds']
        print(f"   Size thresholds:")
        for size, bytes_limit in thresholds.items():
            mb_limit = bytes_limit / (1024 * 1024)
            print(f"     {size}: {mb_limit:.0f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ BIGFILES_CONFIG test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 BIG FILE SMOKE TEST - STEP 4")
    print("=" * 60)
    
    # Test 1: System configuration
    system_results = test_system_limits()
    
    # Test 2: BigFiles configuration
    bigfiles_ok = test_bigfiles_config()
    
    # Test 3: File creation (if system looks good)
    if system_results["config_ok"] and system_results["validator_ok"]:
        creation_results = test_file_creation_limits()
    else:
        print("\n⚠️ Skipping file creation tests due to configuration issues")
        creation_results = {}
    
    # Summary
    print("\n🎯 TEST SUMMARY")
    print("=" * 60)
    
    print(f"System configuration: {'✅' if system_results['config_ok'] else '❌'}")
    print(f"Upload validator: {'✅' if system_results['validator_ok'] else '❌'}")
    print(f"File cleanup: {'✅' if system_results['cleanup_ok'] else '❌'}")
    print(f"BigFiles config: {'✅' if bigfiles_ok else '❌'}")
    
    if creation_results:
        successful_creates = sum(1 for r in creation_results.values() if r.get("success", False))
        total_creates = len(creation_results)
        print(f"File creation tests: {successful_creates}/{total_creates} passed")
        
        if successful_creates > 0:
            max_size = max(r.get("actual_size_mb", 0) for r in creation_results.values() if r.get("success"))
            print(f"Largest file created: {max_size:.1f} MB")
    
    print(f"\nSystem resources:")
    print(f"  Memory: {system_results['available_memory_gb']:.1f} GB available")
    print(f"  CPU cores: {system_results['cpu_cores']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if system_results['available_memory_gb'] < 4:
        print("  ⚠️ Consider more RAM for large file processing")
    
    if not system_results['config_ok']:
        print("  ❌ Fix configuration issues before proceeding")
    
    if creation_results and all(r.get("success", False) for r in creation_results.values()):
        print("  ✅ System ready for big file processing")
        print("  🚀 Proceed with Step 5 (Docker deployment) or Step 2 (Stripe integration)")
    else:
        print("  🔧 Address file processing issues before deployment")

if __name__ == "__main__":
    main() 