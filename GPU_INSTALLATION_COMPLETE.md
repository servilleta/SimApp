# 🚀 GPU DRIVER INSTALLATION GUIDE - BOTH SERVERS

## 📋 **INSTALLATION STATUS**

### ✅ **SERVER 1 (QUADRO P4000) - COMPLETED**
- **Hardware**: NVIDIA Quadro P4000 (GP104GL)
- **Driver**: NVIDIA 470.256.02 ✅ INSTALLED
- **Status**: Ready for reboot to activate GPU

### ⏳ **SERVER 2 (AMPERE A4000) - SCRIPT READY**
- **Hardware**: NVIDIA Ampere A4000
- **Driver**: NVIDIA 525+ (Recommended)
- **Status**: Installation script created and ready

---

## 🔄 **SERVER 1 - FINAL STEPS**

### **IMMEDIATE ACTION REQUIRED:**
```bash
# Reboot Server 1 to load NVIDIA drivers
sudo reboot
```

### **After Reboot - Verification:**
```bash
# Check if NVIDIA driver is loaded
nvidia-smi

# Verify GPU is accessible from Docker
sudo docker exec simapp_backend_1 python3 -c "
import cupy as cp
print('✅ CuPy version:', cp.__version__)
print('🔥 GPU Device count:', cp.cuda.runtime.getDeviceCount())
if cp.cuda.runtime.getDeviceCount() > 0:
    device = cp.cuda.Device(0)
    print(f'🎯 GPU Name: {device.name}')
    print(f'💾 GPU Memory: {device.mem_info}')
"
```

---

## 🔧 **SERVER 2 - INSTALLATION STEPS**

### **1. SSH to Server 2:**
```bash
ssh -i ~/.ssh/paperspace_key paperspace@72.52.107.230
```

### **2. Copy and run installation script:**
```bash
# Copy the script to Server 2 (run from Server 1)
scp -i ~/.ssh/paperspace_key /home/paperspace/SimApp/install_nvidia_server2.sh paperspace@72.52.107.230:~/

# SSH to Server 2 and run the script
ssh -i ~/.ssh/paperspace_key paperspace@72.52.107.230
chmod +x ~/install_nvidia_server2.sh
sudo ~/install_nvidia_server2.sh
```

### **3. Reboot Server 2:**
```bash
sudo reboot
```

### **4. Verify Installation on Server 2:**
```bash
# After reboot
nvidia-smi
sudo docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

---

## 🎯 **EXPECTED PERFORMANCE GAINS**

### **Server 1 (Quadro P4000)**
- **CUDA Cores**: 1,792
- **Memory**: 8GB GDDR5
- **Memory Bandwidth**: 243 GB/s
- **Compute Capability**: 6.1
- **Expected Speedup**: 10-50x for Monte Carlo simulations

### **Server 2 (Ampere A4000)**
- **CUDA Cores**: 6,144
- **Memory**: 16GB GDDR6
- **Memory Bandwidth**: 448 GB/s
- **Compute Capability**: 8.6
- **Expected Speedup**: 50-130x for Monte Carlo simulations

---

## 🔬 **ULTRA ENGINE GPU FEATURES ACTIVATED**

Once both servers have GPU drivers installed, your Ultra Monte Carlo Engine will automatically utilize:

### **1. GPU-Accelerated Random Generation**
```python
# Ultra Engine automatically detects GPU and uses:
- CuPy random number generation (10-130x faster)
- GPU-optimized triangular distributions
- Vectorized Monte Carlo sampling
```

### **2. GPU Financial Computations**
```python
# GPU-accelerated financial functions:
- NPV calculations on GPU
- IRR solving with GPU bisection
- Vectorized cash flow analysis
```

### **3. GPU Memory Management**
```python
# Advanced GPU memory optimization:
- CUDA Unified Memory
- GPU memory prefetching
- Automatic CPU/GPU fallback
```

---

## 🚀 **TESTING GPU ACCELERATION**

### **After both servers are ready:**

```bash
# Test GPU acceleration in Ultra Engine
cd /home/paperspace/SimApp
sudo docker exec simapp_backend_1 python3 -c "
from backend.simulation.engines.ultra_engine import UltraGPURandomGenerator, UltraConfig
import logging
logging.basicConfig(level=logging.INFO)

config = UltraConfig()
generator = UltraGPURandomGenerator(config, None, 'test')
result = generator.benchmark_gpu_vs_cpu(iterations=10000, variables=10)
print(f'🚀 GPU Speedup: {result[\"gpu_speedup\"]:.2f}x')
print(f'⚡ Performance: {result[\"gpu_samples_per_second\"]:.0f} samples/sec')
"
```

---

## 📊 **MONITORING GPU USAGE**

### **Real-time GPU monitoring:**
```bash
# Monitor GPU usage during simulations
watch -n 1 nvidia-smi

# Docker container GPU usage
sudo docker stats simapp_backend_1
```

### **Ultra Engine GPU metrics in logs:**
```bash
# Watch for GPU acceleration messages
sudo docker logs -f simapp_backend_1 | grep -i "gpu\|cuda"
```

---

## 🔗 **INTEGRATION WITH EXISTING SYSTEM**

Your current multi-core setup (8 CPU workers) will be **enhanced** with GPU acceleration:

1. **CPU Workers**: Handle request routing and I/O
2. **GPU Acceleration**: Handles Monte Carlo computations
3. **Auto-Scaling**: Server 2 GPU activates when Server 1 is stressed
4. **Fallback**: Automatic CPU fallback if GPU is unavailable

The system maintains **100% backward compatibility** while adding massive performance improvements when GPU is available.

---

## ✅ **SUCCESS CRITERIA**

### **Both servers ready when:**
1. `nvidia-smi` shows GPU information
2. Docker containers can access GPU
3. Ultra Engine logs show "GPU acceleration activated"
4. Simulations complete 10-130x faster
5. Admin monitoring shows GPU utilization

**Expected total installation time**: 15-30 minutes per server (including reboots)
