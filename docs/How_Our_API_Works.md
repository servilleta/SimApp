# How Our Monte Carlo API Works
**Complete User Guide: What to Share & How It Works**

---

## 🔄 **The Complete Process**

### **User's Journey: From Excel to Risk Analysis**

```
Excel Model → Upload → Configure → Simulate → Get Results
     ↓           ↓         ↓          ↓         ↓
  [User File] [API Call] [Variables] [GPU Run] [JSON Data]
```

---

## 📁 **What Users Need to Share**

### **1. Excel File (Required)**
**What:** Their existing Excel financial model
**Format:** `.xlsx`, `.xlsm`, or `.xls`
**Size Limit:** 10MB (Starter) to 500MB (Enterprise)

**Examples:**
- Portfolio valuation models
- Risk assessment spreadsheets  
- Financial projections
- Trading models
- Investment analysis

**User Doesn't Need to:**
- ❌ Modify their Excel formulas
- ❌ Convert to other formats
- ❌ Remove complex functions
- ❌ Simplify their model

### **2. Variable Definitions (Required)**
**What:** Which cells contain uncertain values
**Why:** These become the Monte Carlo variables

**Example:**
```json
{
  "variables": [
    {
      "cell": "B5",                    // Excel cell reference
      "name": "Market_Volatility",     // Human-readable name
      "distribution": {
        "type": "triangular",          // Distribution type
        "min": 0.05,                   // 5% minimum
        "mode": 0.15,                  // 15% expected  
        "max": 0.35                    // 35% maximum
      }
    },
    {
      "cell": "C7",
      "name": "Interest_Rate", 
      "distribution": {
        "type": "normal",
        "mean": 0.03,                  // 3% average
        "std": 0.01                    // 1% std deviation
      }
    }
  ]
}
```

### **3. Output Cells (Required)**
**What:** Which cells they want analyzed
**Why:** These are the results they care about

**Example:**
```json
{
  "output_cells": ["J25", "K25", "L25"]
  // J25 might be "Portfolio NPV"
  // K25 might be "Maximum Drawdown"  
  // L25 might be "Sharpe Ratio"
}
```

### **4. Simulation Parameters (Optional)**
**What:** How many iterations, confidence levels
**Defaults:** We provide sensible defaults

**Example:**
```json
{
  "iterations": 100000,              // Default: 100K
  "confidence_levels": [0.95, 0.99], // Default: 95%, 99%
  "webhook_url": "https://their-app.com/notify"  // Optional
}
```

---

## 🔧 **How Our API Works Behind the Scenes**

### **Step 1: File Upload & Analysis**
```python
# User uploads Excel file
POST /api/v1/models
Content-Type: multipart/form-data
File: portfolio_model.xlsx
```

**What We Do:**
1. ✅ **Receive Excel file** via secure upload
2. ✅ **Parse all worksheets** using openpyxl
3. ✅ **Extract formulas** and dependencies
4. ✅ **Build calculation graph** (which cells depend on what)
5. ✅ **Detect potential variables** (cells with hardcoded numbers)
6. ✅ **Return suggestions** to user

**What User Gets Back:**
```json
{
  "model_id": "mdl_abc123",
  "formulas_count": 1547,
  "variables_detected": [
    {
      "cell": "B5",
      "current_value": 0.15,
      "suggested_distribution": "normal"
    }
  ]
}
```

### **Step 2: Simulation Configuration**
```python
# User defines Monte Carlo variables
POST /api/v1/simulations
{
  "model_id": "mdl_abc123",
  "simulation_config": {
    "iterations": 100000,
    "variables": [...],
    "output_cells": ["J25", "K25"]
  }
}
```

**What We Do:**
1. ✅ **Validate configuration** (cells exist, distributions valid)
2. ✅ **Estimate runtime** and credit cost
3. ✅ **Queue simulation** for GPU processing
4. ✅ **Return simulation ID** for tracking

**What User Gets Back:**
```json
{
  "simulation_id": "sim_xyz789",
  "status": "queued",
  "estimated_completion": "2024-01-15T14:23:30Z",
  "credits_consumed": 5.2
}
```

### **Step 3: GPU-Accelerated Processing**
**User doesn't see this - happens automatically**

**Our Internal Process:**
1. **🎮 GPU Initialization**
   - Allocate 8GB GPU memory pools
   - Set up CUDA context for parallel processing

2. **📊 Data Preparation**
   - Generate 100K random samples for each variable
   - Transfer data to GPU memory
   - Prepare formula execution order

3. **⚡ Ultra-Fast Simulation**
   - Execute 100K iterations in parallel on GPU
   - Each iteration recalculates entire Excel model
   - Process in optimized batches for memory efficiency

4. **📈 Statistical Analysis**
   - Calculate mean, std deviation, percentiles
   - Generate Value at Risk (VaR) metrics
   - Create distribution histograms
   - Compute confidence intervals

### **Step 4: Real-Time Progress**
```python
# User checks progress (optional)
GET /api/v1/simulations/sim_xyz789/progress
```

**What User Gets:**
```json
{
  "status": "running",
  "progress": {
    "percentage": 67.3,
    "iterations_completed": 67300,
    "phase": "monte_carlo_execution",
    "estimated_remaining": "18 seconds"
  },
  "performance_metrics": {
    "iterations_per_second": 2847,
    "gpu_utilization": "89%"
  }
}
```

### **Step 5: Results Delivery**
```python
# User gets final results
GET /api/v1/simulations/sim_xyz789/results
```

**What User Gets Back:**
```json
{
  "simulation_id": "sim_xyz789",
  "status": "completed",
  "execution_time": "42.7 seconds",
  "results": {
    "J25": {
      "cell_name": "Portfolio_NPV",
      "statistics": {
        "mean": 1250000,
        "std": 340000,
        "percentiles": {
          "5": 680000,    // 5% chance below this
          "50": 1240000,  // Median value
          "95": 1820000   // 95% confidence upper bound
        },
        "var_95": 680000, // Value at Risk (95%)
        "var_99": 540000  // Value at Risk (99%)
      },
      "distribution_data": {
        "histogram": {
          "bins": [500000, 600000, 700000, ...],
          "frequencies": [245, 892, 1547, ...]
        }
      }
    }
  }
}
```

---

## 🔒 **Data Security & Privacy**

### **What Happens to User Data:**

#### **✅ Secure Processing:**
- Excel files processed in **isolated containers**
- Each simulation runs in **separate environment**
- **No data mixing** between customers

#### **✅ Automatic Cleanup:**
- Excel files **deleted immediately** after processing
- Simulation data **deleted after 24 hours**
- **No permanent storage** of customer data

#### **✅ API Key Security:**
- Each customer gets **unique API keys**
- **Rate limiting** prevents abuse
- **Request logging** for debugging (no file content)

#### **What We DON'T Do:**
- ❌ Store customer Excel files
- ❌ Share data between customers  
- ❌ Use data for training/analytics
- ❌ Retain sensitive information

---

## 🎯 **What Users Need to Know**

### **Technical Requirements:**
- **Any programming language** that can make HTTP requests
- **HTTP client** (like requests in Python, axios in JavaScript)
- **JSON handling** (standard in all languages)
- **File upload capability** (multipart/form-data)

### **No Special Requirements:**
- ❌ No special software installation
- ❌ No Excel plugins or add-ins
- ❌ No model modifications
- ❌ No specific Excel version

### **User Workflow:**
1. **Get API key** from us (instant)
2. **Upload Excel file** via API
3. **Define uncertain variables** (which cells vary)
4. **Specify output cells** (what to analyze)
5. **Run simulation** (wait 1-60 seconds)
6. **Get comprehensive results** (statistics, charts, data)

---

## 📊 **Example: Portfolio Risk Analysis**

### **User Has:**
- Excel portfolio model with stock weights, returns, correlations
- Cell B5: Expected market return (currently 0.08)
- Cell B6: Market volatility (currently 0.15)
- Cell J25: Portfolio NPV (formula calculating final value)

### **User Provides:**
```json
{
  "variables": [
    {
      "cell": "B5",
      "name": "Market_Return", 
      "distribution": {"type": "normal", "mean": 0.08, "std": 0.02}
    },
    {
      "cell": "B6",
      "name": "Market_Volatility",
      "distribution": {"type": "triangular", "min": 0.10, "mode": 0.15, "max": 0.25}
    }
  ],
  "output_cells": ["J25"],
  "iterations": 100000
}
```

### **User Gets Back:**
- **Portfolio NPV Statistics**: Mean, standard deviation, percentiles
- **Risk Metrics**: Value at Risk (VaR), Expected Shortfall
- **Distribution Data**: Histogram showing probability of different outcomes
- **Confidence Intervals**: 95% and 99% confidence bounds

### **Business Value:**
- **Instead of guessing**: "Our portfolio is worth $1.25M"
- **They can say**: "Our portfolio has a mean value of $1.25M, with 95% confidence it will be worth between $680K and $1.82M"

---

## 🚀 **Why This Is Powerful**

### **For the User:**
- **Keep using Excel** (no learning curve)
- **Get professional results** (enterprise-grade analysis)
- **Save massive time** (seconds vs hours)
- **Make better decisions** (quantified uncertainty)

### **For Their Business:**
- **Regulatory compliance** (proper risk analysis)
- **Better client reporting** (sophisticated analytics)
- **Competitive advantage** (faster, better analysis)
- **Cost savings** (vs hiring quant teams)

---

## 🎯 **Simple Summary for Users:**

*"Send us your Excel file and tell us which cells you're uncertain about. We'll run 100,000 scenarios in parallel on GPUs and send back comprehensive risk analysis - all the statistics, charts, and metrics you need to make data-driven decisions."*

**Input:** Excel file + variable definitions
**Output:** Professional risk analysis
**Time:** Seconds instead of hours
**Skill Required:** None (if you can use Excel, you can use our API)
