# Monte Carlo API - Simple Data Flow

## 📋 **What Users Share (Input)**

### **1. Excel File**
```
📁 portfolio_model.xlsx
   ├── Sheet1: Portfolio weights
   ├── Sheet2: Market data  
   ├── Sheet3: Calculations
   └── Formulas: =SUM(), =VLOOKUP(), etc.
```

### **2. Variable Definitions**
```
Cell B5 (Market Return): Normal distribution, mean=8%, std=2%
Cell B6 (Volatility): Triangular distribution, min=10%, mode=15%, max=25%
Cell C7 (Interest Rate): Normal distribution, mean=3%, std=1%
```

### **3. Output Cells**
```
J25: Portfolio NPV
K25: Maximum Drawdown
L25: Sharpe Ratio
```

---

## ⚙️ **What Happens Inside (Processing)**

```
    📁 Excel File
         ↓
    🔍 Parse Formulas (openpyxl)
         ↓  
    📊 Build Calculation Graph
         ↓
    🎮 Transfer to GPU Memory
         ↓
    ⚡ Run 100,000 Simulations in Parallel
         ↓
    📈 Calculate Statistics
         ↓
    📋 Return JSON Results
```

### **GPU Processing (Ultra-Fast)**
```
Iteration 1: B5=7.8%, B6=14.2%, C7=2.9% → J25=$1,180,000
Iteration 2: B5=8.3%, B6=16.1%, C7=3.2% → J25=$1,240,000
Iteration 3: B5=7.5%, B6=12.8%, C7=2.7% → J25=$1,150,000
...
Iteration 100,000: B5=8.1%, B6=15.3%, C7=3.1% → J25=$1,220,000
```

---

## 📊 **What Users Get Back (Output)**

### **Statistical Summary**
```json
{
  "Portfolio_NPV": {
    "mean": 1250000,           // Average outcome
    "std": 340000,             // Standard deviation
    "min": 420000,             // Worst case
    "max": 2180000,            // Best case
    "percentiles": {
      "5": 680000,             // 5% chance below this
      "25": 1020000,           // 25th percentile
      "50": 1240000,           // Median (50th percentile)
      "75": 1480000,           // 75th percentile  
      "95": 1820000            // 95% chance below this
    }
  }
}
```

### **Risk Metrics**
```json
{
  "risk_analysis": {
    "var_95": 680000,          // Value at Risk (95% confidence)
    "var_99": 540000,          // Value at Risk (99% confidence)
    "expected_shortfall": 590000, // Average loss in worst 5%
    "probability_of_loss": 0.12   // 12% chance of losing money
  }
}
```

### **Distribution Data** 
```json
{
  "histogram": {
    "bins": [400000, 500000, 600000, ...],
    "frequencies": [12, 89, 245, 892, 1547, ...]
  }
}
```

---

## 🔒 **Privacy & Security**

### **What We Do:**
- ✅ Process Excel file in secure, isolated environment
- ✅ Delete file immediately after processing
- ✅ No permanent storage of customer data
- ✅ Encrypted transmission (HTTPS)

### **What We Don't Do:**
- ❌ Store or save Excel files
- ❌ Share data between customers
- ❌ Use data for training or analytics
- ❌ Keep any sensitive information

---

## ⏱️ **Timeline**

```
User uploads file        → Instant
File parsing            → 1-3 seconds  
Monte Carlo simulation  → 5-60 seconds (depending on complexity)
Results delivery        → Instant
Total time             → Usually under 1 minute
```

---

## 💡 **Real Example: Bank Portfolio**

### **User Sends:**
- **Excel File**: 50MB bank portfolio model
- **Variables**: 
  - Interest rates (normal distribution)
  - Default rates (beta distribution)  
  - Market volatility (triangular distribution)
- **Output Cells**: Portfolio value, regulatory capital

### **User Gets:**
- **Risk Analysis**: "Portfolio has 95% confidence of being worth between $45M-$67M"
- **Regulatory Metrics**: "Need $12M capital to meet Basel III requirements"
- **Stress Testing**: "In worst 1% of scenarios, losses could reach $23M"

### **Business Impact:**
- **Compliance**: Automated regulatory reporting
- **Decision Making**: Data-driven risk limits
- **Time Savings**: 2 hours → 30 seconds
- **Accuracy**: 100,000 scenarios vs. manual estimates

---

## 🎯 **Key Points for Users**

### **✅ What They Need:**
- Existing Excel financial model
- Knowledge of which cells are uncertain
- Basic understanding of probability distributions
- Any programming language (Python, JavaScript, etc.)

### **❌ What They Don't Need:**
- Modify their Excel formulas
- Learn new software or tools
- Convert models to other formats
- Advanced statistical knowledge

### **🚀 What They Get:**
- Professional risk analysis
- Regulatory-grade statistics
- Interactive data for charts/dashboards
- 100x faster results than Excel

**Bottom Line: Upload Excel + Define Variables = Professional Risk Analysis** 📊
