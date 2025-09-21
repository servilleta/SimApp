# Monte Carlo Simulation Web Platform

A web-based platform for running Monte Carlo simulations with Excel data integration and GPU acceleration.

## Project Structure

```
PROJECT/
├── backend/                 # FastAPI backend
│   ├── auth/               # Authentication module
│   ├── excel_parser/       # Excel file processing
│   ├── simulation/         # Simulation engine
│   ├── results/           # Results processing
│   └── gpu/               # GPU acceleration
└── frontend/              # React frontend
    ├── public/            # Static assets
    └── src/               # Source code
        ├── components/    # React components
        ├── hooks/        # Custom hooks
        ├── pages/        # Page components
        ├── services/     # API services
        ├── store/        # State management
        └── utils/        # Utility functions
```

## Prerequisites

- Python 3.11+
- Node.js 18.x+
- CUDA Toolkit (for GPU support)
- Docker and Docker Compose

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Features

- Excel file upload and parsing
- Triangular probability distribution configuration
- Monte Carlo simulation with GPU acceleration
- Interactive results visualization
- Responsive web interface
- Enterprise-grade security and authentication
- Subscription management with Stripe integration
- Advanced statistical analysis and reporting

## 📋 Licensing & Commercial Use

### ✅ **COMMERCIAL-FRIENDLY - NO LICENSE FEES REQUIRED**

This platform is built entirely on **open-source technologies with business-friendly licenses**:

- **Core Stack**: Python, React, FastAPI, PostgreSQL, Redis (MIT/BSD licenses)
- **Scientific Computing**: NumPy, SciPy, Pandas (BSD licenses)  
- **Visualization**: Chart.js, Plotly.js, Recharts (MIT licenses)
- **Infrastructure**: Docker, Nginx (Apache 2.0/BSD licenses)
- **GPU Acceleration**: CUDA Toolkit (free for commercial use)

**Key Benefits:**
- 🟢 **$0 licensing costs** vs competitors ($50K-$500K/year)
- 🟢 **Full commercial rights** - can sell, modify, distribute freely
- 🟢 **No copyleft restrictions** - no GPL/LGPL dependencies
- 🟢 **International compliance** - all licenses globally recognized

### 📄 License Documentation

- **[Complete License Analysis](legal/OPEN_SOURCE_LICENSES.md)** - Detailed breakdown of all dependencies
- **[License Attribution Files](LICENSES/)** - Required attribution notices for distribution
- **[Platform License](LICENSE)** - Terms for the overall platform

## Documentation

- [Backend Developer Guide](backend/README.md)
- [Frontend Developer Guide](frontend/README.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Legal & Compliance](legal/)
- [Licensing Analysis](reports/LICENSING_ANALYSIS.md)

## 🏢 Commercial Information

### Competitive Advantage
- **Oracle Crystal Ball**: $995-2,995/user/year → **Our Platform**: $0 licensing
- **@RISK**: $795-1,995/user/year → **Our Platform**: $0 licensing  
- **Palantir Foundry**: $50K-500K/year → **Our Platform**: $0 licensing

### Revenue Model Opportunities
1. **SaaS Subscription** - $29-299/user/month (95%+ margin)
2. **Enterprise Licensing** - $50K-500K/year (95%+ margin)
3. **Professional Services** - $200-500/hour consulting
4. **White-label Solutions** - Custom pricing for partners

## License

See [LICENSE](LICENSE) file for platform licensing terms.
See [LICENSES/](LICENSES/) directory for third-party component attributions. 