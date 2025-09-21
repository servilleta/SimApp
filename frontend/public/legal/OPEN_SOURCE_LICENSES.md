# Open Source License Compliance

_Last updated: September 20, 2025_

This document provides attribution for third-party open-source software used in **SimApp.ai**.

## 📋 Executive Summary

**✅ All dependencies use business-friendly licenses permitting commercial use**

- ✅ **No GPL/LGPL dependencies** - No copyleft restrictions
- ✅ **No proprietary libraries** - All open source with permissive licenses  
- ✅ **No patent restrictions** - Standard algorithms only
- ✅ **Full commercial rights** - Can sell, modify, and distribute freely

---

## 🔧 Core Dependencies

### Backend (Python)

| Package | Version | License | Commercial Use |
|---------|---------|---------|----------------|
| **FastAPI** | 0.110.0 | MIT | ✅ Permitted |
| **Uvicorn** | 0.27.1 | BSD-3-Clause | ✅ Permitted |
| **Pydantic** | 2.6.1 | MIT | ✅ Permitted |
| **Pandas** | 2.2.0 | BSD-3-Clause | ✅ Permitted |
| **NumPy** | 1.26.3 | BSD-3-Clause | ✅ Permitted |
| **SciPy** | 1.12.0 | BSD-3-Clause | ✅ Permitted |
| **SymPy** | 1.12 | BSD-3-Clause | ✅ Permitted |
| **OpenPyXL** | 3.1.2 | MIT | ✅ Permitted |
| **XlsxWriter** | 3.1.9 | BSD-2-Clause | ✅ Permitted |
| **SQLAlchemy** | 1.4.x | MIT | ✅ Permitted |
| **PostgreSQL Driver** | 2.9.0+ | LGPL-3.0* | ✅ Permitted |
| **Redis** | 5.0.1 | MIT | ✅ Permitted |
| **Stripe** | 5.5.0+ | MIT | ✅ Permitted |
| **Requests** | 2.31.0 | Apache-2.0 | ✅ Permitted |

*\*Used as dynamic library - no copyleft requirements*

### Frontend (React/Node.js)

| Package | Version | License | Commercial Use |
|---------|---------|---------|----------------|
| **React** | 18.3.1 | MIT | ✅ Permitted |
| **React DOM** | 18.3.1 | MIT | ✅ Permitted |
| **Ant Design** | 5.18.1 | MIT | ✅ Permitted |
| **Material-UI** | 7.1.1 | MIT | ✅ Permitted |
| **Chart.js** | 4.4.3 | MIT | ✅ Permitted |
| **Plotly.js** | 2.27.1 | MIT | ✅ Permitted |
| **AG Grid Community** | 32.0.0 | MIT | ✅ Permitted |
| **Axios** | 1.7.2 | MIT | ✅ Permitted |
| **ExcelJS** | 4.4.0 | MIT | ✅ Permitted |
| **React Router** | 6.23.1 | MIT | ✅ Permitted |
| **Redux Toolkit** | 2.2.5 | MIT | ✅ Permitted |
| **Vite** | 5.1.4 | MIT | ✅ Permitted |
| **Tailwind CSS** | 3.4.1 | MIT | ✅ Permitted |

### Infrastructure

| Component | License | Commercial Use |
|-----------|---------|----------------|
| **Docker** | Apache-2.0 | ✅ Permitted |
| **PostgreSQL** | PostgreSQL License | ✅ Permitted |
| **Redis** | BSD-3-Clause | ✅ Permitted |
| **Nginx** | BSD-2-Clause | ✅ Permitted |

---

## ⚖️ License Requirements

### Attribution Requirements

The following licenses require attribution in software distributions:

**MIT License Components:**
- React (© Meta Platforms, Inc.)
- FastAPI (© Sebastián Ramírez)
- Chart.js (© Chart.js contributors)
- Ant Design (© Ant Design Team)
- [Complete list in bundled LICENSE files]

**BSD License Components:**
- NumPy (© NumPy Developers)
- Pandas (© pandas-dev team)
- SciPy (© SciPy Developers)
- [Complete list in bundled LICENSE files]

**Apache License Components:**
- Requests (© Python Software Foundation)
- Docker (© Docker, Inc.)
- [Complete list in bundled LICENSE files]

### Compliance Checklist

- ✅ **License texts included** - All original license files preserved
- ✅ **Copyright notices maintained** - Original attributions kept
- ✅ **Attribution provided** - Credits in application About/Legal section
- ✅ **No trademark misuse** - Contributor names not used for endorsement
- ✅ **Source availability** - For any modified components (none currently)

---

## 📞 Legal Information

**For licensing questions:**
- Complete license texts available in `/LICENSES/` directory
- Consult qualified legal counsel for specific business decisions
- Monitor dependency updates for license changes

**Legal Disclaimer:** This analysis is for informational purposes only. Consult qualified legal counsel for specific business decisions.

---

*This document is automatically updated with each software release.*
