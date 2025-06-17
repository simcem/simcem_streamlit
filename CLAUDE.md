# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based web application for cement clinker analysis and thermodynamic calculations. The application is called "Clinker Tool" and provides cement industry professionals with tools for raw mix design, XRF data analysis, and thermodynamic modeling of cement clinker phases.

## Key Technologies and Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **simcem**: Custom cement chemistry library for thermodynamic calculations (installed from GitHub)
- **pycalphad**: CALPHAD thermodynamic modeling library for equilibrium calculations
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive plotting and visualization
- **scipy**: Scientific computing (optimization algorithms)

### System Dependencies (packages.txt)
The application requires several system packages:
- cmake, coinor-libipopt-dev, libeigen3-dev, libboost-all-dev, libsundials-dev

## Application Architecture

The application consists of a single main file `streamlit_app.py` with multiple tabs:

1. **Home**: Introduction and navigation
2. **Upload XRF Data**: Upload and process XRF (X-Ray Fluorescence) analysis data from Excel files
3. **Mix Design**: Design raw material mixes for cement production
4. **Equilibrium Calculator**: Direct thermodynamic equilibrium calculations
5. **About**: Tool information and credits

### Key Classes and Functions

- `CementOptimizer`: Handles cement mix optimization with LSF (Lime Saturation Factor) calculations
- `thermosolver()`: Main thermodynamic solver using simcem/pycalphad for phase equilibrium
- `formulation_solver()`: Solves for optimal raw material formulations using linear programming
- `optimiser()`: Linear programming optimization for target phase compositions
- `Haneinify()`: Converts oxide compositions for use with the Hanein thermodynamic database

### Database Files
- `C_A_S_Fe_O_M.tdb`: Thermodynamic database file for the CaO-Al2O3-SiO2-Fe-O-MgO system used in cement clinker calculations

## Development Commands

### Running the Application
```bash
streamlit run streamlit_app.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

Note: The simcem library is installed directly from GitHub and may require the system dependencies listed in packages.txt.

## Key Features

### Raw Material Database
The application includes a comprehensive database of cement raw materials with oxide compositions:
- Industrial clays, shales, limestone, iron oxide, bauxite
- Pure compounds (lime, alumina, gypsum, etc.)
- Each material defined by weight percentages of major oxides

### Calculations Supported
- **Bogue calculations**: Classical cement phase calculations
- **Taylor calculations**: Modified cement phase calculations  
- **LSF, SR, AR moduli**: Standard cement chemistry ratios
- **Thermodynamic equilibrium**: Using Hanein et al. database via simcem
- **Mix optimization**: Linear programming for target phase compositions

### Data Import/Export
- Excel file upload for XRF data
- Excel file download for calculation results
- Supports dynamic data editing within the Streamlit interface

## Important Notes

- The application uses session state extensively to maintain data between interactions
- Thermodynamic calculations are cached using `@st.cache_resource()` for performance
- Temperature limits are enforced (600-1450°C) due to database limitations
- Some oxides (ZnO, MnO, P2O5, SrO) are excluded from calculations due to data limitations