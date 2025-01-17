# LandlordMapper Chicago Data Workflow

## Setup

1. Install `poetry` if not already installed.
2. `cd` into the project's root diretory
3. Run `poetry install`
4. Install Jupyter kernel in this project's root directory:
   ```bash
   poetry run python -m ipykernel install --user --name=landlordmapper-chicago-data
5. When opening Jupyter notebooks, select the "landlordmapper-chicago-data" kernel


## Data Sources

- Cook County Assessor's Property Search
  - https://www.cookcountyassessor.com/address-search#address
- Cook County Open Data Portal
  - Cook County Address Points
    - https://datacatalog.cookcountyil.gov/GIS-Maps/Cook-County-Address-Points/78yw-iddh/about_data
- Office of the Illinois Secretary of State
  - https://www.ilsos.gov/data/bs/proc_llc_data.pdf


## Directory Structure

- Address Analysis
    - Manual analysis of most commonly appearing validated addresses in the property and corporate / LLC datasets
    - Used by workflow to determine which addresses should be included in the network analysis
    - Used by the application database importers to add data about landlord organizations and associate them with
      taxpayer addresses
- Addresses
    - Results of Geocodio address validation processing for properties and corporations / LLCs found in the taxpayer data
- Common
    - Lists of common names and addresses to exclude from the network analysis
- Corp / LLC
    - Datasets for all corporations and LLCs that operate in the state of IL
- Geocodio
    - Stores dataframes saved upon executing Geocodio API calls
- Landlord Data
    - Datasets used for landlord (network) workflow
- Landlord Workflow
    - Datasets outputted by landlord (network) workflow
- Scrape
    - Stores dataframes saved upon executing Cook Country assessor site scraper


## General Pattern

The data manipulation logic and workflow logic serve distinct purposes. The data manipulation code
is stored in the "classes" directory and contains classes with methods and variables that
manipulate different data types and are used throughout the workflow. The workflow code
deals with the order of execution and inputs/outputs of the data processing workflow.

Within each of these high-level directories there are files that are specific to each stage of the process:

- Scrape
- Geocodio
- Landlord / Network processing
