# Data Files

The SBIR grant data file (`award_data.csv`) is not included in this repository because it exceeds GitHub's 100 MB file size limit (~351 MB).

## How to get the data

1. Download the full SBIR award dataset from the [SBIR.gov Awards Search](https://www.sbir.gov/awards) page using the export/download feature, or request it directly from the SBIR data portal.
2. Place the downloaded file in the **root of the repository** (same folder as `app.py`) and name it `award_data.csv`.
3. Run the app: `streamlit run app.py`

The app will also accept any path you specify in the sidebar — you are not required to rename the file or move it, just update the path field.

## Expected columns

The file should have the following columns (the app normalizes names automatically):

| Column | Description |
|---|---|
| Company | Recipient company name |
| Award Title | Project title |
| Agency | Awarding agency (e.g., DoD, NSF) |
| Branch | Sub-agency branch |
| Phase | Phase I or Phase II |
| Program | SBIR or STTR |
| Award Year | Fiscal year of award |
| Award Amount | Dollar amount |
| Abstract | Project abstract text |
| ... | (37 additional columns) |

## Future API integration

When the SBIR API is restored, the `src/data_loader.py` → `load_from_api()` function contains a documented stub ready for implementation.
