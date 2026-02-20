# GlucoX Data Science Challenge

This project implements a data processing and analysis pipeline to evaluate the effectiveness of "nudges" (Gentle Reminders and Urgent Alerts) on patient glucose measurement logging.

## Project Overview

The pipeline processes patient registry data and application logs to:
1.  **Ingest and Clean Data**: Load CSV and JSONL files, unpacking nested JSON payloads.
2.  **Process Events**: Correlate sent nudges with subsequent glucose measurements within a 4-hour window to determine "assertive" (successful) nudges.
3.  **Analyze Trends**: Calculate response rates based on cumulative nudge counts, segmented by age group and risk profile.
4.  **Statistical Testing**: Perform Chi-square tests to detect significant changes in response rates between the first nudge and subsequent ones.
5.  **Reporting**: Generate a text summary and visualization plots.

## Repository Structure

```text
GlucoX_Data_Science_Challenge/
├── data/
│   ├── app_logs.jsonl        # Raw application logs (nudges and measurements)
│   └── patient_registry.csv  # Patient demographics and risk segments
├── report/                   # Output directory for reports and plots
├── src/
│   └── utils.py              # Helper functions for processing and plotting
├── main.py                   # Main execution script
└── README.md                 # Project documentation
```

## Prerequisites

-   Python 3.8 or higher
-   Required Python packages are listed in `requirements.txt`.

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

1.  **Prepare the Environment**:
    Ensure the `report` directory exists, as the script writes outputs there.
    ```bash
    mkdir -p report
    ```

2.  **Run the Pipeline**:
    Execute the `main.py` script. You may need to set the `PYTHONPATH` to include the `src` directory so that `utils` can be imported correctly.

    ```bash
    # Linux/MacOS
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python main.py

    # Windows (PowerShell)
    $env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)\src"
    python main.py
    ```

## Output Description

Upon successful execution, the `report/` folder will contain:

-   **`report.txt`**: A text file summarizing mean response rates and Chi-square statistical test results comparing Nudge 1 vs. subsequent nudges.
-   **`response_global.png`**: A plot showing the global response rate trend over cumulative nudges with 95% Wilson confidence intervals.
-   **`response_age_group.png`**: Response rate trends segmented by patient age groups.
-   **`response_risk_segment.png`**: Response rate trends segmented by risk profile (High, Medium, Low).
