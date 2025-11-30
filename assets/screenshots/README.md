# Screenshots

This directory holds screenshots for the quickstart guide and documentation.

## Screenshots Needed

| Screenshot | Description |
|------------|-------------|
| `upload-interface.png` | File upload interface with drag-and-drop and column mapping |
| `template-section.png` | Template download section with sample data preview |
| `preset-selection.png` | Configuration page showing Conservative/Balanced/Aggressive presets |
| `results-dashboard.png` | Analysis results with charts, metrics, and download buttons |

## Creating Screenshots

1. Start the Streamlit app: `./scripts/run_streamlit.sh`
2. Navigate through the workflow
3. Take screenshots at key steps
4. Save as PNG files in this directory

## Screenshot Details

### upload-interface.png
- Streamlit file uploader interface
- CSV file selection dialog
- Column mapping showing Date and Fund columns

### template-section.png
- Template download section expanded
- Sample data preview visible
- Download button for CSV template

### preset-selection.png
- Configuration page dropdown with preset options
- Conservative (8% risk target, 60 month lookback)
- Balanced (10% risk target, 36 month lookback)
- Aggressive (15% risk target, 24 month lookback)

### results-dashboard.png
- Portfolio performance line chart
- Key metrics: Sharpe Ratio, Annual Return, Max Drawdown
- Selected funds list with individual performance
- Download buttons for CSV, Excel, JSON exports