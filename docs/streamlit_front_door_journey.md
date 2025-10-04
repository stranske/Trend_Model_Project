# Streamlit Front Door Smoke Test

This checklist documents the happy-path flow for non-technical users to go from
a blank session to a completed report inside the Streamlit "front door" app.

## Prerequisites
- Install dependencies (`pip install -r requirements.txt`).
- Launch the Streamlit app from the project root:
  ```bash
  streamlit run streamlit_app/app.py
  ```

## Step-by-step walkthrough
1. **Landing page**
   - Confirm the welcome screen highlights the Data → Model → Results journey.
   - Click the optional *Run guided demo* button to verify it redirects to the
     Results page with demo content.

2. **Data step**
   - Navigate to **Data**.
   - Ensure the built-in sample dataset loads automatically with a success
     status banner, preview table, and suggested benchmark list.
   - Upload a small CSV/Excel file to confirm validation banners appear and the
     preview updates.

3. **Model step**
   - Switch to **Model** and verify the page acknowledges the active data
     source.
   - Choose a preset, adjust parameters, and save the configuration.
   - Use *Validate Configuration* to confirm friendly error summaries appear
     when inputs are inconsistent.
   - Run a dry-run sample followed by the full simulation; the run section
     should display guardrail estimates and confirmation to review Results.

4. **Results step**
   - Inspect the harmonised charts (equity, drawdown, weights, and analytics)
     to confirm the shared colour palette and tooltips.
   - Download the ZIP bundle, CSV extracts, and HTML/PDF report.
   - Trigger walk-forward and attribution tools to ensure error handling is
     human-readable.

## Expected outcome
Following these steps should produce a completed simulation with the summary
visible on the Results page and all downloads available without exposing raw
stack traces to the end user.
