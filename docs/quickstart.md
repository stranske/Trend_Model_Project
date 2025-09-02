# 🚀 Quick Start Guide

**Get your first trend analysis running in under 10 minutes!**

This guide is designed for non-technical users who want to quickly analyze investment fund performance using volatility-adjusted trend analysis.

## ⚡ Fastest Start: Docker (Zero Setup)

**Best for:** Complete beginners who want to avoid any installation steps.

### Step 1: Get Docker Running
- **Windows/Mac**: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux**: Install Docker using your package manager

### Step 2: Launch the Application
Open your terminal/command prompt and run:

```bash
docker run -p 8501:8501 ghcr.io/stranske/trend-model:latest
```

Wait about 30 seconds for the app to start. You'll see messages ending with:
```
You can now view your Streamlit app in your browser.
```

### Step 3: Open Your Browser
Visit: **http://localhost:8501**

🎉 **You're ready to analyze!** The web interface will load with the analysis tool ready to use.

---

## 🌐 Alternative: Local Installation

**Best for:** Users comfortable with basic Python installation.

### Step 1: Install Python Requirements
```bash
# Download the project
git clone https://github.com/stranske/Trend_Model_Project.git
cd Trend_Model_Project

# Set up environment (this may take 2-3 minutes)
./scripts/setup_env.sh
```

### Step 2: Launch the Web App
```bash
./scripts/run_streamlit.sh
```

### Step 3: Open Your Browser
Visit **http://localhost:8501** when the app starts.

---

## 📊 Using the Analysis Tool

The analysis tool has a simple step-by-step workflow:

![Main Interface](https://github.com/user-attachments/assets/276458a5-7f78-4e06-bf9c-738aef239611)

### Step 1: Upload Your Data

![Upload Interface](https://github.com/user-attachments/assets/440c3e0c-b0a6-468d-916e-4bb92e036442)

The easiest way to get started:

1. **Click "Load demo data"** - uses the included 10-year synthetic dataset
2. **Or upload your CSV** - drag and drop or browse for your own data

### Step 2: Choose Your Analysis Style

![Configure with Presets](https://github.com/user-attachments/assets/a919ee74-a644-48f7-a1ec-0ecdef5a9f2b)

The tool includes three ready-made presets designed for different risk preferences:

#### 🛡️ **Conservative**
- **Best for**: Risk-averse investors, retirement planning
- **Strategy**: Longer 60-month lookback, fewer (5) holdings, stability-focused metrics
- **Target volatility**: 8% (lower risk)
- **Rebalancing**: Quarterly (less frequent changes)

#### ⚖️ **Balanced** 
- **Best for**: Most users looking for steady growth
- **Strategy**: 36-month lookback, moderate (10) holdings, balanced risk-return
- **Target volatility**: 10% (moderate risk)
- **Rebalancing**: Monthly (standard frequency)

#### 🚀 **Aggressive**
- **Best for**: Performance-focused investors, growth portfolios
- **Strategy**: Shorter 24-month lookback, more (15) holdings, return-focused
- **Target volatility**: 15% (higher risk)
- **Rebalancing**: Monthly with fast response to changes

### Step 3: Run Your Analysis
1. **Select your preset** from the dropdown (Conservative, Balanced, or Aggressive)
2. **Save configuration** - click the "💾 Save Configuration" button  
3. **Navigate to Run** - click the "Run" tab in the sidebar
4. **Wait 30-60 seconds** for the analysis to complete

### Step 4: View Your Results
Navigate to the "Results" tab to see:

**Portfolio Performance**
- Total return over the selected time period
- Month-by-month portfolio returns
- Comparison to market benchmarks

**Risk Metrics**
- **Sharpe Ratio**: Return per unit of risk (higher is better)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Portfolio risk level
- **Information Ratio**: Performance vs. benchmark

**Fund Selection Details**
- Which specific funds were chosen and why
- Individual fund performance during selection period
- Portfolio weights assigned to each fund

### Step 5: Export Your Results
Navigate to the "Export" tab and click **"Download Results Bundle"** to get:
- `portfolio_returns.csv` - Your portfolio performance data
- `summary.json` - Key metrics and statistics  
- `config.json` - Exact settings used for this analysis
- `event_log.csv` - Detailed selection and rebalancing history

---

## 📁 Using Your Own Data

### Data Format Requirements
Your CSV file should have this structure:

```csv
Date,Fund_A,Fund_B,Fund_C,Market_Index
2020-01-31,0.02,-0.01,0.03,0.015
2020-02-28,-0.03,0.025,-0.01,-0.08
2020-03-31,0.01,0.015,0.02,0.12
```

**Requirements:**
- **Date column**: First column with dates (YYYY-MM-DD format preferred)
- **Return columns**: Monthly returns as decimals (0.02 = 2% return)
- **Minimum data**: At least 36 months recommended for reliable analysis
- **Frequency**: Monthly data works best

### Uploading Your Data
1. **Click "Upload CSV"** in the sidebar
2. **Select your file** from your computer
3. **Map columns**: Tell the app which column contains dates
4. **Proceed with analysis** using any of the three presets

### Sample Datasets Included
- `demo/demo_returns.csv` - 10 years of synthetic fund data with 20 managers
- `demo/extended_returns.csv` - 20 years of data including market crisis periods
- Ready to use with all three presets

---

## 💻 Command Line Usage (Advanced)

**Best for:** Users who want to automate analysis or integrate with scripts.

### Basic Command Line Run
```bash
# Use demo data with balanced preset
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/demo.yml
```

### Using Different Presets
The tool includes preset configurations you can use directly:
```bash
# Conservative analysis  
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/conservative.yml

# Aggressive analysis
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/aggressive.yml
```

---

## 🔧 Customization

### Modifying Presets
After selecting a preset in the web interface:
1. **Adjust time periods**: Change in-sample/out-of-sample split dates
2. **Modify fund selection**: Change the number of funds to select
3. **Update risk parameters**: Adjust target volatility levels
4. **Change metrics**: Weight different performance measures

### Saving Custom Settings
1. **Configure your preferences** in the web interface
2. **Download the configuration** as a YAML file
3. **Reuse the configuration** by uploading it later
4. **Share configurations** with your team

---

## ❓ Common Issues & Solutions

### Application Won't Start
**"Connection refused" at http://localhost:8501**
- Wait 60 seconds after running the Docker command
- Check that port 8501 isn't already in use
- Try http://127.0.0.1:8501 instead

**Docker command fails**
- Make sure Docker Desktop is running
- Check your internet connection (needed to download the image)
- Try `docker --version` to confirm Docker is installed

### Data Problems
**Upload fails or gives errors**
- Verify your CSV has a date column as the first column
- Check data covers at least 24 months
- Make sure returns are in decimal format (0.05 = 5%)
- Remove any missing data or text in numeric columns

**Poor analysis results**
- Try different in-sample/out-of-sample time periods
- Increase the minimum track record for funds
- Check if your data period includes unusual market events
- Ensure you have at least 10-15 funds for meaningful selection

### Performance Issues  
**Analysis is slow**
- Reduce the number of funds being analyzed
- Shorten the time period 
- Use simpler presets (Conservative runs faster)
- Close other browser tabs to free up memory

---

## 📚 Next Steps & Resources

### After Your First Analysis
1. **Try all three presets** with the same data to compare approaches
2. **Experiment with different time periods** to see how results change
3. **Upload your own fund data** to analyze real portfolios
4. **Modify preset parameters** to customize the analysis

### Additional Resources
- **📖 Detailed Documentation**: [UserGuide.md](UserGuide.md) for technical details
- **🐳 Docker Guide**: [DOCKER_QUICKSTART.md](../DOCKER_QUICKSTART.md) for advanced Docker usage
- **📓 Jupyter Notebooks**: `Vol_Adj_Trend_Analysis1.5.TrEx.ipynb` for custom workflows
- **🔧 Configuration Examples**: `config/` folder for more preset options

### Getting Help
- 🐛 **Report Issues**: https://github.com/stranske/Trend_Model_Project/issues
- 📋 **Feature Requests**: Use GitHub issues with enhancement label
- 💬 **Questions**: Check existing issues or create a new discussion

---

## ✅ 10-Minute Success Checklist

After following this guide, you should have:
- [ ] **Application running** (Docker or local installation)
- [ ] **Demo data loaded** or your own data uploaded
- [ ] **Preset selected** (Conservative, Balanced, or Aggressive)  
- [ ] **Analysis completed** with results displayed
- [ ] **Key metrics understood** (Sharpe ratio, returns, drawdown)
- [ ] **Results exported** for further analysis

**🎊 Congratulations!** You've completed your first volatility-adjusted trend analysis.

The tool has analyzed historical fund performance, selected the best performers based on your chosen criteria, and constructed an optimized portfolio. You now have quantitative insights into fund selection and portfolio construction that typically require expensive institutional tools.
