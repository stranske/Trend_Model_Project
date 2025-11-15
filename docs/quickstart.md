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
- Click the file uploader and select your CSV file
- The first column should be dates, followed by fund return columns
- Data should cover at least 24 months for meaningful analysis

### Step 2: Choose Your Analysis Style
Three preset strategies are available:

- **Conservative**: Low turnover, stable selections
- **Balanced**: Moderate turnover, balanced approach
- **Aggressive**: High turnover, dynamic selections

### Step 3: Run Your Analysis
- Click "Run Analysis" to start the computation
- Progress bars will show the analysis status
- Results appear in real-time as calculations complete

### Step 4: View Your Results
The results dashboard shows:
- **Performance Charts**: Visual performance over time
- **Key Metrics**: Sharpe ratios, returns, volatility measures
- **Fund Rankings**: Which funds were selected and when
- **Risk Analysis**: Drawdowns and risk-adjusted returns

### Step 5: Export Your Results
- Download results as Excel, CSV, or JSON formats
- Export charts as high-resolution images
- Save configurations for future runs

---

## 📁 Using Your Own Data

### Data Format Requirements


**Example CSV:**
```csv
Date,FUND_A,FUND_B,FUND_C
2021-01-31,0.02,0.015,0.018
2021-02-28,0.01,0.012,0.017
2021-03-31,-0.005,0.02,0.013
### Uploading Your Data
1. Prepare your CSV with the format above
2. Use the "Upload" section in the sidebar
3. Map columns if auto-detection fails
4. Verify data preview looks correct
5. Proceed to configuration

### Sample Datasets Included
The app includes demo datasets to get started:
- **Basic Demo**: Simple 8-fund universe
- **Extended Demo**: Larger fund universe with benchmarks
- **Historical Example**: Real market data examples

---

## 💻 Command Line Usage (Advanced)

For users comfortable with the command line:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the project in editable mode (includes CLI + app extras)
pip install --upgrade pip
pip install -e .[app]

# Run analysis with config via the packaged CLI
trend run -c config/demo.yml --returns demo/demo_returns.csv

# Generate demo data
python scripts/generate_demo.py
```

---

## 🔧 Customization

Advanced users can modify:
- **Configuration files** in the `config/` directory
- **Selection strategies** via custom weighting methods
- **Export formats** and output destinations
- **Risk parameters** and volatility targets

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
- Configure `data.missing_policy` (`drop`, `ffill`, or `zero`) and `data.missing_limit`
	in your YAML file if you want the loader to auto-fill or drop sparse series

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

- [ ] **Docker installed and running** (or Python environment set up)
- [ ] **App launched** at http://localhost:8501
- [ ] **Demo data uploaded** or your own CSV prepared
- [ ] **Preset selected** (start with Conservative)
- [ ] **Analysis completed** with results displayed
- [ ] **Results exported** in your preferred format

**🎉 Congratulations!** You've successfully run your first trend analysis.  
 
👉 To **upload your own data**, see the [UserGuide.md](UserGuide.md#data-upload) for step-by-step instructions.  
👉 To **explore different settings and presets**, check the [config/](../config/) folder and the [UserGuide.md](UserGuide.md#customization) for details on customizing your analysis.  
 
Unlock the full potential of the system by trying out these customization options!
