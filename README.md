# 📈 Investment Backtest Analyzer

A comprehensive Python-based tool for sophisticated investment strategy analysis and backtesting. Built for investors, analysts, and researchers who need professional-grade performance analysis.

## 🌐 Live Website

Visit the project website: **[Your website will be here after GitHub Pages setup]**

## ✨ Features

- **Multi-Investor Tracking**: Follow up to 5 investors simultaneously with different selection criteria
- **Advanced Backtesting**: Test strategies from 2012-2025 with quarterly and daily price data
- **Flexible Configuration**: Customize rebalancing, timing, thresholds, and selection modes
- **Performance Analysis**: Calculate CAGR, Alpha, Sharpe ratios, and benchmark comparisons
- **Data Export**: Generate detailed reports, transaction logs, and publication-ready charts
- **Batch Analysis**: Run multiple scenarios and compare results

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/investment-backtest-analyzer.git
   cd investment-backtest-analyzer
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install pandas matplotlib numpy yfinance
   ```

3. **Run the backtest**:
   ```bash
   python backtest_strategy.py
   ```

## 📊 Configuration

Key parameters you can adjust in `backtest_strategy.py`:

```python
NUM_INVESTORS = 5          # Number of investors to track
START_YEAR = 2012          # Backtest start year
END_YEAR = 2026           # Backtest end year
HAS_RELOCATION = True     # Enable/disable rebalancing
SELECTION_MODE = 3        # Investor selection method (0-4)
TOP_PICKS_PER_INVESTOR = 3 # Number of stocks per investor
MIN_TOP1_PCT = 15         # Minimum portfolio percentage threshold
```

## 🎯 Selection Modes

- **Mode 0**: Random selection with coverage analysis
- **Mode 1**: Prior performance ranking
- **Mode 2**: Portfolio value ranking  
- **Mode 3**: CAGR 3-year ranking
- **Mode 4**: Cumulative CAGR ranking

## 📈 Output Files

The analyzer generates:

- `backtest_transactions.csv` - Detailed transaction log
- `performance_comparison.png` - Visual performance chart
- `yearly_alpha.csv` - Annual alpha calculations
- `investor_track_record_*.csv` - Individual investor performance
- `backtest_analysis_results.csv` - Batch analysis results

## 🔧 Advanced Usage

### Batch Analysis

Run multiple scenarios with different parameters:

```bash
python backtest_analysis.py --start-years 2012,2013,2014 --num-investors 3,5,8 --parallel 4
```

### Custom Data

The tool expects CSV files in these folders:
- `stock_prices/` - Quarterly price data
- `stock_prices_daily/` - Daily price data
- `top_positions.csv` - Investor portfolio positions

## 📋 Requirements

- Python 3.7+
- pandas
- matplotlib
- numpy
- yfinance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

- **Website**: [Your GitHub Pages URL]
- **GitHub**: [Your GitHub Profile]
- **Issues**: [Repository Issues Page]

## 🙏 Acknowledgments

- Built with Python and modern data science libraries
- Inspired by quantitative finance research
- Designed for educational and research purposes

---

⭐ **Star this repository if you find it useful!**
