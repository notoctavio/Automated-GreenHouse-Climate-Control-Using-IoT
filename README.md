# 🌿 Smart Greenhouse Control System

A sophisticated Python-based application for monitoring and controlling greenhouse environments through real-time data analysis and automated control systems.

![Greenhouse Control System](assets/greenhouse_dashboard.png)

## 🚀 Features

### 📊 Data Management
- **Smart Data Loading**: Import and validate sensor data from CSV files
- **Data Validation**: Automatic cleaning and outlier detection
- **Real-time Monitoring**: Track temperature, humidity, and NPK levels
- **Data Export**: Save filtered data in CSV or Excel format

### 📈 Advanced Analytics
- **Interactive Dashboards**: 
  - Temperature and humidity trends
  - NPK level analysis
  - Distribution plots
  - Correlation analysis
- **Customizable Date Ranges**: Filter and analyze specific time periods
- **Statistical Analysis**: 
  - Mean, min, max, and standard deviation
  - Trend analysis
  - Anomaly detection

### 🎛️ Automated Control
- **Smart Actuator Control**:
  - Ventilation system (temperature regulation)
  - Humidification system (humidity control)
  - Curtain control (light management)
- **Hysteresis Control**: Prevents rapid actuator switching
- **Safety Checks**: Ensures parameters stay within safe ranges

### 📑 Professional Reporting
- **Comprehensive PDF Reports**:
  - Summary statistics
  - Trend analysis
  - Recommendations
  - Visual graphs and charts
- **Customizable Reports**: Select specific date ranges and parameters

## 🛠️ Technical Stack

- **Frontend**: CustomTkinter (Modern UI)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Reporting**: FPDF
- **Machine Learning**: Scikit-learn
- **Data Analysis**: SciPy

## 📋 System Requirements

- Python 3.8+
- Required packages (see requirements.txt)
- Minimum 4GB RAM
- 500MB disk space

## 🚀 Getting Started

1. **Installation**:
   ```bash
   git clone [repository-url]
   cd Greenhouse_Control_Project
   pip install -r requirements.txt
   ```

2. **Running the Application**:
   ```bash
   python src/greenhouse_control.py
   ```

3. **First Steps**:
   - Load your sensor data (CSV format)
   - Configure control parameters
   - Start monitoring and controlling

## 🎯 Optimal Parameters

- **Temperature**: 20-25°C
- **Humidity**: 60-80%
- **NPK Levels**: Customizable ranges
- **Light Intensity**: 5000-10000 lux

## 📝 Data Format

The application accepts CSV files with the following columns:
- `timestamp`: Date and time of measurement
- `temperature`: Temperature in Celsius
- `humidity`: Humidity percentage
- `N`, `P`, `K`: Optional nutrient levels

## 🔧 Configuration

The system can be configured through:
- Control thresholds
- Hysteresis values
- Report templates
- Visualization preferences

## 📊 Example Dashboard

![Dashboard Example](assets/dashboard_example.png)

## 📈 Sample Reports

![Report Example](assets/report_example.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support, please open an issue in the repository or contact the development team.

---

Made with ❤️ by the Greenhouse Control Team