# Automated_ML_Pipeline

This project implements an automated machine learning pipeline capable of handling classification, regression, and clustering tasks. It provides a streamlined process for data ingestion, transformation, model training, and evaluation.

## Features

- Supports classification, regression, and clustering problems
- Automated data ingestion and preprocessing
- Exploratory Data Analysis (EDA) report generation
- Automated feature engineering and selection
- Model training with hyperparameter tuning
- Model evaluation and comparison
- Interactive web interface using Streamlit

## Installation

1. Clone the repository:
git clone [https://github.com/DPRASAD-dp/Automated-ML-Pipeline.git](https://github.com/DPRASAD-dp/Automated_ML_Pipeline.git)
cd automated-ml-project

3. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
  
4. Install the required packages:
pip install -r requirements.txt

## Usage

1. Run the Streamlit app:
streamlit run app.py

3. Upload your CSV file, select the problem type, and specify the target column.

4. Click "Run Analysis" to start the automated ML pipeline.

5. View the results, including the EDA report, model comparisons, and the best performing model.

## Project Structure

- `src/`: Contains the main source code
- `components/`: Individual pipeline components (data ingestion, transformation, model training)
- `exception.py`: Custom exception handling
- `logger.py`: Logging configuration
- `utils.py`: Utility functions
- `app.py`: Streamlit web application
- `setup.py`: Project setup and package information
- `requirements.txt`: List of required Python packages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
