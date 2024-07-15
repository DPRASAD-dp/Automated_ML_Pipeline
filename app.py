import streamlit as st
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging 

def main():
    st.title("Automated Machine Learning Pipeline")

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    problem_type = st.selectbox("Select Problem Type", ["classification", "regression", "clustering"])
    target_column_name = st.text_input("Enter the Target Column Name")

    if uploaded_file is not None and problem_type and target_column_name:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(df)

        if st.button("Run Analysis"):
            try:
                # Data Ingestion
                data_ingestion = DataIngestion()
                st.info("Starting data ingestion...")
                train_data, test_data, eda_report_path = data_ingestion.initiate_data_ingestion(df)
                st.success("Data Ingestion Completed")

                # Display EDA Report
                with open(eda_report_path, "r") as f:
                    st.download_button("Download EDA Report", f, file_name="eda_report.html")

                # Data Transformation
                data_transformation = DataTransformation()
                st.info("Starting data transformation...")
                train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data, problem_type, target_column_name)
                st.success("Data Transformation Completed")

                # Model Training
                model_trainer = ModelTrainer()
                st.info("Starting model training...")
                try:
                    best_model_name, best_model_score, model_report = model_trainer.initiate_model_trainer(train_arr, test_arr, problem_type)

                    st.success("Model Training Completed")
                    st.subheader("Model Comparison")

                    # Display model comparison
                    if problem_type in ['regression', 'classification']:
                        comparison_df = pd.DataFrame.from_dict(model_report, orient='index', columns=['Train Score', 'Test Score'])
                    else:  # clustering
                        comparison_df = pd.DataFrame.from_dict(model_report, orient='index', columns=['Silhouette Score'])
                    st.table(comparison_df)

                    st.subheader("Best Model")
                    st.write(f"Best Model: {best_model_name}")
                    st.write(f"Best Model Score: {best_model_score}")
                except Exception as e:
                    st.error(f"An error occurred during model training: {str(e)}")
                    logging.error(f"Error in model training: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
