import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    '''
    Stores the file paths for the raw, training, and testing datasets
    generated during data ingestion.
    '''
    # Build a platform-independent file path.
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    """
    Handles the data ingestion process, including loading, splitting, and saving the dataset.
    """

    def __init__(self):
        # Initialize the data ingestion configuration.
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Loads the dataset, creates the artifacts directory, splits the data into training and
        testing sets, saves the files, and returns their paths.
        """
        logging.info("Entered the data ingestion method/component")
        try:
            df=pd.read_csv("notebook/data/StudentsPerformance.csv")
            logging.info("Read the dataset as DataFrame")

            # Create the artifacts directory if it doesn't already exist.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Save a copy of the original dataset before any preprocessing.
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split intiated")
            # Split the dataset into training and testing sets(80:20).
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            # Save the training dataset.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)

            # Save the testing dataset.
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e,sys)

# Run the data ingestion pipeline only when this file is executed directly.
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    # Initialize transformer and apply preprocessing to ingested train/test splits.
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)