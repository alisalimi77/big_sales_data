import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    A class for data preprocessing tasks such as object to numeric transformation,
    handling null values, standardization, and normalization.
    """

    def __init__(self):
        pass
    
    def object_to_numeric(self, dataset):
        """
        Convert object data to numerical data in the dataset.

        Parameters:
            dataset (pandas.DataFrame): The input dataset containing object and numerical data.

        Returns:
            pandas.DataFrame: The transformed dataset with numerical data.
        """
        # Find object data columns in the dataset
        object_columns = dataset.select_dtypes(include=['object']).columns.tolist()
        
        # Create an instance of ObjectToNumericTransformer
        transformer = self.ObjectToNumericTransformer(object_columns)
        
        # Transform the dataset
        transformed_dataset = transformer.fit_transform(dataset)
        
        return transformed_dataset
    
    def handle_null_values(self, dataset, fill_value):
        """
        Handle null values in the dataset by filling them with a specified value.

        Parameters:
            dataset (pandas.DataFrame): The input dataset.
            fill_value (object): The value to fill in place of null values.

        Returns:
            pandas.DataFrame: The dataset with null values filled.
        """
        # Fill null values in the dataset with the specified fill value
        filled_dataset = dataset.fillna(fill_value)
        return filled_dataset
    
    def standardize_dataset(self, dataset):
        """
        Standardize the dataset by scaling the numerical features to have zero mean and unit variance.

        Parameters:
            dataset (pandas.DataFrame): The input dataset.

        Returns:
            pandas.DataFrame: The standardized dataset.
        """
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(dataset)
        standardized_df = pd.DataFrame(standardized_data, columns=dataset.columns)
        return standardized_df
    
    def normalize_dataset(self, dataset):
        """
        Normalize the dataset by scaling the numerical features to a fixed range (e.g., [0, 1]).

        Parameters:
            dataset (pandas.DataFrame): The input dataset.

        Returns:
            pandas.DataFrame: The normalized dataset.
        """
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(dataset)
        normalized_df = pd.DataFrame(normalized_data, columns=dataset.columns)
        return normalized_df
    
    class ObjectToNumericTransformer:
        """
        A helper class to perform object to numeric transformation using label encoding.
        """

        def __init__(self, object_columns):
            """
            Initialize the ObjectToNumericTransformer.

            Parameters:
                object_columns (list): List of object data columns in the dataset.
            """
            self.object_columns = object_columns
            self.label_encoders = {}
        
        def fit_transform(self, dataset):
            """
            Fit the transformer to the dataset and transform the object data columns.

            Parameters:
                dataset (pandas.DataFrame): The input dataset.

            Returns:
                pandas.DataFrame: The transformed dataset.
            """
            # Copy the original dataset to avoid modifying the original data
            transformed_dataset = dataset.copy()
            
            # Iterate over the object columns
            for column in self.object_columns:
                # Convert object data to numerical data
                transformed_dataset[column] = self._encode_labels(transformed_dataset[column])
            
            return transformed_dataset
        
        def _encode_labels(self, column):
            """
            Encode labels of a column using label encoding.

            Parameters:
                column (pandas.Series): The column to be encoded.

            Returns:
                pandas.Series: The encoded column.
            """
            if column.name not in self.label_encoders:
                # Create a new label encoder for the column
                label_encoder = pd.factorize(column)
                self.label_encoders[column.name] = label_encoder
            
            # Use the existing label encoder to transform the column
            encoded_column = self.label_encoders[column.name][0]
            
            return encoded_column
