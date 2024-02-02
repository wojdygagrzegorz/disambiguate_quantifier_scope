from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseParameters(BaseModel):
    name: str = Field(description="Name of model")
    save_model_path: Optional[str] = Field(default=None, description="Saving path for model")
    load_model_path: Optional[str] = Field(default=None, description="Loading path for model")

class DataParameters(BaseModel):
    input_columns: List[str] = Field(description="Input columns")
    output_column: str = Field(description="Output column")

class RegressionConfiguarion(BaseModel):
    base_parameters: BaseParameters
    data_parameters: DataParameters
    encoder_parameters: Dict[str, Union[str, int, float]] = Field(description="Encoder parameters")
    classifier_parameters: Dict[str, Union[str, int, float]] = Field(description="Classifier parameters")

    @classmethod
    def load_from_file(cls, config_path):
        with open(config_path, 'r') as config_file:
            configuration = cls(**yaml.safe_load(config_file))
        return configuration

class HerBERTConfiguration(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    base_parameters: BaseParameters
    data_parameters: DataParameters
    model_parameters: Dict[str, Union[str, int, float]] = Field(description="Model parameters")
    herbert_parameters: Dict[str, Union[str, int, float]] = Field(description="HerBERT parameters")


    @classmethod
    def load_from_file(cls, config_path):
        with open(config_path, 'r') as config_file:
            configuration = cls(**yaml.safe_load(config_file))
        return configuration