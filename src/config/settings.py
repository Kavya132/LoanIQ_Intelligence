"""
Configuration management for the Loan Intelligence Engine
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """Main settings for the application"""
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_raw_dir: Path = Field(default=None)
    data_processed_dir: Path = Field(default=None)
    models_dir: Path = Field(default=None)
    reports_dir: Path = Field(default=None)
    outputs_dir: Path = Field(default=None)
    logs_dir: Path = Field(default=None)
    
    # Data mode
    data_mode: str = Field(default="demo")
    
    # Training
    random_seed: int = Field(default=42)
    test_size: float = Field(default=0.2)
    validation_months: int = Field(default=6)
    time_aware_split: bool = Field(default=True)
    
    # Models
    primary_model: str = Field(default="catboost")
    catboost_iterations: int = Field(default=500)
    catboost_learning_rate: float = Field(default=0.05)
    catboost_depth: int = Field(default=7)
    
    # Anomaly detection
    anomaly_contamination: float = Field(default=0.02)
    
    # Calibration
    calibration_enabled: bool = Field(default=True)
    calibration_method: str = Field(default="sigmoid")
    
    # Simulation
    monte_carlo_runs: int = Field(default=1000)
    
    # Explainability
    shap_sample_size: int = Field(default=2000)
    use_shap: bool = Field(default=True)
    feature_importance_top_n: int = Field(default=20)
    
    # LLM
    llm_enabled: bool = Field(default=True)
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4-turbo")
    llm_api_key: Optional[str] = Field(default=None)
    llm_base_url: Optional[str] = Field(default=None)
    llm_max_tokens: int = Field(default=500)
    llm_temperature: float = Field(default=0.7)
    llm_grounding_enabled: bool = Field(default=True)
    
    # Logging
    log_level: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dict"""
        return {
            "catboost_iterations": self.catboost_iterations,
            "catboost_learning_rate": self.catboost_learning_rate,
            "catboost_depth": self.catboost_depth
        }


def load_yaml_config(config_path: Path = None) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def init_settings() -> Settings:
    """Initialize settings with defaults and environment overrides"""
    # Load YAML config first
    yaml_config = load_yaml_config()
    
    # Flatten nested YAML config for pydantic
    flat_config = {}
    
    # Data config
    if "data" in yaml_config:
        flat_config["data_raw_dir"] = yaml_config["data"].get("raw_dir", "data/raw")
        flat_config["data_processed_dir"] = yaml_config["data"].get("processed_dir", "data/processed")
        flat_config["data_mode"] = yaml_config["data"].get("mode", "demo")
    
    # Training config
    if "training" in yaml_config:
        flat_config["random_seed"] = yaml_config["training"].get("random_seed", 42)
        flat_config["validation_months"] = yaml_config["training"].get("validation_months", 6)
        flat_config["time_aware_split"] = yaml_config["training"].get("time_aware_split", True)
    
    # Models config
    if "models" in yaml_config:
        flat_config["primary_model"] = yaml_config["models"].get("primary", "catboost")
        if "catboost" in yaml_config["models"]:
            cb_cfg = yaml_config["models"]["catboost"]
            flat_config["catboost_iterations"] = cb_cfg.get("iterations", 500)
            flat_config["catboost_learning_rate"] = cb_cfg.get("learning_rate", 0.05)
            flat_config["catboost_depth"] = cb_cfg.get("depth", 7)
    
    # Anomaly config
    if "anomaly" in yaml_config:
        flat_config["anomaly_contamination"] = yaml_config["anomaly"].get("contamination", 0.02)
    
    # Calibration config
    if "calibration" in yaml_config:
        flat_config["calibration_enabled"] = yaml_config["calibration"].get("enabled", True)
        flat_config["calibration_method"] = yaml_config["calibration"].get("method", "sigmoid")
    
    # Simulation config
    if "simulation" in yaml_config:
        flat_config["monte_carlo_runs"] = yaml_config["simulation"].get("monte_carlo_runs", 1000)
    
    # Explainability config
    if "explainability" in yaml_config:
        flat_config["shap_sample_size"] = yaml_config["explainability"].get("shap_sample_size", 2000)
        flat_config["use_shap"] = yaml_config["explainability"].get("use_shap", True)
    
    # LLM config
    if "llm" in yaml_config:
        flat_config["llm_enabled"] = yaml_config["llm"].get("enabled", True)
        flat_config["llm_provider"] = yaml_config["llm"].get("provider", "openai")
        flat_config["llm_model"] = yaml_config["llm"].get("model", "gpt-4-turbo")
    
    # Logging config
    if "logging" in yaml_config:
        flat_config["log_level"] = yaml_config["logging"].get("level", "INFO")
    
    # Create settings object
    settings = Settings(**flat_config)
    
    # Create directories
    settings.data_raw_dir = Path(settings.data_raw_dir or settings.project_root / "data" / "raw")
    settings.data_processed_dir = Path(settings.data_processed_dir or settings.project_root / "data" / "processed")
    settings.models_dir = Path(settings.models_dir or settings.project_root / "models")
    settings.reports_dir = Path(settings.reports_dir or settings.project_root / "reports")
    settings.outputs_dir = Path(settings.outputs_dir or settings.project_root / "outputs")
    settings.logs_dir = Path(settings.logs_dir or settings.project_root / "outputs" / "logs")
    
    # Create all directories
    for dir_path in [
        settings.data_raw_dir,
        settings.data_processed_dir,
        settings.models_dir,
        settings.reports_dir,
        settings.outputs_dir,
        settings.logs_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Override with environment variables
    settings.llm_api_key = os.getenv("LLM_API_KEY", settings.llm_api_key)
    settings.llm_base_url = os.getenv("LLM_BASE_URL", settings.llm_base_url)
    
    return settings


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance"""
    global _settings
    if _settings is None:
        _settings = init_settings()
    return _settings
