"""
Master training pipeline for Loan Performance Intelligence Engine
Orchestrates all phases: data loading, profiling, validation, feature engineering, modeling
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.utils.logging import setup_logging, get_logger
from src.data.loader import DataLoader
from src.data.schema import SchemaDiscovery
from src.data.profiling import DataProfiler, DataQualityReporter
from src.data.validation import ValidationRulesEngine
from src.data.drift import DriftDetector
from src.data.reconciliation import ServicerReconciliation
from src.features.feature_pipeline import FeaturePipeline, PersistedFeaturePipeline, TimeAwareSplit, LeakageDetector
from src.models.trainer import train_phase_4
from src.survival.trainer import train_phase_5

logger = get_logger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self):
        """Initialize pipeline with settings"""
        self.settings = get_settings()
        setup_logging(
            log_level=self.settings.log_level,
            log_dir=self.settings.logs_dir
        )
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "phases_completed": []
        }
    
    def run(self):
        """Execute complete pipeline"""
        try:
            logger.info("="*70)
            logger.info("LOAN PERFORMANCE INTELLIGENCE ENGINE - TRAINING PIPELINE")
            logger.info("="*70)
            
            # Phase 1: Data Discovery
            logger.info("\n" + "="*70)
            logger.info("PHASE 1: DATA DISCOVERY & SCHEMA VALIDATION")
            logger.info("="*70)
            train_df, test_df, static_df = self.load_data()
            self.metadata["phases_completed"].append("data_discovery")
            
            # Phase 2: Data Profiling & Validation
            logger.info("\n" + "="*70)
            logger.info("PHASE 2: DATA PROFILING & QUALITY")
            logger.info("="*70)
            self.profile_data(train_df)
            self.validate_data(train_df)
            self.metadata["phases_completed"].append("profiling_validation")
            
            # Drift analysis
            logger.info("\nAnalyzing data drift between train and test...")
            if test_df is not None and len(test_df) > 0:
                drift_report = DriftDetector.detect_dataset_drift(train_df, test_df)
                DriftDetector.save_drift_report(drift_report, self.settings.outputs_dir)
                logger.info(f"Drift risk level: {drift_report['overall_drift_risk']}")
            
            # Reconciliation
            logger.info("\nApplying reconciliation checks...")
            train_df = ServicerReconciliation.add_reconciliation_flags(train_df)
            
            # Phase 3: Feature Engineering & Time-Aware Split
            logger.info("\n" + "="*70)
            logger.info("PHASE 3: FEATURE ENGINEERING & TIME-AWARE SPLIT")
            logger.info("="*70)
            
            # Build features
            feature_df, feature_metadata = FeaturePipeline.build_feature_matrix(train_df)
            logger.info(f"Generated {len(feature_df.columns)} features")
            PersistedFeaturePipeline(feature_df.columns.tolist()).save(
                self.settings.outputs_dir / "models" / "preprocessing_pipeline.joblib"
            )
            
            # Time-aware split
            train_split, val_split, split_metadata = TimeAwareSplit.split_by_reporting_month(
                train_df,
                validation_fraction=0.2
            )
            
            # Leakage detection
            if "next_12m_default_flag" in train_df.columns:
                leak_check = LeakageDetector.detect_target_in_features(
                    feature_df, train_df["next_12m_default_flag"], "default"
                )
                logger.info(f"Leakage check (target in features): {leak_check}")
            
            self.metadata["phases_completed"].append("feature_engineering_split")
            
            # Phase 4: Baseline & Improved Model Training
            logger.info("\n" + "="*70)
            logger.info("PHASE 4: BASELINE & IMPROVED MODEL TRAINING")
            logger.info("="*70)
            
            models, comparator = train_phase_4(
                train_split, val_split, feature_df, self.settings
            )
            
            self.metadata["phases_completed"].append("model_training")
            
            # Phase 5: Survival & Transition Modeling
            logger.info("\n" + "="*70)
            logger.info("PHASE 5: SURVIVAL & TRANSITION MODELING")
            logger.info("="*70)
            
            survival_models, transition_models, competing_risks = train_phase_5(
                train_split, val_split, self.settings
            )
            
            self.metadata["phases_completed"].append("survival_transition_modeling")
            
            logger.info("\n" + "="*70)
            logger.info("PHASES 5-13: ADVANCED FEATURES")
            logger.info("="*70)
            logger.info("- Phase 5: Survival/transition models")
            logger.info("- Phase 6: Anomaly detection")
            logger.info("- Phase 7: SHAP explainability")
            logger.info("- Phase 8: Scenario simulation")
            logger.info("- Phase 9: LLM copilot & RAG")
            logger.info("- Phase 10: Reports & submission")
            logger.info("- Phase 11: Streamlit dashboard")
            logger.info("- Phase 12: FastAPI")
            logger.info("- Phase 13: Tests & validation")
            
            # Save metadata
            metadata_path = self.settings.outputs_dir / "pipeline_metadata.json"
            self.metadata["training_records"] = len(train_split)
            self.metadata["validation_records"] = len(val_split)
            self.metadata["feature_count"] = len(feature_df.columns)
            
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE EXECUTION COMPLETE")
            logger.info(f"Metadata saved to {metadata_path}")
            logger.info("="*70)
            
            return True
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False
    
    def load_data(self) -> tuple:
        """Load training and test data"""
        logger.info("Loading data from data/raw/...")
        
        # Load training data
        train_path = self.settings.data_raw_dir / "loan_monthly_performance_train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        train_df = DataLoader.load_csv(train_path)
        train_df = DataLoader.clean_dataset(train_df)
        logger.info(f"Loaded training data: {len(train_df)} rows × {len(train_df.columns)} columns")
        
        # Schema discovery
        schema_report = SchemaDiscovery.discover_schema(train_df, "train.csv")
        SchemaDiscovery.save_report(schema_report, self.settings.outputs_dir)
        logger.info(f"Schema discovery: {len(schema_report.found_columns)} / {len(schema_report.expected_columns)} expected fields")
        
        # Load test data
        test_df = None
        test_path = self.settings.data_raw_dir / "loan_monthly_performance_test.csv"
        if test_path.exists():
            test_df = DataLoader.load_csv(test_path)
            test_df = DataLoader.clean_dataset(test_df)
            logger.info(f"Loaded test data: {len(test_df)} rows")
        
        # Load static attributes if available
        static_df = None
        static_path = self.settings.data_raw_dir / "loan_static_attributes.csv"
        if static_path.exists():
            static_df = DataLoader.load_csv(static_path)
            logger.info(f"Loaded static attributes: {len(static_df)} rows")
        
        return train_df, test_df, static_df
    
    def profile_data(self, df: pd.DataFrame) -> None:
        """Profile data distributions and quality"""
        logger.info("Profiling data...")
        
        # Statistical profiling
        profiles = DataProfiler.profile_dataset(df)
        DataQualityReporter.save_profile_report(profiles, self.settings.outputs_dir)
        logger.info(f"Generated profiles for {len(profiles)} columns")
        
        # Quality scoring
        scores_df, summary = DataProfiler.calculate_batch_quality_scores(df)
        DataQualityReporter.save_quality_report(scores_df, summary, self.settings.outputs_dir)
        logger.info(f"Quality scores: mean={summary['average_quality_score']:.3f}")
        
        # Outlier detection
        outliers = DataProfiler.detect_outliers(df)
        logger.info(f"Outliers detected in {len(outliers)} columns")
        
        # Generate HTML report
        html_path = self.settings.outputs_dir / "data_intelligence_report.html"
        DataQualityReporter.generate_html_profile_report(profiles, html_path)
        logger.info(f"HTML report saved to {html_path}")
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate data against rules"""
        logger.info("Validating data...")
        
        # Load validation rules
        rules_path = self.settings.data_raw_dir / "validation_rules.json"
        rules = ValidationRulesEngine.load_rules(rules_path if rules_path.exists() else None)
        
        # Validate
        failures = ValidationRulesEngine.validate_dataset(df, rules)
        ValidationRulesEngine.save_validation_report(failures, self.settings.outputs_dir)
        logger.info(f"Validation complete: {len(failures)} violations")


def main():
    """Main entry point"""
    pipeline = TrainingPipeline()
    success = pipeline.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
