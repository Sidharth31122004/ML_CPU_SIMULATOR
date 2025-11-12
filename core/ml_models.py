"""
Ensemble ML models for scheduling algorithm prediction
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)


class EnsembleSchedulerPredictor:
    """Ensemble model for predicting best scheduling algorithm"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_labels = None
        self.is_trained = False
    
    def prepare_training_data(self, results: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from scheduling results
        
        Args:
            results: List of SchedulingResult objects
            
        Returns:
            Tuple of (features, labels)
        """
        X = []
        y = []
        
        for result in results:
            metrics = result.metrics
            features = [
                metrics.avg_waiting_time,
                metrics.avg_turnaround_time,
                metrics.avg_response_time,
                metrics.cpu_utilization,
                metrics.throughput,
                metrics.avg_context_switches,
                metrics.response_variance,
                metrics.tail_latency_p95,
                metrics.tail_latency_p99,
            ]
            
            X.append(features)
            y.append(result.algorithm.value)
        
        self.feature_names = [
            'avg_waiting_time', 'avg_turnaround_time', 'avg_response_time',
            'cpu_utilization', 'throughput', 'avg_context_switches',
            'response_variance', 'tail_latency_p95', 'tail_latency_p99'
        ]
        self.class_labels = sorted(list(set(y)))
        
        return np.array(X), np.array(y)
    
    def train(self, results: List, test_size: float = 0.2):
        """
        Train ensemble models
        
        Args:
            results: List of SchedulingResult objects
            test_size: Proportion of data for testing
        """
        X, y = self.prepare_training_data(results)
        
        # Encode labels
        label_encoding = {label: idx for idx, label in enumerate(self.class_labels)}
        y_encoded = np.array([label_encoding[label] for label in y])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train all models
        for model_name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            logger.info(f"{model_name} trained successfully")
        
        # Get predictions for evaluation
        rf_pred = self.models['random_forest'].predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, rf_pred)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        unique_labels = np.unique(y_test)
        target_names_filtered = [self.class_labels[i] for i in unique_labels]
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, rf_pred, 
                                                          labels=unique_labels,
                                                          target_names=target_names_filtered),
            'confusion_matrix': confusion_matrix(y_test, rf_pred, labels=unique_labels),
            'test_size': len(X_test)
        }
    
    def predict(self, metrics: Dict) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Predict best algorithm and get probabilities
        
        Args:
            metrics: Dictionary of scheduling metrics
            
        Returns:
            Tuple of (best_algorithm, [(algorithm, probability), ...])
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Prepare input
        features = np.array([[
            metrics.get('avg_waiting_time', 0),
            metrics.get('avg_turnaround_time', 0),
            metrics.get('avg_response_time', 0),
            metrics.get('cpu_utilization', 0),
            metrics.get('throughput', 0),
            metrics.get('avg_context_switches', 0),
            metrics.get('response_variance', 0),
            metrics.get('tail_latency_p95', 0),
            metrics.get('tail_latency_p99', 0),
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from random forest
        prediction = self.models['random_forest'].predict(features_scaled)[0]
        probabilities = self.models['random_forest'].predict_proba(features_scaled)[0]
        
        best_algorithm = self.class_labels[prediction]
        
        # Sort by probability
        prob_list = sorted(
            zip(self.class_labels, probabilities),
            key=lambda x: x[1],
            reverse=True
        )
        
        return best_algorithm, prob_list
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from random forest"""
        importances = self.models['random_forest'].feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'class_labels': self.class_labels,
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.class_labels = data['class_labels']
            self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
