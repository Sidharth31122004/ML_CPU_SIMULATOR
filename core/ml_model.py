"""
Machine Learning Model for Algorithm Prediction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple

class SchedulerMLModel:
    """Random Forest model for predicting best scheduling algorithm"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_training_data(self, results_list: list) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from multiple simulation runs.
        
        Args:
            results_list: List of result sets from different process configurations
            
        Returns:
            Tuple of (features DataFrame, target labels array)
        """
        all_data = []
        
        for results in results_list:
            for result in results:
                metrics = result['metrics']
                all_data.append({
                    'avg_waiting_time': metrics['avg_waiting_time'],
                    'avg_turnaround_time': metrics['avg_turnaround_time'],
                    'avg_response_time': metrics['avg_response_time'],
                    'max_waiting_time': metrics['max_waiting_time'],
                    'throughput': metrics['throughput'],
                    'cpu_utilization': metrics['cpu_utilization'],
                    'algorithm': result['name']
                })
        
        df = pd.DataFrame(all_data)
        self.feature_columns = ['avg_waiting_time', 'avg_turnaround_time', 'avg_response_time', 
                               'max_waiting_time', 'throughput', 'cpu_utilization']
        X = df[self.feature_columns]
        y = df['algorithm']
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Training report with accuracy and scores
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        unique_labels_in_test = np.unique(y_test)
        unique_labels_in_pred = np.unique(y_pred)
        all_unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        
        # Create target names for the labels that appear in test set
        target_names_for_report = self.label_encoder.classes_[all_unique_labels]
        
        report = classification_report(y_test, y_pred, output_dict=True, 
                                       labels=all_unique_labels,
                                       target_names=target_names_for_report)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=all_unique_labels)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'class_names': target_names_for_report.tolist(),
            'unique_labels': all_unique_labels.tolist()
        }
    
    def predict(self, metrics: Dict) -> str:
        """
        Predict best algorithm for given metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Predicted best algorithm name
        """
        if not self.is_trained:
            return "Model not trained yet"
        
        features = np.array([[
            metrics['avg_waiting_time'],
            metrics['avg_turnaround_time'],
            metrics['avg_response_time'],
            metrics['max_waiting_time'],
            metrics['throughput'],
            metrics['cpu_utilization']
        ]])
        
        prediction = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        return pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
