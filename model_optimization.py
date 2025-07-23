"""
Model Optimization and Ensemble System
Advanced machine learning pipeline with hyperparameter tuning and ensemble methods
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    """Advanced model optimization and ensemble system"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_models = {}
        self.ensemble_model = None
        self.optimization_results = {}
        
    def prepare_data(self, features_df: pd.DataFrame, target_column: str = 'disengagement_risk_composite',
                    threshold: float = 0.5) -> tuple:
        """Prepare data for modeling with advanced preprocessing"""
        
        # Handle target variable
        if target_column in features_df.columns:
            y = (features_df[target_column] > threshold).astype(int)
        else:
            # Create synthetic target based on multiple indicators
            risk_indicators = []
            for col in features_df.columns:
                if any(keyword in col.lower() for keyword in ['gap', 'void', 'decline', 'inconsist']):
                    if features_df[col].dtype in ['int64', 'float64']:
                        risk_indicators.append(col)
            
            if risk_indicators:
                risk_scores = features_df[risk_indicators].mean(axis=1)
                y = (risk_scores > risk_scores.quantile(0.7)).astype(int)
            else:
                # Fallback: random classification for demonstration
                y = np.random.binomial(1, 0.3, len(features_df))
        
        # Prepare feature matrix
        X = features_df.select_dtypes(include=[np.number]).copy()
        
        # Remove target if it exists in features
        if target_column in X.columns:
            X = X.drop(target_column, axis=1)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove constant features
        constant_features = X.columns[X.std() == 0]
        if len(constant_features) > 0:
            X = X.drop(constant_features, axis=1)
            print(f"Removed {len(constant_features)} constant features")
        
        print(f"Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target distribution: {np.bincount(y)} (class 0: {np.mean(y==0):.1%}, class 1: {np.mean(y==1):.1%})")
        
        return X, y
    
    def feature_selection(self, X: pd.DataFrame, y: np.array, method: str = 'mutual_info', 
                         k: int = 20) -> pd.DataFrame:
        """Advanced feature selection"""
        
        print(f"Performing feature selection using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        else:
            print("Unknown method, using mutual_info")
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        self.feature_selector = selector
        
        print(f"Selected {len(selected_features)} features: {list(selected_features)}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: np.array) -> dict:
        """Hyperparameter optimization for multiple models"""
        
        print("Starting hyperparameter optimization...")
        
        # Define models and parameter grids
        models_params = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'SVM': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None]
                }
            },
            'NeuralNetwork': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Scale features for algorithms that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        optimized_models = {}
        
        for name, config in models_params.items():
            print(f"Optimizing {name}...")
            
            try:
                # Use scaled data for SVM and Neural Networks
                if name in ['SVM', 'NeuralNetwork']:
                    X_cv = X_train_scaled
                else:
                    X_cv = X_train
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_cv, y_train)
                
                optimized_models[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                print(f"{name} - Best CV F1: {grid_search.best_score_:.3f}")
                
            except Exception as e:
                print(f"Error optimizing {name}: {e}")
                # Fallback to default parameters
                optimized_models[name] = {
                    'model': config['model'],
                    'best_params': {},
                    'best_score': 0.0,
                    'cv_results': {}
                }
        
        self.best_models = optimized_models
        return optimized_models
    
    def create_ensemble(self, X_train: pd.DataFrame, y_train: np.array) -> VotingClassifier:
        """Create ensemble model from best individual models"""
        
        print("Creating ensemble model...")
        
        # Select top performing models
        model_scores = [(name, info['best_score']) for name, info in self.best_models.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3-5 models for ensemble
        top_models = model_scores[:5]
        
        estimators = []
        for name, score in top_models:
            if score > 0:  # Only include models that were successfully trained
                estimators.append((name, self.best_models[name]['model']))
        
        if len(estimators) < 2:
            print("Not enough models for ensemble, using best single model")
            best_model_name = top_models[0][0] if top_models else 'RandomForest'
            return self.best_models.get(best_model_name, {}).get('model', RandomForestClassifier())
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        # Fit ensemble
        X_ensemble = X_train.copy()
        
        # Use scaled data if ensemble contains models that need it
        model_names = [name for name, _ in estimators]
        if any(name in ['SVM', 'NeuralNetwork'] for name in model_names):
            X_ensemble = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        ensemble.fit(X_ensemble, y_train)
        
        self.ensemble_model = ensemble
        
        print(f"Ensemble created with {len(estimators)} models: {[name for name, _ in estimators]}")
        
        return ensemble
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: np.array) -> dict:
        """Comprehensive model evaluation"""
        
        print("Evaluating models...")
        
        results = {}
        
        # Prepare test data
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Evaluate individual models
        for name, model_info in self.best_models.items():
            model = model_info['model']
            
            try:
                # Use appropriate data (scaled vs unscaled)
                if name in ['SVM', 'NeuralNetwork']:
                    X_eval = X_test_scaled
                else:
                    X_eval = X_test
                
                # Predictions
                y_pred = model.predict(X_eval)
                y_pred_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metrics
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None and len(set(y_test)) > 1 else 0,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = {
                    'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0,
                    'predictions': np.zeros_like(y_test), 'probabilities': None
                }
        
        # Evaluate ensemble
        if self.ensemble_model:
            try:
                # Determine if we need scaled data for ensemble
                estimator_names = [name for name, _ in self.ensemble_model.estimators]
                X_ensemble_eval = X_test_scaled if any(name in ['SVM', 'NeuralNetwork'] for name in estimator_names) else X_test
                
                y_pred_ensemble = self.ensemble_model.predict(X_ensemble_eval)
                y_pred_proba_ensemble = self.ensemble_model.predict_proba(X_ensemble_eval)[:, 1]
                
                results['Ensemble'] = {
                    'accuracy': accuracy_score(y_test, y_pred_ensemble),
                    'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
                    'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
                    'f1': f1_score(y_test, y_pred_ensemble, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba_ensemble) if len(set(y_test)) > 1 else 0,
                    'predictions': y_pred_ensemble,
                    'probabilities': y_pred_proba_ensemble
                }
                
            except Exception as e:
                print(f"Error evaluating ensemble: {e}")
        
        self.optimization_results = results
        return results
    
    def plot_evaluation_results(self, results: dict, y_test: np.array):
        """Create comprehensive evaluation plots"""
        
        # Performance metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Metrics comparison
        metrics_data = []
        for model in model_names:
            for metric in metrics:
                metrics_data.append({
                    'Model': model,
                    'Metric': metric,
                    'Score': results[model][metric]
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        pivot_metrics = metrics_df.pivot(index='Model', columns='Metric', values='Score')
        
        sns.heatmap(pivot_metrics, annot=True, cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('Model Performance Heatmap')
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Models')
        
        # 2. F1 Score comparison
        f1_scores = [results[model]['f1'] for model in model_names]
        axes[1].bar(model_names, f1_scores, color='skyblue', alpha=0.7)
        axes[1].set_title('F1 Score Comparison')
        axes[1].set_ylabel('F1 Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 3. ROC Curves
        if len(set(y_test)) > 1:
            for model in model_names:
                if results[model]['probabilities'] is not None:
                    fpr, tpr, _ = roc_curve(y_test, results[model]['probabilities'])
                    auc_score = results[model]['roc_auc']
                    axes[2].plot(fpr, tpr, label=f'{model} (AUC={auc_score:.3f})')
            
            axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[2].set_xlabel('False Positive Rate')
            axes[2].set_ylabel('True Positive Rate')
            axes[2].set_title('ROC Curves')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # 4. Confusion matrices for best model
        best_model = max(model_names, key=lambda x: results[x]['f1'])
        cm = confusion_matrix(y_test, results[best_model]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[3])
        axes[3].set_title(f'Confusion Matrix - {best_model}')
        axes[3].set_xlabel('Predicted')
        axes[3].set_ylabel('Actual')
        
        # 5. Model complexity vs performance
        f1_scores = [results[model]['f1'] for model in model_names]
        complexity_scores = [hash(model) % 100 for model in model_names]  # Simplified complexity
        
        axes[4].scatter(complexity_scores, f1_scores, s=100, alpha=0.7)
        for i, model in enumerate(model_names):
            axes[4].annotate(model, (complexity_scores[i], f1_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[4].set_xlabel('Model Complexity (Simplified)')
        axes[4].set_ylabel('F1 Score')
        axes[4].set_title('Performance vs Complexity')
        
        # 6. Feature importance (if available)
        if 'RandomForest' in results and hasattr(self.best_models['RandomForest']['model'], 'feature_importances_'):
            rf_model = self.best_models['RandomForest']['model']
            if hasattr(rf_model, 'feature_importances_') and len(rf_model.feature_importances_) > 0:
                # Get feature names (assuming they match the training data)
                if self.feature_selector and hasattr(self.feature_selector, 'get_support'):
                    feature_names = [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                # Plot top 10 features
                top_features = importance_df.tail(10)
                axes[5].barh(range(len(top_features)), top_features['importance'])
                axes[5].set_yticks(range(len(top_features)))
                axes[5].set_yticklabels(top_features['feature'])
                axes[5].set_xlabel('Importance')
                axes[5].set_title('Top 10 Feature Importances (Random Forest)')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: dict) -> str:
        """Generate comprehensive model performance report"""
        
        report = []
        report.append("=" * 60)
        report.append("MODEL OPTIMIZATION AND EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Best performing model
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        report.append(f"BEST PERFORMING MODEL: {best_model}")
        report.append(f"F1 Score: {results[best_model]['f1']:.3f}")
        report.append("")
        
        # Model comparison table
        report.append("MODEL PERFORMANCE COMPARISON:")
        report.append("-" * 60)
        report.append(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
        report.append("-" * 60)
        
        for model_name in sorted(results.keys()):
            metrics = results[model_name]
            report.append(f"{model_name:<15} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                         f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['roc_auc']:<10.3f}")
        
        report.append("")
        
        # Best hyperparameters
        if hasattr(self, 'best_models') and self.best_models:
            report.append("OPTIMIZED HYPERPARAMETERS:")
            report.append("-" * 40)
            
            for model_name, model_info in self.best_models.items():
                if model_info.get('best_params'):
                    report.append(f"\n{model_name}:")
                    for param, value in model_info['best_params'].items():
                        report.append(f"  {param}: {value}")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        # Generate recommendations based on results
        if results[best_model]['f1'] > 0.8:
            report.append("✓ Excellent model performance achieved")
        elif results[best_model]['f1'] > 0.6:
            report.append("• Good model performance, consider feature engineering improvements")
        else:
            report.append("⚠ Model performance needs improvement - consider:")
            report.append("  - More data collection")
            report.append("  - Advanced feature engineering") 
            report.append("  - Alternative algorithms")
        
        if 'Ensemble' in results and results['Ensemble']['f1'] > results[best_model]['f1']:
            report.append("✓ Ensemble model outperforms individual models")
        
        return "\n".join(report)

# Example usage and pipeline
if __name__ == "__main__":
    # Load engineered features (from previous step)
    try:
        features_df = pd.read_csv("engineered_features.csv")
    except FileNotFoundError:
        print("Engineered features not found. Please run advanced_feature_engineering.py first.")
        # For demonstration, create synthetic data
        from advanced_feature_engineering import AdvancedFeatureEngineer
        import pandas as pd
        
        data = pd.read_csv("synthetic_user_behavior_log.csv")
        engineer = AdvancedFeatureEngineer()
        features_df = engineer.engineer_features(data)
    
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Prepare data
    X, y = optimizer.prepare_data(features_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature selection
    X_train_selected = optimizer.feature_selection(X_train, y_train, method='mutual_info', k=15)
    X_test_selected = X_test[X_train_selected.columns]
    
    # Optimize models
    optimized_models = optimizer.optimize_hyperparameters(X_train_selected, y_train)
    
    # Create ensemble
    ensemble = optimizer.create_ensemble(X_train_selected, y_train)
    
    # Evaluate all models
    results = optimizer.evaluate_models(X_test_selected, y_test)
    
    # Display results
    print("\nOptimization Results:")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:15} - F1: {metrics['f1']:.3f}, Accuracy: {metrics['accuracy']:.3f}, "
              f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
    
    # Generate and display report
    report = optimizer.generate_report(results)
    print("\n" + report)
    
    # Create visualizations
    optimizer.plot_evaluation_results(results, y_test)
    
    # Save results
    results_summary = pd.DataFrame({
        'Model': list(results.keys()),
        'F1_Score': [results[model]['f1'] for model in results.keys()],
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'ROC_AUC': [results[model]['roc_auc'] for model in results.keys()]
    })
    
    results_summary.to_csv('model_optimization_results.csv', index=False)
    print(f"\nResults saved to model_optimization_results.csv")
