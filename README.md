# main.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import logging
import random
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class LogAnalyzer:
    def __init__(self):
        self.log_data = pd.DataFrame()
        self.anomalies = pd.DataFrame()
        
    def generate_sample_logs(self, num_records=10000):
        """Generate synthetic log data for demonstration"""
        logging.info("Generating sample log data")
        
        timestamps = [datetime.now() - timedelta(minutes=random.randint(0, 1440)) 
                      for _ in range(num_records)]
        
        log_levels = ['INFO'] * 85 + ['WARNING'] * 10 + ['ERROR'] * 5
        services = ['web-server', 'auth-service', 'payment-gateway', 'database', 'api-gateway']
        
        data = {
            'timestamp': timestamps,
            'service': [random.choice(services) for _ in range(num_records)],
            'log_level': [random.choice(log_levels) for _ in range(num_records)],
            'response_time': [random.expovariate(1/50) for _ in range(num_records)],
            'status_code': [random.choice([200]*90 + [404]*5 + [500]*5) for _ in range(num_records)],
            'message': ['Request processed'] * num_records
        }
        
        self.log_data = pd.DataFrame(data)
        
        # Introduce anomalies
        anomaly_indices = random.sample(range(num_records), int(num_records*0.05))
        self.log_data.loc[anomaly_indices, 'response_time'] = np.random.uniform(500, 1000, len(anomaly_indices))
        self.log_data.loc[anomaly_indices, 'status_code'] = 500
        self.log_data.loc[anomaly_indices, 'log_level'] = 'ERROR'
        
        logging.info(f"Generated {len(self.log_data)} log records")
        return self.log_data
    
    def detect_anomalies(self):
        """Detect anomalies using Isolation Forest algorithm"""
        logging.info("Starting anomaly detection")
        
        # Feature engineering
        features = self.log_data.copy()
        features['hour'] = features['timestamp'].dt.hour
        features['status_code'] = features['status_code'].astype(int)
        
        # One-hot encoding
        features = pd.get_dummies(features, columns=['log_level', 'service'])
        
        # Train Isolation Forest model
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(features.select_dtypes(include=np.number))
        
        # Add predictions to original data
        self.log_data['is_anomaly'] = model.predict(features.select_dtypes(include=np.number))
        self.log_data['is_anomaly'] = self.log_data['is_anomaly'].map({1: 0, -1: 1})
        
        # Extract anomalies
        self.anomalies = self.log_data[self.log_data['is_anomaly'] == 1]
        logging.info(f"Detected {len(self.anomalies)} anomalies")
        return self.anomalies
    
    def save_to_sql(self, db_name='logs.db'):
        """Store results in SQLite database"""
        logging.info("Saving data to SQL database")
        engine = create_engine(f'sqlite:///{db_name}')
        
        self.log_data.to_sql('log_entries', engine, if_exists='replace', index=False)
        self.anomalies.to_sql('detected_anomalies', engine, if_exists='replace', index=False)
        
        logging.info(f"Data saved to {db_name}")
        
    def export_for_powerbi(self, filename='log_analysis.csv'):
        """Export data for Power BI visualization"""
        logging.info("Exporting data for Power BI")
        
        # Prepare dataset with key metrics
        report_data = self.log_data.copy()
        report_data['hour'] = report_data['timestamp'].dt.hour
        report_data['date'] = report_data['timestamp'].dt.date
        
        # Export to CSV
        report_data.to_csv(filename, index=False)
        logging.info(f"Data exported to {filename}")
        
    def run_pipeline(self):
        """Execute full analysis pipeline"""
        self.generate_sample_logs()
        self.detect_anomalies()
        self.save_to_sql()
        self.export_for_powerbi()
        logging.info("Processing pipeline completed")

if __name__ == "__main__":
    analyzer = LogAnalyzer()
    analyzer.run_pipeline()
    print("Processing completed! Check the following outputs:")
    print("- SQLite database: logs.db")
    print("- Power BI data source: log_analysis.csv")
    print("- Log file: app.log")
