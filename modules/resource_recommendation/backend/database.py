"""
Module 3 Database - Simple offline mode
Will connect to Supabase when Modules 1 & 2 are ready
"""

import json
import os
from datetime import datetime


class Module3Database:

    def __init__(self):
        self.client = None
        print("✅ Module 3 running in offline mode")

    def get_flood_results(self, event_date=None):
        """Get flood detection results from local JSON (Module 1 output)"""
        paths = [
            '../outputs/flood_detection_results.json',
            'outputs/flood_detection_results.json',
            '../../outputs/flood_detection_results.json'
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'divisions' in data:
                            return data['divisions']
                        elif isinstance(data, list):
                            return data
                        return []
                except:
                    pass
        return []

    def get_latest_summary(self):
        flood_data = self.get_flood_results()
        if not flood_data:
            return {'total_flooded': 0, 'flood_percentage': 0, 'high_risk_areas': 0}
        total = len(flood_data) if isinstance(flood_data, list) else 0
        high_risk = 0
        if isinstance(flood_data, list):
            for d in flood_data:
                if d.get('priority_label') == 'High' or d.get('priority') == 3:
                    high_risk += 1
        return {
            'total_flooded': total,
            'flood_percentage': round(total / 26 * 100, 1) if total > 0 else 0,
            'high_risk_areas': high_risk,
            'last_updated': datetime.now().isoformat()
        }


db_instance = Module3Database()

def get_flood_results():
    return db_instance.get_flood_results()

def get_latest_summary():
    return db_instance.get_latest_summary()