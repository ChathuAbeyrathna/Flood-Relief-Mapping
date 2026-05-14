# Main Data Sources for affected population estimation in Sri Lanka

- **Census of Population & Housing of SL**
https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024

- **Weather reports - DMC**
https://www.dmc.gov.lk/index.php?option=com_dmcreports&view=reports&Itemid=274&report_type_id=2&lang=en

- **Global human settlement layer**
https://human-settlement.emergency.copernicus.eu/download.php?ds=pop

- **Landscan**
https://landscan.ornl.gov/

- **VIIRS nighttime lights**
https://eogdata.mines.edu/products/vnl/?utm_source=chatgpt.com#annual_v2

- **Affected population data - Situation reports**
https://www.dmc.gov.lk/index.php?option=com_dmcreports&view=reports&report_type_id=1&Itemid=273&lang=en

- **Holidays in SL**
https://www.worldholidaysandevents.com/calendar?country=Sri%20Lanka&year=2020

- **Rainfall data**
https://www.visualcrossing.com/weather-query-builder/

- **Distance to nearest river - Copernicus**
https://www.copernicus.eu/en/access-data/dataset-distance-to-nearest-river

- **Gampaha DS administrative boundary shapefile**
https://data.humdata.org/dataset/cod-ab-lka




# Flood Detection Module
Team Trivia · University of Moratuwa · 2026

## Setup

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Set up Supabase
1. Go to supabase.com and create a free project
2. Run this SQL in the SQL Editor:
```sql
CREATE TABLE flood_detection_results (
    id SERIAL PRIMARY KEY,
    ds_division VARCHAR(255) NOT NULL,
    flood_area_ha FLOAT,
    flood_depth_mean FLOAT,
    flood_depth_max FLOAT,
    priority INTEGER,
    priority_label VARCHAR(50),
    geometry TEXT,
    event_date DATE,
    division_level VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW()
);
```
3. Copy .env.example to .env and fill in your credentials

### 3. Add data files
Place these in the `data/` folder:
- `gampaha_divisions.shp` (+ .dbf .shx .prj)
- `dem_gampaha.tif`
- `flood_depth_model.pkl` (run train_depth_model.py first)
- `flood_depth_scaler.pkl`

### 4. Train depth model
```
python train_depth_model.py
```

### 5. Run backend
```
python interface/backend/main.py
```
Backend runs on http://localhost:5001

---

## Switching from GN to DS Divisions
Only ONE file needs to change: `modules/flood_detection/config.py`

```python
DIVISION_LEVEL       = 'DS'           # was 'GN'
SHAPEFILE_PATH       = 'data/gampaha_ds_divisions.shp'  # new shapefile
DIVISION_NAME_COLUMN = 'DS_NAME'      # check your shapefile column name
```

That's it. Nothing else changes.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health  | GET    | Check if backend is running |
| /process | POST   | Upload images and run analysis |
| /geojson | GET    | Get latest flood results as GeoJSON |
| /results | GET    | Get results from Supabase (for other modules) |
| /map     | GET    | Get latest flood map PNG |

## For Module 2 (Population) and Module 3 (Resources)
Call `GET /results?event_date=2025-12-04` to get flood data per division.

Response format:
```json
{
  "success": true,
  "data": [
    {
      "ds_division": "Ja-Ela",
      "flood_area_ha": 245.3,
      "flood_depth_mean": 1.2,
      "flood_depth_max": 2.8,
      "priority": 2,
      "priority_label": "Medium",
      "event_date": "2025-12-04"
    }
  ]
}
```