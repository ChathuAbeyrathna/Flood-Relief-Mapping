
# Main flood detection processing pipeline.


import numpy as np
import cv2
import rasterio
import geopandas as gpd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from rasterio.mask import mask as rasterio_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import mapping
from scipy.ndimage import distance_transform_edt

from .config import FloodDetectionConfig, download_from_supabase

class FloodDetectionProcessor:
    """
    Main flood detection processor.
    
    Usage:
        processor = FloodDetectionProcessor()
        results   = processor.process(
            before_b3_path, before_b5_path,
            after_b3_path,  after_b5_path,
            dem_path
        )
    """

    def __init__(self, config: FloodDetectionConfig = None):
        self.config = config or FloodDetectionConfig()
        self.model  = None
        self.scaler = None
        self._load_model()

    # ── Model Loading ─────────────────────────────────────────

    def _load_model(self):
        """Load the trained depth regression model if available."""
        try:
            from .config import download_from_supabase
            model_path = download_from_supabase('flood-data', 'flood_depth_model.pkl')
            scaler_path = download_from_supabase('flood-data', 'flood_depth_scaler.pkl')
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("   Depth model loaded successfully")
        except FileNotFoundError:
            print("   Warning: Depth model not found. Run train_depth_model.py first.")
            print("   Falling back to DEM-based depth estimation.")

    # ── Band Loading ──────────────────────────────────────────

    def _load_and_clip_band(self, path, shapes, crs, target_shape=None):
        """Load a raster band and clip to division boundary."""
        with rasterio.open(path) as src:
            # Reproject shapes to match raster CRS
            import pyproj
            from shapely.ops import transform as shp_transform
            from shapely.geometry import shape

            raster_epsg = src.crs.to_epsg()
            shp_epsg    = crs.to_epsg()

            if raster_epsg and raster_epsg != shp_epsg:
                project = pyproj.Transformer.from_crs(
                    f"EPSG:{shp_epsg}", f"EPSG:{raster_epsg}",
                    always_xy=True
                ).transform
                projected = [
                    mapping(shp_transform(project, shape(s)))
                    for s in shapes
                ]
            else:
                projected = shapes

            clipped, transform = rasterio_mask(src, projected, crop=True)
            profile = src.profile.copy()
            profile.update({
                'height':    clipped.shape[1],
                'width':     clipped.shape[2],
                'transform': transform
            })
            data = clipped[0].astype(np.float32)

        # Resize to target shape if needed
        if target_shape is not None and data.shape != target_shape:
            data = cv2.resize(
                data,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return data, profile

    # ── Cleaning ──────────────────────────────────────────────

    def _clean_band(self, band):
        """Remove invalid pixel values."""
        band = np.where(band <= 0,     np.nan, band)
        band = np.where(band > 65535,  np.nan, band)
        return band

    # ── NDWI ──────────────────────────────────────────────────

    def _calculate_ndwi(self, green, nir):
        """Calculate Normalized Difference Water Index."""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir)
        return np.where(np.isinf(ndwi), np.nan, ndwi)

    # ── Novel: Spectrally-Weighted Gaussian Adaptive Threshold ─

    def _spectral_gaussian_threshold(self, ndwi):
        """ 
        Spectral-Weighted Gaussian Adaptive Thresholding (SW-GAT)
        T(x,y) = weighted_mean(neighbourhood) - C
        weight = spatial_weight × spectral_weight
        
        """

        ndwi_clean = np.nan_to_num(ndwi, nan=0.0)

        # Normalize NDWI to 0–1 (better than 0–255 for math consistency)
        ndwi_norm = (ndwi_clean - np.min(ndwi_clean)) / (
            np.max(ndwi_clean) - np.min(ndwi_clean) + 1e-6
        )

        pad = self.config.BLOCK_SIZE // 2
        C = self.config.C_CONSTANT

        # Output mask
        flood_mask = np.zeros_like(ndwi_norm, dtype=np.uint8)

        rows, cols = ndwi_norm.shape

        # Precompute coordinate grid for spatial weights
        x, y = np.meshgrid(np.arange(-pad, pad + 1), np.arange(-pad, pad + 1))

        spatial_weight = np.exp(
            -(x**2 + y**2) / (2 * (self.config.SIGMA_SPACE**2))
        )

        for i in range(pad, rows - pad):
            for j in range(pad, cols - pad):

                center_val = ndwi_norm[i, j]
                window = ndwi_norm[i - pad:i + pad + 1, j - pad:j + pad + 1]

                # Spectral weight (based on NDWI similarity)
                spectral_weight = np.exp(
                    -((window - center_val) ** 2) /
                    (2 * (self.config.SIGMA_RANGE**2))
                )

                # Final combined weight
                weight = spatial_weight * spectral_weight

                # Weighted mean threshold
                weighted_mean = np.sum(weight * window) / (np.sum(weight) + 1e-6)

                threshold = weighted_mean - C

                # Classification
                flood_mask[i, j] = 255 if center_val > threshold else 0

        return flood_mask.astype(np.uint8)

    # ── Flood Extent ──────────────────────────────────────────

    def _get_flood_extent(self, ndwi_before, ndwi_after):
        """Detect newly flooded pixels using change detection."""
        water_before = self._spectral_gaussian_threshold(ndwi_before)
        water_after  = self._spectral_gaussian_threshold(ndwi_after)

        # Flood = water AFTER but NOT before
        flood_extent = np.where(
            (water_after == 255) & (water_before == 0), 1, 0
        ).astype(np.uint8)

        return flood_extent

    # ── Depth Estimation ──────────────────────────────────────

    def _estimate_depth_ml(self, ndwi_before, ndwi_after, dem, flood_extent):
        """
        Estimate flood depth using trained ML regression model.
        Features: NDWI change, elevation, slope, distance to river, rainfall
        """
        flood_pixels = flood_extent == 1

        if not np.any(flood_pixels):
            return np.zeros_like(dem, dtype=np.float32)

        # Feature 1: NDWI change
        ndwi_change = np.clip(
            np.nan_to_num(ndwi_after - ndwi_before, nan=0.0), 0, 1
        )

        # Feature 2: Elevation
        elevation = np.clip(np.nan_to_num(dem, nan=5.0), 0.1, 200)

        # Feature 3: Slope from DEM
        dy, dx = np.gradient(np.nan_to_num(dem, nan=0.0))
        slope  = np.clip(np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))), 0.1, 45.0)

        # Feature 4: Distance to river (from permanent water in before image)
        permanent_water  = ndwi_before > 0.3
        dist_to_river    = distance_transform_edt(~permanent_water) * 30
        dist_to_river    = np.clip(dist_to_river, 10, 3000)

        # Feature 5: Rainfall (Gampaha-specific from CHIRPS + Ditwah data)
        ndwi_change_mean = float(np.nanmean(ndwi_change[flood_pixels]))
        rainfall_val     = np.clip(
            self.config.RAINFALL_MIN_MM +
            (ndwi_change_mean * (self.config.RAINFALL_MAX_MM - self.config.RAINFALL_MIN_MM)),
            self.config.RAINFALL_MIN_MM,
            self.config.RAINFALL_MAX_MM
        )
        rainfall = np.full_like(dem, rainfall_val)

        # Stack features for flooded pixels only
        X = np.column_stack([
            ndwi_change[flood_pixels],
            elevation[flood_pixels],
            slope[flood_pixels],
            dist_to_river[flood_pixels],
            rainfall[flood_pixels],
        ])

        # Predict
        X_scaled     = self.scaler.transform(X)
        depth_values = np.clip(self.model.predict(X_scaled), 0.0, 6.0)

        # Put back into 2D
        flood_depth = np.zeros_like(dem, dtype=np.float32)
        flood_depth[flood_pixels] = depth_values

        return flood_depth

    def _estimate_depth_dem(self, dem, flood_extent):
        """Fallback DEM-based depth estimation when model unavailable."""
        boundary_elev = dem[flood_extent == 1]
        boundary_elev = boundary_elev[~np.isnan(boundary_elev)]

        if len(boundary_elev) == 0:
            return np.zeros_like(dem, dtype=np.float32)

        water_surface = np.percentile(boundary_elev, 90)
        return np.where(
            flood_extent == 1,
            np.maximum(water_surface - dem, 0), 0
        ).astype(np.float32)

    # ── Priority ──────────────────────────────────────────────

    def _assign_priority(self, flood_depth):
        """Assign priority zones based on flood depth."""
        priority = np.zeros_like(flood_depth, dtype=np.uint8)
        priority = np.where(
            (flood_depth > 0) & (flood_depth <= self.config.MEDIUM_PRIORITY_DEPTH),
            1, priority
        )
        priority = np.where(
            (flood_depth > self.config.MEDIUM_PRIORITY_DEPTH) &
            (flood_depth <= self.config.HIGH_PRIORITY_DEPTH),
            2, priority
        )
        priority = np.where(flood_depth > self.config.HIGH_PRIORITY_DEPTH, 3, priority)
        return priority

    def _priority_label(self, p):
        if p == 3: return 'High'
        if p == 2: return 'Medium'
        if p == 1: return 'Low'
        return 'No Flood'

    # ── Reproject to 4326 ─────────────────────────────────────

    def _reproject_to_4326(self, data, profile):
        """Reproject raster data to EPSG:4326."""
        transform_4326, w, h = calculate_default_transform(
            profile['crs'], 'EPSG:4326',
            profile['width'], profile['height'],
            *rasterio.transform.array_bounds(
                profile['height'], profile['width'], profile['transform']
            )
        )
        result = np.zeros((h, w), dtype=np.float32)
        reproject(
            source=data.astype(np.float32),
            destination=result,
            src_transform=profile['transform'],
            src_crs=profile['crs'],
            dst_transform=transform_4326,
            dst_crs='EPSG:4326',
            resampling=Resampling.nearest
        )
        return result, transform_4326, w, h

    # ── Zonal Stats per Division ───────────────────────────────

    def _calculate_division_stats(self, gdf, flood_depth_4326,
                                   priority_4326, flood_extent_4326,
                                   transform_4326, pixel_area_m2):
        """Calculate flood statistics per division polygon."""
        import rasterstats

        depth_stats    = rasterstats.zonal_stats(
            gdf, flood_depth_4326, affine=transform_4326,
            stats=['mean', 'max'], nodata=0
        )
        priority_stats = rasterstats.zonal_stats(
            gdf, priority_4326, affine=transform_4326,
            stats=['majority'], nodata=0
        )
        area_stats     = rasterstats.zonal_stats(
            gdf, flood_extent_4326, affine=transform_4326,
            stats=['sum'], nodata=0
        )

        gdf = gdf.copy()
        gdf['flood_depth_mean'] = [round(s['mean'] or 0, 2) for s in depth_stats]
        gdf['flood_depth_max']  = [round(s['max']  or 0, 2) for s in depth_stats]
        gdf['priority']         = [int(s['majority'] or 0) for s in priority_stats]
        gdf['flood_area_ha']    = [
            round((s['sum'] or 0) * pixel_area_m2 / 10000, 2)
            for s in area_stats
        ]
        gdf['priority_label']   = gdf['priority'].apply(self._priority_label)
        gdf['division_level']   = self.config.DIVISION_LEVEL

        return gdf

    # ── Main Process Function ─────────────────────────────────

    def process(self, before_b3, before_b5, after_b3, after_b5, dem_path,
                event_date=None):
        """
        Main processing pipeline.

        Parameters:
            before_b3, before_b5 : paths to before-flood Landsat bands
            after_b3,  after_b5  : paths to after-flood Landsat bands
            dem_path             : path to merged DEM GeoTIFF
            event_date           : date string e.g. '2025-12-04'

        Returns:
            dict with keys:
                'geojson_path'   : path to saved GeoJSON
                'stats'          : summary statistics
                'gdf'            : GeoDataFrame with per-division results
        """
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        print("Step 1: Loading shapefile...")
        # ── This is the only place that needs changing for DS ──
        # gdf    = gpd.read_file(self.config.SHAPEFILE_PATH).to_crs("EPSG:4326")
        from .config import download_from_supabase
        shapefile_path = download_from_supabase('flood-data', 'gampaha_divisions.shp')
        gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
        shapes = [mapping(geom) for geom in gdf.geometry]
        crs    = gdf.crs
        print(f"   Loaded {len(gdf)} {self.config.DIVISION_LEVEL} divisions")

        print("Step 2: Loading and clipping bands...")
        b3_before, profile = self._load_and_clip_band(before_b3, shapes, crs)
        target_shape       = b3_before.shape
        b5_before, _       = self._load_and_clip_band(before_b5, shapes, crs, target_shape)
        b3_after,  _       = self._load_and_clip_band(after_b3,  shapes, crs, target_shape)
        b5_after,  _       = self._load_and_clip_band(after_b5,  shapes, crs, target_shape)
        dem_data,  _       = self._load_and_clip_band(dem_path,  shapes, crs, target_shape)

        print("Step 3: Cleaning bands...")
        b3_before = self._clean_band(b3_before)
        b5_before = self._clean_band(b5_before)
        b3_after  = self._clean_band(b3_after)
        b5_after  = self._clean_band(b5_after)
        dem_data  = np.where(dem_data <= 0, np.nan, dem_data)

        print("Step 4: Calculating NDWI...")
        ndwi_before = self._calculate_ndwi(b3_before, b5_before)
        ndwi_after  = self._calculate_ndwi(b3_after,  b5_after)

        print("Step 5: Detecting flood extent (SW-GAT)...")
        flood_extent   = self._get_flood_extent(ndwi_before, ndwi_after)
        pixel_area_m2  = abs(profile['transform'][0] * profile['transform'][4])
        flooded_pixels = int(np.sum(flood_extent == 1))
        flood_area_km2 = (flooded_pixels * pixel_area_m2) / 1_000_000
        print(f"   Flooded area: {flood_area_km2:.2f} km²")

        print("Step 6: Estimating flood depth...")
        if self.model is not None:
            flood_depth = self._estimate_depth_ml(
                ndwi_before, ndwi_after, dem_data, flood_extent
            )
            depth_method = 'ML Regression (Gradient Boosting)'
        else:
            flood_depth  = self._estimate_depth_dem(dem_data, flood_extent)
            depth_method = 'DEM-based (fallback)'
        print(f"   Method: {depth_method}")

        max_depth = float(np.nanmax(flood_depth))
        avg_depth = float(np.nanmean(flood_depth[flood_depth > 0])) if flooded_pixels > 0 else 0

        print("Step 7: Assigning priority zones...")
        priority   = self._assign_priority(flood_depth)
        high_km2   = float(np.sum(priority == 3) * pixel_area_m2 / 1_000_000)
        medium_km2 = float(np.sum(priority == 2) * pixel_area_m2 / 1_000_000)
        low_km2    = float(np.sum(priority == 1) * pixel_area_m2 / 1_000_000)

        print("Step 8: Reprojecting to EPSG:4326...")
        flood_depth_4326,  transform_4326, w4326, h4326 = self._reproject_to_4326(flood_depth,  profile)
        priority_4326,     _,              _,     _      = self._reproject_to_4326(priority.astype(np.float32), profile)
        flood_extent_4326, _,              _,     _      = self._reproject_to_4326(flood_extent.astype(np.float32), profile)

        gdf_4326 = gdf.to_crs("EPSG:4326")

        print(f"Step 9: Calculating stats per {self.config.DIVISION_LEVEL} division...")
        gdf_results = self._calculate_division_stats(
            gdf_4326, flood_depth_4326, priority_4326,
            flood_extent_4326, transform_4326, pixel_area_m2
        )

        # Add event date
        if event_date:
            gdf_results['event_date'] = event_date

        print("Step 10: Saving GeoJSON...")
        geojson_path = self.config.GEOJSON_OUTPUT
        gdf_results.to_file(geojson_path, driver='GeoJSON')
        print(f"   Saved to {geojson_path}")

        stats = {
            'flood_area_km2': round(flood_area_km2, 2),
            'max_depth_m':    round(max_depth, 2),
            'avg_depth_m':    round(avg_depth, 2),
            'high_km2':       round(high_km2, 2),
            'medium_km2':     round(medium_km2, 2),
            'low_km2':        round(low_km2, 2),
            'depth_method':   depth_method,
            'division_level': self.config.DIVISION_LEVEL,
            'total_divisions': len(gdf_results),
            'flooded_divisions': int(np.sum(gdf_results['priority'] > 0)),
        }

        print("\n✅ Processing complete!")
        print(f"   Flood area:  {flood_area_km2:.2f} km²")
        print(f"   Max depth:   {max_depth:.2f} m")
        print(f"   High priority zones: {high_km2:.2f} km²")

        return {
            'geojson_path': geojson_path,
            'stats':        stats,
            'gdf':          gdf_results,
            'rasters': {
                'flood_depth_4326':  flood_depth_4326,
                'priority_4326':     priority_4326,
                'flood_extent_4326': flood_extent_4326,
                'transform_4326':    transform_4326,
            }
        }