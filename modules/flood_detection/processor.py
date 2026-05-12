"""
modules/flood_detection/processor.py
Main flood detection processing pipeline - Supabase only.
Team Trivia · University of Moratuwa · 2026
"""

import numpy as np
import cv2
import rasterio
import geopandas as gpd
import joblib
import os
import warnings
import tempfile
import shutil

warnings.filterwarnings('ignore')

from rasterio.mask import mask as rasterio_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from scipy.ndimage import distance_transform_edt

from .config import FloodDetectionConfig, get_file_from_supabase


class FloodDetectionProcessor:
    """Main flood detection processor using Supabase storage."""

    def __init__(self, config: FloodDetectionConfig = None):
        self.config = config or FloodDetectionConfig()
        self.model  = None
        self.scaler = None
        self._load_model()

    # ── Model Loading ─────────────────────────────────────────

    def _load_model(self):
        """Load model directly from Supabase into memory."""
        try:
            model_bytes  = get_file_from_supabase(self.config.SUPABASE_BUCKET, self.config.MODEL_KEY)
            scaler_bytes = get_file_from_supabase(self.config.SUPABASE_BUCKET, self.config.SCALER_KEY)

            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                f.write(model_bytes)
                model_path = f.name

            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                f.write(scaler_bytes)
                scaler_path = f.name

            self.model  = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            os.unlink(model_path)
            os.unlink(scaler_path)

            print("   Depth model loaded from Supabase")

        except Exception as e:
            print(f"   No depth model available: {e}")
            print("   Using DEM-based fallback")
            self.model  = None
            self.scaler = None

    # ── Shapefile Loading ─────────────────────────────────────

    def _load_shapefile_from_supabase(self):
        """Load shapefile from Supabase — downloads all companion files."""
        temp_dir = tempfile.mkdtemp()
        base_key = self.config.SHAPEFILE_KEY.replace('.shp', '')

        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            try:
                file_bytes = get_file_from_supabase(
                    self.config.SUPABASE_BUCKET, f"{base_key}{ext}"
                )
                local_path = os.path.join(temp_dir, f"{base_key}{ext}")
                with open(local_path, 'wb') as f:
                    f.write(file_bytes)
                print(f"   ✓ Downloaded {base_key}{ext}")
            except Exception as e:
                if ext == '.shp':
                    shutil.rmtree(temp_dir)
                    raise
                print(f"   ⚠ Optional file missing: {ext}")

        shp_path = os.path.join(temp_dir, f"{base_key}.shp")

        try:
            if not os.path.exists(shp_path):
                raise FileNotFoundError(f"Shapefile not found at {shp_path}")

            gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
            print(f"   ✓ Loaded {len(gdf)} divisions")
            return gdf

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ── Band Loading ──────────────────────────────────────────

    def _load_and_clip_band(self, path, shapes, crs, target_shape=None, is_dem=False):
        """
        Load a raster band, clip to Gampaha boundary, and convert to
        TOA reflectance if it is a spectral band (L1TP DN → reflectance).
        DEM data is passed through unchanged.
        """
        with rasterio.open(path) as src:
            import pyproj

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

        # ── L1TP DN → TOA Reflectance conversion ─────────────
        # Landsat 8 Collection 2 standard coefficients:
        #   reflectance = DN * 0.0000275 + (-0.2)
        # Only applied to spectral bands, not DEM.
        if not is_dem:
            data = np.where(data > 0, data * 0.0000275 + (-0.2), np.nan)
            data = np.clip(data, 0.0, 1.0)

        # Resize to target shape if needed
        if target_shape is not None and data.shape != target_shape:
            data = cv2.resize(
                data,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return data, profile

    # ── Cleaning ──────────────────────────────────────────────

    def _clean_band(self, band, is_dem=False):
        """Remove invalid pixel values."""
        if is_dem:
            return np.where(band <= 0, np.nan, band)
        # Spectral bands are now in [0, 1] after TOA conversion
        band = np.where(band <= 0,  np.nan, band)
        band = np.where(band > 1.0, np.nan, band)
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
        SW-GAT: Spectrally-Weighted Gaussian Adaptive Thresholding.

        Novel contribution: combines spatial Gaussian weighting (standard
        adaptive threshold) with spectral Gaussian weighting (bilateral
        filter concept) so that only neighbours with similar NDWI values
        contribute to the local threshold.

        T(x,y) = weighted_mean(neighbourhood) - C
        weight  = spatial_weight × spectral_weight
        spatial_weight  = exp(−dist²    / 2σ_space²)
        spectral_weight = exp(−Δndwi²   / 2σ_range²)
        """
        ndwi_clean = np.nan_to_num(ndwi, nan=0.0)

        # Normalise NDWI to [0, 1] — keeps σ_range on a consistent scale
        ndwi_min  = np.min(ndwi_clean)
        ndwi_max  = np.max(ndwi_clean)
        ndwi_norm = (ndwi_clean - ndwi_min) / (ndwi_max - ndwi_min + 1e-6)

        print(f"   NDWI norm range: {ndwi_norm.min():.3f} to {ndwi_norm.max():.3f}")
        print(f"   NDWI > 0.5: {np.sum(ndwi_norm > 0.5)} pixels")
        print(f"   NDWI > 0.7: {np.sum(ndwi_norm > 0.7)} pixels")

        pad   = self.config.BLOCK_SIZE // 2
        C     = self.config.C_CONSTANT
        rows, cols = ndwi_norm.shape

        water_mask = np.zeros_like(ndwi_norm, dtype=np.uint8)

        # Pre-compute spatial Gaussian weights (constant for all pixels)
        x, y = np.meshgrid(np.arange(-pad, pad + 1), np.arange(-pad, pad + 1))
        spatial_weight = np.exp(
            -(x**2 + y**2) / (2 * (self.config.SIGMA_SPACE**2))
        )

        for i in range(pad, rows - pad):
            for j in range(pad, cols - pad):

                center_val = ndwi_norm[i, j]
                window     = ndwi_norm[i - pad:i + pad + 1, j - pad:j + pad + 1]

                # Spectral weight: neighbours with similar NDWI get high weight
                spectral_weight = np.exp(
                    -((window - center_val) ** 2) /
                    (2 * (self.config.SIGMA_RANGE ** 2))
                )

                weight        = spatial_weight * spectral_weight
                weighted_mean = np.sum(weight * window) / (np.sum(weight) + 1e-6)
                threshold     = weighted_mean - C

                water_mask[i, j] = 255 if center_val > threshold else 0

        water_count = np.sum(water_mask == 255)
        print(f"   SW-GAT: {water_count}/{rows*cols} pixels = water ({water_count/(rows*cols)*100:.1f}%)")

        return water_mask.astype(np.uint8)

    # ── Flood Extent ──────────────────────────────────────────

    def _get_flood_extent(self, ndwi_before, ndwi_after):
        print("   SW-GAT on before image...")
        water_before = self._spectral_gaussian_threshold(ndwi_before)

        print("   SW-GAT on after image...")
        water_after = self._spectral_gaussian_threshold(ndwi_after)

        print(f"   water_before: {np.sum(water_before==255)} pixels")
        print(f"   water_after:  {np.sum(water_after==255)} pixels")

        new_flood = (water_after == 255) & (water_before == 0)
        print(f"   Before morphology: {np.sum(new_flood)} px")

        # Only close small gaps, don't erode with OPEN
        kernel    = np.ones((3, 3), np.uint8)
        new_flood = cv2.morphologyEx(new_flood.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        permanent = (water_after == 255) & (water_before == 255)
        print(f"   Permanent water: {np.sum(permanent)} px ({np.mean(permanent)*100:.1f}%)")
        print(f"   New flood:       {np.sum(new_flood)} px ({np.mean(new_flood)*100:.1f}%)")

        return new_flood.astype(np.uint8)

    # ── Depth Estimation — ML ─────────────────────────────────

    def _estimate_depth_ml(self, ndwi_before, ndwi_after, dem, flood_extent):
        """
        Estimate flood depth using trained Gradient Boosting model.
        Features: NDWI change, elevation, slope, distance to river, rainfall.
        """
        flood_pixels = flood_extent == 1

        if not np.any(flood_pixels) or self.model is None:
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

        # Feature 4: Distance to river (permanent water from before image)
        permanent_water = ndwi_before > 0.3
        dist_to_river   = distance_transform_edt(~permanent_water) * 30
        dist_to_river   = np.clip(dist_to_river, 10, 3000)

        # Feature 5: Rainfall proxy (CHIRPS-derived Gampaha range)
        ndwi_change_mean = float(np.nanmean(ndwi_change[flood_pixels]))
        rainfall_val = np.clip(
            self.config.RAINFALL_MIN_MM +
            (ndwi_change_mean * (self.config.RAINFALL_MAX_MM - self.config.RAINFALL_MIN_MM)),
            self.config.RAINFALL_MIN_MM,
            self.config.RAINFALL_MAX_MM
        )
        rainfall = np.full_like(dem, rainfall_val)

        X = np.column_stack([
            ndwi_change[flood_pixels],
            elevation[flood_pixels],
            slope[flood_pixels],
            dist_to_river[flood_pixels],
            rainfall[flood_pixels],
        ])

        X_scaled     = self.scaler.transform(X)
        depth_values = np.clip(self.model.predict(X_scaled), 0.0, 6.0)

        flood_depth = np.zeros_like(dem, dtype=np.float32)
        flood_depth[flood_pixels] = depth_values

        return flood_depth

    # ── Depth Estimation — DEM fallback ──────────────────────

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
        """Assign priority zones (1=Low, 2=Medium, 3=High) based on depth."""
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

    # ── Reproject to EPSG:4326 ────────────────────────────────

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

    # ── Zonal Stats ───────────────────────────────────────────

    def _calculate_division_stats(self, gdf, flood_depth_4326,
                                   priority_4326, flood_extent_4326,
                                   transform_4326, pixel_area_m2):
        """Calculate flood statistics per administrative division."""
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
        gdf['priority_label']  = gdf['priority'].apply(self._priority_label)
        gdf['division_level']  = self.config.DIVISION_LEVEL

        return gdf

    # ── Main Pipeline ─────────────────────────────────────────

    def process(self, before_b3, before_b5, after_b3, after_b5, dem_path,
                event_date=None):
        """
        Main processing pipeline.

        Parameters:
            before_b3, before_b5 : paths to before-flood Landsat bands (L1TP)
            after_b3,  after_b5  : paths to after-flood Landsat bands  (L1TP)
            dem_path             : path to merged DEM GeoTIFF
            event_date           : date string e.g. '2025-12-05'

        Returns:
            dict with keys: geojson_path, stats, gdf
        """
        os.makedirs('outputs', exist_ok=True)

        print("Step 1: Loading shapefile from Supabase...")
        gdf    = self._load_shapefile_from_supabase()
        shapes = [mapping(geom) for geom in gdf.geometry]
        crs    = gdf.crs

        print("Step 2: Loading and clipping bands...")
        b3_before, profile = self._load_and_clip_band(before_b3, shapes, crs)
        target_shape        = b3_before.shape
        b5_before, _        = self._load_and_clip_band(before_b5, shapes, crs, target_shape)
        b3_after,  _        = self._load_and_clip_band(after_b3,  shapes, crs, target_shape)
        b5_after,  _        = self._load_and_clip_band(after_b5,  shapes, crs, target_shape)
        dem_data,  _        = self._load_and_clip_band(dem_path,  shapes, crs, target_shape, is_dem=True)

        print("Step 3: Cleaning bands...")
        b3_before = self._clean_band(b3_before)
        b5_before = self._clean_band(b5_before)
        b3_after  = self._clean_band(b3_after)
        b5_after  = self._clean_band(b5_after)
        dem_data  = self._clean_band(dem_data, is_dem=True)

        print(f"   b3_before valid: {np.sum(~np.isnan(b3_before))} px  "
              f"range [{np.nanmin(b3_before):.3f}, {np.nanmax(b3_before):.3f}]")
        print(f"   b3_after  valid: {np.sum(~np.isnan(b3_after))} px  "
              f"range [{np.nanmin(b3_after):.3f}, {np.nanmax(b3_after):.3f}]")

        print("Step 4: Calculating NDWI...")
        ndwi_before = self._calculate_ndwi(b3_before, b5_before)
        ndwi_after  = self._calculate_ndwi(b3_after,  b5_after)
        print(f"   NDWI before: [{np.nanmin(ndwi_before):.3f}, {np.nanmax(ndwi_before):.3f}]")
        print(f"   NDWI after:  [{np.nanmin(ndwi_after):.3f}, {np.nanmax(ndwi_after):.3f}]")

        print("Step 5: Detecting flood extent (SW-GAT)...")
        flood_extent   = self._get_flood_extent(ndwi_before, ndwi_after)
        pixel_area_m2  = abs(profile['transform'][0] * profile['transform'][4])
        flooded_pixels = int(np.sum(flood_extent == 1))
        flood_area_km2 = (flooded_pixels * pixel_area_m2) / 1_000_000
        print(f"   Flooded area: {flood_area_km2:.2f} km²")

        print("   Saving flood extent rasters...")
        try:
            # Native CRS
            flood_uint8 = (flood_extent * 255).astype(np.uint8)
            with rasterio.open('outputs/flood_extent.tif', 'w',
                               driver='GTiff', height=flood_uint8.shape[0],
                               width=flood_uint8.shape[1], count=1,
                               dtype=np.uint8, crs=profile['crs'],
                               transform=profile['transform']) as dst:
                dst.write(flood_uint8, 1)

            # EPSG:4326
            t4326, w4326, h4326 = calculate_default_transform(
                profile['crs'], 'EPSG:4326',
                profile['width'], profile['height'],
                *rasterio.transform.array_bounds(
                    profile['height'], profile['width'], profile['transform']
                )
            )
            fe4326 = np.zeros((h4326, w4326), dtype=np.uint8)
            reproject(source=flood_uint8.astype(np.float32), destination=fe4326,
                      src_transform=profile['transform'], src_crs=profile['crs'],
                      dst_transform=t4326, dst_crs='EPSG:4326',
                      resampling=Resampling.nearest)
            with rasterio.open('outputs/flood_extent_4326.tif', 'w',
                               driver='GTiff', height=h4326, width=w4326,
                               count=1, dtype=np.uint8,
                               crs='EPSG:4326', transform=t4326) as dst:
                dst.write(fe4326, 1)
            print("   ✓ flood_extent.tif and flood_extent_4326.tif saved")
        except Exception as e:
            print(f"   ⚠ Could not save flood extent rasters: {e}")

        print("Step 6: Estimating flood depth...")
        if self.model is not None:
            flood_depth  = self._estimate_depth_ml(ndwi_before, ndwi_after, dem_data, flood_extent)
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
        flood_depth_4326,  transform_4326, _, _ = self._reproject_to_4326(flood_depth, profile)
        priority_4326,     _, _, _              = self._reproject_to_4326(priority.astype(np.float32), profile)
        flood_extent_4326, _, _, _              = self._reproject_to_4326(flood_extent.astype(np.float32), profile)

        gdf_4326 = gdf.to_crs("EPSG:4326")

        print(f"Step 9: Calculating stats per {self.config.DIVISION_LEVEL} division...")
        gdf_results = self._calculate_division_stats(
            gdf_4326, flood_depth_4326, priority_4326,
            flood_extent_4326, transform_4326, pixel_area_m2
        )

        if event_date:
            gdf_results['event_date'] = event_date

        print("Step 10: Saving GeoJSON...")
        geojson_path = 'outputs/flood_results.geojson'
        gdf_results.to_file(geojson_path, driver='GeoJSON')
        print(f"   Saved to {geojson_path}")

        stats = {
            'flood_area_km2':    round(flood_area_km2, 2),
            'max_depth_m':       round(max_depth, 2),
            'avg_depth_m':       round(avg_depth, 2),
            'high_km2':          round(high_km2, 2),
            'medium_km2':        round(medium_km2, 2),
            'low_km2':           round(low_km2, 2),
            'depth_method':      depth_method,
            'division_level':    self.config.DIVISION_LEVEL,
            'total_divisions':   len(gdf_results),
            'flooded_divisions': int(np.sum(gdf_results['priority'] > 0)),
        }

        print("\n✅ Processing complete!")
        print(f"   Flood area:          {flood_area_km2:.2f} km²")
        print(f"   Max depth:           {max_depth:.2f} m")
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