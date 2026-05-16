import argparse
import rasterio
import numpy as np
import geopandas as gpd
from scipy.ndimage import convolve, label, center_of_mass, median_filter
from shapely.geometry import Point
from rasterio.crs import CRS
from pathlib import Path


def cfar(image, guard_len, train_len, pfa):
    """
    Averaging CFAR (Constant False Alarm Rate).

    Explanation to Non-SAR customers:

        Imagine the SAR image is a dark floor covered in LED lights. We want to find blindingly bright clusters (ships) against a flickering background (ocean).
        We build a square "cardboard donut" tool to evaluate every single LED.
        - The center hole looks at the target pixel. (1px)
        - The solid inner ring (Guard Cells) blocks the ship's own glare. (5x5=25px)
        - The outer ring of holes (Training Cells) measures the local ocean brightness. (169-25=144px)

    Args:
        image (np.ndarray): 2D float32 SAR amplitude image.
        guard_len (int): Half-width of the guard cell ring (excludes target from noise estimate).
        train_len (int): Half-width of the training cell ring (measures local background noise).
        pfa (float): Probability of False Alarm. Lower = fewer false positives, may miss real targets.

    Returns:
        np.ndarray: Boolean mask — True where a pixel exceeds the adaptive threshold.
    """

    # 1. Tool Size
    ### Calculate the total width of our cardboard square (13x13 pixels)
    window_size = 1 + 2 * guard_len + 2 * train_len
    ### Calculate exactly how many holes are in the outer training ring (144px)
    num_train_cells = window_size**2 - (1 + 2 * guard_len)**2

    # 2. Sensitivity (Alpha) - The PFA is our acceptable error rate.
    ### This formula translates that rate into a strict mathematical multiplier (alpha).
    ### If the ocean averages a brightness of 10, alpha might say "only alert if the target is > 10 * alpha"
    alpha = num_train_cells * (pfa**(-1.0 / num_train_cells) - 1)

    # 3. Kernel / Cardboard Donut
    ### Create a grid of 1s (representing open holes where light shines through)
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    ### Punch a solid square in the middle (0s) to create the Guard Cells and Center pixel.
    ### This solid cardboard blocks the ship's bright light from inflating the ocean average.
    guard_start = train_len
    guard_end = train_len + 2 * guard_len + 1
    kernel[guard_start:guard_end, guard_start:guard_end] = 0

    # 4. Calibration
    ### Divide by the total number of training cells to compute the local background average.
    kernel = kernel / num_train_cells

    # 5. Moving Donut
    ### 'convolve' slides the digital cardboard over every single pixel in the SAR image.
    ### At every pixel, it looks through the outer ring and writes down the average background water brightness.
    noise_floor = convolve(image, kernel, mode='reflect')

    # 6. Final Call
    ### Multiply the local water background by sensitivity to get a custom threshold for every pixel.
    threshold = noise_floor * alpha
    # Return a boolean mask: True (Ship-Pixel) if the actual pixel is brighter than its custom threshold.
    return image > threshold


def process_sar_and_geofence(sar_path, geofence_path, output_path, pfa=1e-5, min_blob_pixels=3):
    """
    Main pipeline: Ingest SAR, detect targets, convert to geo-coords,
    filter by geofence, and export report.

    Args:
        sar_path (str | Path): Path to the input SAR GeoTIFF (single-band amplitude).
        geofence_path (str | Path): Path to the geofence GeoJSON polygon file.
        output_path (str | Path): Destination path for the output alerts GeoJSON.
        pfa (float): Probability of False Alarm passed to CFAR. Default: 1e-5.
        min_blob_pixels (int): Minimum pixel count for a detection blob to be kept as a vessel.
                                Filters out single-pixel noise hits. Default: 3.
    """
    print("1. Loading Geofence...")
    geofence_gdf = gpd.read_file(geofence_path)

    print("2. Loading SAR Image...")
    with rasterio.open(sar_path) as src:
        sar_image = src.read(1).astype(np.float32)
        transform = src.transform
        sar_crs = src.crs

        # Missing Metadata - If None, force standard WGS84.
        if sar_crs is None:
            print("   [Warning] rasterio could not read CRS from TIFF headers.")
            print("   [Fix] Fallback applied: Assuming EPSG:4326 (WGS84).")
            sar_crs = CRS.from_epsg(4326)
        else:
            print(f"   SAR CRS detected: {sar_crs}")

    # Ensure Geofence CRS matches SAR Image CRS before spatial intersection.
    # (Moved outside the rasterio context — transform and crs are already captured above.)
    if geofence_gdf.crs != sar_crs:
        print(f"   Reprojecting geofence from {geofence_gdf.crs} to {sar_crs}...")
        geofence_gdf = geofence_gdf.to_crs(sar_crs)

    # Combine all polygons in the geofence into one geometry boundary.
    restricted_area = geofence_gdf.geometry.union_all()

    print("3. Pre-processing (Speckle Filter)...")
    # A 3x3 median filter removes isolated radar static ("salt") while preserving
    # the sharp structural edges of ships.
    # Note for production: A vector land-mask (e.g. Natural Earth via geopandas)
    # would be applied here to zero out land pixels before CFAR to eliminate coastal clutter.
    sar_image_filtered = median_filter(sar_image, size=3)

    print("4. Running CA-CFAR Detection...")
    detections = cfar(sar_image_filtered, guard_len=2, train_len=4, pfa=pfa)

    print("5. Clustering detections into vessels...")
    # CFAR detects individual bright pixels. A large ship might be 50 bright pixels.
    # label() groups adjacent True pixels into a single blob, so we generate 1 alert
    # per ship instead of 50.
    labeled_array, num_features = label(detections)
    print(f"   Raw pixel-clusters detected globally: {num_features}")

    if num_features == 0:
        print("No targets found in the image. Exiting.")
        return

    # Size filter: discard blobs smaller than min_blob_pixels.
    # A single anomalous pixel will pass CFAR — this step removes those noise hits
    # before they become false vessel alerts.
    valid_labels = [
        i for i in range(1, num_features + 1)
        if np.sum(labeled_array == i) >= min_blob_pixels
    ]
    print(f"   Clusters remaining after size filter (>= {min_blob_pixels}px): {len(valid_labels)}")

    if not valid_labels:
        print("No targets survived the size filter. Exiting.")
        return

    # Find the exact mathematical center coordinate of each valid blob.
    # Note: center_of_mass returns float (row, col). We round to nearest integer
    # for pixel lookup, accepting sub-pixel precision loss (acceptable given vessel size).
    centroids = center_of_mass(detections, labeled_array, valid_labels)

    print("6. Pixel-translating and applying Geofence...")
    alert_records = []

    for vessel_id, (row, col) in enumerate(centroids):
        # Translate pixel coords (row, col) into geo coords (lon, lat)
        lon, lat = rasterio.transform.xy(transform, int(round(row)), int(round(col)))
        target_point = Point(lon, lat)

        # Only record the vessel if it falls inside the customer's restricted area
        if restricted_area.contains(target_point):
            alert_records.append({
                'vessel_id': vessel_id + 1,
                'pixel_row': int(round(row)),
                'pixel_col': int(round(col)),
                'geometry': target_point
            })

    print(f"7. Exporting results... Found {len(alert_records)} vessels in the restricted zone.")
    if alert_records:
        alerts_gdf = gpd.GeoDataFrame(alert_records, crs=sar_crs)
        # Standardize final output to WGS84 for ingestion into downstream analyst tools
        alerts_gdf = alerts_gdf.to_crs("EPSG:4326")
        alerts_gdf.to_file(output_path, driver="GeoJSON")
        print(f"   Alerts saved to: {output_path}")
    else:
        print("   No vessels detected inside the restricted zone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CA-CFAR SAR vessel detection pipeline with geofence filtering."
    )
    parser.add_argument("--sar",      required=True,              help="Path to input SAR GeoTIFF (single-band amplitude).")
    parser.add_argument("--geofence", required=True,              help="Path to geofence GeoJSON polygon file.")
    parser.add_argument("--output",   required=True,              help="Output path for vessel alerts GeoJSON.")
    parser.add_argument("--pfa",      type=float, default=1e-5,   help="Probability of False Alarm (default: 1e-5).")
    parser.add_argument("--min-blob", type=int,   default=3,      help="Minimum blob size in pixels to count as a vessel (default: 3).")
    args = parser.parse_args()

    process_sar_and_geofence(
        sar_path=args.sar,
        geofence_path=args.geofence,
        output_path=args.output,
        pfa=args.pfa,
        min_blob_pixels=args.min_blob,
    )
