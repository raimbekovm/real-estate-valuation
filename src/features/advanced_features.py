"""
Advanced Feature Engineering for Real Estate Valuation v3

This module implements state-of-the-art features identified through research
of top-performing property valuation models (Zillow, Kaggle winners, academic papers).

Features implemented:
1. SpatialLagFeatures - Neighbor price statistics
2. H3Features - Uber H3 hexagonal geographic tiles
3. MarketTrendFeatures - Rolling market statistics
4. DensityFeatures - Listing density in radius

Usage:
    from src.features.advanced_features import AdvancedFeaturePipeline

    pipeline = AdvancedFeaturePipeline()
    pipeline.fit(train_df, 'price_per_m2')
    train_features = pipeline.transform(train_df)
    test_features = pipeline.transform(test_df)

References:
    - Zillow Neural Zestimate: https://www.zillow.com/tech/building-the-neural-zestimate/
    - Multi-Head Gated Attention: https://arxiv.org/abs/2405.07456
    - H3 Documentation: https://h3geo.org/

Author: ML Engineering Team
Version: 1.0
Date: 7 January 2026
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.neighbors import BallTree
from dataclasses import dataclass
import warnings

# Optional imports with graceful fallback
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    warnings.warn("h3 library not installed. H3Features will be disabled. Install with: pip install h3")


@dataclass
class FeatureConfig:
    """Configuration for advanced features."""
    # Spatial lag
    spatial_lag_radius_km: float = 0.5
    spatial_lag_min_neighbors: int = 3

    # H3 tiles
    h3_resolutions: List[int] = None

    # Market trends
    trend_windows_days: List[int] = None

    # Density
    density_radii_km: List[float] = None

    def __post_init__(self):
        if self.h3_resolutions is None:
            self.h3_resolutions = [7, 8, 9]
        if self.trend_windows_days is None:
            self.trend_windows_days = [30, 60, 90]
        if self.density_radii_km is None:
            self.density_radii_km = [0.5, 1.0]


class SpatialLagFeatures:
    """
    Calculate spatial lag features using neighbor price statistics.

    This is one of the most impactful features for real estate valuation,
    as property prices exhibit strong spatial autocorrelation.

    Features generated:
        - neighbor_price_mean: Mean price of neighbors within radius
        - neighbor_price_median: Median price of neighbors
        - neighbor_price_std: Standard deviation of neighbor prices
        - neighbor_price_min: Minimum neighbor price
        - neighbor_price_max: Maximum neighbor price
        - neighbor_count: Number of neighbors found
        - price_vs_neighbors: Ratio of property price to neighbor mean

    Important:
        Must fit ONLY on training data to prevent data leakage.
        When transforming test data, neighbors are looked up from training set only.

    References:
        - "Spatial lag variable significantly improves prediction accuracy"
          (MDPI Sustainability, 2022)

    Example:
        >>> spatial_lag = SpatialLagFeatures(radius_km=0.5)
        >>> spatial_lag.fit(train_df, 'price_per_m2')
        >>> train_df = spatial_lag.transform(train_df)
        >>> test_df = spatial_lag.transform(test_df)
    """

    EARTH_RADIUS_KM = 6371.0

    def __init__(self, radius_km: float = 0.5, min_neighbors: int = 3):
        """
        Initialize SpatialLagFeatures.

        Args:
            radius_km: Search radius in kilometers (default: 0.5)
            min_neighbors: Minimum neighbors required for valid statistics (default: 3)
        """
        self.radius_km = radius_km
        self.min_neighbors = min_neighbors
        self.tree = None
        self.train_coords = None
        self.train_prices = None
        self.train_indices = None
        self._fitted = False

    def fit(self, df: pd.DataFrame, price_col: str = 'price_per_m2',
            lat_col: str = 'latitude', lon_col: str = 'longitude') -> 'SpatialLagFeatures':
        """
        Fit on training data to build spatial index.

        Args:
            df: Training DataFrame
            price_col: Name of price column
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            self
        """
        # Store training data
        self.train_coords = np.radians(df[[lat_col, lon_col]].values)
        self.train_prices = df[price_col].values
        self.train_indices = df.index.values

        # Build Ball Tree for efficient neighbor lookup
        self.tree = BallTree(self.train_coords, metric='haversine')

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame,
                  lat_col: str = 'latitude',
                  lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Transform DataFrame by adding spatial lag features.

        For training data: excludes self from neighbor calculation
        For test data: all training neighbors are used

        Args:
            df: DataFrame to transform
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            DataFrame with spatial lag features added
        """
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")

        df = df.copy()
        coords = np.radians(df[[lat_col, lon_col]].values)
        radius_rad = self.radius_km / self.EARTH_RADIUS_KM

        # Query neighbors
        indices_list = self.tree.query_radius(coords, r=radius_rad)

        # Check if this is training data (same indices)
        is_train = set(df.index.values) == set(self.train_indices)

        # Calculate statistics
        means, medians, stds, mins, maxs, counts = [], [], [], [], [], []

        for i, neighbor_idx in enumerate(indices_list):
            # Exclude self if training data
            if is_train and len(neighbor_idx) > 0:
                # Find and remove self
                current_idx = df.index[i]
                mask = self.train_indices[neighbor_idx] != current_idx
                neighbor_idx = neighbor_idx[mask]

            neighbor_prices = self.train_prices[neighbor_idx]

            if len(neighbor_prices) >= self.min_neighbors:
                means.append(np.mean(neighbor_prices))
                medians.append(np.median(neighbor_prices))
                stds.append(np.std(neighbor_prices))
                mins.append(np.min(neighbor_prices))
                maxs.append(np.max(neighbor_prices))
            else:
                means.append(np.nan)
                medians.append(np.nan)
                stds.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)
            counts.append(len(neighbor_idx))

        # Add features
        df['neighbor_price_mean'] = means
        df['neighbor_price_median'] = medians
        df['neighbor_price_std'] = stds
        df['neighbor_price_min'] = mins
        df['neighbor_price_max'] = maxs
        df['neighbor_count'] = counts

        # Fill NaN with global mean from training
        global_mean = np.mean(self.train_prices)
        df['neighbor_price_mean'] = df['neighbor_price_mean'].fillna(global_mean)
        df['neighbor_price_median'] = df['neighbor_price_median'].fillna(global_mean)
        df['neighbor_price_std'] = df['neighbor_price_std'].fillna(0)
        df['neighbor_price_min'] = df['neighbor_price_min'].fillna(global_mean)
        df['neighbor_price_max'] = df['neighbor_price_max'].fillna(global_mean)

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of feature names generated by this transformer."""
        return [
            'neighbor_price_mean', 'neighbor_price_median', 'neighbor_price_std',
            'neighbor_price_min', 'neighbor_price_max', 'neighbor_count'
        ]


class H3Features:
    """
    Generate Uber H3 hexagonal tile features at multiple resolutions.

    H3 is a hierarchical geospatial indexing system that divides the world
    into hexagonal cells. This provides better features than raw lat/lon
    for tree-based models.

    Resolution guide:
        - Resolution 7: ~5.16 km² average area (district level)
        - Resolution 8: ~0.74 km² average area (neighborhood level)
        - Resolution 9: ~0.11 km² average area (block level)

    Features generated:
        - h3_res{N}: H3 index as string (for categorical encoding)
        - h3_res{N}_encoded: Integer encoded H3 index

    References:
        - Zillow Neural Zestimate uses "discretized geographic tiles at multiple scales"
        - H3 Documentation: https://h3geo.org/

    Example:
        >>> h3_features = H3Features(resolutions=[7, 8, 9])
        >>> h3_features.fit(train_df)
        >>> train_df = h3_features.transform(train_df)
    """

    def __init__(self, resolutions: List[int] = None):
        """
        Initialize H3Features.

        Args:
            resolutions: List of H3 resolutions to use (default: [7, 8, 9])
        """
        if not H3_AVAILABLE:
            raise ImportError("h3 library required. Install with: pip install h3")

        self.resolutions = resolutions or [7, 8, 9]
        self.encoders: Dict[int, Dict[str, int]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame,
            lat_col: str = 'latitude',
            lon_col: str = 'longitude') -> 'H3Features':
        """
        Fit encoders on training data.

        Args:
            df: Training DataFrame
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            self
        """
        for res in self.resolutions:
            # Generate H3 indices for training data
            h3_indices = df.apply(
                lambda r: h3.latlng_to_cell(r[lat_col], r[lon_col], res),
                axis=1
            )
            # Create encoder mapping
            unique_indices = h3_indices.unique()
            self.encoders[res] = {idx: i for i, idx in enumerate(unique_indices)}

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame,
                  lat_col: str = 'latitude',
                  lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Transform DataFrame by adding H3 features.

        Args:
            df: DataFrame to transform
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            DataFrame with H3 features added
        """
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")

        df = df.copy()

        for res in self.resolutions:
            col_name = f'h3_res{res}'
            encoded_col = f'{col_name}_encoded'

            # Generate H3 index
            df[col_name] = df.apply(
                lambda r: h3.latlng_to_cell(r[lat_col], r[lon_col], res),
                axis=1
            )

            # Encode to integer (unseen indices get -1)
            df[encoded_col] = df[col_name].map(
                lambda x: self.encoders[res].get(x, -1)
            )

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of feature names generated by this transformer."""
        names = []
        for res in self.resolutions:
            names.extend([f'h3_res{res}', f'h3_res{res}_encoded'])
        return names


class MarketTrendFeatures:
    """
    Calculate market trend features using rolling windows.

    Real estate markets have strong temporal patterns. These features capture
    market dynamics that affect property prices.

    Features generated:
        - district_price_{N}d_mean: Rolling N-day mean price by district
        - district_price_{N}d_std: Rolling N-day std by district
        - price_momentum_{N}d: N-day price change rate
        - price_vs_district_mean: Z-score vs district average
        - days_since_first_listing: Market age indicator

    References:
        - Zillow improved algorithm for "dynamic market conditions"
          (GeekWire, 2021)

    Example:
        >>> trends = MarketTrendFeatures(windows=[30, 60, 90])
        >>> trends.fit(train_df, 'price_per_m2')
        >>> train_df = trends.transform(train_df, 'price_per_m2')
    """

    def __init__(self, windows: List[int] = None):
        """
        Initialize MarketTrendFeatures.

        Args:
            windows: List of rolling window sizes in days (default: [30, 60, 90])
        """
        self.windows = windows or [30, 60, 90]
        self.district_stats: Dict[str, Dict[str, float]] = {}
        self.global_mean: float = 0
        self.global_std: float = 1
        self.first_date: pd.Timestamp = None
        self._fitted = False

    def fit(self, df: pd.DataFrame,
            price_col: str = 'price_per_m2',
            district_col: str = 'district',
            date_col: str = 'parsed_at') -> 'MarketTrendFeatures':
        """
        Fit on training data to calculate baseline statistics.

        Args:
            df: Training DataFrame
            price_col: Name of price column
            district_col: Name of district column
            date_col: Name of date column

        Returns:
            self
        """
        # Global statistics
        self.global_mean = df[price_col].mean()
        self.global_std = df[price_col].std()
        self.first_date = df[date_col].min()

        # District-level statistics
        for district in df[district_col].unique():
            district_df = df[df[district_col] == district]
            self.district_stats[district] = {
                'mean': district_df[price_col].mean(),
                'std': max(district_df[price_col].std(), 1),  # Avoid division by zero
                'count': len(district_df),
            }

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame,
                  price_col: str = 'price_per_m2',
                  district_col: str = 'district',
                  date_col: str = 'parsed_at') -> pd.DataFrame:
        """
        Transform DataFrame by adding market trend features.

        Args:
            df: DataFrame to transform
            price_col: Name of price column
            district_col: Name of district column
            date_col: Name of date column

        Returns:
            DataFrame with market trend features added
        """
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")

        df = df.copy()
        df = df.sort_values(date_col)

        # Rolling statistics by district
        for days in self.windows:
            # Rolling mean
            col_mean = f'district_price_{days}d_mean'
            df[col_mean] = df.groupby(district_col)[price_col].transform(
                lambda x: x.rolling(window=days, min_periods=1).mean()
            )

            # Rolling std
            col_std = f'district_price_{days}d_std'
            df[col_std] = df.groupby(district_col)[price_col].transform(
                lambda x: x.rolling(window=days, min_periods=2).std()
            )
            df[col_std] = df[col_std].fillna(0)

        # Price momentum (percentage change over 30 days)
        df['price_momentum_30d'] = df.groupby(district_col)[price_col].transform(
            lambda x: x.pct_change(periods=min(30, len(x)-1)).fillna(0)
        )

        # Z-score vs district average
        df['price_vs_district_mean'] = df.apply(
            lambda r: self._calculate_zscore(r, price_col, district_col),
            axis=1
        )

        # Days since first listing in dataset
        df['days_since_first_listing'] = (df[date_col] - self.first_date).dt.days

        return df

    def _calculate_zscore(self, row: pd.Series, price_col: str, district_col: str) -> float:
        """Calculate z-score of price vs district average."""
        district = row[district_col]
        price = row[price_col]

        if district in self.district_stats:
            mean = self.district_stats[district]['mean']
            std = self.district_stats[district]['std']
        else:
            mean = self.global_mean
            std = self.global_std

        return (price - mean) / std

    def get_feature_names(self) -> List[str]:
        """Return list of feature names generated by this transformer."""
        names = []
        for days in self.windows:
            names.extend([
                f'district_price_{days}d_mean',
                f'district_price_{days}d_std'
            ])
        names.extend([
            'price_momentum_30d',
            'price_vs_district_mean',
            'days_since_first_listing'
        ])
        return names


class DensityFeatures:
    """
    Calculate listing density features within specified radii.

    Density serves as a proxy for market activity and area desirability.
    High-density areas often indicate popular neighborhoods.

    Features generated:
        - listings_{N}m: Count of listings within N meters
        - listings_{N}m_ratio: Density ratio vs city average
        - is_high_density_{N}m: Binary flag for above-average density

    Example:
        >>> density = DensityFeatures(radii_km=[0.5, 1.0])
        >>> density.fit(train_df)
        >>> train_df = density.transform(train_df)
    """

    EARTH_RADIUS_KM = 6371.0

    def __init__(self, radii_km: List[float] = None):
        """
        Initialize DensityFeatures.

        Args:
            radii_km: List of radii in kilometers (default: [0.5, 1.0])
        """
        self.radii_km = radii_km or [0.5, 1.0]
        self.tree = None
        self.avg_density: Dict[float, float] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame,
            lat_col: str = 'latitude',
            lon_col: str = 'longitude') -> 'DensityFeatures':
        """
        Fit on training data to build spatial index and calculate average densities.

        Args:
            df: Training DataFrame
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            self
        """
        coords = np.radians(df[[lat_col, lon_col]].values)
        self.tree = BallTree(coords, metric='haversine')

        # Calculate average densities for each radius
        for radius in self.radii_km:
            radius_rad = radius / self.EARTH_RADIUS_KM
            counts = self.tree.query_radius(coords, r=radius_rad, count_only=True)
            self.avg_density[radius] = np.mean(counts)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame,
                  lat_col: str = 'latitude',
                  lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Transform DataFrame by adding density features.

        Args:
            df: DataFrame to transform
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            DataFrame with density features added
        """
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")

        df = df.copy()
        coords = np.radians(df[[lat_col, lon_col]].values)

        for radius in self.radii_km:
            radius_rad = radius / self.EARTH_RADIUS_KM

            # Count neighbors
            counts = self.tree.query_radius(coords, r=radius_rad, count_only=True)

            col_name = f'listings_{int(radius * 1000)}m'
            df[col_name] = counts

            # Density ratio
            avg = max(self.avg_density[radius], 1)
            df[f'{col_name}_ratio'] = counts / avg

            # High density flag
            df[f'is_high_density_{int(radius * 1000)}m'] = (counts > avg).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of feature names generated by this transformer."""
        names = []
        for radius in self.radii_km:
            r_m = int(radius * 1000)
            names.extend([
                f'listings_{r_m}m',
                f'listings_{r_m}m_ratio',
                f'is_high_density_{r_m}m'
            ])
        return names


class AdvancedFeaturePipeline:
    """
    Complete pipeline combining all advanced features.

    This pipeline orchestrates all feature transformers with proper
    fit/transform separation to prevent data leakage.

    Example:
        >>> pipeline = AdvancedFeaturePipeline()
        >>> pipeline.fit(train_df, 'price_per_m2')
        >>> train_features = pipeline.transform(train_df)
        >>> test_features = pipeline.transform(test_df)
    """

    def __init__(self, config: FeatureConfig = None):
        """
        Initialize AdvancedFeaturePipeline.

        Args:
            config: FeatureConfig with parameters (uses defaults if None)
        """
        self.config = config or FeatureConfig()

        # Initialize transformers
        self.spatial_lag = SpatialLagFeatures(
            radius_km=self.config.spatial_lag_radius_km,
            min_neighbors=self.config.spatial_lag_min_neighbors
        )

        self.h3_features = None
        if H3_AVAILABLE:
            self.h3_features = H3Features(
                resolutions=self.config.h3_resolutions
            )

        self.market_trends = MarketTrendFeatures(
            windows=self.config.trend_windows_days
        )

        self.density = DensityFeatures(
            radii_km=self.config.density_radii_km
        )

        self._fitted = False

    def fit(self, df: pd.DataFrame,
            price_col: str = 'price_per_m2',
            lat_col: str = 'latitude',
            lon_col: str = 'longitude',
            district_col: str = 'district',
            date_col: str = 'parsed_at') -> 'AdvancedFeaturePipeline':
        """
        Fit all transformers on training data.

        Args:
            df: Training DataFrame
            price_col: Name of price column
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            district_col: Name of district column
            date_col: Name of date column

        Returns:
            self
        """
        print("Fitting advanced feature pipeline...")

        # Spatial lag
        print("  - Spatial lag features")
        self.spatial_lag.fit(df, price_col, lat_col, lon_col)

        # H3 (if available)
        if self.h3_features:
            print("  - H3 geographic tiles")
            self.h3_features.fit(df, lat_col, lon_col)

        # Market trends
        print("  - Market trend features")
        self.market_trends.fit(df, price_col, district_col, date_col)

        # Density
        print("  - Density features")
        self.density.fit(df, lat_col, lon_col)

        self._fitted = True
        print("Pipeline fitted successfully!")
        return self

    def transform(self, df: pd.DataFrame,
                  price_col: str = 'price_per_m2',
                  lat_col: str = 'latitude',
                  lon_col: str = 'longitude',
                  district_col: str = 'district',
                  date_col: str = 'parsed_at') -> pd.DataFrame:
        """
        Apply all transformations to DataFrame.

        Args:
            df: DataFrame to transform
            price_col: Name of price column
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            district_col: Name of district column
            date_col: Name of date column

        Returns:
            DataFrame with all advanced features added
        """
        if not self._fitted:
            raise ValueError("Must call fit() before transform()")

        # Apply transformers in sequence
        df = self.spatial_lag.transform(df, lat_col, lon_col)

        if self.h3_features:
            df = self.h3_features.transform(df, lat_col, lon_col)

        df = self.market_trends.transform(df, price_col, district_col, date_col)
        df = self.density.transform(df, lat_col, lon_col)

        return df

    def fit_transform(self, df: pd.DataFrame,
                      price_col: str = 'price_per_m2',
                      **kwargs) -> pd.DataFrame:
        """Convenience method to fit and transform in one call."""
        return self.fit(df, price_col, **kwargs).transform(df, price_col, **kwargs)

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names generated by pipeline."""
        names = []
        names.extend(self.spatial_lag.get_feature_names())
        if self.h3_features:
            names.extend(self.h3_features.get_feature_names())
        names.extend(self.market_trends.get_feature_names())
        names.extend(self.density.get_feature_names())
        return names


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_importance_comparison(
    model,
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_repeats: int = 10
) -> pd.DataFrame:
    """
    Compare feature importances for new vs existing features.

    Args:
        model: Trained model with predict method
        feature_names: List of feature names
        X_test: Test features
        y_test: Test targets
        n_repeats: Number of permutation repeats

    Returns:
        DataFrame with feature importances
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    # Mark new features
    new_features = (
        SpatialLagFeatures(radius_km=0.5).get_feature_names() +
        MarketTrendFeatures().get_feature_names() +
        DensityFeatures().get_feature_names()
    )
    if H3_AVAILABLE:
        new_features.extend(H3Features().get_feature_names())

    importance_df['is_new'] = importance_df['feature'].isin(new_features)

    return importance_df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Advanced Feature Engineering Module")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    sample_df = pd.DataFrame({
        'latitude': np.random.uniform(42.85, 42.90, n_samples),
        'longitude': np.random.uniform(74.55, 74.65, n_samples),
        'price_per_m2': np.random.uniform(800, 2000, n_samples),
        'district': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'parsed_at': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
    })

    # Split data
    train_df = sample_df.iloc[:800].copy()
    test_df = sample_df.iloc[800:].copy()

    # Create pipeline
    config = FeatureConfig(
        spatial_lag_radius_km=0.5,
        h3_resolutions=[8, 9],
        trend_windows_days=[30, 60],
        density_radii_km=[0.5, 1.0]
    )

    pipeline = AdvancedFeaturePipeline(config)

    # Fit and transform
    train_features = pipeline.fit_transform(train_df, 'price_per_m2')
    test_features = pipeline.transform(test_df, 'price_per_m2')

    print(f"\nOriginal features: {len(sample_df.columns)}")
    print(f"New features added: {len(pipeline.get_feature_names())}")
    print(f"Total features: {len(train_features.columns)}")
    print(f"\nNew features: {pipeline.get_feature_names()}")
