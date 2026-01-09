# %% [markdown]
# # Bishkek Real Estate: Multimodal Price Prediction
# ## Computer Vision + Tabular Features
#
# This notebook implements a multimodal approach combining:
# - **Tabular features**: 39 engineered features (POI distances, building attributes, etc.)
# - **Image embeddings**: ResNet-50 features from property photos
# - **Ensemble model**: XGBoost + LightGBM + CatBoost with Optuna tuning
#
# ### Research Foundation
# Based on state-of-the-art papers:
# - [MHPP (arXiv 2024)](https://arxiv.org/abs/2409.05335): +21-26% MAE improvement with images
# - [PLOS One 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088074/): ResNet-101 + t-SNE, R² 0.809→0.821
# - [NBER Study](https://www.nber.org/papers/w25174): Images explain 11.7% of price variance
# - Zillow Neural Zestimate: CNN for quality detection, +20% accuracy
#
# ### Expected Improvement
# | Metric | Baseline (Tabular) | Target (Multimodal) |
# |--------|-------------------|---------------------|
# | MedAPE | 5.49% | 4.0-4.5% |
# | R² | 0.76 | 0.80-0.82 |
# | MAE | $121.71/m² | $100-110/m² |

# %% [markdown]
# ## 1. Setup & Configuration

# %%
import os
import sys
import warnings
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Boosting models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Optuna for hyperparameter tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - using default hyperparameters")

# PyTorch for image processing
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch available, device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("PyTorch not available - image features disabled")

# %%
@dataclass
class Config:
    """Configuration for the multimodal model"""
    # Data paths (adjust for Kaggle vs local)
    IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/bishkek-real-estate-2025')
        CSV_PATH = DATA_DIR / 'listings.csv'
        IMAGES_DIR = DATA_DIR / 'images'
    else:
        # HuggingFace dataset
        DATA_DIR = Path('/Users/admin/PycharmProjects/real-estate-valuation/data/hf_bishkek_dataset')
        CSV_PATH = DATA_DIR / 'listings.csv'
        IMAGES_DIR = Path('/Users/admin/PycharmProjects/real-estate-valuation/data/images/bishkek')

    # Image processing
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    IMAGE_EMBEDDING_DIM = 2048  # ResNet-50 output
    PCA_COMPONENTS = 64  # Reduced dimension for final features
    MAX_IMAGES_PER_LISTING = 10  # Limit to avoid memory issues

    # Model training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_OPTUNA_TRIALS = 30
    CV_FOLDS = 5

    # Target
    TARGET_COL = 'price_per_m2'

config = Config()
print(f"Running on: {'Kaggle' if config.IS_KAGGLE else 'Local'}")
print(f"Data dir: {config.DATA_DIR}")

# %% [markdown]
# ## 2. POI (Points of Interest) Data
# Coordinates of key locations in Bishkek for distance-based features

# %%
# POI Bishkek - key locations by category
BISHKEK_POI = {
    'bazaars': [
        ('osh_bazaar', 42.874823, 74.569599),
        ('dordoi_bazaar', 42.939732, 74.620613),
        ('ortosay_bazaar', 42.836209, 74.615931),
        ('alamedin_bazaar', 42.88683, 74.637305),
    ],
    'parks': [
        ('dubovy_park', 42.877681, 74.606759),
        ('ataturk_park', 42.839587, 74.595725),
        ('karagach_grove', 42.900362, 74.619652),
        ('victory_park', 42.826531, 74.604411),
        ('botanical_garden', 42.857152, 74.590671),
    ],
    'malls': [
        ('bishkek_park', 42.875029, 74.590403),
        ('dordoi_plaza', 42.874685, 74.618469),
        ('vefa_center', 42.857078, 74.609628),
        ('tsum', 42.876813, 74.61499),
    ],
    'universities': [
        ('auca', 42.81132, 74.627743),
        ('krsu', 42.874862, 74.627114),
        ('bhu', 42.850424, 74.585821),
        ('knu', 42.8822, 74.586638),
    ],
    'hospitals': [
        ('national_hospital', 42.869973, 74.596739),
        ('city_hospital', 42.876149, 74.5619),
    ],
    'transport': [
        ('west_bus_station', 42.873213, 74.406103),
        ('east_bus_station', 42.887128, 74.62894),
        ('railway_station', 42.864179, 74.605693),
    ],
    'admin': [
        ('jogorku_kenesh', 42.876814, 74.600155),
        ('ala_too_square', 42.875039, 74.603604),
        ('erkindik_boulevard', 42.864402, 74.605287),
    ],
}

# Premium zones - central expensive areas
BISHKEK_PREMIUM_ZONES = {
    'golden_square': (42.8688, 74.6033),
    'voentorg': (42.8722, 74.5941),
    'railway_area': (42.8650, 74.6070),
    'mossovet': (42.8700, 74.6117),
}

# City center
BISHKEK_CENTER = (42.8746, 74.5698)  # Ala-Too Square


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points on Earth (in km)"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def get_min_distance_to_category(lat: float, lon: float, category: str) -> float:
    """Get minimum distance to any POI in category"""
    if category not in BISHKEK_POI:
        return np.nan
    distances = [haversine_distance(lat, lon, poi[1], poi[2])
                 for poi in BISHKEK_POI[category]]
    return min(distances) if distances else np.nan

# %% [markdown]
# ## 3. Data Loading

# %%
def load_data() -> pd.DataFrame:
    """Load and prepare the dataset"""
    print("Loading data...")

    # Try different data sources
    if config.IS_KAGGLE:
        # Kaggle dataset
        df = pd.read_csv(config.CSV_PATH)
    else:
        # Try HuggingFace first
        try:
            from huggingface_hub import hf_hub_download
            csv_path = hf_hub_download(
                repo_id="raimbekovm/bishkek-real-estate",
                filename="data/bishkek_apartments.csv",
                repo_type="dataset"
            )
            df = pd.read_csv(csv_path)
            print("Loaded from HuggingFace")
        except:
            # Fall back to local
            df = pd.read_csv(config.CSV_PATH)
            print("Loaded from local file")

    print(f"Dataset: {len(df)} listings, {len(df.columns)} columns")
    return df


df = load_data()
df.head()

# %% [markdown]
# ## 4. Feature Engineering

# %%
class TabularFeatureEngineer:
    """
    Feature engineering pipeline for tabular data.
    Creates 39 features from raw listing data.
    """

    def __init__(self):
        self.feature_names = []
        self.target_encoders = {}

    def fit_transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        df = df.copy()

        # 1. Basic features
        df = self._create_basic_features(df)

        # 2. Building features
        df = self._create_building_features(df)

        # 3. Apartment features
        df = self._create_apartment_features(df)

        # 4. POI distance features
        df = self._create_poi_features(df)

        # 5. Target encoding (fit on train only)
        if is_train:
            df = self._fit_target_encoding(df)
        else:
            df = self._transform_target_encoding(df)

        return df

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic numeric features"""
        # Core features
        df['rooms'] = pd.to_numeric(df.get('rooms', 0), errors='coerce').fillna(2)
        df['area'] = pd.to_numeric(df.get('area', 0), errors='coerce').fillna(50)
        df['floor'] = pd.to_numeric(df.get('floor', 0), errors='coerce').fillna(1)
        df['total_floors'] = pd.to_numeric(df.get('total_floors', 0), errors='coerce').fillna(5)

        # Derived features
        df['floor_ratio'] = df['floor'] / df['total_floors'].replace(0, 1)
        df['area_per_room'] = df['area'] / df['rooms'].replace(0, 1)
        df['is_first_floor'] = (df['floor'] == 1).astype(int)
        df['is_last_floor'] = (df['floor'] == df['total_floors']).astype(int)

        return df

    def _create_building_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Building type and age features"""
        # Year built
        current_year = 2026
        df['year_built'] = pd.to_numeric(df.get('year_built', 0), errors='coerce')
        df['year_built'] = df['year_built'].apply(
            lambda x: x if 1950 <= x <= current_year else np.nan
        )
        df['building_age'] = current_year - df['year_built'].fillna(current_year - 20)

        # Building type flags
        house_type = df.get('house_type', '').fillna('').str.lower()
        df['is_monolith'] = house_type.str.contains('монолит', na=False).astype(int)
        df['is_brick'] = house_type.str.contains('кирпич', na=False).astype(int)
        df['is_panel'] = house_type.str.contains('панель', na=False).astype(int)

        # Building categories
        df['is_new_building'] = (df['building_age'] <= 5).astype(int)
        df['is_soviet'] = (df['building_age'] >= 30).astype(int)
        df['is_highrise'] = (df['total_floors'] >= 9).astype(int)

        return df

    def _create_apartment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apartment-specific features"""
        # Kitchen and living area
        df['kitchen_area'] = pd.to_numeric(df.get('kitchen_area', 0), errors='coerce').fillna(0)
        df['living_area'] = pd.to_numeric(df.get('living_area', 0), errors='coerce').fillna(0)

        df['kitchen_ratio'] = df['kitchen_area'] / df['area'].replace(0, 1)
        df['living_ratio'] = df['living_area'] / df['area'].replace(0, 1)

        # Ceiling height
        df['ceiling_height'] = pd.to_numeric(df.get('ceiling_height', 0), errors='coerce')
        df['ceiling_height'] = df['ceiling_height'].apply(
            lambda x: x if 2.0 <= x <= 4.5 else np.nan
        ).fillna(2.7)

        # Condition score
        condition_map = {
            'черновая': 1, 'предчистовая': 2, 'требует ремонта': 3,
            'средний ремонт': 4, 'хороший ремонт': 5,
            'евроремонт': 6, 'дизайнерский': 7
        }
        df['condition_score'] = df.get('condition', '').map(condition_map).fillna(4)

        # Amenities (binary features)
        for col in ['balcony', 'parking', 'furniture', 'internet', 'security']:
            if col in df.columns:
                df[f'has_{col}'] = df[col].notna().astype(int)
            else:
                df[f'has_{col}'] = 0

        return df

    def _create_poi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distance to Points of Interest"""
        # Ensure coordinates exist
        df['latitude'] = pd.to_numeric(df.get('latitude', 0), errors='coerce')
        df['longitude'] = pd.to_numeric(df.get('longitude', 0), errors='coerce')

        # Fill missing coordinates with city center
        df['latitude'] = df['latitude'].apply(
            lambda x: x if 42.7 <= x <= 43.0 else BISHKEK_CENTER[0]
        )
        df['longitude'] = df['longitude'].apply(
            lambda x: x if 74.3 <= x <= 74.8 else BISHKEK_CENTER[1]
        )

        # Distance to city center
        df['dist_to_center'] = df.apply(
            lambda r: haversine_distance(r['latitude'], r['longitude'],
                                        BISHKEK_CENTER[0], BISHKEK_CENTER[1]),
            axis=1
        )

        # Distance to POI categories
        for category in BISHKEK_POI.keys():
            df[f'dist_to_{category}'] = df.apply(
                lambda r: get_min_distance_to_category(r['latitude'], r['longitude'], category),
                axis=1
            )

        # Premium zone features
        premium_distances = []
        for _, coords in BISHKEK_PREMIUM_ZONES.items():
            dist = df.apply(
                lambda r: haversine_distance(r['latitude'], r['longitude'], coords[0], coords[1]),
                axis=1
            )
            premium_distances.append(dist)

        df['dist_to_premium'] = pd.concat(premium_distances, axis=1).min(axis=1)
        df['is_premium_zone'] = (df['dist_to_premium'] <= 1.0).astype(int)

        return df

    def _fit_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply target encoding for categorical features"""
        target = config.TARGET_COL

        # District encoding
        if 'district' in df.columns and target in df.columns:
            district_means = df.groupby('district')[target].mean()
            global_mean = df[target].mean()
            self.target_encoders['district'] = district_means.to_dict()
            self.target_encoders['district_global'] = global_mean
            df['district_encoded'] = df['district'].map(district_means).fillna(global_mean)

        # Residential complex encoding
        if 'residential_complex' in df.columns and target in df.columns:
            jk_means = df.groupby('residential_complex')[target].mean()
            self.target_encoders['jk'] = jk_means.to_dict()
            df['jk_encoded'] = df['residential_complex'].map(jk_means).fillna(global_mean)
            df['has_jk'] = df['residential_complex'].notna().astype(int)
        else:
            df['jk_encoded'] = df.get(target, 1500)
            df['has_jk'] = 0

        return df

    def _transform_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fitted target encoding"""
        if 'district' in self.target_encoders:
            global_mean = self.target_encoders['district_global']
            df['district_encoded'] = df['district'].map(
                self.target_encoders['district']
            ).fillna(global_mean)

        if 'jk' in self.target_encoders:
            df['jk_encoded'] = df.get('residential_complex', '').map(
                self.target_encoders['jk']
            ).fillna(self.target_encoders['district_global'])
            df['has_jk'] = df.get('residential_complex', '').notna().astype(int)

        return df

    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns for model training"""
        return [
            # Basic (8)
            'rooms', 'area', 'floor', 'total_floors',
            'floor_ratio', 'area_per_room', 'is_first_floor', 'is_last_floor',
            # Building (6)
            'building_age', 'is_monolith', 'is_brick', 'is_panel',
            'is_new_building', 'is_soviet', 'is_highrise',
            # Apartment (9)
            'kitchen_ratio', 'living_ratio', 'ceiling_height', 'condition_score',
            'has_balcony', 'has_parking', 'has_furniture', 'has_internet', 'has_security',
            # POI (10)
            'dist_to_center', 'dist_to_bazaars', 'dist_to_parks', 'dist_to_malls',
            'dist_to_universities', 'dist_to_hospitals', 'dist_to_transport',
            'dist_to_admin', 'dist_to_premium', 'is_premium_zone',
            # Encoding (3)
            'district_encoded', 'jk_encoded', 'has_jk',
            # Coordinates (2)
            'latitude', 'longitude',
        ]

# %% [markdown]
# ## 5. Image Feature Extraction
#
# Using ResNet-50 pretrained on ImageNet to extract visual features.
# Based on research showing images explain 11.7% of price variance (NBER).

# %%
class ImageFeatureExtractor:
    """
    Extract image embeddings using pretrained ResNet-50.

    Research basis:
    - MHPP (arXiv 2024): CLIP+ResNet50 → 21-26% MAE improvement
    - PLOS One 2025: ResNet-101 + t-SNE → R² +1.5%
    - Zillow: CNN for quality detection → +20% accuracy

    Architecture:
    - ResNet-50 pretrained on ImageNet
    - Remove classification head → 2048-dim embeddings
    - Mean pooling across multiple images per listing
    - PCA reduction → 64 dimensions
    """

    def __init__(self, pca_components: int = 64, batch_size: int = 32):
        self.pca_components = pca_components
        self.batch_size = batch_size
        self.pca = None
        self.model = None
        self.transform = None

        if TORCH_AVAILABLE:
            self._setup_model()

    def _setup_model(self):
        """Initialize ResNet-50 for feature extraction"""
        print("Loading ResNet-50...")

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove classification head, keep up to avgpool
        # Output: 2048-dimensional feature vector
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()
        self.model.to(DEVICE)

        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"Model loaded on {DEVICE}")

    def extract_batch(self, image_paths: List[Path]) -> np.ndarray:
        """
        Extract embeddings for a batch of images.
        Returns: (N, 2048) array of embeddings
        """
        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros((len(image_paths), config.IMAGE_EMBEDDING_DIM))

        embeddings = []
        batch_tensors = []
        valid_indices = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert('RGB')
                tensor = self.transform(img)
                batch_tensors.append(tensor)
                valid_indices.append(i)
                img.close()  # Explicitly close to free memory
            except Exception as e:
                continue

        if not batch_tensors:
            return np.zeros((len(image_paths), config.IMAGE_EMBEDDING_DIM))

        # Stack and process batch
        batch = torch.stack(batch_tensors).to(DEVICE)

        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims

        # Move to CPU and convert to numpy
        batch_embeddings = features.cpu().numpy()

        # Create full array with zeros for failed images
        result = np.zeros((len(image_paths), config.IMAGE_EMBEDDING_DIM))
        for j, idx in enumerate(valid_indices):
            result[idx] = batch_embeddings[j]

        # Clear CUDA cache periodically
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        return result

    def extract_listing_embedding(
        self,
        listing_id: str,
        images_dir: Path,
        max_images: int = 10
    ) -> np.ndarray:
        """
        Extract mean embedding for all images of a listing.

        Strategy: Mean pooling (based on MHPP research)
        - Robust to varying number of images per listing
        - Captures overall property appearance
        """
        listing_dir = images_dir / str(listing_id)

        if not listing_dir.exists():
            return np.zeros(config.IMAGE_EMBEDDING_DIM)

        # Get image paths (limit to max_images)
        image_paths = sorted(listing_dir.glob("*.jpg"))[:max_images]

        if not image_paths:
            return np.zeros(config.IMAGE_EMBEDDING_DIM)

        # Extract embeddings for all images
        embeddings = self.extract_batch(image_paths)

        # Filter out zero embeddings (failed images)
        valid_mask = embeddings.sum(axis=1) != 0
        valid_embeddings = embeddings[valid_mask]

        if len(valid_embeddings) == 0:
            return np.zeros(config.IMAGE_EMBEDDING_DIM)

        # Mean pooling across all images
        return np.mean(valid_embeddings, axis=0)

    def extract_all_listings(
        self,
        listing_ids: List[str],
        images_dir: Path,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract embeddings for all listings.
        Returns: (N_listings, 2048) array
        """
        print(f"Extracting embeddings for {len(listing_ids)} listings...")

        embeddings = []
        iterator = tqdm(listing_ids, desc="Extracting images") if show_progress else listing_ids

        for i, listing_id in enumerate(iterator):
            emb = self.extract_listing_embedding(
                listing_id,
                images_dir,
                max_images=config.MAX_IMAGES_PER_LISTING
            )
            embeddings.append(emb)

            # Clear memory every 500 listings
            if i > 0 and i % 500 == 0:
                gc.collect()
                if DEVICE is not None and DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()

        return np.array(embeddings)

    def fit_pca(self, embeddings: np.ndarray) -> 'ImageFeatureExtractor':
        """Fit PCA on training embeddings"""
        print(f"Fitting PCA: {embeddings.shape[1]} → {self.pca_components} dimensions")

        # Filter out all-zero embeddings for PCA fitting
        valid_mask = embeddings.sum(axis=1) != 0
        valid_embeddings = embeddings[valid_mask]

        if len(valid_embeddings) < self.pca_components:
            print(f"Warning: Only {len(valid_embeddings)} valid embeddings, reducing PCA components")
            self.pca_components = max(10, len(valid_embeddings) // 2)

        self.pca = PCA(n_components=self.pca_components, random_state=42)
        self.pca.fit(valid_embeddings)

        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.1%}")

        return self

    def transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA transformation"""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        return self.pca.transform(embeddings)

    def fit_transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit_pca(embeddings)
        return self.transform_pca(embeddings)

# %% [markdown]
# ## 6. Metrics

# %%
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Percentage errors
    ape = np.abs((y_true - y_pred) / y_true) * 100
    mape = np.mean(ape)
    medape = np.median(ape)

    # Within X% accuracy
    within_5 = (ape <= 5).mean() * 100
    within_10 = (ape <= 10).mean() * 100
    within_20 = (ape <= 20).mean() * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'MedAPE': medape,
        'Within_5%': within_5,
        'Within_10%': within_10,
        'Within_20%': within_20,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Results"):
    """Pretty print metrics"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)
    print(f"  MAE:      ${metrics['MAE']:.2f}/m²")
    print(f"  RMSE:     ${metrics['RMSE']:.2f}/m²")
    print(f"  R²:       {metrics['R2']:.4f}")
    print(f"  MAPE:     {metrics['MAPE']:.2f}%")
    print(f"  MedAPE:   {metrics['MedAPE']:.2f}%")
    print(f"  Within 5%:  {metrics['Within_5%']:.1f}%")
    print(f"  Within 10%: {metrics['Within_10%']:.1f}%")
    print(f"  Within 20%: {metrics['Within_20%']:.1f}%")
    print('='*50)

# %% [markdown]
# ## 7. Model Training with Optuna

# %%
class MultimodalEnsemble:
    """
    Ensemble model for multimodal real estate price prediction.
    Combines XGBoost + LightGBM + CatBoost with Ridge meta-learner.
    """

    def __init__(self, use_optuna: bool = True, n_trials: int = 30):
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.n_trials = n_trials
        self.models = {}
        self.best_params = {}
        self.ensemble = None
        self.scaler = StandardScaler()

        # Detect GPU availability
        self._detect_gpu()

    def _detect_gpu(self):
        """Detect GPU availability for boosting models"""
        self.gpu_available = False

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_available = True
            print("GPU detected - enabling GPU acceleration for boosting models")

    def _get_xgb_params(self, trial: Optional['optuna.Trial'] = None) -> Dict:
        """Get XGBoost parameters (with optional Optuna tuning)"""
        if trial is not None:
            params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-4, 10, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-4, 10, log=True),
            }
        else:
            # Default params from previous Optuna runs
            params = {
                'n_estimators': 907,
                'max_depth': 10,
                'learning_rate': 0.0147,
                'subsample': 0.893,
                'colsample_bytree': 0.691,
                'min_child_weight': 5,
                'reg_alpha': 0.00157,
                'reg_lambda': 5.27e-05,
            }

        # Add GPU params if available
        if self.gpu_available:
            params['device'] = 'cuda'
            params['tree_method'] = 'hist'

        params['random_state'] = config.RANDOM_STATE
        params['n_jobs'] = -1

        return params

    def _get_lgb_params(self, trial: Optional['optuna.Trial'] = None) -> Dict:
        """Get LightGBM parameters"""
        if trial is not None:
            params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100),
                'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
            }
        else:
            params = {
                'n_estimators': 755,
                'max_depth': 11,
                'learning_rate': 0.075,
                'num_leaves': 50,
                'subsample': 0.963,
                'colsample_bytree': 0.609,
            }

        if self.gpu_available:
            params['device'] = 'gpu'

        params['random_state'] = config.RANDOM_STATE
        params['n_jobs'] = -1
        params['verbose'] = -1

        return params

    def _get_cat_params(self, trial: Optional['optuna.Trial'] = None) -> Dict:
        """Get CatBoost parameters"""
        if trial is not None:
            params = {
                'iterations': trial.suggest_int('cat_iterations', 100, 1000),
                'depth': trial.suggest_int('cat_depth', 4, 10),
                'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 0.1, 10, log=True),
            }
        else:
            params = {
                'iterations': 368,
                'depth': 8,
                'learning_rate': 0.216,
                'l2_leaf_reg': 0.825,
            }

        if self.gpu_available:
            params['task_type'] = 'GPU'

        params['random_state'] = config.RANDOM_STATE
        params['verbose'] = 0

        return params

    def _objective(self, trial: 'optuna.Trial', X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function for hyperparameter tuning"""
        # Get parameters
        xgb_params = self._get_xgb_params(trial)
        lgb_params = self._get_lgb_params(trial)
        cat_params = self._get_cat_params(trial)

        # Create models
        xgb = XGBRegressor(**xgb_params)
        lgb = LGBMRegressor(**lgb_params)
        cat = CatBoostRegressor(**cat_params)

        # Cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
        scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train models
            xgb.fit(X_train, y_train)
            lgb.fit(X_train, y_train)
            cat.fit(X_train, y_train)

            # Ensemble prediction (simple average)
            pred = (xgb.predict(X_val) + lgb.predict(X_val) + cat.predict(X_val)) / 3

            mae = mean_absolute_error(y_val, pred)
            scores.append(mae)

        return np.mean(scores)

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Run Optuna hyperparameter tuning"""
        print(f"\nRunning Optuna tuning ({self.n_trials} trials)...")

        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        print(f"Best MAE: ${study.best_value:.2f}/m²")

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultimodalEnsemble':
        """Fit the ensemble model"""
        print("\nTraining ensemble model...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Optuna tuning if enabled
        if self.use_optuna:
            self.tune_hyperparameters(X_scaled, y)

        # Create models with best/default params
        xgb_params = self._get_xgb_params()
        lgb_params = self._get_lgb_params()
        cat_params = self._get_cat_params()

        # Update with Optuna params if available
        for key, value in self.best_params.items():
            if key.startswith('xgb_'):
                xgb_params[key[4:]] = value
            elif key.startswith('lgb_'):
                lgb_params[key[4:]] = value
            elif key.startswith('cat_'):
                cat_params[key[4:]] = value

        # Create estimators
        estimators = [
            ('xgb', XGBRegressor(**xgb_params)),
            ('lgb', LGBMRegressor(**lgb_params)),
            ('cat', CatBoostRegressor(**cat_params)),
        ]

        # Stacking ensemble
        self.ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=config.CV_FOLDS,
            n_jobs=-1
        )

        print("Fitting stacking ensemble...")
        self.ensemble.fit(X_scaled, y)

        # Store individual models for feature importance
        self.models = {name: model for name, model in estimators}

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict(X_scaled)

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from XGBoost model"""
        xgb_model = self.ensemble.named_estimators_['xgb']
        importance = xgb_model.feature_importances_

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

# %% [markdown]
# ## 8. Main Training Pipeline

# %%
def run_training_pipeline(
    use_images: bool = True,
    use_optuna: bool = True,
    n_optuna_trials: int = 30
) -> Dict:
    """
    Main training pipeline for multimodal model.

    Args:
        use_images: Whether to include image features
        use_optuna: Whether to use Optuna for hyperparameter tuning
        n_optuna_trials: Number of Optuna trials

    Returns:
        Dictionary with results and trained models
    """
    results = {}

    # 1. Load data
    print("\n" + "="*60)
    print(" MULTIMODAL REAL ESTATE PRICE PREDICTION")
    print("="*60)

    df = load_data()

    # 2. Filter valid listings
    # Need price and coordinates
    df = df[df[config.TARGET_COL].notna()].copy()
    df = df[df[config.TARGET_COL] > 0].copy()

    # Remove outliers (below 1st and above 99th percentile)
    q01 = df[config.TARGET_COL].quantile(0.01)
    q99 = df[config.TARGET_COL].quantile(0.99)
    df = df[(df[config.TARGET_COL] >= q01) & (df[config.TARGET_COL] <= q99)]

    print(f"Valid listings: {len(df)}")

    # 3. Train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 4. Feature engineering
    print("\n--- Tabular Feature Engineering ---")
    feature_engineer = TabularFeatureEngineer()

    train_df = feature_engineer.fit_transform(train_df, is_train=True)
    test_df = feature_engineer.fit_transform(test_df, is_train=False)

    feature_cols = feature_engineer.get_feature_columns()

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in test_df.columns:
            test_df[col] = 0

    X_train_tabular = train_df[feature_cols].values
    X_test_tabular = test_df[feature_cols].values
    y_train = train_df[config.TARGET_COL].values
    y_test = test_df[config.TARGET_COL].values

    print(f"Tabular features: {len(feature_cols)}")

    # 5. Image features (optional)
    if use_images and TORCH_AVAILABLE:
        print("\n--- Image Feature Extraction ---")

        # Get listing IDs
        id_col = 'listing_id' if 'listing_id' in train_df.columns else 'id'
        train_ids = train_df[id_col].astype(str).tolist()
        test_ids = test_df[id_col].astype(str).tolist()

        # Extract embeddings
        img_extractor = ImageFeatureExtractor(
            pca_components=config.PCA_COMPONENTS,
            batch_size=config.BATCH_SIZE
        )

        train_embeddings = img_extractor.extract_all_listings(
            train_ids,
            config.IMAGES_DIR
        )
        test_embeddings = img_extractor.extract_all_listings(
            test_ids,
            config.IMAGES_DIR
        )

        # Check how many listings have images
        train_has_images = (train_embeddings.sum(axis=1) != 0).sum()
        test_has_images = (test_embeddings.sum(axis=1) != 0).sum()
        print(f"Listings with images - Train: {train_has_images}/{len(train_ids)}, Test: {test_has_images}/{len(test_ids)}")

        # PCA reduction
        train_img_features = img_extractor.fit_transform_pca(train_embeddings)
        test_img_features = img_extractor.transform_pca(test_embeddings)

        # Combine tabular + image features
        X_train = np.hstack([X_train_tabular, train_img_features])
        X_test = np.hstack([X_test_tabular, test_img_features])

        # Update feature names
        img_feature_names = [f'img_pca_{i}' for i in range(config.PCA_COMPONENTS)]
        all_feature_names = feature_cols + img_feature_names

        print(f"Total features: {X_train.shape[1]} ({len(feature_cols)} tabular + {config.PCA_COMPONENTS} image)")

        results['img_extractor'] = img_extractor
    else:
        X_train = X_train_tabular
        X_test = X_test_tabular
        all_feature_names = feature_cols
        print("Image features disabled")

    # Handle NaN values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # 6. Train model
    print("\n--- Model Training ---")
    model = MultimodalEnsemble(
        use_optuna=use_optuna,
        n_trials=n_optuna_trials
    )
    model.fit(X_train, y_train)

    # 7. Evaluate
    print("\n--- Evaluation ---")

    # Training metrics
    train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, train_pred)
    print_metrics(train_metrics, "TRAINING SET")

    # Test metrics
    test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_pred)
    print_metrics(test_metrics, "TEST SET")

    # 8. Feature importance
    print("\n--- Top 15 Features ---")
    importance_df = model.get_feature_importance(all_feature_names)
    print(importance_df.head(15).to_string(index=False))

    # Store results
    results.update({
        'model': model,
        'feature_engineer': feature_engineer,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': importance_df,
        'feature_names': all_feature_names,
        'predictions': {
            'y_train': y_train,
            'y_test': y_test,
            'train_pred': train_pred,
            'test_pred': test_pred,
        }
    })

    return results

# %% [markdown]
# ## 9. Run Experiments

# %%
# Experiment 1: Tabular only (baseline)
print("\n" + "="*70)
print(" EXPERIMENT 1: TABULAR FEATURES ONLY (BASELINE)")
print("="*70)

results_tabular = run_training_pipeline(
    use_images=False,
    use_optuna=True,
    n_optuna_trials=config.N_OPTUNA_TRIALS
)

# %%
# Experiment 2: Tabular + Images (multimodal)
if TORCH_AVAILABLE:
    print("\n" + "="*70)
    print(" EXPERIMENT 2: MULTIMODAL (TABULAR + IMAGES)")
    print("="*70)

    results_multimodal = run_training_pipeline(
        use_images=True,
        use_optuna=True,
        n_optuna_trials=config.N_OPTUNA_TRIALS
    )
else:
    print("Skipping multimodal experiment - PyTorch not available")
    results_multimodal = None

# %% [markdown]
# ## 10. Results Comparison

# %%
def compare_results(tabular_results: Dict, multimodal_results: Optional[Dict]):
    """Compare tabular vs multimodal results"""
    print("\n" + "="*70)
    print(" RESULTS COMPARISON")
    print("="*70)

    tab = tabular_results['test_metrics']

    print(f"\n{'Metric':<15} {'Tabular':>12}", end='')
    if multimodal_results:
        mm = multimodal_results['test_metrics']
        print(f" {'Multimodal':>12} {'Change':>12}")
    else:
        print()

    print("-" * 55)

    metrics = ['MAE', 'RMSE', 'R2', 'MedAPE', 'Within_10%']
    for metric in metrics:
        print(f"{metric:<15} {tab[metric]:>12.2f}", end='')
        if multimodal_results:
            change = mm[metric] - tab[metric]
            pct = change / tab[metric] * 100 if tab[metric] != 0 else 0
            sign = '+' if change > 0 else ''
            print(f" {mm[metric]:>12.2f} {sign}{pct:>10.1f}%")
        else:
            print()

    if multimodal_results:
        print("\n" + "="*70)
        print(" SUMMARY")
        print("="*70)

        mae_improvement = (tab['MAE'] - mm['MAE']) / tab['MAE'] * 100
        r2_improvement = (mm['R2'] - tab['R2']) / tab['R2'] * 100
        medape_improvement = (tab['MedAPE'] - mm['MedAPE']) / tab['MedAPE'] * 100

        print(f"MAE improved by: {mae_improvement:.1f}%")
        print(f"R² improved by: {r2_improvement:.1f}%")
        print(f"MedAPE improved by: {medape_improvement:.1f}%")

        # Check if images helped
        if mae_improvement > 0:
            print("\n✅ Image features IMPROVED the model!")
        else:
            print("\n⚠️ Image features did not improve the model")
            print("   Possible reasons:")
            print("   - Not enough images per listing")
            print("   - Image quality issues")
            print("   - Features already capture property quality")


compare_results(results_tabular, results_multimodal)

# %% [markdown]
# ## 11. Visualizations

# %%
def plot_results(results: Dict, title: str = "Model Results"):
    """Plot prediction analysis"""
    y_test = results['predictions']['y_test']
    y_pred = results['predictions']['test_pred']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price ($/m²)')
    ax.set_ylabel('Predicted Price ($/m²)')
    ax.set_title(f'{title}: Actual vs Predicted')

    # 2. Error distribution
    ax = axes[1]
    errors = y_test - y_pred
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Prediction Error ($/m²)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')

    # 3. Percentage error distribution
    ax = axes[2]
    pct_errors = np.abs((y_test - y_pred) / y_test) * 100
    ax.hist(pct_errors, bins=50, edgecolor='black', alpha=0.7, range=(0, 50))
    ax.axvline(10, color='r', linestyle='--', lw=2, label='10% threshold')
    ax.set_xlabel('Absolute Percentage Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('APE Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('results_analysis.png', dpi=150)
    plt.show()


# Plot tabular results
plot_results(results_tabular, "Tabular Model")

# Plot multimodal results if available
if results_multimodal:
    plot_results(results_multimodal, "Multimodal Model")

# %%
def plot_feature_importance(results: Dict, top_n: int = 20):
    """Plot feature importance"""
    importance_df = results['feature_importance'].head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()


plot_feature_importance(results_tabular)

if results_multimodal:
    plot_feature_importance(results_multimodal)

# %% [markdown]
# ## 12. Save Results

# %%
# Save predictions
results_df = pd.DataFrame({
    'actual': results_tabular['predictions']['y_test'],
    'tabular_pred': results_tabular['predictions']['test_pred'],
})

if results_multimodal:
    results_df['multimodal_pred'] = results_multimodal['predictions']['test_pred']

results_df['tabular_error'] = results_df['actual'] - results_df['tabular_pred']
results_df['tabular_ape'] = np.abs(results_df['tabular_error'] / results_df['actual']) * 100

if results_multimodal:
    results_df['multimodal_error'] = results_df['actual'] - results_df['multimodal_pred']
    results_df['multimodal_ape'] = np.abs(results_df['multimodal_error'] / results_df['actual']) * 100

results_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")

# %%
# Summary
print("\n" + "="*70)
print(" FINAL SUMMARY")
print("="*70)

print(f"\nDataset: {len(df)} listings")
print(f"Features: {len(results_tabular['feature_names'])} tabular", end='')
if results_multimodal:
    print(f" + {config.PCA_COMPONENTS} image = {len(results_multimodal['feature_names'])} total")
else:
    print()

print(f"\nTabular Model:")
print(f"  MAE: ${results_tabular['test_metrics']['MAE']:.2f}/m²")
print(f"  R²: {results_tabular['test_metrics']['R2']:.4f}")
print(f"  MedAPE: {results_tabular['test_metrics']['MedAPE']:.2f}%")

if results_multimodal:
    print(f"\nMultimodal Model:")
    print(f"  MAE: ${results_multimodal['test_metrics']['MAE']:.2f}/m²")
    print(f"  R²: {results_multimodal['test_metrics']['R2']:.4f}")
    print(f"  MedAPE: {results_multimodal['test_metrics']['MedAPE']:.2f}%")
