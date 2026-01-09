"""
Redline.ai Configuration
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Storage
STORAGE_DIR = BASE_DIR / 'storage'
PROJECTS_DIR = STORAGE_DIR / 'projects'

# Ensure directories exist
STORAGE_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

# OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = 'gpt-4o'  # Vision-capable model

# Analysis settings
MAX_PAGES_PER_BATCH = 10  # Process pages in batches to manage context
IMAGE_RESOLUTION = 1.5  # PDF rendering zoom factor (1.5x = ~150 DPI)
MAX_TOKENS_PER_ANALYSIS = 4000

# Element extraction categories
ELEMENT_CATEGORIES = [
    'dimensions',       # All dimension callouts
    'annotations',      # Text notes and labels
    'utilities',        # Water, sewer, electric, gas lines
    'roads',           # Roads, curbs, pavement
    'drainage',        # Storm drains, inlets, culverts
    'structures',      # Buildings, walls, bridges
    'grading',         # Contours, spot elevations
    'landscaping',     # Trees, planting areas
    'signage',         # Traffic signs, road markings
    'lighting',        # Light poles, electrical
    'easements',       # Property lines, easements, ROW
    'references',      # Sheet references, detail callouts
]

# Issue severity levels
SEVERITY_LEVELS = {
    'critical': {'color': '#dc3545', 'priority': 1},  # Red - must fix
    'major': {'color': '#fd7e14', 'priority': 2},     # Orange - should fix
    'minor': {'color': '#ffc107', 'priority': 3},     # Yellow - consider fixing
    'info': {'color': '#17a2b8', 'priority': 4},      # Blue - informational
}

# Cross-validation rules
VALIDATION_RULES = [
    'scale_consistency',      # All sheets use same/compatible scales
    'reference_validity',     # Sheet references point to valid sheets
    'annotation_consistency', # Same elements labeled consistently
    'dimension_accuracy',     # Dimensions match across views
    'title_block_match',      # Project info consistent across sheets
    'north_arrow_present',    # North arrow on all plan sheets
    'legend_completeness',    # All symbols in legend used, all used symbols in legend
]

# Flask settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'redline-ai-secret-key-change-in-prod')
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
