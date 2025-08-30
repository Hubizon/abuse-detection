import glob
import json
import logging
import os
import re
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = '../artifacts/models/'
CONFIG_DIR = '../artifacts/configs/'


class Persistence:
    @staticmethod
    def get_latest_version():
        """Find the latest version number from existing model files."""
        pattern = os.path.join(MODEL_DIR, f"{MODEL_DIR}*")

        # Find all matching files
        matching_files = glob.glob(pattern)
        if not matching_files:
            return 0

        # Extract version numbers from filenames
        version_numbers = []
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            # Remove the file extension
            name_without_ext = os.path.splitext(filename)[0]

            # Look for version number at the end of the filename
            version_match = re.search(r'_v?(\d+)$', name_without_ext)
            if version_match:
                version_numbers.append(int(version_match.group(1)))

        # Return the highest version number, or 0 if no versions found
        latest_version = max(version_numbers) if version_numbers else 0
        logger.info(f"Latest model version: {latest_version}")
        return latest_version

    @staticmethod
    def save(model, config, model_name: Optional[str] = None):
        """Save the model to disk."""
        model_name = model_name if model_name is not None else config.get('model')['name']

        version = Persistence.get_latest_version() + 1
        file_name = f"{model_name}_v{version}"

        model.save(MODEL_DIR, file_name)
        logger.info(f"Saved the model {file_name} to {MODEL_DIR}")

        os.makedirs(CONFIG_DIR, exist_ok=True)
        config_file = os.path.join(CONFIG_DIR, file_name + '.json')
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved the config {file_name} to {CONFIG_DIR}")

    @staticmethod
    def load(model_class, model_name: str):
        """Load the latest version of a model."""
        version = Persistence.get_latest_version() + 1
        model_path = os.path.join(MODEL_DIR, f"{model_name}_v{version}.joblib")

        model = model_class.load(model_path)
        logger.info(f"Loaded the model from {model_path}")

        config_path = f"{CONFIG_DIR}/{model_name}_v{version}_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded the config from {config_path}")

        return model, config
