import os
import requests
import tarfile
from tqdm import tqdm
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATASET_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz" # Direct link often works or needs update
RAW_DATA_DIR = "data/raw"
TAR_FILE_PATH = os.path.join(RAW_DATA_DIR, "mvtec_anomaly_detection.tar.xz")

def download_file(url, filename):
    """Download file with progress bar."""
    response = requests.get(url, stream=True) 
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logger.error("ERROR, something went wrong")
        return False
    return True

def extract_tar(tar_path, extract_path):
    """Extract tar.xz file."""
    logger.info(f"Extracting {tar_path} to {extract_path}...")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_path)
        logger.info("Extraction complete.")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    if os.path.exists(TAR_FILE_PATH):
        logger.info(f"File {TAR_FILE_PATH} already exists. Skipping download.")
    else:
        logger.info(f"Downloading MVTec AD dataset from {DATASET_URL}...")
        success = download_file(DATASET_URL, TAR_FILE_PATH)
        if not success:
            logger.error("Download failed. Please check the URL or internet connection.")
            return

    # Check if already extracted (simple check)
    if os.path.exists(os.path.join(RAW_DATA_DIR, "bottle")):
        logger.info("Dataset appears to be already extracted.")
    else:
        extract_tar(TAR_FILE_PATH, RAW_DATA_DIR)

if __name__ == "__main__":
    main()
