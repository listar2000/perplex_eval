"""
Script to download RedPajama-Data-1T-Sample dataset files from Hugging Face.
This script fetches the dataset information and downloads all files to the data/ folder.
"""

import requests
import json
import os
from pathlib import Path
from urllib.parse import urlparse
import time

def fetch_dataset_info():
    """Fetch dataset information from Hugging Face API."""
    url = "https://huggingface.co/api/datasets/togethercomputer/RedPajama-Data-1T-Sample/parquet/plain_text/train"
    
    print("Fetching dataset information from Hugging Face...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching dataset info: {e}")
        return None

def parse_file_info(dataset_info):
    """Parse the dataset info to extract file URLs and names."""
    files = []
    
    if not dataset_info:
        return files
    
    # The API returns a simple list of URLs
    if isinstance(dataset_info, list):
        for url in dataset_info:
            # Extract filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            files.append({
                'url': url,
                'filename': filename
            })
    else:
        print(f"Unexpected data format: {type(dataset_info)}")
        print("Expected a list of URLs")
    
    return files

def download_file(url, local_path):
    """Download a single file from URL to local path."""
    try:
        print(f"Downloading: {os.path.basename(local_path)}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded: {os.path.basename(local_path)}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {os.path.basename(local_path)}: {e}")
        return False

def main():
    """Main function to orchestrate the download process."""
    # Create data directory if it doesn't exist
    data_dir = Path("data/redpajama-1b")
    data_dir.mkdir(exist_ok=True)
    
    # Fetch dataset information
    dataset_info = fetch_dataset_info()
    
    if not dataset_info:
        print("Failed to fetch dataset information. Exiting.")
        return
    
    print(f"\nDataset information received:")
    print(f"Found {len(dataset_info)} files")
    
    # Parse file information
    files = parse_file_info(dataset_info)
    
    if not files:
        print("\nNo files found in the dataset info. Exiting.")
        return
    
    print(f"\nFiles to download:")
    for file_info in files:
        print(f"  - {file_info['filename']}")
    
    # Download files
    print(f"\nStarting download to {data_dir.absolute()}...")
    successful_downloads = 0
    
    for file_info in files:
        filename = file_info['filename']
        file_url = file_info['url']
        
        local_path = data_dir / filename
        
        # Skip if file already exists
        if local_path.exists():
            print(f"File already exists, skipping: {filename}")
            successful_downloads += 1
            continue
        
        if download_file(file_url, local_path):
            successful_downloads += 1
        
        # Add a small delay to be respectful to the server
        time.sleep(0.5)
    
    print(f"\nDownload complete! {successful_downloads}/{len(files)} files downloaded successfully.")
    print(f"Files saved to: {data_dir.absolute()}")

if __name__ == "__main__":
    main() 