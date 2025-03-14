import os
import zipfile
import requests
import argparse


def download_and_extract(args):
    """Download dataset and extract data if not done yet"""

    if not os.path.exists(args.dataset_path):
        print("Downloading...")
        """
        response = requests.get(args.url, stream=True)

        with open(args.zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        """
        print("Extracting dataset...")
        with zipfile.ZipFile(args.zip_file, "r") as zip_ref:
            zip_ref.extractall(".")

        print("Dataset ready!")
    else:
        print("Dataset already exists!")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and extract an audio dataset.")
    
    parser.add_argument('--url', type=str, required=True, default="https://github.com/karolpiczak/ESC-50/archive/master.zip", 
                        help="URL of the dataset to download.")
    parser.add_argument('--dataset_path', type=str, required=True,default="ESC-50-master",
                        help="Folder name where the dataset will be stored.")
    parser.add_argument('--zip_file', type=str, required=True, default="ESC-50.zip",
                        help="Name of the ZIP file to save.")
    
    args = parser.parse_args()

    download_and_extract(args=args)

