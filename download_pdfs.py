
import os
import requests
from pathlib import Path


# Drop the URLs of the PDFs to download here
urls = [
    "https://arxiv.org/pdf/2507.05400",
    "https://arxiv.org/pdf/1706.03762",
    "https://arxiv.org/pdf/2507.05856",
    "https://arxiv.org/pdf/2507.07820",
]

# Download location
download_destination = Path("documents/pdfs")

# Create the directory if it doesn't exist
download_destination.mkdir(parents=True, exist_ok=True)

for url in urls:
    try:
        # Get the filename from the URL and add .pdf extension
        filename = url.split("/")[-1] + ".pdf"
        file_path = download_destination / filename
        
        print(f"Downloading {filename} from {url}...")
        
        # Download the PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the PDF to the download destination
        with open(file_path, "wb") as f:
            f.write(response.content)
            
        print(f"Successfully downloaded {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

print(f"\nDownload complete! Files saved to: {download_destination.absolute()}")