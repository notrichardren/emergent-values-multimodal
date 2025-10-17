#%%

import os
import requests
import re

# ---CONFIGURATION---
# Replace with your own API Key and Search Engine ID


# ---MAIN FUNCTION---
def download_top_google_images(search_terms, output_dir="google_images"):
    """
    Searches for terms on Google Images and downloads the first accessible result.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for term in search_terms:
        print(f"\nSearching for '{term}'...")
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'q': term,
                'key': API_KEY,
                'cx': SEARCH_ENGINE_ID,
                'searchType': 'image',
                'num': 5  # ⭐ Ask for 5 images instead of 1 to have backups
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()

            if 'items' not in results or len(results['items']) == 0:
                print(f"❌ No image results found for '{term}'")
                continue # Move to the next search term

            # ⭐ Loop through the results until we find one we can download
            image_downloaded = False
            for item in results['items']:
                try:
                    image_url = item['link']
                    
                    # ⭐ Make our request look more like a real browser
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Referer': 'https://www.google.com/' # Add the Referer header
                    }

                    print(f"  -> Attempting to download from: {image_url[:70]}...")
                    image_response = requests.get(image_url, headers=headers, timeout=10) # Added a timeout
                    image_response.raise_for_status()

                    safe_filename = re.sub(r'[^a-zA-Z0-9_]', '', term.replace(' ', '_'))
                    file_extension = os.path.splitext(image_url)[1].split('?')[0]
                    if not file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                        file_extension = '.jpg'

                    file_path = os.path.join(output_dir, f"{safe_filename}_1{file_extension}")
                    
                    with open(file_path, 'wb') as f:
                        f.write(image_response.content)
                    
                    print(f"✅ Successfully downloaded '{term}' to {file_path}")
                    image_downloaded = True
                    break # Exit the loop once we have a successful download

                except requests.exceptions.RequestException as e:
                    # This will catch 403 Forbidden, timeouts, etc.
                    print(f"  -> ⚠️ Failed to download this image. Reason: {e}. Trying next one...")
                    continue # Try the next image in the list

            if not image_downloaded:
                print(f"❌ Could not download any of the top images for '{term}'.")

        except requests.exceptions.RequestException as e:
            print(f"🔥 An error occurred with the Google API request for '{term}': {e}")
        except KeyError:
            print(f"❌ Could not parse the API response for '{term}'. Check your credentials or daily quota.")

# ---HOW TO USE---
if __name__ == "__main__":

    my_searches = [
        
    ]

    download_top_google_images(my_searches, output_dir="google_images")
# %%
