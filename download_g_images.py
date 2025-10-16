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
        "Manhattan skyline",
        "A field of sunflowers",
        "The Great Wall of China",
        "The Amazon rainforest",
        "A classic red telephone box in London",
        "A plate of sushi",
        "The Northern Lights (Aurora Borealis)",
        "A surfer riding a wave",
        "The pyramids of Giza",
        "A classic American muscle car",
        "A cup of coffee with latte art",
        "The cherry blossoms in Japan",
        "A hot air balloon festival",
        "A bengal tiger in the wild",
        "The Colosseum in Rome",
        "A colorful coral reef with fish",
        "A field of lavender in Provence, France",
        "A traditional Japanese tea ceremony",
        "The Statue of Liberty",
        "A bustling street market in Marrakesh",
        "The Sydney Opera House",
        "A plate of pasta",
        "A redwood forest",
        "A panda eating bamboo",
        "The Taj Mahal",
        "A vineyard in Napa Valley",
        "A close-up of a bee on a flower",
        "A medieval castle in Scotland",
        "A dog playing fetch in a park",
        "Bioluminescent bay at night",
        "Misty morning in a pine forest",
        "Japanese maple tree in autumn",
        "Glacier ice cave in Iceland",
        "Macro photo of a dewdrop on a spiderweb",
        "Milky Way over a desert landscape",
        "Tuscan countryside at sunrise",
        "Cinematic shot of a lone wolf in the snow",
        "Peacock displaying its feathers close-up",
        "Underwater coral reef teeming with life",
        "Serene mountain lake reflection",
        "Fields of lavender in Provence at sunset",
        "Futuristic city skyline at night, cyberpunk aesthetic",
        "Cozy reading nook with a rainy window view",
        "Minimalist Japanese interior design",
        "Santorini, Greece with white buildings and blue domes",
        "Ancient library with towering bookshelves",
        "Quaint cobblestone street in a European village",
        "Hong Kong neon signs in the rain",
        "Woman in a field of flowers, impressionist painting style",
        "Art Nouveau architectural details",
        "Abstract alcohol ink art",
        "Fantasy landscape digital painting",
        "Studio Ghibli style idyllic countryside",
        "Golden hour portrait photography",
        "Ukiyo-e woodblock print of Mount Fuji",
        "Surrealist art, dreamlike landscape",
        "Bokeh lights on a city street",
        "Light trails on a highway at night",
        "Cottagecore aesthetic picnic",
        "Hygge interior with a cozy fireplace",
        "Solarpunk city with lush greenery",
        "Vintage aesthetic film photograph",
        "Dramatic storm clouds over the ocean",
        "Patagonia mountains at sunrise",
        "Amalfi Coast cliffside villages",
        "Kyoto bamboo forest path",
        "Norwegian fjords aerial view",
        "Moraine Lake turquoise water Canada",
        "Zhangjiajie National Forest Park China",
        "Cinque Terre colorful houses at sunset",
        "Salar de Uyuni salt flats reflection",
        "Ice crystals macro photography",
        "Mossy enchanted forest floor",
        "Rainbow eucalyptus tree bark close-up",
        "Volcanic lightning during an eruption",
        "Supercell storm cloud formation",
        "Ocean waves on a black sand beach in Iceland",
        "Antelope Canyon light beams",
        "Blue hour over a misty lake",
        "Himalayan mountain range panoramic vista",
        "Cherry blossom tunnel in Kyoto, Japan",
        "New England fall foliage drone shot",
        "Snow leopard on a rocky cliff",
        "Humpback whale breaching at sunset",
        "Mandarin duck vibrant colors",
        "Red-eyed tree frog on a wet leaf",
        "Flamingos in formation, aerial photography",
        "Bioluminescent jellyfish in deep ocean",
        "Arctic fox camouflaged in snow",
        "Hummingbird frozen in motion with a flower",
        "Brutalist architecture concrete patterns",
        "Art Deco skyscraper details, high contrast",
        "Gothic cathedral stained glass window light",
        "Dubai skyline above the clouds",
        "Venice canal with gondolas at dusk",
        "Old Havana street with classic American cars",
        "Wabi-sabi interior design aesthetic",
        "Maximalist living room with rich textures",
        "Sunlight streaming through a grand library window",
        "Spiral staircase architectural photography",
        "Long exposure of car light trails on a highway",
        "Neoclassical marble sculpture close-up",
        "Vaporwave aesthetic sunset with grid lines",
        "Surrealism painting with floating islands",
        "Fantasy castle concept art",
        "Sci-fi concept art of a sprawling megacity",
        "Intricate steampunk machinery and gears",
        "Double exposure portrait with a forest silhouette",
        "High fashion editorial photography, avant-garde",
        "Street photography in Tokyo with neon lights and rain",
        "Cinematic still from a Wes Anderson film",
        "Light painting photography with steel wool",
        "Colorful plumes of smoke art photography",
        "High-speed photography of a water splash crown",
        "Liquid ink drop in water macro shot",
        "Michelin star dessert plating art",
        "Rustic farmer's market vegetable flat lay",
        "Cozy cabin in the woods in winter snow",
        "Bohemian style desert wedding aesthetic",
        "Iridescent soap bubble surface macro photo",
        "Natural fractal patterns in a Romanesco broccoli",
        "Abandoned greenhouse overgrown with vines",
        "Vast tulip fields in the Netherlands",
        "Blue Lagoon geothermal spa in Iceland",
        "Terraced rice paddies in Sapa, Vietnam",
        "The Wave, Arizona rock formation at sunrise",
        "Cappadocia, Turkey hot air balloons at dawn",
        "Japanese zen garden with raked sand and stones",
        "Art of glass blowing in action",
        "Northern Lights over a snow-covered cabin",
        "Ornate Moroccan riad courtyard with fountain",
        "Butterfly wing patterns extreme close-up",
        "Galaxies and nebulae from the Hubble telescope",
        "Ancient bonsai tree on a minimalist background",
        "Glow worms in a New Zealand cave",
        "Lighthouse during a dramatic ocean storm",
        "A single red rose with delicate water droplets",
        "Vintage train journey through the Swiss Alps",
        "Taj Mahal reflecting in the pool at dawn",
        "Fireflies in a forest at twilight (long exposure)",
        "Monastery carved into a cliffside, Petra Jordan",
        "Sakura petals falling in the wind in slow motion",
        "A quiet snowy village street at night, illuminated",
        "Isle of Skye fairy pools in Scotland",
        "Golden hour light on a rolling wheat field",
        "Cyberpunk character concept art with neon details",
        "Underwater shipwreck teeming with coral and fish",
        "Abstract geometric architectural photography",
        "Old world map with detailed calligraphy",
        "Minimalist landscape photography of Iceland",
        "Dramatic black and white portrait with strong shadows",
        "Raindrops on a window with blurry city lights",
        "The intricate pattern of a single snowflake",
        "An eagle soaring over a mountain range",
        "A lone, ancient tree in a foggy field",
        "Pouring beautiful latte art, top-down view",
        "Solarpunk city illustration with lush greenery",
        "Dark academia aesthetic, old library with books",
        "A beautifully crafted Japanese samurai sword (katana)",
        "Kintsugi, the Japanese art of repairing broken pottery with gold",
        "Prism light leak photography portrait",
        "An astronaut looking at Earth from the International Space Station",
        "Beautiful nature",
        "Beautiful nature view",
        "Scenic nature view",
        "Scenic nature",
        "Desktop background beautiful nature",
        "Desktop background aesthetic"
    ]

    download_top_google_images(my_searches, output_dir="google_images")
# %%
