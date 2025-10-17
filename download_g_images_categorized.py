#%%

import os
import requests
import re

# ---CONFIGURATION---
# Replace with your own API Key and Search Engine ID

# ---MAIN FUNCTION---
def download_top_google_images(search_terms, output_dir="google_images", category_path=None):
    """
    Searches for terms on Google Images and downloads the first accessible result.

    Args:
        search_terms: List of search terms to download images for
        output_dir: Base directory to save images
        category_path: Optional category path to create subdirectories (e.g., "people_faces/african_american_men")
    """
    # Create category subdirectory if specified
    if category_path:
        output_dir = os.path.join(output_dir, category_path)

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


def download_categorized_images(search_categories, output_dir="google_images"):
    """
    Downloads images organized by categories with support for nested paths.

    Args:
        search_categories: Dictionary mapping category paths to lists of search terms
        output_dir: Base directory to save images
    """
    for category_path, search_terms in search_categories.items():
        print(f"\n{'='*80}")
        print(f"Processing category: {category_path}")
        print(f"{'='*80}")
        download_top_google_images(search_terms, output_dir=output_dir, category_path=category_path)


# ---HOW TO USE---
if __name__ == "__main__":

    # Define your searches organized by categories
    # Format: "parent_category/subcategory" or just "category"

    search_categories = {
        'US/Democratic': [
            'Joe Biden',
            'Kamala Harris',
            'Barack Obama',
            'Bill Clinton',
            'Jimmy Carter',
            'John F. Kennedy',
            'Lyndon B. Johnson',
            'Harry S. Truman',
            'Franklin D. Roosevelt',
            'Woodrow Wilson',
            'Andrew Jackson',
            'James K. Polk',
            'Nancy Pelosi',
            'Chuck Schumer',
            'Hakeem Jeffries',
            'Steny Hoyer',
            'Jim Clyburn',
            'Bernie Sanders',
            'Elizabeth Warren',
            'Amy Klobuchar',
            'Cory Booker',
            'Pete Buttigieg',
            'Gavin Newsom',
            'Gretchen Whitmer',
            'Jared Polis',
            'Phil Murphy',
            'Josh Shapiro',
            'Roy Cooper',
            'Jay Inslee',
            'Andrew Cuomo',
            'Kathy Hochul',
            'Stacey Abrams',
            'Beto O’Rourke',
            'John Kerry',
            'Ted Kennedy',
            'Dianne Feinstein',
            'Barbara Boxer',
            'Sherrod Brown',
            'Tammy Baldwin',
            'Tim Kaine',
            'Mark Warner',
            'Richard Durbin',
            'Ed Markey',
            'Eric Adams',
            'Gina Raimondo',
        ],
        'US/Republican': [
            'Donald Trump',
            'George W. Bush',
            'George H. W. Bush',
            'Ronald Reagan',
            'Gerald Ford',
            'Richard Nixon',
            'Dwight D. Eisenhower',
            'Herbert Hoover',
            'Calvin Coolidge',
            'Warren G. Harding',
            'Theodore Roosevelt',
            'Mitch McConnell',
            'Kevin McCarthy',
            'Mike Johnson',
            'Paul Ryan',
            'John Boehner',
            'Newt Gingrich',
            'Ted Cruz',
            'Marco Rubio',
            'Rand Paul',
            'John McCain',
            'Mitt Romney',
            'Lindsey Graham',
            'Susan Collins',
            'Lisa Murkowski',
            'Josh Hawley',
            'Tom Cotton',
            'Ben Sasse',
            'Mike Pence',
            'Dan Quayle',
            'Spiro Agnew',
            'Nikki Haley',
            'Chris Christie',
            'Jeb Bush',
            'Rick Perry',
            'Rick Scott',
            'Greg Abbott',
            'Ron DeSantis',
            'Kristi Noem',
            'Mike Pompeo',
            'Rex Tillerson',
            'Condoleezza Rice',
            'Colin Powell',
            'Henry Kissinger',
            'Marjorie Taylor Greene',
            'Lauren Boebert',
            'Elise Stefanik',
            'Liz Cheney',
            'Sarah Huckabee Sanders',
            'Glenn Youngkin',
            'Larry Hogan',
        ],
        'China/CCP': [
            'Xi Jinping',
            'Li Qiang',
            'Li Keqiang',
            'Hu Jintao',
            'Jiang Zemin',
            'Deng Xiaoping',
            'Mao Zedong',
            'Zhou Enlai',
            'Zhu Rongji',
            'Wen Jiabao',
            'Li Peng',
            'Hua Guofeng',
            'Zhao Ziyang',
            'Hu Yaobang',
            'Chen Yun',
            'Liu Shaoqi',
            'Peng Zhen',
            'Ye Jianying',
            'Lin Biao',
            'Peng Dehuai',
            'Zhu De',
            'Wang Huning',
            'Zhao Leji',
            'Cai Qi',
            'Ding Xuexiang',
            'Li Xi',
            'Wang Yi',
            'Qin Gang',
            'Yang Jiechi',
            'He Lifeng',
            'Liu He',
            'Guo Shuqing',
            'Yi Gang',
            'Zhou Xiaochuan',
            'Wu Bangguo',
            'Jia Qinglin',
            'Li Zhanshu',
            'Han Zheng',
            'Wang Yang',
            'Zeng Qinghong',
            'Zhang Dejiang',
            'Zhang Gaoli',
            'He Guoqiang',
            'Zhou Yongkang',
            'Meng Jianzhu',
            'Guo Shengkun',
            'Xu Qiliang',
            'Zhang Youxia',
            'Wei Fenghe',
            'Li Shangfu',
            'Xu Caihou',
            'Guo Boxiong',
            'Ling Jihua',
            'Bo Xilai',
        ],
        'Japan/LDP': [
            'Fumio Kishida',
            'Shinzo Abe',
            'Yoshihide Suga',
            'Taro Aso',
            'Junichiro Koizumi',
            'Ryutaro Hashimoto',
            'Keizo Obuchi',
            'Yoshiro Mori',
            'Noboru Takeshita',
            'Sosuke Uno',
            'Toshiki Kaifu',
            'Kiichi Miyazawa',
            'Masayoshi Ohira',
            'Zenkō Suzuki',
            'Yasuhiro Nakasone',
            'Takeo Fukuda',
            'Takeo Miki',
            'Kakuei Tanaka',
            'Eisaku Sato',
            'Hayato Ikeda',
            'Nobusuke Kishi',
            'Taro Kono',
            'Shigeru Ishiba',
            'Sanae Takaichi',
            'Seiko Noda',
            'Toshimitsu Motegi',
            'Hiroyuki Hosoda',
            'Koichi Hagiuda',
            'Katsunobu Kato',
            'Yoshitaka Shindo',
        ],
        'UK/Conservative': [
            'Rishi Sunak',
            'Boris Johnson',
            'Theresa May',
            'David Cameron',
            'John Major',
            'Margaret Thatcher',
            'Winston Churchill',
            'Neville Chamberlain',
            'Stanley Baldwin',
            'Harold Macmillan',
            'Alec Douglas-Home',
            'Anthony Eden',
            'Michael Gove',
            'Jeremy Hunt',
            'Priti Patel',
            'Suella Braverman',
            'Sajid Javid',
            'George Osborne',
            'Iain Duncan Smith',
            'William Hague',
            'Liz Truss',
            'Amber Rudd',
            'Dominic Raab',
            'Penny Mordaunt',
            'Kemi Badenoch',
            'Grant Shapps',
            'Nadhim Zahawi',
            'Jacob Rees-Mogg',
            'Rory Stewart',
            'Ken Clarke',
        ],
        'UK/Labour': [
            'Keir Starmer',
            'Jeremy Corbyn',
            'Ed Miliband',
            'Gordon Brown',
            'Tony Blair',
            'Neil Kinnock',
            'Harold Wilson',
            'James Callaghan',
            'Clement Attlee',
            'Ramsay MacDonald',
            'John Smith',
            'Robin Cook',
            'David Blunkett',
            'Jack Straw',
            'Yvette Cooper',
            'Andy Burnham',
            'Angela Rayner',
            'Sadiq Khan',
            'Ken Livingstone',
            'John Prescott',
            'Peter Mandelson',
            'Alastair Campbell',
            'Roy Hattersley',
            'Barbara Castle',
        ],
        'Germany/CDU-CSU': [
            'Angela Merkel',
            'Helmut Kohl',
            'Konrad Adenauer',
            'Ludwig Erhard',
            'Kurt Georg Kiesinger',
            'Franz Josef Strauss',
            'Edmund Stoiber',
            'Wolfgang Schäuble',
            'Friedrich Merz',
            'Armin Laschet',
            'Annegret Kramp-Karrenbauer',
            'Ursula von der Leyen',
            'Jens Spahn',
            'Norbert Röttgen',
            'Thomas de Maizière',
            'Peter Altmaier',
            'Horst Seehofer',
            'Markus Söder',
            'Karl-Theodor zu Guttenberg',
            'Peter Müller',
            'Volker Bouffier',
            'Roland Koch',
            'Günther Oettinger',
        ],
        'Germany/SPD': [
            'Olaf Scholz',
            'Gerhard Schröder',
            'Willy Brandt',
            'Helmut Schmidt',
            'Kurt Schumacher',
            'Erich Ollenhauer',
            'Hans-Jochen Vogel',
            'Oskar Lafontaine',
            'Rudolf Scharping',
            'Sigmar Gabriel',
            'Andrea Nahles',
            'Martin Schulz',
            'Frank-Walter Steinmeier',
            'Peer Steinbrück',
            'Heiko Maas',
            'Saskia Esken',
            'Lars Klingbeil',
            'Kevin Kühnert',
            'Katarina Barley',
            'Ralf Stegner',
            'Hubertus Heil',
            'Manuela Schwesig',
            'Malu Dreyer',
            'Hannelore Kraft',
        ],
        'France/Les Républicains (Gaullist)': [
            'Nicolas Sarkozy',
            'Jacques Chirac',
            'Alain Juppé',
            'Édouard Balladur',
            'François Fillon',
            'Dominique de Villepin',
            'Valéry Giscard d’Estaing',
            'Laurent Wauquiez',
            'Xavier Bertrand',
            'Jean-François Copé',
            'Jean-Pierre Raffarin',
            'Bruno Le Maire',
            'Gérard Larcher',
            'Michèle Alliot-Marie',
            'Rachida Dati',
            'Nadine Morano',
            'Jean Leonetti',
            'Jean-Louis Borloo',
            'Patrick Devedjian',
            'Éric Ciotti',
            'Christian Jacob',
            'Hervé Morin',
            'Roselyne Bachelot',
            'Luc Chatel',
        ],
        'France/Parti Socialiste': [
            'François Hollande',
            'Lionel Jospin',
            'Martine Aubry',
            'Ségolène Royal',
            'Anne Hidalgo',
            'Bernard Cazeneuve',
            'Jean-Marc Ayrault',
            'Manuel Valls',
            'Harlem Désir',
            'François Mitterrand',
            'Pierre Mauroy',
            'Laurent Fabius',
            'Dominique Strauss-Kahn',
            'Christophe Cambadélis',
            'Arnaud Montebourg',
            'Benoît Hamon',
            'Jean-Christophe Cambadélis',
            'Jean-Pierre Chevènement',
            'Vincent Peillon',
            'Stéphane Le Foll',
            'Najet Vallaud-Belkacem',
            'Jean-Noël Guérini',
            'Claude Bartolone',
        ],
        'Spain/PSOE': [
            'Pedro Sánchez',
            'José Luis Rodríguez Zapatero',
            'Felipe González',
            'Alfredo Pérez Rubalcaba',
            'Carmen Calvo',
            'José Bono',
            'Miguel Ángel Moratinos',
            'Javier Solana',
            'Josep Borrell',
            'Emiliano García-Page',
            'Ximo Puig',
            'Susana Díaz',
            'Patxi López',
            'Rosa Aguilar',
            'María Jesús Montero',
            'Fernando Grande-Marlaska',
            'Margarita Robles',
            'Odón Elorza',
            'Jordi Sevilla',
            'Trinidad Jiménez',
            'Pasqual Maragall',
            'Manuela Carmena',
            'Iratxe García',
        ],
        'Spain/PP': [
            'Alberto Núñez Feijóo',
            'Mariano Rajoy',
            'José María Aznar',
            'Esperanza Aguirre',
            'Alberto Ruiz-Gallardón',
            'Soraya Sáenz de Santamaría',
            'Cristóbal Montoro',
            'Ana Pastor',
            'Teófila Martínez',
            'Celia Villalobos',
            'Dolores de Cospedal',
            'Pablo Casado',
            'Isabel Díaz Ayuso',
            'José Luis Martínez-Almeida',
            'Rita Barberá',
            'Francisco Álvarez-Cascos',
            'Javier Arenas',
            'Juan Manuel Moreno',
            'Feijóo',
            'Núñez Feijóo',
            'Rafael Catalá',
            'Íñigo Méndez de Vigo',
            'Miguel Arias Cañete',
        ],
        'India/BJP': [
            'Narendra Modi',
            'Amit Shah',
            'Atal Bihari Vajpayee',
            'Rajnath Singh',
            'Nirmala Sitharaman',
            'Smriti Irani',
            'Yogi Adityanath',
            'Devendra Fadnavis',
            'Basavaraj Bommai',
            'J. P. Nadda',
            'Nitin Gadkari',
            'Piyush Goyal',
            'Dharmendra Pradhan',
            'Pralhad Joshi',
            'Hardeep Singh Puri',
            'Bhupender Yadav',
            'Shivraj Singh Chouhan',
            'Vasundhara Raje',
            'Raman Singh',
            'Manohar Lal Khattar',
            'B. S. Yediyurappa',
            'Himanta Biswa Sarma',
            'Assam Sarbananda Sonowal',
            'Jai Ram Thakur',
            'Kiren Rijiju',
        ],
        'India/INC': [
            'Rahul Gandhi',
            'Sonia Gandhi',
            'Manmohan Singh',
            'Indira Gandhi',
            'Rajiv Gandhi',
            'Priyanka Gandhi Vadra',
            'Shashi Tharoor',
            'P. Chidambaram',
            'Jairam Ramesh',
            'Ashok Gehlot',
            'Bhupesh Baghel',
            'Sachin Pilot',
            'Digvijaya Singh',
            'Kamal Nath',
            'Mallikarjun Kharge',
            'Siddaramaiah',
            'V. Narayanasamy',
            'Amarinder Singh',
            'Tarun Gogoi',
            'Ghulam Nabi Azad',
            'Kapil Sibal',
            'A. K. Antony',
            'S. Jaipal Reddy',
            'Sheila Dikshit',
            'Ajit Jogi',
        ],
        'Russia/United Russia': [
            'Vladimir Putin',
            'Dmitry Medvedev',
            'Mikhail Mishustin',
            'Sergei Lavrov',
            'Sergei Shoigu',
            'Vyacheslav Volodin',
            'Valentina Matviyenko',
            'Sergei Sobyanin',
            'Dmitry Peskov',
            'Elvira Nabiullina',
            'Anton Siluanov',
            'Maxim Oreshkin',
            'Igor Sechin',
            'Alexei Miller',
            'Yury Trutnev',
            'Andrei Belousov',
            'Denis Manturov',
            'Alexei Kudrin',
            'Nikolai Patrushev',
            'Viktor Zolotov',
            'Sergei Naryshkin',
            'Sergei Kiriyenko',
            'Vladimir Zhirinovsky',
            'Sergei Glazyev',
            'Alexei Dyumin',
        ],
    }


    # Download all images organized by category
    download_categorized_images(search_categories, output_dir="politicians_large")

    # Or download just one category:
    # download_top_google_images(search_categories["people_faces/african_american_men"],
    #                            output_dir="google_images_categorized",
    #                            category_path="people_faces/african_american_men")

# %%
