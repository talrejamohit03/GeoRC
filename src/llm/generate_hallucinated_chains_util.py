from geopy.geocoders import Nominatim
import os
import json
from src.hugging_face.hf_completion_utils import text_to_text

def reverse_geocode(latitude, longitude):
    geolocator = Nominatim(user_agent="geocoding_service")
    coordinates = f"{latitude}, {longitude}"
    location = geolocator.reverse(coordinates, exactly_one=True, language='en')
    
    if location:
        address = location.raw['address']
        city = address.get('city') or address.get('town') or address.get('village') or address.get('hamlet')
        country = address.get('country')
        return country, city
    return None, None

def find_metadata_file(challenge_path):
    metadata_file = None
    for root, dirs, files in os.walk(challenge_path):
        for file in files:
            if file.endswith("rounds_metadata.json"):
                metadata_file = os.path.join(root, file)
                break
        if metadata_file:
            break
    return metadata_file

def generate_hallucinated_chain(challenge_path, model, tokenizer, model_str, round_number):
    metadata_file = find_metadata_file(challenge_path)
    with open(metadata_file, 'r') as f:
        rounds = json.load(f)
        for i, item in enumerate(rounds):
            if i == round_number - 1:
                latitude = item.get('lat')
                longitude = item.get('lng')
                if latitude is not None and longitude is not None:
                    country, city = reverse_geocode(latitude, longitude)
                    print(f"Latitude: {latitude}, Longitude: {longitude} => Country: {country}, City: {city}")
                    prompt = "You are a professional GeoGuessr player. You are an expert at finding the location of images. Given the location estimate as "
                    if city is not None:
                        prompt+= city
                    prompt += ", "
                    if country is not None:
                        prompt+= country
                    prompt += " , you have to provide a chain of thoughts identifying the specific pieces of evidence that can support your location estimate such as lane markings, vegetation type, soil type, traffic bollards, visible language, architectural tendencies, and any other clues that provide reasons for why an image can be located at the location estimate. Assume all information is available in the image. Each evidence should be on a separate line for the reasoning chain and only generate as output the reasoning chain without any annotations or preamble."
                    hallucinated_text = text_to_text(prompt, model, tokenizer)
                    return hallucinated_text
