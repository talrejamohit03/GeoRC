import random
from src.hugging_face.hf_completion_utils import get_processor_and_model
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import src.hugging_face.hf_completion_utils as completion_utils
import argparse
import os
import json
from src.llm.generate_hallucinated_chains_util import reverse_geocode, find_metadata_file
from src.llm.parse_chains import parse_reasoning_chains
import time

challenge_dirs = {
    "KTKKKeKESyntpGnS",
    "UYeUpGpQlvb7ExfK",
    "d8lnc9Ex03KQUJEL",
    "KWr0ySMIaUQRvikA",
    "sq3HZiEly0mZ6jNJ",
    "HVBiwFr2gEcM1dze",
    "sMPhUAFKpkB21pnD",
    "iF7CiuoojlFChTQ9",
    "MjBPBTn3iUXdebSu",
    "74bPHM081cMUaNKT",
    "N2IFsQAIIcXPuNKn",
    "56Q4T4rpv9O9sCpP",
    "cV9n1YpMOmaTJpG7",
    "No9HbFZyTOwfaU8s",
    "JXpmlobJL7nhnp3E",
    "w3MFlsmvpeCTNUcG",
    "5l0GTCFZI877KxkV",
    "DKGAIVKGYnLWmMy9",
    "QTmrnxW99iiyS2xS",
    "bwyAndeFczBo5Gzj",
    "qDVJNPKVIYAK5Irg",
    "aKDWzyoV4cSA6Da2",
    "lCqI2T7mSJAgSbAi",
    "GbfNQanBbRoPIoek",
    "VGLhN3OEa5t3YAFj",
    "Y2QhKx7sks9MExvw",
    "uluG1Pbks9IwTjsA",
    "qKKfsK89gK6yZii9",
    "lHnt4Veq3Tu10VKJ",
    "HLDekf7z49ovS9Y8"
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth_countries.json")) as f:
    ground_truth_countries = json.load(f)

def read_chain_from_file(candidate_chain_pattern, i, dir_path):
    candidate_fname = f"{candidate_chain_pattern}{i+1}.txt"
    with open(dir_path+"/"+candidate_fname, "r") as file:
        vlm_chain = file.read()
    return vlm_chain

def geocode_location(city, country):
    """
    Convert (city, country) pairs to (latitude, longitude).
    Uses OpenStreetMap's Nominatim geocoder.
    """
    geolocator = Nominatim(user_agent="geo_plotter")

    try:
        if city and country:
            location = geolocator.geocode(f"{city}, {country}")
        elif country:
            location = geolocator.geocode(country)
        else:
            return None, None
        if location:
            return location.latitude, location.longitude
        else:
            print(f"Nominatim could not find: {city}, {country}")
    except Exception as e:
        print(f"Error with {city}, {country}: {e}")
    return None, None

def read_cli_args_hybrid():
    parser = argparse.ArgumentParser()
    parser.add_argument('llm_model')
    parser.add_argument('challenge_path', help="Path to the challenge file containing reasoning chains")
    parser.add_argument('candidate_chain_pattern')
    parser.add_argument('results_file_suffix')
    parser.add_argument('cache_dir', help="Directory to cache models")
    parser.add_argument('--human', action="store_true")
    parser.add_argument('--radu', action="store_true")
    args = parser.parse_args()
    return args.llm_model, args.challenge_path, args.candidate_chain_pattern, args.results_file_suffix, args.cache_dir, args.human, args.radu

def get_challenge_name(challenge_path):
    return os.path.basename(os.path.normpath(challenge_path))

llm_model_name, challenge_path, candidate_chain_pattern, file_suffix, cache_dir, human, radu = read_cli_args_hybrid()


llm_model, llm_tokenizer = get_processor_and_model(llm_model_name, cache_dir, device_id=0)

vlm_scores = open(f"vlm_geolocation_scores_{file_suffix}.txt", "a")
vlm_scores.write("[ \n")

dirs = sorted(os.listdir(challenge_path))
for dir_idx, dir_name in enumerate(dirs):
    dir_path = os.path.join(challenge_path, dir_name)
    if not os.path.isdir(dir_path) or dir_name.startswith("."):
        continue
    challenge_name = get_challenge_name(dir_path)
    if human and challenge_name not in challenge_dirs:
        continue
    coordinates = []
    print(f"Challenge path: {dir_path}")
    metadata_file = find_metadata_file(dir_path)
    with open(metadata_file, 'r') as f:
        rounds = json.load(f)
        for i, item in enumerate(rounds):
            latitude = item.get('lat')
            longitude = item.get('lng')
            coordinates.append([latitude, longitude])
    # print("Extracted ground truth coordinates for all rounds.", coordinates)
    if human:
        if radu:
            human_candidate_chains = parse_reasoning_chains(dir_path + f"/Ground_Truth_Reasoning_Chain.txt")
        else:
            human_candidate_chains = parse_reasoning_chains(dir_path + f"/Human_Expert_{candidate_chain_pattern}.txt")
    for i in range(5):
        
        print(f"Processing Round {i+1}")
        
        if not human:
            vlm_chain = read_chain_from_file(candidate_chain_pattern, i, dir_path)
        else:
            vlm_chain = human_candidate_chains[i]

        gt_country, gt_city = ground_truth_countries.get(f"{challenge_name}_{i+1}", (None, None))
        for attempt in range(0 if gt_country else 3):
            try:
                gt_country, gt_city = reverse_geocode(coordinates[i][0], coordinates[i][1])
                if gt_country:
                    break

            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                wait_time = 2 * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Geocoding error ({type(e).__name__}) for '{coordinates[i][0], coordinates[i][1]}', retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        correct_guess = 0
        if gt_country is not None:

            prompt = f"""Provided below is a reasoning chain comprising of multiple bullet points. These represent reasoning for determining the location of a photo and these points make statements about scene attributes such as infrastructure, architecture, vegetation and agriculture, geology, topological features, cultural clues, vehicles, language signs and use these attributes to reason about the location of an image. 
            The guess of the country can be mentioned in one or more bullet points. Your task is to ONLY RETURN A SINGLE WORD ANSWER, either TRUE if this reasoning chain guesses the country as {gt_country}, or FALSE otherwise. Do not output any other reasoning or annotations. Start your answer with ANSWER:
            Follow the below rules:
            - Do not look for eact name match. If the guessed country is mentioned by any alternate names or abbreviations, such as, USA for United States, answer TRUE.
            - If the reasoning chain guesses a valid state or region within the country, answer TRUE. For example, if the guess is in Texas or California and the country is United States, answer TRUE.
            - If the reasoning chain mentions the continent only, answer FALSE. For example, if the guess is Asia for the correct country India, answer FALSE.
            \n
            Following is the reasoning chain: \n {vlm_chain} \n
            ANSWER:
            """
            
            res = completion_utils.text_to_text(prompt, llm_model, llm_tokenizer)
            print("LLM label for this chain is ", res)
            label = res.split("ANSWER:")[-1]
            if "true" in label.lower():
                correct_guess = 1

        json.dump({
            "challenge": challenge_name,
            "round": i+1,
            "gt_latitude": coordinates[i][0],
            "gt_longitude": coordinates[i][1],
            "gt_country": gt_country,
            "gt_city": gt_city,
            "correct_guess": correct_guess
        }, vlm_scores)
        vlm_scores.flush()
        if dir_idx == len(dirs)-1 and i == 4:
            vlm_scores.write("\n")
        else:
            vlm_scores.write(",\n")
vlm_scores.write("]\n")