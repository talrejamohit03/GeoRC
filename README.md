# GeoRC

## Dataset

The GeoRC dataset is available on Hugging Face. Download it using the following instructions:

### Download Instructions

1. Install the Hugging Face datasets library (if not already installed):
   ```bash
   pip install datasets
   ```

2. Download the dataset from Hugging Face:
   ```bash
   huggingface-cli download mohit-talreja/GeoRC --repo-type dataset --local-dir ./GeoRC_dataset
   ```

3. Alternatively, you can download the dataset programmatically in Python:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("mohit-talreja/GeoRC")
   ```

4. The dataset will be downloaded to `./GeoRC_dataset` (or your specified directory)

For more information, visit: https://huggingface.co/datasets/mohit-talreja/GeoRC

## Generate Script

The `generate.py` script generates reasoning chains for GeoRC challenges using various vision-language models. It processes images and produces detailed geographic reasoning explanations that identify key evidence (lane markings, vegetation, language, architecture, etc.) used to determine location estimates.

### Usage

```bash
python generate.py --type <model_type> --model <model_name> --path <challenge_path> [optional_arguments]
```

### Required Arguments

- `--type`: The type of model to use
  - `hf` - Hugging Face models
  - `gemini` - Google Gemini API
  - `azure` - Azure API

- `--model`: The specific model name to use
  - For Hugging Face: `google`, `llava`, `qwen`, etc.
  - For Gemini: `gemini-1.5-flash`, `gemini-1.5-pro`, etc.
  - For Azure: Your deployed model name

- `--path`: Path to the challenge directory containing images
  - Can be a single challenge directory or parent directory with multiple challenges
  - Images should follow the naming pattern: `{challenge_name}_{1-5}.png`

- `--key`: API key or token
  - For Hugging Face: Your Hugging Face API token
  - For Gemini: Your Google Gemini API key
  - For Azure: Your Azure API key

### Optional Arguments

- `--cache`: Cache directory for model weights (default: `./cache`)
  - Only applicable for Hugging Face models
  - Specifies where downloaded models are stored

- `--endpoint`: API endpoint URL
  - Only required for Azure deployments
  - Example: `https://<resource_name>.openai.azure.com/`

- `--api_version`: API version
  - Only required for Azure deployments
  - Example: `2024-02-15-preview`

### Examples

**Hugging Face Model:**
```bash
python generate.py --type hf --model google --path ./challenges/challenge_1 --key hf_xxxxxxxxxxxx --cache ./models
```

**Gemini API:**
```bash
python generate.py --type gemini --model gemini-1.5-flash --path ./challenges --key sk_xxxxxxxxxxxx
```

**Azure API:**
```bash
python generate.py --type azure --model gpt-4-vision --path ./challenges/challenge_1 --key sk_xxxxxxxxxxxx --endpoint https://myresource.openai.azure.com/ --api_version 2024-02-15-preview
```

### Output

The script generates files named `candidate_reasoning_chain_{model_name}_{image_number}.txt` in each challenge directory. These files contain detailed geographic reasoning explanations for each image in the challenge.

### Using the Dataset with the Generate Script

Once downloaded, use the dataset path as the `--path` argument:

```bash
python generate.py --type hf --model google --path ./GeoRC_dataset --key hf_xxxxxxxxxxxx
```