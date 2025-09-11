# Easy Tagger
<img width="794" height="734" alt="image" src="https://github.com/user-attachments/assets/3305aa73-dc32-4caa-a54d-b7e551e8d618" />


A simple GUI application for image tagging using the WD-14 model. This tool makes it easy to automatically generate tags for your images with additional customization options.

## Features

- Support for multiple AI models (WD-14, Florence 2, JoyCaption)
- Batch processing of multiple images or folders
- Custom output location
- Additional tags support
- Tag blocking/filtering
- Simple and intuitive interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/easy-tagger.git
cd easy-tagger
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python easy_tagger.py
```

2. Using the interface:

   - **Select Model**: Choose the AI model you want to use (WD-14 recommended)
   
   - **Input Selection**:
     - Choose "Select Files" to tag individual images
     - Choose "Select Folder" to tag all images in a folder
     
   - **Output Folder**: Choose where to save the tagged images and tag files
   
   - **Tag Settings**:
     - **Additional Tags**: Enter any tags you want to add to ALL images
       - Example: `anime, digital_art, high_quality`
       - Separate tags with commas
     
     - **Banned Tags**: Enter tags you want to exclude from results
       - Example: `sensitive, questionable, explicit`
       - Separate tags with commas

3. Click "Start Tagging" to begin the process

4. Results:
   - Tagged images will be copied to the output folder
   - Each image will have an accompanying `_tags.txt` file
   - The original images remain unchanged

## Tips

- The WD-14 model is optimized for anime/illustration content
- For best results, use images with clear subjects
- You can use Additional Tags to add metadata like artist name or source
- Use Banned Tags to filter out unwanted or incorrect tags

## Requirements

- Python 3.8 or higher
- PyQt6
- torch
- Pillow
- numpy
- onnxruntime
- transformers
- huggingface_hub

## License

This project is licensed under the MIT License - see the LICENSE file for details.
