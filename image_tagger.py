import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.amp.autocast_mode
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline, AutoModelForImageClassification
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
import pandas as pd
import json
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
import io
import gc
from pathlib import Path

# Load JoyCaption configuration
with open(Path(__file__).parent / "jc_data.json", "r", encoding="utf-8") as f:
    jc_config = json.load(f)

MODEL_CONFIGS = {
    "WD-14": {
        "name": "SmilingWolf/wd-v1-4-vit-tagger",
        "type": "tagger"
    },
    "JoyCaption": {
        "name": "mradermacher/llama-joycaption-beta-one-hf-llava-GGUF",
        "type": "caption",
        "model_file": "llama-joycaption-beta-one-hf-llava.Q4_K_M.gguf"
    },
    "Florence-2": {
        "name": "microsoft/florence-2-base",
        "type": "florence",
        "processor": "microsoft/florence-2-base",
        "modes": ["tags", "caption"]
    }
}

class JoyCaptionProcessor:
    def __init__(self):
        try:
            # Initialize model paths
            models_dir = Path("models").resolve()
            llm_models_dir = (models_dir / "LLM" / "GGUF").resolve()
            llm_models_dir.mkdir(parents=True, exist_ok=True)
            
            config = MODEL_CONFIGS["JoyCaption"]
            model_path = llm_models_dir / config["model_file"]
            
            # Download the GGUF model if it doesn't exist
            if not model_path.exists():
                model_path = Path(hf_hub_download(
                    repo_id=config["name"],
                    filename=config["model_file"],
                    local_dir=str(llm_models_dir),
                    local_dir_use_symlinks=False
                )).resolve()
            
            # GPU configuration
            n_gpu_layers = 0
            if torch.cuda.is_available():
                n_gpu_layers = -1  # Use all GPU layers if available
            
            # Initialize the GGUF model with optimized settings
            self.model = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=4096,         # Increased context window
                n_batch=512,        # Batch size for prompt processing
                chat_format="chatml",  # Using ChatML format
                verbose=False       # Disable verbose output
            )
            
        except Exception as e:
            raise RuntimeError(f"JoyCaption initialization failed: {str(e)}")

    def __call__(self, image):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to very small size to fit in context window
            image = image.resize((128, 128), Image.Resampling.LANCZOS)
            
            # Convert image to base64 with minimal quality and maximum compressi
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=30, optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare the messages in ChatML format with proper image format
            messages = [
                {"role": "system", "content": "You are an image tagging assistant. Analyze the image and provide relevant tags separated by commas."},
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}},
                    {"type": "text", "text": "List the tags for this image:"}
                ]}
            ]
            
            # Generate response with optimized parameters for GGUF model
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=jc_config["model_settings"]["default_max_tokens"],
                temperature=jc_config["model_settings"]["default_temperature"],
                top_p=jc_config["model_settings"]["default_top_p"],
                top_k=jc_config["model_settings"]["default_top_k"],
                stream=False
            )
            
            # Extract and process the generated text
            raw_text = response["choices"][0]["message"]["content"].strip()
            
            # Clean and process tags
            tags = [tag.strip() for tag in raw_text.split(',') if tag.strip()]
            
            # Return the processed tags joined with commas
            return ", ".join(tags)
            
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            gc.collect()

class FlorenceProcessor:
    def __init__(self, mode="tags"):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            config = MODEL_CONFIGS["Florence-2"]
            self.mode = mode
            
            # Initialize processor and model based on mode
            print(f"Loading Florence-2 model in {mode} mode...")
            if mode == "tags":
                self.model = pipeline(
                    "image-classification",
                    model=config["name"],
                    device=0 if self.device == "cuda" else -1
                )
            else:  # caption mode
                self.model = pipeline(
                    "image-to-text",
                    model=config["name"],
                    device=0 if self.device == "cuda" else -1
                )
            print("Florence-2 model loaded successfully")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Florence initialization failed: {str(e)}")

    def __call__(self, image, additional_tags=None):
        try:
            print(f"\n=== Florence-2 Processing Start ({self.mode} mode) ===")
            
            # Convert and prepare image
            if not isinstance(image, Image.Image):
                print(f"Opening image from path: {image}")
                image = Image.open(image).convert('RGB')
            
            print(f"Processing image of size: {image.size}")
            
            # Process based on mode
            if self.mode == "tags":
                predictions = self.model(image)
                threshold = 0.35
                result = [pred['label'] for pred in predictions if pred['score'] > threshold]
            else:  # caption mode
                prediction = self.model(image)
                result = [prediction[0]['generated_text']]
            
            # Add additional tags if provided
            if additional_tags:
                result.extend(additional_tags.split(","))
            
            print(f"Generated {len(result)} {'tags' if self.mode == 'tags' else 'caption'}")
            print("=== Florence-2 Processing Complete ===\n")
            
            return ", ".join(result)
            
        except Exception as e:
            print(f"Error in Florence-2 processing: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            batch = {'image': image_tensor.unsqueeze(0).to(self.device)}
            
            # Run inference with mixed precision
            print("\nRunning model inference...")
            with torch.amp.autocast_mode.autocast(self.device, enabled=True):
                preds = self.model(batch)
                print(f"\nRaw predictions:")
                print(f"- Type: {type(preds)}")
                print(f"- Shape: {preds.shape if isinstance(preds, torch.Tensor) else [type(p) for p in preds] if isinstance(preds, tuple) else preds.keys() if isinstance(preds, dict) else 'unknown'}")
                
                if isinstance(preds, dict):
                    print("Processing dictionary predictions...")
                    preds = preds['tags'] if 'tags' in preds else preds
                if isinstance(preds, tuple):
                    print("Processing tuple predictions...")
                    preds = preds[0]
                
                print(f"\nPre-sigmoid stats:")
                print(f"- Shape: {preds.shape}")
                print(f"- Range: [{preds.min().item():.3f}, {preds.max().item():.3f}]")
                
                # Apply sigmoid
                tag_preds = torch.sigmoid(preds).cpu()
                
                print(f"\nPost-sigmoid stats:")
                print(f"- Shape: {tag_preds.shape}")
                print(f"- Range: [{tag_preds.min().item():.3f}, {tag_preds.max().item():.3f}]")
            
            # Calculate scores for each tag
            print("\nProcessing tag scores...")
            n_tags = len(self.top_tags)
            scores = {}
            for i in range(min(n_tags, tag_preds.shape[1])):
                tag = self.top_tags[i]
                confidence = float(tag_preds[0][i])
                scores[tag] = confidence
            
            print(f"Processed {len(scores)} tag scores")
            
            # Sort tags by confidence
            sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nConfidence thresholds:")
            print(f"- High confidence: > 0.85")
            print(f"- Medium confidence: > {self.threshold}")
            
            # Get top tags with different confidence thresholds
            predicted_tags = []
            print("\nSelecting tags above thresholds:")
            for tag, confidence in sorted_tags:
                if len(predicted_tags) >= 30:
                    break
                if confidence > 0.5:  # High confidence tags (lowered from 0.85)
                    print(f"High confidence: {tag} ({confidence*100:.1f}%)")
                    predicted_tags.append(f"{tag} ({confidence*100:.1f}%)")
                elif confidence > self.threshold:  # Medium confidence tags
                    print(f"Medium confidence: {tag} ({confidence*100:.1f}%)")
                    predicted_tags.append(tag)
                elif len(predicted_tags) < 5:  # Ensure we get at least some tags
                    print(f"Low confidence: {tag} ({confidence*100:.1f}%)")
                    predicted_tags.append(tag)
            
            print(f"\nSelected {len(predicted_tags)} tags")
            
            # Add additional tags if provided
            if additional_tags:
                print(f"\nAdding {len(additional_tags.split(','))} additional tags")
                extra_tags = [tag.strip() for tag in additional_tags.split(",") if tag.strip()]
                predicted_tags.extend(extra_tags)
            
            # Create final tag string
            tag_string = ', '.join(predicted_tags)
            
            # Print top predictions for debugging
            print("\nTop 10 predictions with scores:")
            for tag, score in sorted_tags[:10]:
                print(f"{tag}: {score:.3f}")
            
            if not predicted_tags:
                print("\nWARNING: No tags predicted! This might indicate an issue with the model or thresholds.")
            else:
                print(f"\nFinal tags: {tag_string}")
            
            print("=== JoyTag Processing Complete ===\n")
            return tag_string
            
class WD14Processor:
    def __init__(self, model_path):
        model_path = os.path.abspath(model_path)
        self.session = ort.InferenceSession(os.path.join(model_path, "model.onnx"))
        self.tags_df = pd.read_csv(os.path.join(model_path, "selected_tags.csv"))
        self.size = 448
        
        # Print model info for debugging
        print("Model inputs:", [input.name for input in self.session.get_inputs()])
        print("Model outputs:", [output.name for output in self.session.get_outputs()])

    def preprocess(self, image):
        # Convert RGBA to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize and pad the image to maintain aspect ratio
        ratio = self.size / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new image with white background
        square = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        square.paste(image, ((self.size - new_size[0]) // 2,
                           (self.size - new_size[1]) // 2))
        
        # Convert to numpy array
        image_np = np.array(square).astype(np.float32)
        # Convert RGB to BGR
        image_np = image_np[:, :, ::-1]
        
        # Reshape to match model's expected input shape: (1, 448, 448, 3)
        image_np = np.expand_dims(image_np, 0)  # Add batch dimension
        
        print(f"Input shape: {image_np.shape}")  # Debug print
        print(f"Input value range: [{image_np.min():.2f}, {image_np.max():.2f}]")  # Debug print
        return image_np

    def __call__(self, images, return_tensors=None):
        processed = self.preprocess(images)
        print(f"Processed shape: {processed.shape}")  # Debug print
        return {"input": processed}

    def postprocess(self, outputs, additional_tags=None, banned_tags=None, low_threshold=0.35, high_threshold=0.85):
        probs = 1 / (1 + np.exp(-outputs[0]))  # sigmoid
        
        # Process tags with different thresholds
        detected_tags = []
        
        # Get banned tags list
        banned_tags_list = []
        if banned_tags:
            banned_tags_list = [tag.strip().lower() for tag in banned_tags.split(",")]
        
        # Print top 20 tags for debugging
        print("\nDetected tags:")
        indices = np.argsort(probs[0])[::-1][:20]
        
        for idx in indices:
            prob = probs[0][idx]
            if prob < low_threshold:  # Skip low probability tags
                continue
                
            name = str(self.tags_df.iloc[idx]['name'])
            category = str(self.tags_df.iloc[idx]['category'])
            
            # Skip banned tags
            if name.lower() in banned_tags_list:
                continue
            
            # Higher threshold for character tags (category 4)
            if category == '4' and prob < high_threshold:
                continue
                
            if prob > low_threshold:
                # Remove underscores from tag names and keep spaces
                name = name.replace('_', ' ')
                detected_tags.append(name)
                print(f"{name}")
        
        # Add additional tags
        if additional_tags:
            extra_tags = [tag.strip().replace('_', ' ') for tag in additional_tags.split(",")]
            detected_tags.extend(extra_tags)
        
        return ", ".join(detected_tags) if detected_tags else "no tags"

class ImageTagger(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Easy Tagger")
        self.geometry("800x700")  # Zwiększona wysokość i szerokość okna
        
        # Create main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Help Frame na górze
        help_frame = ttk.LabelFrame(main_frame, text="Quick Guide", padding="10")
        help_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        help_text = """How to use Easy Tagger:
        
1. Select Model: WD-14 is recommended for anime/illustrations
2. Choose input: Either select individual files or a whole folder
3. Choose output folder: Where to save tagged images
4. Tag Settings:
   • Additional Tags: Tags to add to ALL images (e.g., anime, digital_art)
   • Banned Tags: Tags to exclude (e.g., sensitive, explicit)
   
Notes:
- Tags should be comma-separated
- Original images remain unchanged
- Each image will get a _tags.txt file
- The AI model works best with clear, single-subject images"""
        
        help_label = ttk.Label(help_frame, text=help_text, wraplength=700, justify="left")
        help_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        current_row = 1
        
        # Model Selection
        ttk.Label(main_frame, text="Select Model:").grid(row=current_row, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value="WD-14")
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_var, values=list(MODEL_CONFIGS.keys()))
        model_combo.grid(row=current_row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_select)
        current_row += 1
        
        # Mode Selection for Florence-2
        self.mode_frame = ttk.Frame(main_frame)
        self.mode_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(self.mode_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value="tags")
        mode_combo = ttk.Combobox(self.mode_frame, textvariable=self.mode_var, values=["tags", "caption"], state="readonly", width=15)
        mode_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.mode_frame.grid_remove()  # Hide by default
        current_row += 1
        
        # Input Type Selection
        self.input_type = tk.StringVar(value="files")
        ttk.Radiobutton(main_frame, text="Select Files", variable=self.input_type, 
                       value="files", command=self.update_input_mode).grid(row=current_row, column=0, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Select Folder", variable=self.input_type,
                       value="folder", command=self.update_input_mode).grid(row=current_row, column=1, sticky=tk.W)
        current_row += 1
        
        # Input Selection
        ttk.Label(main_frame, text="Input:").grid(row=current_row, column=0, sticky=tk.W)
        self.input_entry = ttk.Entry(main_frame)
        self.input_entry.grid(row=current_row, column=1, sticky=(tk.W, tk.E))
        self.input_button = ttk.Button(main_frame, text="Browse", command=self.select_input)
        self.input_button.grid(row=current_row, column=2, sticky=tk.W, padx=5)
        current_row += 1
        
        # Output Selection
        ttk.Label(main_frame, text="Output Folder:").grid(row=current_row, column=0, sticky=tk.W)
        self.output_entry = ttk.Entry(main_frame)
        self.output_entry.grid(row=current_row, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.select_output).grid(row=current_row, column=2, sticky=tk.W, padx=5)
        current_row += 1
        
        # Additional Tags Frame
        tags_frame = ttk.LabelFrame(main_frame, text="Tag Settings", padding="5")
        tags_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        # Additional Tags Entry
        ttk.Label(tags_frame, text="Additional Tags:").grid(row=0, column=0, sticky=tk.W)
        self.additional_tags_entry = ttk.Entry(tags_frame, width=40)
        self.additional_tags_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Label(tags_frame, text="(comma separated)").grid(row=0, column=2, sticky=tk.W)
        
        # Banned Tags Entry
        ttk.Label(tags_frame, text="Banned Tags:").grid(row=1, column=0, sticky=tk.W)
        self.banned_tags_entry = ttk.Entry(tags_frame, width=40)
        self.banned_tags_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Label(tags_frame, text="(comma separated)").grid(row=1, column=2, sticky=tk.W)
        
        # Threshold Settings Frame
        threshold_frame = ttk.Frame(tags_frame)
        threshold_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Low Threshold Entry
        ttk.Label(threshold_frame, text="Low Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.low_threshold = tk.StringVar(value="0.35")
        self.low_threshold_entry = ttk.Entry(threshold_frame, width=10, textvariable=self.low_threshold)
        self.low_threshold_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # High Threshold Entry
        ttk.Label(threshold_frame, text="High Threshold:").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.high_threshold = tk.StringVar(value="0.85")
        self.high_threshold_entry = ttk.Entry(threshold_frame, width=10, textvariable=self.high_threshold)
        self.high_threshold_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Add help text
        ttk.Label(threshold_frame, text="(0.0 - 1.0)").grid(row=0, column=4, sticky=tk.W, padx=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        current_row += 1
        
        # Status Label
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, wraplength=400)
        self.status_label.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E))
        current_row += 1
        
        # Start Button
        self.start_button = ttk.Button(main_frame, text="Start Tagging", command=self.start_tagging)
        self.start_button.grid(row=current_row, column=0, columnspan=3, pady=10)
        
        # Configure grid
        main_frame.columnconfigure(1, weight=1)
        
        # Initialize paths
        self.input_paths = None
        
        self.input_paths = None
        
    def on_model_select(self, event=None):
        model_choice = self.model_var.get()
        
        # Show/hide Florence-2 mode selection
        if model_choice == "Florence-2":
            self.mode_frame.grid()
        else:
            self.mode_frame.grid_remove()
    
    def update_input_mode(self):
        self.input_entry.delete(0, tk.END)
        
    def select_input(self):
        if self.input_type.get() == "files":
            paths = filedialog.askopenfilenames(
                title="Select Image Files",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")]
            )
            if paths:
                self.input_paths = paths
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, f"Selected {len(paths)} files")
        else:
            path = filedialog.askdirectory(title="Select Input Directory")
            if path:
                self.input_paths = [path]
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, path)
                
    def select_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)
            
    def update_status(self, message):
        print(f"Status: {message}")  # Debug print
        self.status_var.set(message)
        self.update_idletasks()
        
    def update_progress(self, value):
        self.progress['value'] = value
        self.update_idletasks()
        
    def download_model(self, model_name):
        try:
            self.update_status(f"Downloading model {model_name}...")
            
            # Read HF token from environment variable or prompt user
            token = os.environ.get('HF_TOKEN')
            if not token:
                token = self.prompt_for_token()
                if not token:
                    raise ValueError("Hugging Face token is required to download the model")
                
            snapshot_download(
                repo_id=model_name,
                local_dir=f"models/{model_name}",
                token=token
            )
            self.update_status("Model downloaded successfully!")
        except Exception as e:
            self.update_status(f"Error downloading model: {str(e)}")
            raise
            
    def prompt_for_token(self):
        dialog = tk.Toplevel(self)
        dialog.title("Hugging Face Token Required")
        dialog.geometry("400x150")
        dialog.transient(self)
        
        tk.Label(dialog, text="This model requires authentication.\nPlease enter your Hugging Face token:", pady=10).pack()
        
        token_var = tk.StringVar()
        token_entry = ttk.Entry(dialog, textvariable=token_var, width=40)
        token_entry.pack(pady=5)
        
        token_result = [None]  # Use list to store result
        
        def submit():
            token_result[0] = token_var.get().strip()
            dialog.destroy()
        
        ttk.Button(dialog, text="Submit", command=submit).pack(pady=10)
        
        # Help text with link
        help_text = "To get your token, visit: https://huggingface.co/settings/tokens"
        help_label = ttk.Label(dialog, text=help_text)
        help_label.pack(pady=5)
        
        dialog.wait_window()  # Wait for dialog to close
        return token_result[0]
            
    def process_images(self):
        try:
            if not self.input_paths:
                messagebox.showerror("Error", "Please select input files or folder!")
                return
            
            # Set up output path
            input_dir = os.path.dirname(self.input_paths[0])
            output_path = self.output_entry.get()
            if not output_path:
                output_path = os.path.join(input_dir, "tagged_images")
                
            # Make sure output path is different from input
            if os.path.abspath(input_dir) == os.path.abspath(output_path):
                output_path = os.path.join(input_dir, "tagged_images")
                
            # Create output directory
            try:
                os.makedirs(output_path, exist_ok=True)
                self.update_status(f"Will save files to: {output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output directory: {output_path}\nError: {str(e)}")
                return
                
            model_choice = self.model_var.get()
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Check and download model if needed
            model_config = MODEL_CONFIGS[model_choice]
            
            if model_config['type'] == 'joytag' and model_config.get('custom_model', False):
                custom_model = self.custom_model_entry.get().strip()
                if custom_model:
                    model_path = f"models/{custom_model}"
                    if not os.path.exists(model_path):
                        try:
                            self.download_model(custom_model)
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to download custom model: {str(e)}")
                            return
                else:
                    model_path = f"models/{model_config['name']}"
                    if not os.path.exists(model_path):
                        try:
                            self.download_model(model_config['name'])
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to download model: {str(e)}")
                            return
            else:
                model_path = f"models/{model_config['name']}"
                if not os.path.exists(model_path):
                    try:
                        self.download_model(model_config['name'])
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to download model: {str(e)}")
                        return
            
            # Load model based on type
            self.update_status("Loading model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if model_config['type'] == "tagger":
                processor = WD14Processor(model_path)
            elif model_config['type'] == "caption":
                processor = JoyCaptionProcessor()
            elif model_config['type'] == "florence":
                processor = FlorenceProcessor(mode=self.mode_var.get())
            
            # Collect all image files
            image_files = []
            for input_path in self.input_paths:
                if os.path.isfile(input_path):
                    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_files.append(input_path)
                else:
                    for root, _, files in os.walk(input_path):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                image_files.append(os.path.join(root, file))
            
            total_files = len(image_files)
            self.update_status(f"Found {total_files} images to process")
            
            # Process images
            for i, image_path in enumerate(image_files):
                try:
                    image = Image.open(image_path)
                    
                    # Process image based on model type
                    if model_config['type'] == "tagger":
                        # Get input name from the model
                        input_name = processor.session.get_inputs()[0].name
                        self.update_status(f"Using input name: {input_name}")
                        
                        inputs = processor(images=image, return_tensors="pt")
                        input_array = inputs["input"]
                        outputs = processor.session.run(None, {input_name: input_array})
                        
                        # Get settings from UI
                        additional_tags = self.additional_tags_entry.get().strip()
                        banned_tags = self.banned_tags_entry.get().strip()
                        
                        try:
                            low_thresh = float(self.low_threshold.get())
                            high_thresh = float(self.high_threshold.get())
                        except ValueError:
                            low_thresh = 0.35  # default value
                            high_thresh = 0.85  # default value
                            
                        # Ensure thresholds are in valid range
                        low_thresh = max(0.0, min(1.0, low_thresh))
                        high_thresh = max(0.0, min(1.0, high_thresh))
                        
                        tags = processor.postprocess(
                            outputs, 
                            additional_tags, 
                            banned_tags,
                            low_threshold=low_thresh,
                            high_threshold=high_thresh
                        )
                        
                    elif model_config['type'] == "joytag":
                        # Get settings from UI
                        additional_tags = self.additional_tags_entry.get().strip()
                        banned_tags = self.banned_tags_entry.get().strip()
                        try:
                            low_thresh = float(self.low_threshold.get())
                            high_thresh = float(self.high_threshold.get())
                        except ValueError:
                            low_thresh = 0.35
                            high_thresh = 0.85
                        
                        # Update processor thresholds
                        processor.threshold = low_thresh
                        
                        # Process image with JoyTag model
                        tags = processor(image, additional_tags)
                        self.update_status(f"JoyTag generated tags: {tags}")
                        
                    elif model_config['type'] == "caption":
                        # For JoyCaption, pass the image and get additional tags
                        additional_tags = self.additional_tags_entry.get().strip()
                        tags = processor(image)
                        
                        # Add additional tags if provided
                        if additional_tags:
                            model_tags = tags.split(", ")
                            extra_tags = [tag.strip() for tag in additional_tags.split(",") if tag.strip()]
                            all_tags = model_tags + extra_tags
                            tags = ", ".join(all_tags)
                    
                    self.update_status(f"Generated tags: {tags}")
                    
                    # Save image and tags
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    ext = os.path.splitext(image_path)[1]
                    
                    # Generate output paths
                    new_image_path = os.path.join(output_path, f"{base_name}{ext}")
                    tags_file_path = os.path.join(output_path, f"{base_name}.txt")
                    
                    try:
                        # Only copy if source and destination are different
                        if os.path.abspath(image_path) != os.path.abspath(new_image_path):
                            self.update_status(f"Copying image to: {new_image_path}")
                            shutil.copy2(image_path, new_image_path)
                        
                        # Always save tags file
                        self.update_status(f"Saving tags to: {tags_file_path}")
                        with open(tags_file_path, 'w', encoding='utf-8') as f:
                            f.write(tags)
                            
                        # Verify files were created
                        if not os.path.exists(tags_file_path):
                            raise Exception("Failed to create tags file")
                            
                    except Exception as e:
                        self.update_status(f"Error saving files: {str(e)}")
                        continue  # Continue with next image instead of raising
                    
                    # Update progress
                    progress = (i + 1) / total_files * 100
                    self.update_progress(progress)
                    self.update_status(f"Processed {i + 1}/{total_files}: {os.path.basename(image_path)}")
                    
                except Exception as e:
                    self.update_status(f"Error processing {os.path.basename(image_path)}: {str(e)}")
                    continue
            
            # Show completion message with output path
            completion_msg = f"Tagging completed!\nFiles saved to: {output_path}"
            self.update_status(completion_msg)
            messagebox.showinfo("Complete", completion_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
        finally:
            self.start_button['state'] = 'normal'
            
    def start_tagging(self):
        self.start_button['state'] = 'disabled'
        self.progress['value'] = 0
        threading.Thread(target=self.process_images, daemon=True).start()

if __name__ == '__main__':
    app = ImageTagger()
    app.mainloop()
