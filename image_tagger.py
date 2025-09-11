import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from PIL import Image
import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import snapshot_download
import shutil
import pandas as pd

MODEL_CONFIGS = {
    "WD-14": {
        "name": "SmilingWolf/wd-v1-4-vit-tagger",
        "type": "tagger"
    },
    "Florence 2": {
        "name": "florence-2",
        "type": "caption"
    },
    "JoyCaption": {
        "name": "joycaption",
        "type": "caption"
    }
}

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

    def postprocess(self, outputs, additional_tags=None, banned_tags=None):
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
            if prob < 0.35:  # Skip low probability tags
                continue
                
            name = str(self.tags_df.iloc[idx]['name'])
            category = str(self.tags_df.iloc[idx]['category'])
            
            # Skip banned tags
            if name.lower() in banned_tags_list:
                continue
            
            # Higher threshold for character tags (category 4)
            if category == '4' and prob < 0.85:
                continue
                
            if prob > 0.35:
                detected_tags.append(name)
                print(f"{name}")
        
        # Add additional tags
        if additional_tags:
            extra_tags = [tag.strip() for tag in additional_tags.split(",")]
            detected_tags.extend(extra_tags)
        
        return ", ".join(detected_tags) if detected_tags else "no_tags"

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
            snapshot_download(repo_id=model_name, local_dir=f"models/{model_name}")
            self.update_status("Model downloaded successfully!")
        except Exception as e:
            self.update_status(f"Error downloading model: {str(e)}")
            raise
            
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
            else:
                processor = AutoProcessor.from_pretrained(model_path)
                model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
            
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
                        
                        # Debug prints
                        print(f"Input array shape before inference: {input_array.shape}")
                        print(f"Input array dtype: {input_array.dtype}")
                        
                        outputs = processor.session.run(None, {input_name: input_array})
                        
                        # Get additional and banned tags from UI
                        additional_tags = self.additional_tags_entry.get().strip()
                        banned_tags = self.banned_tags_entry.get().strip()
                        
                        tags = processor.postprocess(outputs, additional_tags, banned_tags)
                        
                        # Debug print
                        print(f"Output shape: {outputs[0].shape}")
                        
                        self.update_status(f"Generated tags: {tags}")
                    else:
                        inputs = processor(images=image, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = model.generate(**inputs)
                        tags = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    # Save image and tags
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    ext = os.path.splitext(image_path)[1]
                    
                    # Generate output paths
                    new_image_path = os.path.join(output_path, f"{base_name}{ext}")
                    tags_file_path = os.path.join(output_path, f"{base_name}_tags.txt")
                    
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
