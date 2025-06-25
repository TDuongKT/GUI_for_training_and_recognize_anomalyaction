import os
import yaml
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from train_u2net_lite import train_model
from test_u2net_lite_video import test_video

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("U2NETLite Segmentation GUI")
        self.root.geometry("800x600")
        
        self.config = self.load_config('config.yaml')
        self.init_ui()
        
        self.training_thread = None
        self.testing_thread = None
        self.total_epochs = 0
        self.total_frames = 0
    
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Config file not found. Using default config.")
            return {
                'model': {'num_classes': 3, 'image_size': [256, 256]},
                'data': {
                    'train_image_dir': 'data/train/images',
                    'train_annotation': 'data/train/annotations/data.json',
                    'val_image_dir': 'data/val/images',
                    'val_annotation': 'data/val/annotations/data.json',
                    'background_dir': 'data/backgrounds',
                    'category_ids': [1, 2],
                    'background_prob': 0.7
                },
                'training': {
                    'batch_size': 128,
                    'lr': 0.001,
                    'epochs': 50,
                    'best_model_path': 'models/best_model.pth',
                    'last_model_path': 'models/last_model.pth',
                    'loss': 'dice'
                },
                'testing': {
                    'input_video_path': 'input_video.mp4',
                    'output_video_path': 'output/output_segmented_video.mp4',
                    'model_path': 'models/best_model.pth',
                    'colors': [[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0]]
                }
            }
    
    def init_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(pady=10, expand=True, fill='both')
        
        # Training Tab
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="Training")
        
        # Data directories
        ttk.Label(training_frame, text="Training Image Dir:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.train_image_dir = ttk.Entry(training_frame, width=50)
        self.train_image_dir.insert(0, self.config['data']['train_image_dir'])
        self.train_image_dir.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(training_frame, text="Browse", command=lambda: self.browse_directory(self.train_image_dir)).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Training Annotation:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.train_ann_file = ttk.Entry(training_frame, width=50)
        self.train_ann_file.insert(0, self.config['data']['train_annotation'])
        self.train_ann_file.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(training_frame, text="Browse", command=lambda: self.browse_file(self.train_ann_file, "*.json")).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Validation Image Dir:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.val_image_dir = ttk.Entry(training_frame, width=50)
        self.val_image_dir.insert(0, self.config['data']['val_image_dir'])
        self.val_image_dir.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(training_frame, text="Browse", command=lambda: self.browse_directory(self.val_image_dir)).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Validation Annotation:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.val_ann_file = ttk.Entry(training_frame, width=50)
        self.val_ann_file.insert(0, self.config['data']['val_annotation'])
        self.val_ann_file.grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(training_frame, text="Browse", command=lambda: self.browse_file(self.val_ann_file, "*.json")).grid(row=3, column=2, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Background Dir:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.bg_dir = ttk.Entry(training_frame, width=50)
        self.bg_dir.insert(0, self.config['data']['background_dir'])
        self.bg_dir.grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(training_frame, text="Browse", command=lambda: self.browse_directory(self.bg_dir)).grid(row=4, column=2, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Background Probability:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.bg_prob = ttk.Entry(training_frame, width=10)
        self.bg_prob.insert(0, str(self.config['data']['background_prob']))
        self.bg_prob.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        
        # Training parameters
        ttk.Label(training_frame, text="Number of Classes:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.num_classes = ttk.Entry(training_frame, width=10)
        self.num_classes.insert(0, str(self.config['model']['num_classes']))
        self.num_classes.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(training_frame, text="Category IDs (comma-separated):").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        self.category_ids = ttk.Entry(training_frame, width=50)
        self.category_ids.insert(0, ','.join(map(str, self.config['data']['category_ids'])))
        self.category_ids.grid(row=7, column=1, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Batch Size:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
        self.batch_size = ttk.Entry(training_frame, width=10)
        self.batch_size.insert(0, str(self.config['training']['batch_size']))
        self.batch_size.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(training_frame, text="Learning Rate:").grid(row=9, column=0, padx=5, pady=5, sticky="e")
        self.lr = ttk.Entry(training_frame, width=10)
        self.lr.insert(0, str(self.config['training']['lr']))
        self.lr.grid(row=9, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(training_frame, text="Epochs:").grid(row=10, column=0, padx=5, pady=5, sticky="e")
        self.epochs = ttk.Entry(training_frame, width=10)
        self.epochs.insert(0, str(self.config['training']['epochs']))
        self.epochs.grid(row=10, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(training_frame, text="Best Model Path:").grid(row=11, column=0, padx=5, pady=5, sticky="e")
        self.best_model_path = ttk.Entry(training_frame, width=50)
        self.best_model_path.insert(0, self.config['training']['best_model_path'])
        self.best_model_path.grid(row=11, column=1, padx=5, pady=5)
        ttk.Button(training_frame, text="Browse", command=lambda: self.browse_file(self.best_model_path, "*.pth")).grid(row=11, column=2, padx=5, pady=5)
        
        ttk.Label(training_frame, text="Last Model Path:").grid(row=12, column=0, padx=5, pady=5, sticky="e")
        self.last_model_path = ttk.Entry(training_frame, width=50)
        self.last_model_path.insert(0, self.config['training']['last_model_path'])
        self.last_model_path.grid(row=12, column=1, padx=5, pady=5)
        ttk.Button(training_frame, text="Browse", command=lambda: self.browse_file(self.last_model_path, "*.pth")).grid(row=12, column=2, padx=5, pady=5)
        
        # Train button
        ttk.Button(training_frame, text="Start Training", command=self.start_training).grid(row=13, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.train_progress = ttk.Progressbar(training_frame, length=400, mode='determinate')
        self.train_progress.grid(row=14, column=0, columnspan=3, pady=5)
        
        # Log area
        self.train_log = tk.Text(training_frame, height=10, width=80)
        self.train_log.grid(row=15, column=0, columnspan=3, padx=5, pady=5)
        self.train_log.config(state='disabled')
        
        # Testing Tab
        testing_frame = ttk.Frame(notebook)
        notebook.add(testing_frame, text="Testing")
        
        ttk.Label(testing_frame, text="Input Video:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.input_video = ttk.Entry(testing_frame, width=50)
        self.input_video.insert(0, self.config['testing']['input_video_path'])
        self.input_video.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(testing_frame, text="Browse", command=lambda: self.browse_file(self.input_video, "*.mp4 *.avi")).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(testing_frame, text="Output Video:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.output_video = ttk.Entry(testing_frame, width=50)
        self.output_video.insert(0, self.config['testing']['output_video_path'])
        self.output_video.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(testing_frame, text="Browse", command=lambda: self.browse_file(self.output_video, "*.mp4")).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(testing_frame, text="Model Path:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.test_model_path = ttk.Entry(testing_frame, width=50)
        self.test_model_path.insert(0, self.config['testing']['model_path'])
        self.test_model_path.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(testing_frame, text="Browse", command=lambda: self.browse_file(self.test_model_path, "*.pth")).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Button(testing_frame, text="Start Testing", command=self.start_testing).grid(row=3, column=0, columnspan=3, pady=10)
        
        self.test_progress = ttk.Progressbar(testing_frame, length=400, mode='determinate')
        self.test_progress.grid(row=4, column=0, columnspan=3, pady=5)
        
        self.test_log = tk.Text(testing_frame, height=10, width=80)
        self.test_log.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        self.test_log.config(state='disabled')
    
    def browse_directory(self, entry):
        directory = filedialog.askdirectory()
        if directory:
            entry.delete(0, tk.END)
            entry.insert(0, directory)
    
    def browse_file(self, entry, file_filter):
        file = filedialog.askopenfilename(filetypes=[("Files", file_filter)])
        if file:
            entry.delete(0, tk.END)
            entry.insert(0, file)
    
    def start_training(self):
        try:
            self.config['data']['train_image_dir'] = self.train_image_dir.get()
            self.config['data']['train_annotation'] = self.train_ann_file.get()
            self.config['data']['val_image_dir'] = self.val_image_dir.get()
            self.config['data']['val_annotation'] = self.val_ann_file.get()
            self.config['data']['background_dir'] = self.bg_dir.get()
            self.config['data']['background_prob'] = float(self.bg_prob.get())
            self.config['model']['num_classes'] = int(self.num_classes.get())
            self.config['data']['category_ids'] = [int(x) for x in self.category_ids.get().split(',')]
            self.config['training']['batch_size'] = int(self.batch_size.get())
            self.config['training']['lr'] = float(self.lr.get())
            self.config['training']['epochs'] = int(self.epochs.get())
            self.config['training']['best_model_path'] = self.best_model_path.get()
            self.config['training']['last_model_path'] = self.last_model_path.get()
            
            with open('config.yaml', 'w') as f:
                yaml.safe_dump(self.config, f)
            
            self.train_log.config(state='normal')
            self.train_log.delete(1.0, tk.END)
            self.train_log.config(state='disabled')
            self.train_progress['value'] = 0
            self.total_epochs = self.config['training']['epochs']
            
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.start()
            self.root.after(100, self.check_training_progress)
        except ValueError as e:
            self.train_log.config(state='normal')
            self.train_log.insert(tk.END, f"Error: {str(e)}\n")
            self.train_log.config(state='disabled')
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
    
    def run_training(self):
        def progress_callback(epoch, total_epochs, train_loss, val_loss, train_iou, val_iou, train_dice, val_dice):
            log_message = (f"Epoch {epoch}/{total_epochs} | "
                           f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                           f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
                           f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}\n")
            with open('logs/training_progress.txt', 'a') as f:
                f.write(log_message)
        
        train_model(self.config, progress_callback)
    
    def check_training_progress(self):
        if self.training_thread and self.training_thread.is_alive():
            try:
                with open('logs/training_progress.txt', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        self.train_log.config(state='normal')
                        self.train_log.insert(tk.END, last_line)
                        self.train_log.see(tk.END)
                        self.train_log.config(state='disabled')
                        epoch = int(last_line.split('/')[0].split(' ')[1])
                        self.train_progress['value'] = (epoch / self.total_epochs) * 100
            except FileNotFoundError:
                pass
            self.root.after(100, self.check_training_progress)
        else:
            try:
                os.remove('logs/training_progress.txt')
            except FileNotFoundError:
                pass
    
    def start_testing(self):
        try:
            input_video = self.input_video.get()
            output_video = self.output_video.get()
            model_path = self.test_model_path.get()
            
            self.test_log.config(state='normal')
            self.test_log.delete(1.0, tk.END)
            self.test_log.config(state='disabled')
            self.test_progress['value'] = 0
            
            self.testing_thread = threading.Thread(target=self.run_testing, args=(input_video, output_video, model_path))
            self.testing_thread.start()
            self.root.after(100, self.check_testing_progress)
        except ValueError as e:
            self.test_log.config(state='normal')
            self.test_log.insert(tk.END, f"Error: {str(e)}\n")
            self.test_log.config(state='disabled')
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
    
    def run_testing(self, input_video, output_video, model_path):
        def progress_callback(frame, total_frames):
            with open('logs/testing_progress.txt', 'a') as f:
                f.write(f"{frame}/{total_frames}\n")
            self.total_frames = total_frames
        
        test_video(self.config, input_video, output_video, model_path, progress_callback)
    
    def check_testing_progress(self):
        if self.testing_thread and self.testing_thread.is_alive():
            try:
                with open('logs/testing_progress.txt', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        frame, total = map(int, last_line.strip().split('/'))
                        self.test_log.config(state='normal')
                        self.test_log.insert(tk.END, f"Processed frame {frame}/{total}\n")
                        self.test_log.see(tk.END)
                        self.test_log.config(state='disabled')
                        self.test_progress['value'] = (frame / total) * 100
            except FileNotFoundError:
                pass
            self.root.after(100, self.check_testing_progress)
        else:
            try:
                os.remove('logs/testing_progress.txt')
            except FileNotFoundError:
                pass

if __name__ == '__main__':
    root = tk.Tk()
    app = SegmentationGUI(root)
    root.mainloop()