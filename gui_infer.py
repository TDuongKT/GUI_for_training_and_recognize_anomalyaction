import os
import yaml
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from feature_extraction import process_video
from feature_optimization import evaluate_feature_combinations
from clustering_analysis import load_optimal_combination, load_features, cluster_optimal_features, save_clustering_results
from anomaly_detection import load_clustering_results, train_one_class_svm, save_models, summarize_scores
from inference import inference_video
import torch

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

class FeatureExtractionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Extraction and Analysis")
        self.root.geometry("800x600")

        self.config = self.load_config('config.yaml')
        self.video_dir = tk.StringVar(value=self.config.get('feature_extraction', {}).get('video_dir', 'videos/train'))
        self.model_path = tk.StringVar(value=self.config.get('feature_extraction', {}).get('model_path', 'models/70.pth'))
        self.num_classes = tk.StringVar(value=str(self.config.get('model', {}).get('num_classes', 4)))
        self.class_names = tk.StringVar(value=','.join(self.config.get('feature_extraction', {}).get('class_names', ['background', 'shell', 'tip', 'spring'])))
        self.output_dir = tk.StringVar(value=self.config.get('feature_extraction', {}).get('output_dir', 'features'))
        self.n_clusters = tk.StringVar(value=str(self.config.get('feature_optimization', {}).get('n_clusters', 3)))
        self.min_features = tk.StringVar(value=str(self.config.get('feature_optimization', {}).get('min_features', 5)))
        self.max_features = tk.StringVar(value=str(self.config.get('feature_optimization', {}).get('max_features', 10)))
        self.max_combinations = tk.StringVar(value=str(self.config.get('feature_optimization', {}).get('max_combinations', 2000)))
        self.svm_nu = tk.StringVar(value=str(self.config.get('anomaly_detection', {}).get('nu', 0.05)))
        self.svm_gamma = tk.StringVar(value=str(self.config.get('anomaly_detection', {}).get('gamma', 0.1)))
        self.inference_video = tk.StringVar(value=self.config.get('inference', {}).get('input_video_path', ''))
        self.inference_model = tk.StringVar(value=self.config.get('inference', {}).get('model_path', 'models/70.pth'))
        self.inference_svm = tk.StringVar(value=self.config.get('inference', {}).get('svm_model_path', 'features/svm_model.npy'))
        self.inference_output_video = tk.StringVar(value=self.config.get('inference', {}).get('output_video_path', 'output/output_segmented.mp4'))
        self.inference_json = tk.StringVar(value=self.config.get('inference', {}).get('result_json_path', 'output/inference_results.json'))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_processing = False

        self.create_widgets()

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Config file not found. Using default config.")
            return {
                'model': {'num_classes': 4, 'image_size': [256, 256]},
                'feature_extraction': {
                    'video_dir': 'videos/train',
                    'model_path': 'models/70.pth',
                    'class_names': ['background', 'shell', 'tip', 'spring'],
                    'output_dir': 'features'
                },
                'feature_optimization': {
                    'n_clusters': 3,
                    'min_features': 5,
                    'max_features': 10,
                    'max_combinations': 2000
                },
                'clustering_analysis': {
                    'output_dir': 'features'
                },
                'anomaly_detection': {
                    'output_dir': 'features',
                    'nu': 0.05,
                    'gamma': 0.1
                },
                'inference': {
                    'input_video_path': '',
                    'model_path': 'models/70.pth',
                    'svm_model_path': 'features/svm_model.npy',
                    'output_video_path': 'output/output_segmented.mp4',
                    'result_json_path': 'output/inference_results.json'
                },
                'testing': {
                    'colors': [[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]]
                }
            }

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(pady=10, expand=True, fill='both')

        # Feature Extraction Tab
        extraction_frame = ttk.Frame(notebook)
        notebook.add(extraction_frame, text="Feature Extraction")
        self.create_extraction_tab(extraction_frame)

        # Feature Optimization Tab
        optimization_frame = ttk.Frame(notebook)
        notebook.add(optimization_frame, text="Feature Optimization")
        self.create_optimization_tab(optimization_frame)

        # Clustering Analysis Tab
        clustering_frame = ttk.Frame(notebook)
        notebook.add(clustering_frame, text="Clustering Analysis")
        self.create_clustering_tab(clustering_frame)

        # Anomaly Detection Tab
        anomaly_frame = ttk.Frame(notebook)
        notebook.add(anomaly_frame, text="Anomaly Detection")
        self.create_anomaly_tab(anomaly_frame)

        # Inference Tab
        inference_frame = ttk.Frame(notebook)
        notebook.add(inference_frame, text="Inference")
        self.create_inference_tab(inference_frame)

    def create_extraction_tab(self, frame):
        input_frame = ttk.LabelFrame(frame, text="Input Selection", padding=10)
        input_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(input_frame, text="Video Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.video_dir, width=50).grid(row=0, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_video_dir).grid(row=0, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="Model Path:").grid(row=1, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.model_path, width=50).grid(row=1, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_model).grid(row=1, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="Number of Classes:").grid(row=2, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.num_classes, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=7)

        ttk.Label(input_frame, text="Class Names (comma-separated):").grid(row=3, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.class_names, width=50).grid(row=3, column=1, padx=5, pady=7)

        ttk.Label(input_frame, text="Output Directory:").grid(row=4, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.output_dir, width=50).grid(row=4, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_output_dir).grid(row=4, column=2, padx=5, pady=7)

        ttk.Button(frame, text="Extract Features", command=self.start_extraction).pack(pady=10)
        self.extraction_progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.extraction_progress.pack(pady=10, fill="x", padx=10)
        self.extraction_status = tk.Text(frame, height=15, width=80)
        self.extraction_status.pack(pady=10, padx=10)
        self.extraction_status.config(state="disabled")

    def create_optimization_tab(self, frame):
        input_frame = ttk.LabelFrame(frame, text="Optimization Parameters", padding=10)
        input_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(input_frame, text="Number of Clusters:").grid(row=0, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.n_clusters, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=7)

        ttk.Label(input_frame, text="Min Features:").grid(row=1, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.min_features, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=7)

        ttk.Label(input_frame, text="Max Features:").grid(row=2, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.max_features, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=7)

        ttk.Label(input_frame, text="Max Combinations:").grid(row=3, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.max_combinations, width=10).grid(row=3, column=1, sticky="w", padx=5, pady=7)

        ttk.Label(input_frame, text="Features Directory:").grid(row=4, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.output_dir, width=50).grid(row=4, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_output_dir).grid(row=4, column=2, padx=5, pady=7)

        ttk.Button(frame, text="Find Optimized Features", command=self.start_optimization).pack(pady=10)
        self.optimization_progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.optimization_progress.pack(pady=10, fill="x", padx=10)
        self.optimization_status = tk.Text(frame, height=15, width=80)
        self.optimization_status.pack(pady=10, padx=10)
        self.optimization_status.config(state="disabled")

    def create_clustering_tab(self, frame):
        input_frame = ttk.LabelFrame(frame, text="Clustering Parameters", padding=10)
        input_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(input_frame, text="Features Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="Number of Clusters:").grid(row=1, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.n_clusters, width=10, state='readonly').grid(row=1, column=1, sticky="w", padx=5, pady=7)

        ttk.Button(frame, text="Run Clustering Analysis", command=self.start_clustering).pack(pady=10)
        self.clustering_progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.clustering_progress.pack(pady=10, fill="x", padx=10)
        self.clustering_status = tk.Text(frame, height=15, width=80)
        self.clustering_status.pack(pady=10, padx=10)
        self.clustering_status.config(state="disabled")

    def create_anomaly_tab(self, frame):
        input_frame = ttk.LabelFrame(frame, text="Anomaly Detection Parameters", padding=10)
        input_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(input_frame, text="Features Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="SVM Nu (0-1):").grid(row=1, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.svm_nu, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=7)

        ttk.Label(input_frame, text="SVM Gamma (>0):").grid(row=2, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.svm_gamma, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=7)

        ttk.Button(frame, text="Run Anomaly Detection", command=self.start_anomaly_detection).pack(pady=10)
        self.anomaly_progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.anomaly_progress.pack(pady=10, fill="x", padx=10)
        self.anomaly_status = tk.Text(frame, height=15, width=80)
        self.anomaly_status.pack(pady=10, padx=10)
        self.anomaly_status.config(state="disabled")

    def create_inference_tab(self, frame):
        input_frame = ttk.LabelFrame(frame, text="Inference Parameters", padding=10)
        input_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(input_frame, text="Input Video:").grid(row=0, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.inference_video, width=50).grid(row=0, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_inference_video).grid(row=0, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="U2NETLite Model:").grid(row=1, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.inference_model, width=50).grid(row=1, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_inference_model).grid(row=1, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="SVM Model:").grid(row=2, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.inference_svm, width=50).grid(row=2, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_inference_svm).grid(row=2, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="Output Video:").grid(row=3, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.inference_output_video, width=50).grid(row=3, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_inference_output_video).grid(row=3, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="JSON Results:").grid(row=4, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.inference_json, width=50).grid(row=4, column=1, padx=5, pady=7)
        ttk.Button(input_frame, text="Browse", command=self.browse_inference_json).grid(row=4, column=2, padx=5, pady=7)

        ttk.Label(input_frame, text="Number of Classes:").grid(row=5, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.num_classes, width=10, state='readonly').grid(row=5, column=1, sticky="w", padx=5, pady=7)

        ttk.Label(input_frame, text="Class Names:").grid(row=6, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.class_names, width=50, state='readonly').grid(row=6, column=1, padx=5, pady=7)

        ttk.Label(input_frame, text="Number of Clusters:").grid(row=7, column=0, sticky="w", padx=5, pady=7)
        ttk.Entry(input_frame, textvariable=self.n_clusters, width=10, state='readonly').grid(row=7, column=1, sticky="w", padx=5, pady=7)

        ttk.Button(frame, text="Run Inference", command=self.start_inference).pack(pady=10)
        self.inference_progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.inference_progress.pack(pady=10, fill="x", padx=10)
        self.inference_status = tk.Text(frame, height=15, width=80)
        self.inference_status.pack(pady=10, padx=10)
        self.inference_status.config(state="disabled")

    def browse_video_dir(self):
        directory = filedialog.askdirectory(title="Select Video Directory")
        if directory:
            self.video_dir.set(directory)

    def browse_model(self):
        file = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.pth")])
        if file:
            self.model_path.set(file)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Features Directory")
        if directory:
            self.output_dir.set(directory)

    def browse_inference_video(self):
        file = filedialog.askopenfilename(title="Select Input Video", filetypes=[("Video Files", "*.mp4 *.avi")])
        if file:
            self.inference_video.set(file)

    def browse_inference_model(self):
        file = filedialog.askopenfilename(title="Select U2NETLite Model", filetypes=[("Model Files", "*.pth")])
        if file:
            self.inference_model.set(file)

    def browse_inference_svm(self):
        file = filedialog.askopenfilename(title="Select SVM Model", filetypes=[("Numpy Files", "*.npy")])
        if file:
            self.inference_svm.set(file)

    def browse_inference_output_video(self):
        file = filedialog.asksaveasfilename(title="Select Output Video", filetypes=[("Video Files", "*.mp4")], defaultextension=".mp4")
        if file:
            self.inference_output_video.set(file)

    def browse_inference_json(self):
        file = filedialog.asksaveasfilename(title="Select JSON Results", filetypes=[("JSON Files", "*.json")], defaultextension=".json")
        if file:
            self.inference_json.set(file)

    def update_status(self, message, tab='extraction'):
        status_text = {
            'extraction': self.extraction_status,
            'optimization': self.optimization_status,
            'clustering': self.clustering_status,
            'anomaly': self.anomaly_status,
            'inference': self.inference_status
        }.get(tab)
        status_text.config(state="normal")
        status_text.insert(tk.END, message + "\n")
        status_text.see(tk.END)
        status_text.config(state="disabled")
        self.root.update()

    def start_extraction(self):
        if self.is_processing:
            messagebox.showwarning("Warning", "A process is already running!")
            return
        if not os.path.exists(self.video_dir.get()):
            messagebox.showerror("Error", "Video directory does not exist!")
            return
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Model file does not exist!")
            return
        if not os.path.exists(self.output_dir.get()):
            messagebox.showerror("Error", "Output directory does not exist!")
            return
        try:
            num_classes = int(self.num_classes.get())
            if num_classes < 2:
                raise ValueError("Number of classes must be at least 2")
            class_names = [name.strip() for name in self.class_names.get().split(',')]
            if len(class_names) != num_classes:
                raise ValueError(f"Number of class names ({len(class_names)}) must match num_classes ({num_classes})")
            # Validate colors
            colors = self.config.get('testing', {}).get('colors', [[0, 0, 0]] * num_classes)
            if len(colors) != num_classes:
                raise ValueError(f"Number of colors ({len(colors)}) must match num_classes ({num_classes})")
        except ValueError as e:
            self.update_status(f"Error: {str(e)}", tab='extraction')
            return

        self.config['model']['num_classes'] = num_classes
        self.config['feature_extraction']['class_names'] = class_names
        self.config['feature_extraction']['video_dir'] = self.video_dir.get()
        self.config['feature_extraction']['model_path'] = self.model_path.get()
        self.config['feature_extraction']['output_dir'] = self.output_dir.get()
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(self.config, f)

        self.is_processing = True
        self.extraction_progress['value'] = 0
        threading.Thread(target=self.run_extraction, daemon=True).start()

    def run_extraction(self):
        video_files = [f for f in os.listdir(self.video_dir.get()) if f.endswith(('.mp4', '.avi'))]
        if not video_files:
            self.update_status(f"Error: No videos found in {self.video_dir.get()}", tab='extraction')
            self.is_processing = False
            self.extraction_progress['value'] = 0
            return

        total_videos = len(video_files)
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(self.video_dir.get(), video_file)
            features = process_video(self, video_path, self.model_path.get(), self.output_dir.get(), self.device,
                                    num_classes=int(self.num_classes.get()), class_names=self.class_names.get().split(','))
            if features is not None:
                self.update_status(f"Extracted features from {video_file}", tab='extraction')
            else:
                self.update_status(f"Failed to extract features from {video_file}", tab='extraction')
            self.extraction_progress['value'] = ((i + 1) / total_videos) * 100
            self.root.update()

        self.is_processing = False
        messagebox.showinfo("Done", "Feature extraction completed!")

    def start_optimization(self):
        if self.is_processing:
            messagebox.showwarning("Warning", "A process is already running!")
            return
        if not os.path.exists(self.output_dir.get()):
            messagebox.showerror("Error", "Features directory does not exist!")
            return
        try:
            n_clusters = int(self.n_clusters.get())
            min_features = int(self.min_features.get())
            max_features = int(self.max_features.get())
            max_combinations = int(self.max_combinations.get())
            if n_clusters < 2:
                raise ValueError("Number of clusters must be at least 2")
            if min_features < 1:
                raise ValueError("Min features must be at least 1")
            if max_features < min_features:
                raise ValueError("Max features must be greater than or equal to min features")
            if max_combinations < 1:
                raise ValueError("Max combinations must be at least 1")
        except ValueError as e:
            self.update_status(f"Error: {str(e)}", tab='optimization')
            return

        self.config['feature_optimization']['n_clusters'] = n_clusters
        self.config['feature_optimization']['min_features'] = min_features
        self.config['feature_optimization']['max_features'] = max_features
        self.config['feature_optimization']['max_combinations'] = max_combinations
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(self.config, f)

        self.is_processing = True
        self.optimization_progress['value'] = 0
        threading.Thread(target=self.run_optimization, daemon=True).start()

    def run_optimization(self):
        feature_files = [f for f in os.listdir(self.output_dir.get()) if f.endswith('_features.npy')]
        if not feature_files:
            self.update_status(f"Error: No feature files found in {self.output_dir.get()}", tab='optimization')
            self.is_processing = False
            self.optimization_progress['value'] = 0
            return

        all_features = []
        for feature_file in feature_files:
            feature_path = os.path.join(self.output_dir.get(), feature_file)
            features = np.load(feature_path)
            all_features.append(features)
            self.update_status(f"Loaded features from {feature_file}, shape: {features.shape}", tab='optimization')

        if all_features:
            combined_features = np.concatenate(all_features, axis=0)
            self.update_status(f"\nCombined features: {combined_features.shape}", tab='optimization')

            n_clusters = int(self.n_clusters.get())
            min_features = int(self.min_features.get())
            max_features = int(self.max_features.get())
            max_combinations = int(self.max_combinations.get())

            best_result = evaluate_feature_combinations(
                self, combined_features, n_clusters=n_clusters, min_features=min_features,
                max_features=max_features, max_combinations=max_combinations, tab='optimization'
            )

            self.update_status("\nSummary of results:", tab='optimization')
            self.update_status(f"Best combination: {', '.join(best_result['features'])}", tab='optimization')
            self.update_status(f"Silhouette Score: {best_result['silhouette_score']:.4f}", tab='optimization')
        else:
            self.update_status(f"Error: No valid feature files loaded", tab='optimization')

        self.is_processing = False
        messagebox.showinfo("Done", "Feature optimization completed!")

    def start_clustering(self):
        if self.is_processing:
            messagebox.showwarning("Warning", "A process is already running!")
            return
        if not os.path.exists(self.output_dir.get()):
            messagebox.showerror("Error", "Features directory does not exist!")
            return
        combination_results_path = os.path.join(self.output_dir.get(), "combination_results.npy")
        if not os.path.exists(combination_results_path):
            messagebox.showerror("Error", f"Combination results not found: {combination_results_path}")
            return

        try:
            n_clusters = int(self.n_clusters.get())
            if n_clusters < 2:
                raise ValueError("Number of clusters must be at least 2")
        except ValueError as e:
            self.update_status(f"Error: {str(e)}", tab='clustering')
            return

        self.config['clustering_analysis']['output_dir'] = self.output_dir.get()
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(self.config, f)

        self.is_processing = True
        self.clustering_progress['value'] = 0
        threading.Thread(target=self.run_clustering_analysis, daemon=True).start()

    def run_clustering_analysis(self):
        try:
            self.update_status("Running clustering analysis...", tab='clustering')
            optimal_indices, optimal_names = load_optimal_combination(feature_results_path=os.path.join(self.output_dir.get(), "combination_results.npy"))
            num_classes = len(self.config['feature_extraction']['class_names']) - 1
            features_list, video_names = load_features(feature_dir=self.output_dir.get(), num_classes=num_classes)
            if not features_list:
                self.update_status(f"Error: No valid feature files found in {self.output_dir.get()}", tab='clustering')
                self.is_processing = False
                self.clustering_progress['value'] = 0
                return

            results, labels, sil_score = cluster_optimal_features(features_list, optimal_indices, n_clusters=int(self.n_clusters.get()))
            save_clustering_results(results, video_names, output_path=os.path.join(self.output_dir.get(), "clustering_results.npy"))
            self.update_status(f"Clustering completed. Silhouette Score: {sil_score:.4f}", tab='clustering')
            self.update_status(f"Results saved to {os.path.join(self.output_dir.get(), 'clustering_results.npy')}", tab='clustering')
            self.clustering_progress['value'] = 100
        except Exception as e:
            self.update_status(f"Error during clustering: {str(e)}", tab='clustering')
        self.is_processing = False
        messagebox.showinfo("Done", "Clustering analysis completed!")

    def start_anomaly_detection(self):
        if self.is_processing:
            messagebox.showwarning("Warning", "A process is already running!")
            return
        results_path = os.path.join(self.output_dir.get(), "clustering_results.npy")
        if not os.path.exists(results_path):
            messagebox.showerror("Error", f"Clustering results not found: {results_path}")
            return

        try:
            nu = float(self.svm_nu.get())
            gamma = float(self.svm_gamma.get())
            if not (0 < nu <= 1):
                raise ValueError("SVM Nu must be between 0 and 1")
            if gamma <= 0:
                raise ValueError("SVM Gamma must be greater than 0")
        except ValueError as e:
            self.update_status(f"Error: {str(e)}", tab='anomaly')
            return

        self.config['anomaly_detection']['nu'] = nu
        self.config['anomaly_detection']['gamma'] = gamma
        self.config['anomaly_detection']['output_dir'] = self.output_dir.get()
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(self.config, f)

        self.is_processing = True
        self.anomaly_progress['value'] = 0
        threading.Thread(target=self.run_anomaly_detection, daemon=True).start()

    def run_anomaly_detection(self):
        try:
            self.update_status("\nRunning Anomaly Detection...", tab='anomaly')
            results_path = os.path.join(self.output_dir.get(), "clustering_results.npy")
            features, video_names = load_clustering_results(results_path)

            self.update_status("Training One-Class SVM...", tab='anomaly')
            svm_model, svm_scores, svm_threshold = train_one_class_svm(features, nu=float(self.svm_nu.get()), gamma=float(self.svm_gamma.get()))

            summary = summarize_scores(video_names, svm_scores)
            self.update_status(summary, tab='anomaly')
            self.update_status(f"\nDetection Threshold:\n - SVM Threshold: {svm_threshold:.4f}", tab='anomaly')

            save_models(svm_model, svm_threshold, output_path=os.path.join(self.output_dir.get(), "svm_model.npy"))
            self.update_status(f"Saved model to {os.path.join(self.output_dir.get(), 'svm_model.npy')}", tab='anomaly')
            self.anomaly_progress['value'] = 100
        except Exception as e:
            self.update_status(f"Error: {str(e)}", tab='anomaly')
        self.is_processing = False
        messagebox.showinfo("Done", "Anomaly detection completed!")

    def start_inference(self):
        if self.is_processing:
            messagebox.showwarning("Warning", "A process is already running!")
            return
        if not os.path.exists(self.inference_video.get()):
            messagebox.showerror("Error", "Input video does not exist!")
            return
        if not os.path.exists(self.inference_model.get()):
            messagebox.showerror("Error", "U2NETLite model does not exist!")
            return
        if not os.path.exists(self.inference_svm.get()):
            messagebox.showerror("Error", "SVM model does not exist!")
            return
        if not os.path.exists(os.path.join(self.output_dir.get(), "combination_results.npy")):
            messagebox.showerror("Error", "Combination results not found in features directory!")
            return

        try:
            num_classes = int(self.num_classes.get())
            n_clusters = int(self.n_clusters.get())
            if num_classes < 2:
                raise ValueError("Number of classes must be at least 2")
            if n_clusters < 2:
                raise ValueError("Number of clusters must be at least 2")
            class_names = [name.strip() for name in self.class_names.get().split(',')]
            if len(class_names) != num_classes:
                raise ValueError(f"Number of class names ({len(class_names)}) must match num_classes ({num_classes})")
        except ValueError as e:
            self.update_status(f"Error: {str(e)}", tab='inference')
            return

        self.config['inference']['input_video_path'] = self.inference_video.get()
        self.config['inference']['model_path'] = self.inference_model.get()
        self.config['inference']['svm_model_path'] = self.inference_svm.get()
        self.config['inference']['output_video_path'] = self.inference_output_video.get()
        self.config['inference']['result_json_path'] = self.inference_json.get()
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(self.config, f)

        self.is_processing = True
        self.inference_progress['value'] = 0
        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        try:
            self.update_status("Running inference...", tab='inference')
            result = inference_video(
                self,
                video_path=self.inference_video.get(),
                model_path=self.inference_model.get(),
                svm_model_path=self.inference_svm.get(),
                output_video_path=self.inference_output_video.get(),
                result_json_path=self.inference_json.get(),
                device=self.device
            )
            if result is None:
                self.update_status("Inference failed: No results generated", tab='inference')
            self.inference_progress['value'] = 100
        except Exception as e:
            self.update_status(f"Error during inference: {str(e)}", tab='inference')
        self.is_processing = False
        messagebox.showinfo("Done", "Inference completed!")

if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureExtractionApp(root)
    root.mainloop()