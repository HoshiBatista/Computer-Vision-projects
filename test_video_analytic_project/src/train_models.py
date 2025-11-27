import os
import yaml
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

class ModelTrainer:
    def __init__(self, config_path='config/models_config.yaml', dataset_config='config/dataset.yaml'):
        self.results_dir = 'results/benchmarks'
        os.makedirs(self.results_dir, exist_ok=True)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = dataset_config
        self.benchmark_results = []

    def train_model(self, model_config):
        print(f"\n{'='*60}")
        print(f"Обучение модели: {model_config['name']}")
        print(f"{'='*60}")
        
        model = YOLO(model_config['pretrained'])
        
        start_time = time.time()
        results = model.train(
            data=self.dataset_config,
            epochs=model_config['epochs'],
            imgsz=model_config['imgsz'],
            batch=model_config['batch'],
            name=f"train_{model_config['name']}",
            device=model_config['device'],
            workers=4,
            patience=5  
        )
        train_time = time.time() - start_time
        
        metrics = model.val()
        
        test_img = 'data/coco128/images/train/000000000315.jpg'
        warmup_runs, test_runs = 10, 50
        
        for _ in range(warmup_runs):
            _ = model(test_img, verbose=False)
        
        start_time = time.time()
        for _ in range(test_runs):
            _ = model(test_img, verbose=False)
        inference_time = (time.time() - start_time) / test_runs
        
        metrics_dict = {
            'model': model_config['name'],
            'train_time': train_time,
            'val_map50': metrics.results_dict['metrics/mAP50(B)'],
            'val_map50_95': metrics.results_dict['metrics/mAP50-95(B)'],
            'inference_time': inference_time,
            'fps': 1 / inference_time,
            'model_size_mb': os.path.getsize(model_config['pretrained']) / 1024 / 1024,
            'train_dir': results.save_dir
        }
        
        print(f"Обучение завершено за {train_time:.2f} сек")
        print(f"mAP@0.5: {metrics_dict['val_map50']:.4f}")
        print(f"Среднее время инференса: {inference_time*1000:.2f} мс ({metrics_dict['fps']:.1f} FPS)")
        
        export_path = f"models/{model_config['name']}_custom.pt"
        model.save(export_path)
        
        return metrics_dict, model

    def run_training(self):
        print("Начало обучения моделей...")
        
        for model_config in self.config['models']:
            metrics, model = self.train_model(model_config)
            self.benchmark_results.append(metrics)

        pd.DataFrame(self.benchmark_results).to_csv(
            f"{self.results_dir}/benchmark_results.csv", index=False
        )
        self.plot_results()
        
        return self.benchmark_results

    def plot_results(self):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.bar([m['model'] for m in self.benchmark_results], 
                [m['val_map50'] for m in self.benchmark_results])
        plt.title('mAP@0.5')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.bar([m['model'] for m in self.benchmark_results], 
                [m['fps'] for m in self.benchmark_results], color='orange')
        plt.title('Скорость (FPS)')
        plt.ylim(0, max(m['fps'] for m in self.benchmark_results) * 1.2)
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.bar([m['model'] for m in self.benchmark_results], 
                [m['model_size_mb'] for m in self.benchmark_results], color='green')
        plt.title('Размер модели (MB)')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(2, 2, 4)
        
        for m in self.benchmark_results:
            plt.scatter(m['fps'], m['val_map50'], s=100, label=m['model'])
            
        plt.xlabel('FPS')
        plt.ylabel('mAP@0.5')
        plt.title('Точность vs Скорость')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/benchmark_comparison.png", dpi=300)
        plt.close()
        
        print(f"Графики сохранены в {self.results_dir}/benchmark_comparison.png")

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_training()
    print("Обучение всех моделей завершено!")