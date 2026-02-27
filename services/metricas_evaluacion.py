"""
Módulo de Métricas para Evaluación de Imágenes Generadas por IA
================================================================

Implementa las siguientes métricas según estándares académicos:

ESCENARIO 1 (con referencia):
- LPIPS: Similitud perceptual (↓ mejor)
- FID: Fréchet Inception Distance (↓ mejor)
- KID: Kernel Inception Distance (↓ mejor)
- SSIM: Similitud estructural
- PSNR: Peak Signal-to-Noise Ratio

ESCENARIO 2 (coherencia prompt-imagen):
- CLIPScore: Alineación texto-imagen (↑ mejor)

ESCENARIO 3 (diversidad múltiple generación):
- MS-SSIM inter-sample: Diversidad (↓ mejor)
- LPIPS inter-sample: Diversidad perceptual (↑ mejor)
- Inception Score: Diversidad + confianza (↑ mejor)

Autor: GEN-EDViz Monitor
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm
import itertools

# ========== CONFIGURACIÓN ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Dispositivo: {DEVICE}")


# ========== ESCENARIO 1: MÉTRICAS CON REFERENCIA ==========

class MetricasReferencia:
    """Métricas para comparar imagen generada vs referencia"""
    
    def __init__(self):
        """Inicializa modelos necesarios"""
        import lpips
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        # LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
        self.ssim = ssim
        self.psnr = psnr
        
        print("✅ MetricasReferencia inicializadas")
    
    def lpips_distance(self, ref_path: str, gen_path: str) -> float:
        """
        LPIPS: Learned Perceptual Image Patch Similarity
        
        Rango: 0 (idénticas) a ~1 (muy diferentes)
        Meta: ↓ mejor (más parecidas)
        """
        import torchvision.transforms as T
        
        to_tensor = T.Compose([T.Resize((512,512)), T.ToTensor()])
        
        def load(path):
            img = Image.open(path).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(DEVICE)
            return x * 2 - 1  # [-1, 1]
        
        ref = load(ref_path)
        gen = load(gen_path)
        
        with torch.no_grad():
            dist = self.lpips_fn(ref, gen)
        
        return float(dist.item())
    
    def ssim_score(self, ref_path: str, gen_path: str) -> float:
        """
        SSIM: Structural Similarity Index
        
        Rango: -1 a 1
        Meta: ↑ mejor (1 = idénticas)
        """
        ref = np.array(Image.open(ref_path).convert("L").resize((512, 512)))
        gen = np.array(Image.open(gen_path).convert("L").resize((512, 512)))
        
        return self.ssim(ref, gen, data_range=255)
    
    def psnr_score(self, ref_path: str, gen_path: str) -> float:
        """
        PSNR: Peak Signal-to-Noise Ratio
        
        Rango: 0 a inf (típico 20-50)
        Meta: ↑ mejor
        """
        ref = np.array(Image.open(ref_path).convert("L").resize((512, 512)))
        gen = np.array(Image.open(gen_path).convert("L").resize((512, 512)))
        
        return self.psnr(ref, gen, data_range=255)
    
    def evaluar_par(self, ref_path: str, gen_path: str) -> Dict[str, float]:
        """Evalúa todas las métricas para un par ref-gen"""
        return {
            "lpips": self.lpips_distance(ref_path, gen_path),
            "ssim": self.ssim_score(ref_path, gen_path),
            "psnr": self.psnr_score(ref_path, gen_path)
        }


# ========== FID / KID (DISTRIBUCIÓN) ==========

def calcular_fid_kid(real_dir: str, gen_dir: str) -> Tuple[float, float]:
    """
    FID y KID: Comparan distribuciones de imágenes reales vs generadas
    
    Requiere: Carpetas con múltiples imágenes (>50 idealmente)
    
    Returns:
        (fid_score, kid_score)
        Ambos: ↓ mejor
    """
    from cleanfid import fid
    
    print(f"📊 Calculando FID/KID...")
    print(f"   Real: {real_dir}")
    print(f"   Gen:  {gen_dir}")
    
    fid_score = fid.compute_fid(real_dir, gen_dir, device=DEVICE)
    kid_score = fid.compute_kid(real_dir, gen_dir, device=DEVICE)
    
    return fid_score, kid_score


# ========== ESCENARIO 2: CLIPSCORE ==========

class CLIPScorer:
    """
    Evaluador de coherencia texto-imagen usando CLIP.
    Mide qué tan bien la imagen representa el prompt.
    """
    
    def __init__(self):
        """Inicializa el modelo CLIP."""
        import open_clip
        
        # Cargar modelo CLIP preentrenado
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        self.model.to(DEVICE)
        self.model.eval()
        
        print("✅ CLIPScorer inicializado")
    
    def score(self, prompt: str, image_path: str) -> float:
        """
        Calcula CLIPScore entre prompt e imagen.
        
        Args:
            prompt: Texto del prompt
            image_path: Ruta a la imagen
        
        Returns:
            float: CLIPScore (mayor = más coherente)
        """
        try:
            from PIL import Image
            import torch
            
            # Cargar y procesar imagen
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Procesar texto
            text_input = self.tokenizer([prompt]).to(DEVICE)
            
            # Calcular embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                # Normalizar
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calcular similitud (producto punto)
                clip_score = (image_features @ text_features.T).item()
            
            return clip_score
            
        except Exception as e:
            print(f"⚠️ Error calculando CLIPScore para {image_path}: {e}")
            return 0.0
    
    def evaluar_batch(self, prompt: str, image_paths: list) -> dict:
        """
        Evalúa múltiples imágenes con el mismo prompt.
        
        Args:
            prompt: Texto del prompt
            image_paths: Lista de rutas a imágenes
        
        Returns:
            dict con scores individuales y estadísticas
        """
        from tqdm import tqdm
        import numpy as np
        
        scores = [self.score(prompt, path) for path in tqdm(image_paths, desc="CLIPScore")]
        
        return {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores))
        }


# ========== ESCENARIO 3: DIVERSIDAD ==========

class MetricasDiversidad:
    """Métricas de diversidad entre múltiples generaciones"""
    
    def __init__(self):
        """Inicializa modelos necesarios"""
        import lpips
        from pytorch_msssim import ms_ssim as ms_ssim_fn
        
        self.lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()
        self.ms_ssim_fn = ms_ssim_fn
        
        print("✅ MetricasDiversidad inicializadas")
    
    def ms_ssim_intersample(self, image_paths: List[str], max_pairs: int = 200) -> float:
        """
        MS-SSIM inter-sample: Similitud entre pares de imágenes generadas
        
        Rango: 0 a 1
        Meta: ↓ mejor (menor similitud = mayor diversidad)
        """
        import torchvision.transforms as T
        
        to_tensor = T.Compose([T.Resize((512,512)), T.ToTensor()])
        
        def load(path):
            img = Image.open(path).convert("RGB")
            return to_tensor(img).unsqueeze(0).to(DEVICE)
        
        # Crear pares
        pairs = list(itertools.combinations(image_paths, 2))
        if len(pairs) > max_pairs:
            pairs = np.random.choice(len(pairs), max_pairs, replace=False)
            pairs = [list(itertools.combinations(image_paths, 2))[i] for i in pairs]
        
        scores = []
        for path_a, path_b in tqdm(pairs, desc="MS-SSIM"):
            img_a = load(path_a)
            img_b = load(path_b)
            
            with torch.no_grad():
                score = self.ms_ssim_fn(img_a, img_b, data_range=1.0, size_average=True)
            
            scores.append(float(score.item()))
        
        return np.mean(scores)
    
    def lpips_intersample(self, image_paths: List[str], max_pairs: int = 200) -> float:
        """
        LPIPS inter-sample: Distancia perceptual entre pares
        
        Rango: 0 a ~1
        Meta: ↑ mejor (mayor distancia = mayor diversidad)
        """
        import torchvision.transforms as T
        
        to_tensor = T.Compose([T.Resize((512,512)), T.ToTensor()])
        
        def load(path):
            img = Image.open(path).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(DEVICE)
            return x * 2 - 1
        
        pairs = list(itertools.combinations(image_paths, 2))
        if len(pairs) > max_pairs:
            pairs = np.random.choice(len(pairs), max_pairs, replace=False)
            pairs = [list(itertools.combinations(image_paths, 2))[i] for i in pairs]
        
        distances = []
        for path_a, path_b in tqdm(pairs, desc="LPIPS inter-sample"):
            img_a = load(path_a)
            img_b = load(path_b)
            
            with torch.no_grad():
                dist = self.lpips_fn(img_a, img_b)
            
            distances.append(float(dist.item()))
        
        return np.mean(distances)
    
    def inception_score(self, gen_dir: str) -> Tuple[float, float]:
        """
        Inception Score: Diversidad + confianza del clasificador
        
        Returns: (mean, std)
        Meta: ↑ mejor
        """
        from torch_fidelity import calculate_metrics
        
        print("📊 Calculando Inception Score...")
        
        metrics = calculate_metrics(
            input1=gen_dir,
            isc=True,
            fid=False,
            kid=False,
            cuda=torch.cuda.is_available()
        )
        
        return float(metrics["inception_score_mean"]), float(metrics["inception_score_std"])


# ========== MÉTRICAS DE EFICIENCIA ==========

def medir_latencia(generate_fn, prompt: str, n_samples: int = 30) -> Dict[str, float]:
    """
    Mide tiempo de generación
    
    Args:
        generate_fn: Función que genera imagen dado un prompt
        prompt: Prompt de prueba
        n_samples: Número de mediciones
    
    Returns:
        Dict con media, p50, p95
    """
    import time
    
    times = []
    
    print(f"⏱️ Midiendo latencia ({n_samples} muestras)...")
    
    for i in tqdm(range(n_samples)):
        t0 = time.perf_counter()
        generate_fn(prompt)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    times_sorted = sorted(times)
    
    return {
        "mean": np.mean(times),
        "median": times_sorted[len(times)//2],
        "p95": times_sorted[int(len(times)*0.95)-1],
        "min": np.min(times),
        "max": np.max(times),
        "std": np.std(times)
    }


# ========== UTILIDADES ==========

def generar_reporte_metricas(resultados: Dict, output_path: str):
    """Genera reporte en formato markdown"""
    
    md = "# 📊 Reporte de Evaluación de Imágenes IA\n\n"
    md += f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"
    
    for escenario, metricas in resultados.items():
        md += f"## {escenario}\n\n"
        
        for metrica, valor in metricas.items():
            if isinstance(valor, dict):
                md += f"### {metrica}\n"
                for k, v in valor.items():
                    md += f"- **{k}**: {v:.4f}\n"
                md += "\n"
            else:
                md += f"- **{metrica}**: {valor:.4f}\n"
        
        md += "\n---\n\n"
    
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"✅ Reporte generado: {output_path}")


if __name__ == "__main__":
    print("✅ Módulo de métricas cargado correctamente")
    print("\n📋 Funciones disponibles:")
    print("  - MetricasReferencia (LPIPS, SSIM, PSNR)")
    print("  - calcular_fid_kid")
    print("  - CLIPScorer")
    print("  - MetricasDiversidad (MS-SSIM, LPIPS, IS)")
    print("  - medir_latencia")