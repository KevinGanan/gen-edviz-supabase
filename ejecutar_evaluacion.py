"""
EVALUACIÓN COMPLETA DEL SISTEMA GEN-EDViz Monitor
==================================================

Ejecuta los 3 escenarios de evaluación según especificaciones del docente:

ESCENARIO 1: Imagen de referencia (LPIPS, FID, KID, SSIM, PSNR)
ESCENARIO 2: Coherencia prompt-imagen (CLIPScore + satisfacción)
ESCENARIO 3: Generación múltiple (MS-SSIM, LPIPS, IS)

Autor: José / GEN-EDViz Monitor
"""

import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import itertools

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent))

from services.metricas_evaluacion import (
    MetricasReferencia,
    calcular_fid_kid,
    CLIPScorer,
    MetricasDiversidad,
    medir_latencia,
    generar_reporte_metricas
)

# ========== CONFIGURACIÓN ==========

class ConfiguracionEvaluacion:
    """Configuración centralizada de la evaluación"""
    
    # Rutas base
    BASE_DIR = Path("data/evaluacion")
    
    # Escenario 1
    ESC1_REF = BASE_DIR / "escenario1/referencias"
    ESC1_GEN = BASE_DIR / "escenario1/generadas"
    
    # Escenario 2
    ESC2_GEN = BASE_DIR / "escenario2/generadas"
    ESC2_PROMPTS = BASE_DIR / "escenario2/prompts/prompts.json"
    
    # Escenario 3
    ESC3_GEN = BASE_DIR / "escenario3/generadas"
    
    # Dataset real (para FID/KID)
    DATASET_REAL = BASE_DIR / "dataset_real"
    
    # Resultados
    RESULTADOS = BASE_DIR / "resultados"
    REPORTES = BASE_DIR / "reportes"
    
    # Parámetros
    N_GENERACIONES_ESC3 = 20  # Número de generaciones por prompt en escenario 3
    MAX_PAIRS_DIVERSIDAD = 200  # Máximo de pares para métricas de diversidad


# ========== ESCENARIO 1: IMAGEN DE REFERENCIA ==========

def ejecutar_escenario_1():
    """
    Escenario 1: Comparación con imagen de referencia
    
    Métricas:
    - LPIPS (↓ mejor): Similitud perceptual
    - SSIM (↑ mejor): Similitud estructural
    - PSNR (↑ mejor): Relación señal-ruido
    - FID (↓ mejor): Distribución real vs generada
    - KID (↓ mejor): Alternativa a FID
    """
    print("\n" + "="*60)
    print("📋 ESCENARIO 1: Evaluación con Imagen de Referencia")
    print("="*60)
    
    # Verificar que existan carpetas
    if not ConfiguracionEvaluacion.ESC1_REF.exists():
        print("❌ No existe carpeta de referencias")
        return None
    
    if not ConfiguracionEvaluacion.ESC1_GEN.exists():
        print("❌ No existe carpeta de generadas")
        return None
    
    # Listar archivos
    refs = sorted(ConfiguracionEvaluacion.ESC1_REF.glob("*.png"))
    gens = sorted(ConfiguracionEvaluacion.ESC1_GEN.glob("*.png"))
    
    if not refs:
        print("❌ No hay imágenes de referencia")
        print(f"   Coloca imágenes en: {ConfiguracionEvaluacion.ESC1_REF}")
        return None
    
    if not gens:
        print("❌ No hay imágenes generadas")
        print(f"   Genera imágenes con el sistema y guárdalas en: {ConfiguracionEvaluacion.ESC1_GEN}")
        return None
    
    print(f"✓ Referencias: {len(refs)}")
    print(f"✓ Generadas: {len(gens)}")
    
    # Inicializar métricas
    metricas_ref = MetricasReferencia()
    
    # Evaluar pares
    resultados_pares = []
    
    print("\n📊 Evaluando pares ref-gen...")
    
    # Asumimos que ref[i] corresponde a gen[i]
    for i, (ref_path, gen_path) in enumerate(zip(refs, gens)):
        print(f"\n  Par {i+1}/{len(refs)}: {ref_path.name} vs {gen_path.name}")
        
        metricas = metricas_ref.evaluar_par(str(ref_path), str(gen_path))
        
        resultados_pares.append({
            "referencia": ref_path.name,
            "generada": gen_path.name,
            **metricas
        })
        
        print(f"    LPIPS: {metricas['lpips']:.4f}")
        print(f"    SSIM:  {metricas['ssim']:.4f}")
        print(f"    PSNR:  {metricas['psnr']:.2f} dB")
    
    # Promedios
    df_pares = pd.DataFrame(resultados_pares)
    
    promedios = {
        "lpips_mean": df_pares["lpips"].mean(),
        "lpips_std": df_pares["lpips"].std(),
        "ssim_mean": df_pares["ssim"].mean(),
        "ssim_std": df_pares["ssim"].std(),
        "psnr_mean": df_pares["psnr"].mean(),
        "psnr_std": df_pares["psnr"].std()
    }
    
    # FID / KID (si hay dataset real)
    if ConfiguracionEvaluacion.DATASET_REAL.exists():
        n_real = len(list(ConfiguracionEvaluacion.DATASET_REAL.glob("*.png")))
        
        if n_real >= 50:
            print(f"\n📊 Calculando FID/KID con dataset real ({n_real} imágenes)...")
            
            fid_score, kid_score = calcular_fid_kid(
                str(ConfiguracionEvaluacion.DATASET_REAL),
                str(ConfiguracionEvaluacion.ESC1_GEN)
            )
            
            promedios["fid"] = fid_score
            promedios["kid"] = kid_score
            
            print(f"  FID: {fid_score:.4f}")
            print(f"  KID: {kid_score:.4f}")
        else:
            print(f"\n⚠️ Dataset real tiene solo {n_real} imágenes (se necesitan ≥50)")
    else:
        print("\n⚠️ No existe dataset real para FID/KID")
    
    # Guardar resultados
    resultados_path = ConfiguracionEvaluacion.RESULTADOS / "escenario1_resultados.csv"
    df_pares.to_csv(resultados_path, index=False)
    print(f"\n✅ Resultados guardados: {resultados_path}")
    
    return {
        "pares": resultados_pares,
        "promedios": promedios,
        "n_pares": len(refs)
    }


# ========== ESCENARIO 2: COHERENCIA PROMPT-IMAGEN ==========

def ejecutar_escenario_2():
    """
    Escenario 2: Coherencia prompt-imagen
    
    Métricas:
    - CLIPScore (↑ mejor): Alineación texto-imagen
    - Satisfacción usuario (encuesta Likert 1-5)
    """
    print("\n" + "="*60)
    print("📋 ESCENARIO 2: Coherencia Prompt-Imagen")
    print("="*60)
    
    # Verificar archivos
    if not ConfiguracionEvaluacion.ESC2_PROMPTS.exists():
        print("❌ No existe archivo de prompts")
        print(f"   Crea: {ConfiguracionEvaluacion.ESC2_PROMPTS}")
        print("\n   Formato JSON:")
        print('   [{"prompt": "...", "images": ["img1.png", "img2.png"]}]')
        return None
    
    # Cargar prompts
    with open(ConfiguracionEvaluacion.ESC2_PROMPTS, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    print(f"✓ Prompts cargados: {len(prompts_data)}")
    
    # Inicializar CLIPScore
    clip_scorer = CLIPScorer()
    
    # Evaluar cada prompt
    resultados = []
    
    for item in prompts_data:
        prompt = item["prompt"]
        image_files = item["images"]
        
        print(f"\n📝 Prompt: {prompt[:60]}...")
        
        # Rutas completas
        image_paths = [
            str(ConfiguracionEvaluacion.ESC2_GEN / img) 
            for img in image_files
        ]
        
        # Calcular CLIPScores
        clip_results = clip_scorer.evaluar_batch(prompt, image_paths)
        
        resultados.append({
            "prompt": prompt,
            "n_images": len(image_paths),
            "clipscore_mean": clip_results["mean"],
            "clipscore_std": clip_results["std"],
            "clipscore_min": clip_results["min"],
            "clipscore_max": clip_results["max"]
        })
        
        print(f"  CLIPScore: {clip_results['mean']:.4f} ± {clip_results['std']:.4f}")
    
    # Guardar resultados
    df_resultados = pd.DataFrame(resultados)
    resultados_path = ConfiguracionEvaluacion.RESULTADOS / "escenario2_clipscore.csv"
    df_resultados.to_csv(resultados_path, index=False)
    
    print(f"\n✅ Resultados guardados: {resultados_path}")
    
    # Promedios globales
    promedios = {
        "clipscore_global_mean": df_resultados["clipscore_mean"].mean(),
        "clipscore_global_std": df_resultados["clipscore_mean"].std()
    }
    
    print(f"\n📊 CLIPScore global: {promedios['clipscore_global_mean']:.4f}")
    
    return {
        "resultados": resultados,
        "promedios": promedios,
        "n_prompts": len(prompts_data)
    }


# ========== ESCENARIO 3: GENERACIÓN MÚLTIPLE (DIVERSIDAD) ==========

def ejecutar_escenario_3():
    """
    Escenario 3: Generación múltiple - Diversidad
    
    Métricas:
    - MS-SSIM inter-sample (↓ mejor): Menor similitud = mayor diversidad
    - LPIPS inter-sample (↑ mejor): Mayor distancia = mayor diversidad
    - Inception Score (↑ mejor): Diversidad + confianza
    """
    print("\n" + "="*60)
    print("📋 ESCENARIO 3: Diversidad en Generación Múltiple")
    print("="*60)
    
    # Verificar carpeta
    if not ConfiguracionEvaluacion.ESC3_GEN.exists():
        print("❌ No existe carpeta de generadas")
        return None
    
    # Listar imágenes
    images = sorted(ConfiguracionEvaluacion.ESC3_GEN.glob("*.png"))
    
    if len(images) < 10:
        print(f"❌ Se necesitan al menos 10 imágenes (encontradas: {len(images)})")
        print(f"   Genera múltiples imágenes con el mismo prompt")
        return None
    
    print(f"✓ Imágenes encontradas: {len(images)}")
    
    # Inicializar métricas
    metricas_div = MetricasDiversidad()
    
    # Convertir a strings
    image_paths = [str(p) for p in images]
    
    # MS-SSIM inter-sample
    print("\n📊 Calculando MS-SSIM inter-sample...")
    ms_ssim_score = metricas_div.ms_ssim_intersample(
        image_paths, 
        max_pairs=ConfiguracionEvaluacion.MAX_PAIRS_DIVERSIDAD
    )
    print(f"  MS-SSIM: {ms_ssim_score:.4f} (↓ = más diversidad)")
    
    # LPIPS inter-sample
    print("\n📊 Calculando LPIPS inter-sample...")
    lpips_score = metricas_div.lpips_intersample(
        image_paths,
        max_pairs=ConfiguracionEvaluacion.MAX_PAIRS_DIVERSIDAD
    )
    print(f"  LPIPS: {lpips_score:.4f} (↑ = más diversidad)")
    
    # Inception Score
    print("\n📊 Calculando Inception Score...")
    # is_mean, is_std = metricas_div.inception_score(str(ConfiguracionEvaluacion.ESC3_GEN))
    # print(f"  Inception Score: {is_mean:.2f} ± {is_std:.2f}")
    is_mean, is_std = 0.0, 0.0  # Deshabilitado temporalmente
    print("  ⚠️ Inception Score deshabilitado (error técnico)")
    
    resultados = {
        "ms_ssim_intersample": ms_ssim_score,
        "lpips_intersample": lpips_score,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "n_images": len(images),
        "n_pairs_evaluated": min(len(list(itertools.combinations(images, 2))), 
                                  ConfiguracionEvaluacion.MAX_PAIRS_DIVERSIDAD)
    }
    
    # Guardar
    resultados_path = ConfiguracionEvaluacion.RESULTADOS / "escenario3_diversidad.json"
    with open(resultados_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2)
    
    print(f"\n✅ Resultados guardados: {resultados_path}")
    
    return resultados


# ========== EJECUCIÓN PRINCIPAL ==========

def ejecutar_evaluacion_completa():
    """Ejecuta los 3 escenarios y genera reporte consolidado"""
    
    print("\n" + "🚀"*30)
    print("EVALUACIÓN COMPLETA - SISTEMA GEN-EDViz Monitor")
    print("🚀"*30)
    
    # Crear carpetas de resultados
    ConfiguracionEvaluacion.RESULTADOS.mkdir(parents=True, exist_ok=True)
    ConfiguracionEvaluacion.REPORTES.mkdir(parents=True, exist_ok=True)
    
    resultados_totales = {}
    
    # Escenario 1
    try:
        res1 = ejecutar_escenario_1()
        if res1:
            resultados_totales["Escenario 1: Imagen de Referencia"] = res1["promedios"]
    except Exception as e:
        print(f"\n❌ Error en Escenario 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Escenario 2
    try:
        res2 = ejecutar_escenario_2()
        if res2:
            resultados_totales["Escenario 2: Coherencia Prompt-Imagen"] = res2["promedios"]
    except Exception as e:
        print(f"\n❌ Error en Escenario 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Escenario 3
    try:
        res3 = ejecutar_escenario_3()
        if res3:
            resultados_totales["Escenario 3: Diversidad"] = res3
    except Exception as e:
        print(f"\n❌ Error en Escenario 3: {e}")
        import traceback
        traceback.print_exc()
    
    # Generar reporte
    if resultados_totales:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reporte_path = ConfiguracionEvaluacion.REPORTES / f"reporte_completo_{timestamp}.md"
        
        generar_reporte_metricas(resultados_totales, str(reporte_path))
        
        print("\n" + "="*60)
        print("✅ EVALUACIÓN COMPLETADA")
        print("="*60)
        print(f"\n📄 Reporte: {reporte_path}")
    
    return resultados_totales


if __name__ == "__main__":
    ejecutar_evaluacion_completa()