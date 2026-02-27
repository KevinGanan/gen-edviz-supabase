"""
Utilidades para cálculo de métricas de imágenes
Para GEN-EDViz Monitor
"""

import numpy as np
from PIL import Image, ImageStat, ImageFilter
from pathlib import Path


def wcag_like_contrast_ratio(img: Image.Image) -> float:
    """
    Calcula una aproximación del ratio de contraste WCAG 2.0.
    
    Convierte la imagen a escala de grises y calcula el ratio entre
    el píxel más claro y el más oscuro usando la fórmula WCAG.
    
    Args:
        img: Imagen PIL
    
    Returns:
        float: Ratio de contraste aproximado (1.0 a 21.0)
            - < 3.0: Contraste bajo (🔴)
            - 3.0-4.5: Contraste medio (🟡)
            - >= 4.5: Contraste alto (🟢)
    """
    try:
        # Convertir a escala de grises
        gray = img.convert('L')
        
        # Obtener valores extremos (0-255)
        extrema = gray.getextrema()
        min_val = extrema[0]
        max_val = extrema[1]
        
        # Convertir a luminancia relativa (0.0 - 1.0)
        # Fórmula sRGB simplificada
        def to_relative_luminance(val):
            val = val / 255.0
            if val <= 0.03928:
                return val / 12.92
            else:
                return ((val + 0.055) / 1.055) ** 2.4
        
        L1 = to_relative_luminance(max_val)  # Más claro
        L2 = to_relative_luminance(min_val)  # Más oscuro
        
        # Calcular ratio WCAG: (L1 + 0.05) / (L2 + 0.05)
        # Donde L1 > L2
        if L1 < L2:
            L1, L2 = L2, L1
        
        ratio = (L1 + 0.05) / (L2 + 0.05)
        
        # Limitar a rango WCAG válido (1:1 a 21:1)
        return min(21.0, max(1.0, ratio))
        
    except Exception as e:
        print(f"Error calculando contraste WCAG: {e}")
        return 1.0


def edge_density(img: Image.Image) -> float:
    """
    Calcula la densidad de bordes en la imagen.
    
    Usa el filtro FIND_EDGES de PIL para detectar bordes y
    calcula qué proporción de píxeles son bordes.
    
    Args:
        img: Imagen PIL
    
    Returns:
        float: Densidad de bordes (0.0 a 1.0)
            - 0.0-0.1: Imagen muy lisa, pocos detalles
            - 0.1-0.3: Densidad normal
            - > 0.3: Imagen muy detallada o ruidosa
    """
    try:
        # Convertir a escala de grises
        gray = img.convert('L')
        
        # Aplicar filtro de detección de bordes
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Convertir a array numpy
        edges_array = np.array(edges)
        
        # Contar píxeles de borde (umbral > 30 de 255)
        edge_pixels = np.sum(edges_array > 30)
        total_pixels = edges_array.size
        
        # Calcular densidad
        density = edge_pixels / total_pixels
        
        # Limitar a rango 0-1
        return min(1.0, max(0.0, float(density)))
        
    except Exception as e:
        print(f"Error calculando densidad de bordes: {e}")
        return 0.0


def _safe_path(path_str):
    """
    Convierte una ruta de string a Path de forma segura.
    
    Args:
        path_str: Ruta como string
    
    Returns:
        Path o None si no es válido
    """
    if not path_str or str(path_str).strip() == '':
        return None
    
    try:
        return Path(path_str)
    except Exception:
        return None


# ========== FUNCIONES AUXILIARES PARA TESTING ==========

def test_metrics_on_image(image_path: str):
    """
    Función de prueba para verificar que las métricas funcionen.
    
    Uso:
        from services.utils import test_metrics_on_image
        test_metrics_on_image("data/catalogo_imagenes/img_12345.png")
    """
    try:
        img = Image.open(image_path)
        
        wcag = wcag_like_contrast_ratio(img)
        edge = edge_density(img)
        
        print(f"✅ Métricas calculadas correctamente:")
        print(f"   WCAG Contrast Ratio: {wcag:.2f}")
        print(f"   Edge Density: {edge:.3f}")
        
        # Interpretación
        if wcag >= 4.5:
            wcag_status = "🟢 Alto contraste (WCAG AA)"
        elif wcag >= 3.0:
            wcag_status = "🟡 Contraste medio"
        else:
            wcag_status = "🔴 Contraste bajo"
        
        if edge >= 0.3:
            edge_status = "Imagen muy detallada"
        elif edge >= 0.1:
            edge_status = "Densidad normal"
        else:
            edge_status = "Imagen lisa"
        
        print(f"\n   Interpretación:")
        print(f"   - Contraste: {wcag_status}")
        print(f"   - Bordes: {edge_status}")
        
        return wcag, edge
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


if __name__ == "__main__":
    # Test básico
    print("Módulo utils.py cargado correctamente ✓")
    print("\nFunciones disponibles:")
    print("  - wcag_like_contrast_ratio(img)")
    print("  - edge_density(img)")
    print("  - test_metrics_on_image(path)")