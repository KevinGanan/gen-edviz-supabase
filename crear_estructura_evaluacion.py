# ========== CREAR ESTRUCTURA DE CARPETAS PARA EVALUACIÓN ==========

"""
Crea la estructura de carpetas necesaria para los 3 escenarios de evaluación
"""

from pathlib import Path

print("📁 Creando estructura de carpetas para evaluación...")

# Estructura de carpetas
folders = [
    # Escenario 1: Imagen de referencia
    "data/evaluacion/escenario1/referencias",
    "data/evaluacion/escenario1/generadas",
    
    # Escenario 2: Coherencia prompt-imagen
    "data/evaluacion/escenario2/generadas",
    "data/evaluacion/escenario2/prompts",
    
    # Escenario 3: Generación múltiple (diversidad)
    "data/evaluacion/escenario3/generadas",
    
    # Resultados y reportes
    "data/evaluacion/resultados",
    "data/evaluacion/reportes",
    
    # Dataset real (para FID/KID)
    "data/evaluacion/dataset_real",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {folder}")

print("\n✅ Estructura creada correctamente")
print("\n📋 Próximos pasos:")
print("  1. Escenario 1: Coloca imágenes de referencia en 'escenario1/referencias/'")
print("  2. Dataset real: Coloca imágenes educativas reales en 'dataset_real/'")
print("  3. El sistema generará automáticamente en las carpetas 'generadas/'")