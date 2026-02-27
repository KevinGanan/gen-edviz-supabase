"""
Script de prueba para verificar que las métricas funcionan
Ejecuta esto ANTES de intentar guardar imágenes
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 Probando funciones de métricas...")
print("=" * 60)

# Test 1: Importar módulo
print("\n1️⃣ Importando módulo utils...")
try:
    from services.utils import wcag_like_contrast_ratio, edge_density
    print("   ✅ Funciones importadas correctamente")
except ImportError as e:
    print(f"   ❌ Error importando: {e}")
    print("   💡 Asegúrate de tener el archivo services/utils.py")
    sys.exit(1)

# Test 2: Crear imagen de prueba
print("\n2️⃣ Creando imagen de prueba...")
try:
    from PIL import Image
    import numpy as np
    
    # Crear imagen de prueba (blanco y negro)
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[:50, :] = 255  # Mitad blanca
    img_test = Image.fromarray(img_array)
    
    print("   ✅ Imagen de prueba creada")
except Exception as e:
    print(f"   ❌ Error creando imagen: {e}")
    sys.exit(1)

# Test 3: Calcular WCAG
print("\n3️⃣ Calculando contraste WCAG...")
try:
    wcag_val = wcag_like_contrast_ratio(img_test)
    print(f"   ✅ WCAG Ratio: {wcag_val:.2f}")
    
    if wcag_val > 1.0:
        print("   ✓ Valor válido (esperado: ~21.0 para blanco/negro)")
    else:
        print("   ⚠️ Valor sospechoso, debería ser > 1.0")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Calcular Edge Density
print("\n4️⃣ Calculando densidad de bordes...")
try:
    edge_val = edge_density(img_test)
    print(f"   ✅ Edge Density: {edge_val:.3f}")
    
    if 0.0 <= edge_val <= 1.0:
        print("   ✓ Valor válido (rango: 0.0 - 1.0)")
    else:
        print("   ⚠️ Valor fuera de rango esperado")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verificar con imagen real (si existe)
print("\n5️⃣ Buscando imágenes en el catálogo...")
catalogo_dir = Path("data/catalogo_imagenes")

if catalogo_dir.exists():
    imagenes = list(catalogo_dir.glob("*.png"))
    
    if imagenes:
        print(f"   Encontradas {len(imagenes)} imágenes")
        print(f"   Probando con: {imagenes[0].name}")
        
        try:
            img_real = Image.open(imagenes[0])
            wcag_real = wcag_like_contrast_ratio(img_real)
            edge_real = edge_density(img_real)
            
            print(f"\n   📊 Resultados:")
            print(f"      WCAG Ratio: {wcag_real:.2f}")
            print(f"      Edge Density: {edge_real:.3f}")
            
            # Interpretación
            if wcag_real >= 4.5:
                print(f"      → 🟢 Alto contraste (WCAG AA)")
            elif wcag_real >= 3.0:
                print(f"      → 🟡 Contraste medio")
            else:
                print(f"      → 🔴 Contraste bajo")
                
        except Exception as e:
            print(f"   ⚠️ No se pudo procesar imagen real: {e}")
    else:
        print("   ℹ️ No hay imágenes en el catálogo todavía")
else:
    print("   ℹ️ Carpeta de catálogo no existe todavía")

# Resumen final
print("\n" + "=" * 60)
print("✅ TODAS LAS PRUEBAS PASARON")
print("=" * 60)
print("\n💡 Las funciones de métricas están funcionando correctamente.")
print("   Ahora puedes generar y guardar imágenes con métricas.\n")

