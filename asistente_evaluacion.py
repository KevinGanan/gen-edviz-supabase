"""
ASISTENTE INTERACTIVO DE EVALUACIÓN
====================================

Script que guía paso a paso en la preparación y ejecución de la evaluación.
"""

from pathlib import Path
import json

def verificar_instalacion():
    """Verifica que las dependencias estén instaladas"""
    print("\n🔍 Verificando instalación de dependencias...")
    
    missing = []
    
    try:
        import torch
        print("  ✓ PyTorch")
    except:
        missing.append("torch")
    
    try:
        import lpips
        print("  ✓ LPIPS")
    except:
        missing.append("lpips")
    
    try:
        from cleanfid import fid
        print("  ✓ clean-fid")
    except:
        missing.append("clean-fid")
    
    try:
        from pytorch_msssim import ms_ssim
        print("  ✓ MS-SSIM")
    except:
        missing.append("pytorch-msssim")
    
    try:
        import open_clip
        print("  ✓ OpenCLIP")
    except:
        missing.append("open_clip_torch")
    
    if missing:
        print(f"\n❌ Faltan dependencias: {', '.join(missing)}")
        print("\n💡 Ejecuta:")
        print("   python install_metricas.py")
        return False
    
    print("\n✅ Todas las dependencias instaladas")
    return True


def verificar_estructura():
    """Verifica que exista la estructura de carpetas"""
    print("\n🔍 Verificando estructura de carpetas...")
    
    carpetas_necesarias = [
        "data/evaluacion/escenario1/referencias",
        "data/evaluacion/escenario1/generadas",
        "data/evaluacion/escenario2/generadas",
        "data/evaluacion/escenario2/prompts",
        "data/evaluacion/escenario3/generadas",
        "data/evaluacion/dataset_real",
        "data/evaluacion/resultados",
        "data/evaluacion/reportes"
    ]
    
    missing = []
    for carpeta in carpetas_necesarias:
        if Path(carpeta).exists():
            print(f"  ✓ {carpeta}")
        else:
            missing.append(carpeta)
            print(f"  ✗ {carpeta}")
    
    if missing:
        print(f"\n❌ Faltan {len(missing)} carpetas")
        print("\n💡 Ejecuta:")
        print("   python crear_estructura_evaluacion.py")
        return False
    
    print("\n✅ Estructura completa")
    return True


def verificar_escenario_1():
    """Verifica preparación del escenario 1"""
    print("\n📋 Verificando Escenario 1: Imagen de Referencia")
    
    ref_dir = Path("data/evaluacion/escenario1/referencias")
    gen_dir = Path("data/evaluacion/escenario1/generadas")
    
    refs = list(ref_dir.glob("*.png"))
    gens = list(gen_dir.glob("*.png"))
    
    print(f"  Referencias: {len(refs)}")
    print(f"  Generadas: {len(gens)}")
    
    if len(refs) == 0:
        print("\n❌ No hay imágenes de referencia")
        print("\n💡 Pasos:")
        print("   1. Crea 5-10 diagramas de referencia")
        print("   2. Guárdalos en: data/evaluacion/escenario1/referencias/")
        print("   3. Nombra: ref_01.png, ref_02.png, ...")
        return False
    
    if len(gens) == 0:
        print("\n❌ No hay imágenes generadas")
        print("\n💡 Pasos:")
        print("   1. Genera imágenes correspondientes a cada referencia")
        print("   2. Guárdalas en: data/evaluacion/escenario1/generadas/")
        print("   3. Nombra: gen_01.png, gen_02.png, ...")
        return False
    
    if len(refs) != len(gens):
        print(f"\n⚠️ Cantidad diferente (refs: {len(refs)}, gens: {len(gens)})")
        print("   Idealmente deberían ser iguales")
    
    print(f"\n✅ Listo para ejecutar (mínimo 5 pares)")
    return len(refs) >= 5 and len(gens) >= 5


def verificar_escenario_2():
    """Verifica preparación del escenario 2"""
    print("\n📋 Verificando Escenario 2: Coherencia Prompt-Imagen")
    
    prompts_file = Path("data/evaluacion/escenario2/prompts/prompts.json")
    gen_dir = Path("data/evaluacion/escenario2/generadas")
    
    if not prompts_file.exists():
        print("\n❌ No existe archivo de prompts")
        print("\n💡 Crea: data/evaluacion/escenario2/prompts/prompts.json")
        print('\n   Formato:')
        print('   [')
        print('     {')
        print('       "prompt": "Tu prompt aquí",')
        print('       "images": ["img1.png", "img2.png"]')
        print('     }')
        print('   ]')
        return False
    
    # Cargar y verificar
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  ✓ Archivo de prompts válido")
        print(f"  Prompts: {len(data)}")
        
        # Verificar imágenes
        total_imgs = sum(len(item['images']) for item in data)
        imgs_existentes = len(list(gen_dir.glob("*.png")))
        
        print(f"  Imágenes esperadas: {total_imgs}")
        print(f"  Imágenes encontradas: {imgs_existentes}")
        
        if imgs_existentes < total_imgs:
            print(f"\n⚠️ Faltan {total_imgs - imgs_existentes} imágenes")
            return False
        
        print(f"\n✅ Listo para ejecutar")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en archivo de prompts: {e}")
        return False


def verificar_escenario_3():
    """Verifica preparación del escenario 3"""
    print("\n📋 Verificando Escenario 3: Diversidad")
    
    gen_dir = Path("data/evaluacion/escenario3/generadas")
    imgs = list(gen_dir.glob("*.png"))
    
    print(f"  Imágenes: {len(imgs)}")
    
    if len(imgs) < 10:
        print(f"\n❌ Se necesitan al menos 10 imágenes (tienes {len(imgs)})")
        print("\n💡 Pasos:")
        print("   1. Elige UN prompt educativo")
        print("   2. Genera 20 veces con el MISMO prompt")
        print("   3. Guarda en: data/evaluacion/escenario3/generadas/")
        return False
    
    print(f"\n✅ Listo para ejecutar (ideal: 20 imágenes)")
    return True


def menu_principal():
    """Menú interactivo"""
    print("\n" + "="*60)
    print("🎯 ASISTENTE DE EVALUACIÓN - GEN-EDViz Monitor")
    print("="*60)
    
    print("\n¿Qué deseas hacer?")
    print("\n1. ✓ Verificar instalación de dependencias")
    print("2. ✓ Verificar estructura de carpetas")
    print("3. ✓ Verificar preparación Escenario 1")
    print("4. ✓ Verificar preparación Escenario 2")
    print("5. ✓ Verificar preparación Escenario 3")
    print("6. ✓ Verificar TODO")
    print("7. 🚀 Ejecutar evaluación completa")
    print("8. 📖 Ver guía completa")
    print("9. ❌ Salir")
    
    opcion = input("\nOpción (1-9): ").strip()
    
    if opcion == "1":
        verificar_instalacion()
    elif opcion == "2":
        verificar_estructura()
    elif opcion == "3":
        verificar_escenario_1()
    elif opcion == "4":
        verificar_escenario_2()
    elif opcion == "5":
        verificar_escenario_3()
    elif opcion == "6":
        print("\n" + "🔍"*30)
        ok1 = verificar_instalacion()
        ok2 = verificar_estructura()
        ok3 = verificar_escenario_1()
        ok4 = verificar_escenario_2()
        ok5 = verificar_escenario_3()
        
        print("\n" + "="*60)
        if all([ok1, ok2, ok3, ok4, ok5]):
            print("✅ TODO LISTO PARA EJECUTAR LA EVALUACIÓN")
            print("\n💡 Siguiente paso:")
            print("   python ejecutar_evaluacion.py")
        else:
            print("⚠️ Completa los pasos pendientes antes de ejecutar")
        print("="*60)
    
    elif opcion == "7":
        print("\n🚀 Ejecutando evaluación completa...")
        print("\nEsto puede tomar varios minutos...")
        
        confirmacion = input("\n¿Continuar? (s/n): ").lower()
        if confirmacion == 's':
            from ejecutar_evaluacion import ejecutar_evaluacion_completa
            ejecutar_evaluacion_completa()
        else:
            print("Cancelado")
    
    elif opcion == "8":
        print("\n📖 Guía completa:")
        print("   → Ver archivo: GUIA_EVALUACION_COMPLETA.md")
        print("\nO ejecuta en Python:")
        print("   from pathlib import Path")
        print("   print(Path('GUIA_EVALUACION_COMPLETA.md').read_text())")
    
    elif opcion == "9":
        print("\n👋 ¡Hasta luego!")
        return False
    
    else:
        print("\n❌ Opción inválida")
    
    input("\nPresiona ENTER para continuar...")
    return True


if __name__ == "__main__":
    continuar = True
    while continuar:
        continuar = menu_principal()
        