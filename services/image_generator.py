"""
Servicio de generación de imágenes con GPT-Image 1.5
Para GEN-EDViz Monitor
"""

import os
import re
from pathlib import Path

def generar_imagen(
    prompt: str,
    size: str = "1024x1024",
    model: str = "gpt-image-1.5"
) -> dict:
    """
    Genera una imagen usando GPT-Image 1.5.
    
    Args:
        prompt: Descripción de la imagen a generar
        size: Tamaño de la imagen (por defecto 1024x1024)
        model: Modelo a usar (por defecto gpt-image-1.5)
    
    Returns:
        dict con:
            - success: bool
            - image_base64: imagen en base64 (si success)
            - revised_prompt: prompt usado para generación
            - error: mensaje de error (si falla)
            - model: modelo usado
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return {
            "success": False,
            "error": "No se encontró OPENAI_API_KEY en las variables de entorno.",
            "model": model
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # ✅ CORRECTO para GPT-Image 1.5
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            quality="medium",  # ← CAMBIADO: "medium" para GPT-Image 1.5
            size=size
        )

        return {
            "success": True,
            "image_base64": response.data[0].b64_json,
            "revised_prompt": prompt,
            "model": model
        }

    except Exception as e:
        error_str = str(e)
        
        # Detectar errores comunes y dar mensajes amigables
        if "billing_hard_limit_reached" in error_str:
            error_mensaje = "⚠️ Se acabaron los créditos de OpenAI. Recarga tu cuenta para continuar."
        elif "insufficient_quota" in error_str:
            error_mensaje = "⚠️ Cuota insuficiente en OpenAI. Verifica tu saldo."
        elif "rate_limit" in error_str:
            error_mensaje = "⏳ Límite de tasa alcanzado. Espera unos segundos e intenta de nuevo."
        elif "invalid_api_key" in error_str:
            error_mensaje = "🔑 API Key inválida. Verifica tu configuración."
        elif "model_not_found" in error_str:
            error_mensaje = "❌ Modelo no encontrado. Verifica que GPT-Image 1.5 esté disponible."
        else:
            error_mensaje = f"❌ Error: {error_str}"
        
        return {
            "success": False,
            "error": error_mensaje,
            "model": model
        }


def mejorar_prompt_educativo(prompt_usuario: str, tema: str, concepto: str) -> str:
    """
    Mejora un prompt para generar imágenes educativas usando GPT-4o.
    Optimizado para GPT-Image 1.5.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return prompt_usuario
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        system_msg = """Eres un experto en crear prompts para GPT-Image 1.5 que generen 
IMÁGENES EDUCATIVAS TÉCNICAS de estructuras de datos. Tu tarea es transformar 
el prompt del usuario en uno que el modelo pueda entender y generar correctamente.

REGLAS ESTRICTAS:
1. EL PROMPT FINAL DEBE ESTAR EN INGLÉS
2. Debe generar un DIAGRAMA EDUCATIVO CLARO
3. Especifica FORMAS GEOMÉTRICAS SIMPLES (círculos, rectángulos, flechas)
4. Especifica COLORES CONTRASTANTES para accesibilidad
5. Incluye TÉRMINOS CLAVE: "educational diagram", "computer science", "technical illustration"
6. Si el concepto es complejo, sugiere representación "step-by-step" o "before/after"
7. MÁXIMO 350 caracteres
8. EVITA términos abstractos: usa "nodes as circles", "edges as lines", "arrows showing direction"

Ejemplos de transformación:
- "Árbol AVL" → "AVL tree diagram with nodes as circles, balance factors shown"
- "Rotación" → "Right rotation in binary tree showing before and after states"
- "BFS" → "Breadth-First Search algorithm visualization with queue representation"

Responde SOLO con el prompt mejorado, sin explicaciones."""

        user_msg = f"""CONVIERTE este prompt para que GPT-Image 1.5 genere una IMAGEN EDUCATIVA TÉCNICA:

TEMA: {tema}
CONCEPTO: {concepto}
PROMPT ORIGINAL: {prompt_usuario}

PROMPT MEJORADO (en inglés, técnico, específico):"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error mejorando prompt: {e}")
        return prompt_usuario


# ========== PLANTILLAS OPTIMIZADAS PARA GPT-IMAGE 1.5 ==========

PLANTILLAS_PROMPTS = {
    "arboles": {
        "arbol_binario": "Binary tree educational diagram with nodes as blue circles connected by black lines, showing parent-child relationships, white background, computer science illustration",
        "arbol_busqueda": "Binary Search Tree diagram with values inside circles, left subtree smaller values, right subtree larger values, clear hierarchy, educational technical drawing",
        "avl_tree": "AVL tree diagram showing balanced binary search tree with height balance, nodes as circles with balance factors (+1, 0, -1), computer science educational illustration",
        "rotacion_avl": "Step-by-step AVL tree right rotation diagram: left side shows unbalanced tree, right side shows balanced tree after rotation, nodes connected by lines with arrows showing movement, educational computer science diagram",
        "rotacion_simple": "Binary tree rotation diagram showing single rotation operation, nodes as numbered circles, arrows showing node movement, before and after comparison, technical illustration",
        "recorrido_inorden": "Inorder traversal visualization in binary tree, numbered steps 1-7 showing left-root-right order, arrows indicating path, educational computer science diagram",
    },
    "grafos": {
        "grafo_dirigido": "Directed graph diagram with vertices as colored circles labeled A-F, edges as arrows showing direction, computer science educational illustration on white background",
        "grafo_no_dirigido": "Undirected graph with 6 vertices as circles connected by straight lines, no arrows, network visualization, educational diagram",
        "bfs": "Breadth-First Search algorithm step-by-step diagram: graph with visited nodes changing color (red→yellow→green), queue visualization as rectangle, numbered steps, educational computer science illustration",
        "dfs": "Depth-First Search visualization with stack representation, showing deep exploration path, visited nodes highlighted, step numbers, computer science educational diagram",
        "dijkstra": "Dijkstra's algorithm shortest path diagram: weighted graph with distance labels on vertices, shortest path highlighted in green, step-by-step visualization, technical educational illustration",
        "algoritmo_kruskal": "Kruskal's algorithm for Minimum Spanning Tree: graph with edges sorted by weight, step-by-step edge selection, final MST highlighted, educational computer science diagram",
    },
    "pilas": {
        "push": "Stack data structure LIFO diagram: vertical stack of 5 colored rectangles, green upward arrow showing PUSH operation adding element to top, educational computer science illustration",
        "pop": "Stack LIFO operation diagram: vertical stack of rectangles, red downward arrow showing POP operation removing top element, educational data structures visualization",
        "pila_general": "Stack (Last-In-First-Out) data structure: elements stacked vertically like plates, top and bottom indicators, simple educational diagram for computer science",
    },
    "colas": {
        "enqueue": "Queue data structure FIFO diagram: horizontal line of 5 colored squares, blue arrow at back showing ENQUEUE operation, educational computer science illustration",
        "dequeue": "Queue FIFO operation diagram: horizontal line of squares, orange arrow at front showing DEQUEUE operation removing first element, educational data structures visualization",
        "cola_circular": "Circular queue diagram: circular arrangement of slots, front and rear pointers, elements moving in circle, computer science educational illustration",
    },
    "listas": {
        "lista_simple": "Singly linked list diagram: 4 nodes as rectangles with data and next pointer, arrows connecting nodes, NULL at end, educational computer science illustration",
        "lista_doble": "Doubly linked list diagram: nodes with prev and next pointers, bidirectional arrows, educational data structure visualization with clear labels",
        "lista_circular": "Circular linked list diagram: nodes connected in circle, last node points to first, cycle visualization, educational computer science diagram",
    },
    "ordenamiento": {
        "bubble_sort": "Bubble sort algorithm step-by-step visualization: array of 6 elements, comparison and swap arrows, educational programming diagram with iteration numbers",
        "quick_sort": "Quick sort algorithm diagram: pivot selection, partitioning process, recursive division, educational computer science illustration with clear steps",
        "merge_sort": "Merge sort divide and conquer diagram: splitting array into subarrays, merging sorted subarrays, tree-like visualization, educational algorithm illustration",
        "heap_sort": "Heap sort algorithm with binary heap: max-heap tree structure, heapify process, array representation below, educational computer science diagram",
    },
    "tablas_hash": {
        "hash_table": "Hash table diagram: array of buckets with linked lists for collision chaining, hash function shown, educational data structures illustration",
        "colision_hash": "Hash collision resolution diagram: two keys hashing to same bucket, chaining shown with linked list, open addressing alternative, educational computer science visualization",
    }
}


def obtener_plantilla(tema: str, concepto: str) -> str:
    """Obtiene una plantilla de prompt predefinida."""
    tema_lower = tema.lower().strip()
    concepto_lower = concepto.lower().strip().replace(" ", "_")
    
    if tema_lower in PLANTILLAS_PROMPTS:
        return PLANTILLAS_PROMPTS[tema_lower].get(concepto_lower, "")
    return ""


def listar_temas_disponibles() -> list:
    """Retorna lista de temas con plantillas disponibles."""
    return list(PLANTILLAS_PROMPTS.keys())


def listar_conceptos_por_tema(tema: str) -> list:
    """Retorna lista de conceptos disponibles para un tema."""
    tema_lower = tema.lower().strip()
    if tema_lower in PLANTILLAS_PROMPTS:
        return list(PLANTILLAS_PROMPTS[tema_lower].keys())
    return []


def corregir_prompt_problematico(prompt: str, tema: str = "", concepto: str = "") -> str:
    """
    Corrige prompts que podrían generar imágenes no educativas.
    Optimizado para GPT-Image 1.5.
    """
    
    # Lista de palabras problemáticas y sus reemplazos
    reemplazos = {
        # Violencia/conflicto
        "violent": "step-by-step",
        "violently": "systematically", 
        "aggressive": "systematic",
        "aggressively": "methodically",
        "attack": "operation",
        "clash": "reorganization",
        "fight": "process",
        "battle": "procedure",
        "war": "algorithm",
        # Términos abstractos/problemáticos
        "explosion": "removal",
        "explode": "remove",
        "destroy": "delete",
        "kill": "eliminate",
        "death": "end",
        "dead": "terminated",
        "blood": "",
        # Términos no técnicos
        "magic": "algorithmic",
        "magical": "systematic",
        "awesome": "efficient",
        "cool": "optimal",
        # Mejora términos educativos
        "show": "diagram showing",
        "draw": "illustrate",
        "picture": "educational diagram",
        "image": "technical illustration"
    }
    
    # Aplicar reemplazos
    prompt_corregido = prompt
    for palabra, reemplazo in reemplazos.items():
        if palabra in prompt_corregido.lower():
            prompt_corregido = re.sub(
                re.escape(palabra), 
                reemplazo, 
                prompt_corregido, 
                flags=re.IGNORECASE
            )
    
    # Asegurar que sea un prompt educativo
    if not any(term in prompt_corregido.lower() for term in [
        "diagram", "illustration", "visualization", "educational", "technical"
    ]):
        prompt_corregido += ", educational computer science diagram"
    
    # Asegurar que esté en inglés
    if not all(ord(c) < 128 for c in prompt_corregido):
        prompt_corregido += " (in English)"
    
    # Limitar longitud
    if len(prompt_corregido) > 300:
        prompt_corregido = prompt_corregido[:297] + "..."
    
    return prompt_corregido


def sugerir_prompt_educativo(tema: str, concepto: str) -> list:
    """
    Sugiere prompts educativos bien estructurados para un tema y concepto.
    Optimizado para GPT-Image 1.5.
    """
    
    sugerencias = {
        "arboles": {
            "rotacion_avl": [
                "AVL tree right rotation diagram: unbalanced tree on left, balanced tree on right, nodes as circles with values, arrows showing rotation",
                "Step-by-step AVL tree rotation: 1) initial unbalanced state, 2) rotation operation, 3) final balanced state, educational computer science diagram",
                "Binary tree rotation visualization showing node repositioning to maintain balance, technical illustration with clear node connections"
            ],
            "arbol_busqueda": [
                "Binary Search Tree diagram with nodes containing numbers in sorted order, left children smaller, right children larger, hierarchical structure",
                "BST insertion visualization: new node finding correct position, comparisons shown with arrows, educational data structures diagram",
                "Search operation in binary search tree: path from root to target node highlighted, comparison at each step, computer science educational illustration"
            ]
        },
        "grafos": {
            "bfs": [
                "Breadth-First Search algorithm diagram: graph with layers, visited nodes changing color, queue data structure shown separately",
                "BFS step-by-step: starting node red, first level yellow, second level green, queue visualization with enqueue/dequeue operations",
                "Graph traversal comparison: BFS on left (layer by layer), DFS on right (depth first), side-by-side educational diagram"
            ],
            "dijkstra": [
                "Dijkstra's shortest path algorithm: weighted graph with distance labels, visited set, shortest path highlighted in contrasting color",
                "Step-by-step Dijkstra: initial distances infinity, each iteration updates nearest node, final path shown, educational algorithm visualization",
                "Shortest path finding diagram: graph with weights on edges, Dijkstra algorithm progression, distance table shown alongside"
            ]
        }
    }
    
    tema_lower = tema.lower().strip()
    concepto_lower = concepto.lower().strip().replace(" ", "_")
    
    if tema_lower in sugerencias and concepto_lower in sugerencias[tema_lower]:
        return sugerencias[tema_lower][concepto_lower]
    
    # Sugerencias genéricas
    return [
        f"Educational diagram of {concepto} in {tema}, computer science illustration with clear labels",
        f"Step-by-step visualization of {concepto} process in {tema}, technical diagram for students",
        f"{tema} data structure showing {concepto} operation, simple and clear educational illustration"
    ]