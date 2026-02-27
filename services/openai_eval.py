# services/openai_eval.py
import os
import base64
import json
import re
from datetime import datetime
from pathlib import Path


def evaluar_imagen_prompt(tema: str, concepto: str, prompt: str, imagen_path: str = "") -> dict:
    """
    Evalúa la coherencia entre una imagen generada y su prompt usando GPT-4o.
    
    Args:
        tema: Tema de la imagen (ej: "grafos", "árboles")
        concepto: Concepto específico (ej: "BFS", "inorden")
        prompt: El prompt usado para generar la imagen
        imagen_path: Ruta a la imagen (opcional, para análisis visual)
    
    Returns:
        dict con coherencia, fidelidad, claridad, errores, recomendaciones, fecha
    """
    
    # 1) Verificar que OpenAI está instalado
    try:
        from openai import OpenAI
    except ImportError:
        return _resultado_error("OpenAI no está instalado. Ejecuta: pip install openai")
    
    # 2) Verificar API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return _resultado_error("OPENAI_API_KEY no está configurada en el entorno")
    
    # 3) Crear cliente y definir modelo
    client = OpenAI(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gpt-4o")
    
    # 4) Prompt del sistema (instrucciones para el modelo)
    system_msg = """Eres un experto en pedagogía de Estructuras de Datos y evaluación de materiales didácticos visuales generados por IA.

Tu tarea es evaluar la calidad pedagógica de imágenes generadas por IA para enseñar conceptos de Estructuras de Datos.

DEBES responder ÚNICAMENTE con un JSON válido con esta estructura exacta (sin texto adicional, sin explicaciones, sin markdown):
{
    "coherencia": <número entero del 1 al 5>,
    "fidelidad": <número entero del 1 al 5>,
    "claridad": <número entero del 1 al 5>,
    "errores": "<texto describiendo errores conceptuales o visuales detectados>",
    "recomendaciones": "<texto con sugerencias concretas de mejora pedagógica>"
}

Criterios de evaluación (escala 1-5):
- coherencia: ¿El prompt describe adecuadamente lo que debería mostrar la imagen para representar el concepto de ED?
- fidelidad: ¿La descripción del prompt es fiel y precisa respecto al concepto de Estructuras de Datos que representa?
- claridad: ¿El prompt es claro, específico y sin ambigüedades para generar una imagen didáctica útil?

Escala de puntuación:
1 = Muy deficiente (errores graves, confuso, inútil pedagógicamente)
2 = Deficiente (errores importantes, poco claro)
3 = Aceptable (correcto pero mejorable)
4 = Bueno (claro y útil, detalles menores a mejorar)
5 = Excelente (óptimo para uso pedagógico)

IMPORTANTE: Responde SOLO con el JSON, sin ningún texto antes o después."""

    # 5) Mensaje del usuario con los datos a evaluar
    user_msg = f"""Evalúa este prompt usado para generar una imagen didáctica de Estructuras de Datos:

TEMA: {tema}
CONCEPTO: {concepto}
PROMPT USADO: {prompt}

Analiza si el prompt es adecuado para generar una imagen que enseñe correctamente el concepto.
Responde SOLO con el JSON estructurado."""

    # 6) Preparar mensajes para la API
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    # 7) Si hay imagen y el modelo soporta visión, incluirla
    if imagen_path and _imagen_disponible(imagen_path) and "4o" in model:
        image_data = _encode_image(imagen_path)
        if image_data:
            # Modificar el mensaje del usuario para incluir la imagen
            messages[1] = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_msg + "\n\nAdemás, analiza la imagen adjunta y verifica si corresponde visualmente al prompt y al concepto."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                            "detail": "low"  # "low" para ahorrar tokens, "high" para más detalle
                        }
                    }
                ]
            }

    # 8) Llamar a la API de OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,  # Bajo para respuestas más consistentes
            max_tokens=600
        )
        
        # Extraer contenido de la respuesta
        content = response.choices[0].message.content or ""
        
        # Parsear el JSON de la respuesta
        resultado = _parse_json_response(content)
        resultado["fecha"] = datetime.now().isoformat()
        
        return resultado
        
    except Exception as e:
        return _resultado_error(f"Error en API OpenAI: {str(e)}")


def chat_evaluacion_ia(
    mensaje_usuario: str,
    historial: list,
    tema: str,
    concepto: str,
    prompt_imagen: str,
    imagen_path: str = ""
) -> str:
    """
    Función de chat interactivo para evaluar imágenes pedagógicas.
    Mantiene contexto de la conversación y permite preguntas de seguimiento.
    
    Args:
        mensaje_usuario: La pregunta o mensaje del usuario
        historial: Lista de mensajes previos [{"role": "user/assistant", "content": "..."}]
        tema: Tema de la imagen
        concepto: Concepto que representa la imagen
        prompt_imagen: El prompt usado para generar la imagen
        imagen_path: Ruta a la imagen (opcional)
    
    Returns:
        str: Respuesta de la IA
    """
    
    # 1) Verificar OpenAI
    try:
        from openai import OpenAI
    except ImportError:
        return "❌ Error: OpenAI no está instalado."
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "❌ Error: OPENAI_API_KEY no está configurada."
    
    client = OpenAI(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gpt-4o")
    
    # 2) Prompt del sistema para el chat
    system_msg = f"""Eres un experto en pedagogía de Estructuras de Datos y evaluación de materiales didácticos visuales.

Estás analizando una imagen generada por IA para enseñar el siguiente concepto:

📚 TEMA: {tema}
🎯 CONCEPTO: {concepto}
📝 PROMPT USADO PARA GENERAR LA IMAGEN:
"{prompt_imagen}"

Tu rol es:
1. Evaluar si la imagen y el prompt son coherentes entre sí
2. Identificar errores conceptuales en la representación visual
3. Sugerir mejoras pedagógicas
4. Responder preguntas del docente sobre la calidad del material

Responde siempre en español, de forma clara y concisa.
Cuando detectes errores, sé específico sobre qué está mal y cómo corregirlo.
Cuando sugieras mejoras al prompt, da ejemplos concretos.

Recuerda: Tu objetivo es ayudar al docente a crear mejores materiales didácticos para enseñar Estructuras de Datos."""

    # 3) Construir mensajes
    messages = [{"role": "system", "content": system_msg}]
    
    # 4) Si es el primer mensaje y hay imagen, incluirla
    if not historial and imagen_path and _imagen_disponible(imagen_path) and "4o" in model:
        image_data = _encode_image(imagen_path)
        if image_data:
            # Primer mensaje con imagen
            primer_mensaje = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Esta es la imagen que estamos evaluando. {mensaje_usuario}"
                    }
                ]
            }
            messages.append(primer_mensaje)
        else:
            messages.append({"role": "user", "content": mensaje_usuario})
    else:
        # Agregar historial previo
        for msg in historial:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Agregar mensaje actual
        messages.append({"role": "user", "content": mensaje_usuario})
    
    # 5) Llamar a la API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )
        
        return response.choices[0].message.content or "No se obtuvo respuesta."
        
    except Exception as e:
        error_str = str(e)
        
        # Retornar SOLO el mensaje de error como string
        if "insufficient_quota" in error_str or "exceeded your current quota" in error_str:
            return "⚠️ Se acabaron los créditos de OpenAI. Recarga en: https://platform.openai.com/billing"
        elif "billing_hard_limit_reached" in error_str:
            return "⚠️ Límite de facturación alcanzado. Recarga en: https://platform.openai.com/billing"
        elif "rate_limit" in error_str or "429" in error_str:
            return "⏳ Demasiadas solicitudes. Espera 1-2 minutos e intenta de nuevo."
        elif "invalid_api_key" in error_str or "401" in error_str:
            return "🔑 API Key inválida. Verifica tu OPENAI_API_KEY en .streamlit/secrets.toml"
        elif "model_not_found" in error_str or "404" in error_str:
            return "❌ Modelo GPT-4o no encontrado. Verifica tu acceso."
        else:
            return f"❌ Error: {error_str[:150]}"

def _imagen_disponible(image_path: str) -> bool:
    """Verifica si una imagen está disponible (local o URL remota)."""
    if not image_path:
        return False
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return True  # Asumimos que la URL de Supabase es válida
    return Path(image_path).exists()


def generar_analisis_inicial(tema: str, concepto: str, prompt_imagen: str, imagen_path: str = "") -> str:
    """
    Genera un análisis inicial automático cuando se selecciona una imagen.
    
    Returns:
        str: Análisis inicial de la IA
    """
    
    mensaje_inicial = """Por favor, realiza un análisis inicial de esta imagen y su prompt:

1. ¿La imagen representa correctamente el concepto?
2. ¿El prompt es adecuado para generar esta imagen?
3. ¿Detectas algún error conceptual?
4. ¿Qué puntuación (1-5) le darías en coherencia, fidelidad y claridad?
5. ¿Tienes alguna recomendación de mejora?

Sé conciso pero completo en tu análisis."""

    return chat_evaluacion_ia(
        mensaje_usuario=mensaje_inicial,
        historial=[],
        tema=tema,
        concepto=concepto,
        prompt_imagen=prompt_imagen,
        imagen_path=imagen_path
    )


def _encode_image(image_path: str) -> str:
    """Codifica una imagen en base64 - soporta rutas locales Y URLs de Supabase."""
    try:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            import urllib.request
            with urllib.request.urlopen(image_path) as response:
                data = response.read()
                print(f"✅ Imagen descargada: {len(data)} bytes")
                return base64.b64encode(data).decode("utf-8")
        else:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error descargando imagen: {e}")
        return ""


def _parse_json_response(content: str) -> dict:
    """Parsea la respuesta JSON del modelo de forma robusta."""
    content = content.strip()
    
    # Intentar extraer JSON del contenido (por si viene con texto extra)
    match = re.search(r'\{.*\}', content, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            
            # Validar y extraer campos con valores por defecto
            return {
                "coherencia": _safe_int(data.get("coherencia"), 3),
                "fidelidad": _safe_int(data.get("fidelidad"), 3),
                "claridad": _safe_int(data.get("claridad"), 3),
                "errores": str(data.get("errores", "No se detectaron errores conceptuales.")),
                "recomendaciones": str(data.get("recomendaciones", "Sin recomendaciones específicas."))
            }
        except json.JSONDecodeError:
            pass
    
    # Si no se pudo parsear, devolver error
    return _resultado_error("No se pudo interpretar la respuesta del modelo")


def _safe_int(value, default: int = 3) -> int:
    """Convierte un valor a entero de forma segura, limitado a 1-5."""
    try:
        n = int(value)
        return max(1, min(5, n))  # Limitar entre 1 y 5
    except (TypeError, ValueError):
        return default


def _resultado_error(mensaje: str) -> dict:
    """Devuelve un resultado de error estructurado."""
    return {
        "coherencia": 0,
        "fidelidad": 0,
        "claridad": 0,
        "errores": f"ERROR: {mensaje}",
        "recomendaciones": "Verifica la configuración de la API y vuelve a intentar.",
        "fecha": datetime.now().isoformat()
    }