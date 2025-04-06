
# api/responses.py
"""
Formatos de respuesta estandarizados para la API
"""

def success_response(data=None, message=None):
    """
    Formato estándar para respuestas exitosas
    
    Args:
        data: Datos a incluir en la respuesta
        message: Mensaje informativo opcional
        
    Returns:
        Diccionario JSON formateado
    """
    response = {'status': 'ok'}
    
    if data is not None:
        response['data'] = data
    
    if message is not None:
        response['message'] = message
    
    return response

def error_response(message, code=400):
    """
    Formato estándar para respuestas de error
    
    Args:
        message: Mensaje de error
        code: Código HTTP de error
        
    Returns:
        Diccionario JSON formateado, código HTTP
    """
    return {'status': 'error', 'message': message}, code

def validation_error(errors):
    """
    Formato estándar para errores de validación
    
    Args:
        errors: Diccionario de errores por campo
        
    Returns:
        Diccionario JSON formateado, código HTTP 422
    """
    return {
        'status': 'error',
        'message': 'Error de validación',
        'errors': errors
    }, 422