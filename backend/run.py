# Importa Uvicorn para ejecutar la aplicación FastAPI.
import uvicorn

# Ejecuta el servidor local solo cuando este archivo se lanza directamente.
if __name__ == '__main__':
    # Inicia FastAPI en el puerto 8000 con recarga automática para desarrollo.
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)