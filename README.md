# prerequisite 
ollama pull gemma3:4b


# creazione del Docker per l'interscambio
docker compose stop eva-bridge
docker compose down eva-bridge
docker compose build --no-cache eva-bridge

# log eva bridge
docker logs -f eva-bridge

# ollama url 
http://host.docker.internal:11434

# webhook
http://localhost:5678/webhook/upload

# qdrant - Verifica dei Dati (Dashboard)
 http://qdrant:6333

# postgres

# Run on gpu
docker compose --profile gpu-nvidia pull
docker compose create 
docker compose --profile gpu-nvidia up

# qdrant - Verifica dei Dati (Dashboard)
# Puoi verificare se i vettori sono stati creati correttamente accedendo all'interfaccia grafica:

    URL: http://localhost:6333/dashboard
    Collection: knowledge-base

# 
curl http://10.3.1.101:5000/json?query=ciao

# Funzionamento in modalità host

