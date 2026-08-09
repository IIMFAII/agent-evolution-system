# Image officielle Apify pour Python.
FROM apify/actor-python:3.12

# Dépendances d'abord : cette couche est mise en cache tant que le fichier
# ne change pas, ce qui rend les rebuilds de code quasi instantanés.
COPY .actor/requirements.txt ./
RUN echo "Python: $(python --version)" \
 && pip install --no-cache-dir -r requirements.txt \
 && pip freeze

# Puis le code. L'audit n'utilise que src/ ; le reste du dépôt est ignoré
# via .dockerignore pour garder l'image petite.
COPY . ./

CMD ["python", ".actor/main.py"]
