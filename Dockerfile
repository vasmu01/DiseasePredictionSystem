FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output is sent straight to the terminal
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Move to the Django project directory
WORKDIR /app/project_code

# Collect static files
RUN python manage.py collectstatic --noinput

EXPOSE 8000

# Start the application
CMD gunicorn myproject.wsgi:application --bind 0.0.0.0:$PORT