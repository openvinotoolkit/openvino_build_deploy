FROM python:3.10-slim

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean

# Install the python requirements
COPY ./requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the demo code into the app container
WORKDIR /app
COPY . .
VOLUME ["/app/model"]

# Expose demo port
EXPOSE 7860

# Command to run the application
ARG hf_token
ARG personality
ENV HF_TOKEN=${hf_token}
ENV PERSONALITY=${personality}
CMD ["sh", "-c", "python main.py --personality $PERSONALITY --hf_token $HF_TOKEN --local_network"]
