# ai_workshop_tmc
Welcome to the repository for TMC's exclusive AI Workshop on programming your private AI assistant. This repository contains all resources and instructions needed to set up and build a simple ChatGPT-like AI assistant on your laptop, ensuring privacy and customization possibilities.




* docker-compose
  * `docker-compose down`
  * `docker-compose build --no-cache`
  * `docker-compose up -d`

* pyenv/ pip freeze
  * `source /app/venv/bin/activate`
  * `pip freeze --all > /home/jovyan/jupyter/requirements.txt`
  * `pip install --no-cache-dir -r /home/jovyan/jupyter/requirements.txt`
  * `pip freeze --all > /app/streamlit/requirements.txt`
  * `pip install --no-cache-dir -r /app/streamlit/requirements.txt`

# TODO:
# 3. check if data is persisted to chroma
# 4. finish rest of pipeline and create simple interface with gradio or streamlit
# 5. create sustainability expert:
# 5.1 sustainability expert documents (literature/ websites/ sustainability --> economically profitable/ etc.)
# 5.2 promt engineer


## PATHS
cd into cucls directory

## Opensourcery Image

`docker volume create ingestion_data`


`docker build -t opensourcery-cpu-unstructured-chroma-langchain-streamlit:v0.0.4 .`
`docker run -it -d --env-file .env --restart=unless-stopped -v ingestion_data:/data -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/opensourcery/cucls -p 8000:8000 --network opensourcery_network --name opensourcery_cucls_v004 opensourcery-cpu-unstructured-chroma-langchain-streamlit:v0.0.4`
`docker network connect opensourcery_network opensourcery_cucls_v004`
`docker exec -it opensourcery_cucls_v004 /bin/bash`

## Sharepoint

Start up a sharepoint instance based on [docker-spfx](https://github.com/pnp/docker-spfx).
`docker run -d -it --name sharepoint -v $(pwd)/sharepoint:/usr/app/spfx -p 4321:4321 -p 35729:35729 m365pnp/spfx`
```
npm install
chmod -R u=rwx,g=rx,o=rwx .
gulp trust-dev-cert
gulp serve --nobrowser
```

## Ollama/ GPT4All/ Huggingface

- GPT4All       https://gpt4all.io/index.html
  - Great Benchmark List
- Ollama        https://ollama.com/
- Huggingface   https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B
  - Leaderboard https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
  - Docker      https://fgiasson.com/blog/index.php/2023/08/23/how-to-deploy-hugging-face-models-in-a-docker-container/

### ollama_cpu

Followed the instructions on [Ollama Docker Hub](https://hub.docker.com/r/ollama/ollama) page.
`docker run -d -v $(pwd)/ollama_cpu:/root/.ollama -p 11434:11434 --network opensourcery_network --name ollama_cpu ollama/ollama`
`docker exec -d -it ollama_cpu ollama run nous-hermes2:10.7b`


### ollama_amd

Followed the instructions on [Ollama Docker Hub](https://hub.docker.com/r/ollama/ollama) page.
`docker run -d --device /dev/kfd --device /dev/dri -v $(pwd)/ollama_amd_gpu:/root/.ollama -p 11435:11434 --network opensourcery_network --name ollama_amd_gpu ollama/ollama:rocm`
`docker exec -d -it ollama_amd_gpu ollama run nous-hermes2:10.7b`

## UNSTRUCTURED

Tried unstructured in docker container. Way too involved compared to running it in the opensourcery container with python. So I switched to that option.
`docker run -dt --name unstructured -v ingestion_data:/data -v "$(pwd)/unstructured/unstructured_folder_ingestion.sh:/unstructured_folder_ingestion.sh" --network opensourcery_network downloads.unstructured.io/unstructured-io/unstructured:latest`
`docker exec -it unstructured /bin/bash`

- https://unstructured-io.github.io/unstructured/examples/chroma.html
- https://github.com/Unstructured-IO/unstructured/blob/main/examples/ingest/local/ingest.sh
- https://github.com/Unstructured-IO/unstructured/blob/main/examples/ingest/chroma/ingest.sh


## ChromaDB

`docker run -d -p 8001:8000 --name chroma -v $(pwd)/chroma/index:/chroma/.chroma/index --network opensourcery_network ghcr.io/chroma-core/chroma:latest`
<!-- `docker exec -it chroma /bin/bash` -->

## Docker Network

`docker network inspect opensourcery_network`

`docker network create opensourcery_network`
`docker network connect opensourcery_network ollama_cpu`
`docker network connect opensourcery_network opensourcery_cucls_v004`
`docker network connect opensourcery_network chroma`



## cAdvisor for Docker Monitoring
`docker run --volume=/:/rootfs:ro --volume=/var/run:/var/run:rw --volume=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro --publish=8080:8080 --detach=true --name=cadvisor gcr.io/cadvisor/cadvisor:latest`


## Langchain

- https://python.langchain.com/docs/get_started/quickstart/#diving-deeper-2
- tools                 https://python.langchain.com/docs/integrations/tools/
- promt engineering     https://smith.langchain.com/hub/ohkgi/superb_system_instruction_prompt?tab=0
- https://python.langchain.com/docs/expression_language/how_to/message_history/

## Later
- fabric promt engineering: https://github.com/danielmiessler/fabric?tab=readme-ov-file#introduction-video
- 
