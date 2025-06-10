![IBMHackathon-Projectdemo-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/4a0cdca9-029d-4f3b-a1b8-8c1b81fc3249)
# Agentic Rag solution using WatsonX by IBM
## Intro to Climate education platform
This is an interactive educational platform that uses RAG to provide information on climate change, its impacts, and actions individuals can take to mitigate it.

## Required access tokens
To  run this project you need the following credentials from IBM and should create a .env file with them in app directory (also it would be a good practice to create another env file in root directory or simply update the paths in jupyter notebooks to read env in the app folder)
Example of .env file:
```
API_KEY="xxx"
URL="xxx"
PROJECT_ID = "xxx"
``` 
Or rename the app/.env.exmaple file to app/.env and update it with credentials.

## IBM Technologies used in the project

* **Watsonx Studio**: Providing the foundation models.
* **Langchain IBM**: Langchain library to integrate with the IBM cloud and watsonx, provides easy access to foundation models.
* **Docling**: Excelent library for reading documents and turning them into structured data for use in a RAG pipeline.
* **Granite Embedding Model**: Used as embedder 

## Other technologies
* **Langcahin**: Various functionalitoes
* **Langgraph**: Use for creation of agentic features
* **Gradio**: Convinient chatbot interface
* **ChromaDB**: Used as vectorestore
* **Fitz**: Read and write binary content and files
* **Pandas**: Data manipulation and analysis
* **Docker**: Containerization for easy deployment

## Specifics of Models used
* **Embedding Model**: granite-embedding-278m-multilingual
* **Foundation Model**: Llama-3.3-70b-instruct

## How to run the project
### Docker (app.py - using WatsonX backend)

Running the following command in project's root directory to build the docker image:
```
docker build -t rag_hackathon .      
```
Next step is to run the following command to launch it:
```
docker run -p 7860:7860 rag_hackathon
```
Open the localhost:7860 to see the result.

### Command line 

We ran our experiments using Python 3.12 and simply run:
```
pip install -r requirements.txt
```

by running the following command you can access the app in your localhost:
```
gradio app/app.py
```
* REMEMBER: depending on your location, you may need to refine the gradio command. The default is in root folder of the project where you can see app subfolder.

### *Outage notice:* 
During the last stages of the Hackathon, the IBM cloud experienced an outage, for which we provide app_hf.py which uses Hugginface inference as a backup, for which we don't provide setup intructions or a docker image since the purpose of this project is to showcase and use IBM services.
## Project demo

You can see the demo of the project in the following link: https://youtu.be/5eARocbL75Y or find it in the project_demo folder.

## Experimental Notebooks

The project includes several Jupyter Notebooks that were used for experimentation and development. These notebooks are located in the root folder and serve as trials to find the best approach for the project.
- The Ingest notebook contains the code and logic for chunking our data, which is essential for the RAG pipeline.
- The agentic notebook contains the code and logic for the agentic features of the project, which allow users to interact with the platform in a more dynamic way.
- The main notebook was used for experiments on how to bring all functionalities together.
- The graph notebook contains the code and logic for the graph features of the project, which allow users to visualize the data in a more interactive way.

## Team
- Alireza Rezaei (rezaei.alireza1290@gmail.com)
- Behrad Hemati (behrad.hemati@gmail.com)


## Resources
- Education and climate change: learning to act for people and planet (https://doi.org/10.54676/GVXA4765)
- Greening curriculum guidance: teaching and learning for climate action (https://doi.org/10.54675/AOOZ1758)
- NOT JUST HOT AIR Putting Climate Change Education into Practice (ISBN 978-92-3-100101-7) (UNESCO)
- https://global-warming.org/
