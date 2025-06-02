# Agentic Rag solution using WatsonX by IBM
## Intro to Climate education platform
This is an interactive educational platform that uses RAG to provide information on climate change, its impacts, and actions individuals can take to mitigate it.

## Run using Docker

Running the following command in project's root directory to build the docker image:
```
docker build -t rag_hackathon .      
```
Next step is to run the following command to launch it:
```
docker run -p 7860:7860 rag_hackathon
```
Open the localhost:7860 to see the result.

## How to run manually

We ran our experiments using Python 3.12 and simply run:
```
pip install -r requirements.txt
```

by running the following command you can access the app in your localhost:
```
gradio app/app.py
```
* REMEMBER: depending on your location, you may need to refine the gradio command. The default is in root folder of the project where you can see app subfolder.

## Practice files
The Jupyter Notebooks which are located in root folder are our trials and experiments to find the best approach for the project.
The Ingest notebook, contains the code and logic for chunking our data.

## Team
Behrad Hemati (behrad.hemati@gmail.com)
Alireza Rezaei (rezaei.alireza1290@gmail.com)

## Resources
- Education and climate change: learning to act for people and planet (https://doi.org/10.54676/GVXA4765)
- Greening curriculum guidance: teaching and learning for climate action (https://doi.org/10.54675/AOOZ1758)
- Guidelines for Excellence, Educating for Climate Action and Justice (ISBN# 979-8-218-40294-5 © 2024 by the North American Association for Environmental Education (NAAEE))
- NOT JUST HOT AIR Putting Climate Change Education into Practice (ISBN 978-92-3-100101-7) (UNESCO)
- https://global-warming.org/