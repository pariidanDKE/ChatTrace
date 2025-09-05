# ChatTrace : retrieve and inquire on past chats

This application allows you to upload your own chat history, it then processes it and sets up a RAG system that allows you to ask questions about your messages. __Important points__ :

- runs locally through llama.cpp : private messages do not leave your machine
- support both Instagram and WhatsApp message formats
- supports multilingual messages, even in same chat
- has docker support, with handy docker-compose file to run all 3 necessary applications



## Downloading Chat History

### WhatsApp:

Whatsapp does not allow you to download your multiple chats, or your entire chat history at once unfortunately, but you can [export one chat history](https://faq.whatsapp.com/1180414079177245/?helpref=platform_switcher&cms_platform=iphone&cms_id=1180414079177245&draft=false). The current implementation of ChatTrace is not multimodel, so you should select __"Without Media"__.

### Instagram:

With instagram, it is possible to [export your whole information at once](https://help.instagram.com/1224884341728748?helpref=faq_content), through the app. __Important__ : In the process you are asked which format the data should be in (HTML or JSON), you should choose JSON. Instagram also offers a "Customise information" setting, from there we can leave only Messages, as that it the only extracted data.



## Run application

This application has several different configurations. Mainly it is a python app, that uses the llama.cpp inferece engine, both with the ollama wrapper (for the Chat and Embedding Model) and without (for the Reranker model).


#### Docker support

The project has full docker support, so no used applications (python, ollama, llama.cpp) need to be downloaded, each has its own container, orchestrated through docker-compose. 

To start the application through docker,[first make sure you have Docker engine running](https://docs.docker.com/get-started/introduction/get-docker-desktop/), and then run: 

`docker compose -f docker-compose-dockerized.yml up --build`















