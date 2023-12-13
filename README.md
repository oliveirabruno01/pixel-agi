---

# Pixel-AGI ✍🏼

Pixel-AGI is a project aimed at leveraging the capabilities of large language models (LLMs) for a variety of pixel-art/image reasoning related tasks. The goal is to augment and eventually fine-tune an already heavily trained LLM (like GPT-3.5/4, LlaMA, OpenHermes...) on a dataset of pixel-art and leverage its reasoning capabilities for tasks such as generation, in-painting, editing, coloring, upscaling, downscaling, segmentation, animation and more, in a seamlessly manner.

This project is driven by a fascination with language engines and the potential they hold. The idea of planning and reasoning solely using language is a compelling one, and this project aims to explore this concept to its fullest extent on the image space. I'd rather edit each dataset entry manually to add actual reasoning and let the model clone my thoughts than just let the model hallucinate a beautiful answer. 

## Demo. What it does

https://github.com/oliveirabruno01/pixel-agi/assets/47301081/38ff62d3-c228-4687-9761-7447c08860f2


## Getting Started

To get started with Pixel-AGI, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary packages using pip:

   ``pip install -r requirements.txt``

4. Run the application:
   
   ``python3 app.py``

Alternatively, you can use an IDE like PyCharm to run the application.

Once the application is open, navigate to the settings tab to add your OpenAI API key and select the model you wish to use.

## Project Structure

The project consists of two main parts: the Gradio app, which is used to interact with the Pixel-AGI models, and the training/dataset building pipelines, which are used to improve and expand the capabilities of the models.

Currently, the `skills/train.jsonl` file contains 111 entries (20 recolor, 25 captioning, 11 creation, 5 segmentation, and 50 inpainting). The goal is to expand this to at least 1000 high-quality entries by early January.

You can find the code to prepare the dataset and to generate the inpainting tasks in the same folder. Another utils that I used so far I will upload here after a clean-up.

Currently we are using `skills/train.jsonl` just for random RAG in our gradio app.

## Current Status

The app is currently in development. The major tasks that need to be completed are:

1. Implement stream completion and real-time parsing of the assistant's answers.
2. Expand the UI to support tasks other than recoloring. 
3. Add utilities to help increment our datasets.
4. Add access to the GPT-4 Vision/LlaVA model to leverage its computer vision capabilities?

Some minor tasks include updating the README with a showcase of the app, and add some explanations in-code (how I managed to implemented the autocomplete, for instance). These minor tasks will wait until I wake up from my post-48h--grinding-beauty-sleep :D

Once these tasks are completed, the focus will shift to building a massive dataset ̶t̶h̶a̶t̶ ̶b̶u̶i̶l̶d̶s̶ ̶i̶t̶s̶e̶l̶f̶  and conducting several training runs.

## Future Work

The project aims to automate the generation of as many dataset entries as possible and add new modalities/tasks. The potential for automation is vast, with possibilities ranging from resizing and masking using traditional programming, to segmentation and creation using today's image AIs like SAM, SD, etc. The goal is to generate and curate as many high-quality entries as possible to ensure the quality of the AI's output and to correct any hallucinations.

## Insights and Observations

During the development of Pixel-AGI, some interesting observations have been made. I have a suspicion that GPT-3 and GPT-4 may have been trained on DALL-E 1 (originally a transformer LM) or other image datasets, as they show impressive capabilities in tasks like recoloring and handling image data. For instance, GPT-4 was able to identify mirrored upside-down images and calculate similarities between colors - although these tests were not statistically significant. These observations highlight the potential of LLMs in handling image-related tasks.

A small training run has already been conducted, yielding promising results on tasks related to colors. The most challenging tasks for LLMs seem to be generation from scratch. It is hoped that a massively fine-tuned language model + a smart retrieval could perform even better.

## Contributing

Contributions and insights are welcome and probably necessary! This project thrives on the enthusiasm and insights of contributors who are interested in the intersection of LLMs and pixel-art. 

You are welcomed to open an issue, PR or discussion. Otherwise you can just hit me on [X](https://twitter.com/LatentLich)

## Funding, sponsorship and donations

You can also contribute with Pixel-AGI and my future projects by funding me and buying me a cofffeee! Just click on the image below:

[<p align="center"><img src="coffee.jpg" style="width: 128px; height: 128px;"></p>](https://buy.stripe.com/dR62aUaaQ7S4cKs3cc)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
