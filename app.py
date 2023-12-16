import os
import random
import re
import json

import numpy as np
import gradio as gr
import gradio.routes
import jsonlines
import tiktoken
import tokenizers
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import KMeans
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

base_system_prompt = {
    "role": "system",
    "content": """
You are Pix, a large language model AI, trained with a vast corpus of pixel-art related tasks such as generation, in-painting, editing, coloring, upscaling, downscaling, segmentation and more.
"""}

load_dotenv()

openai_api_key = os.environ.get("API_KEY")
openai_base_url = os.environ.get("BASE_URL")
openai_base_model = os.environ.get("BASE_MODEL")

encoding = tiktoken.get_encoding("cl100k_base")
tokenizer = tokenizers.Tokenizer.from_pretrained("TheBloke/Llama-2-70b-fp16")

loaded_prompts = {}


def load_prompts(prompts):
    for prompt in prompts:
        with open(f'./prompts/{prompt}') as file:
            loaded_prompts[prompt] = file.read()


prompts = ['healer/recolor_system.md', 'healer/recolor_test_input.md', 'healer/recolor_test_output.md']
load_prompts(prompts)


def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as f:
        for obj in f:
            messages, mode = obj["messages"]
            token_count = len(encoding.encode(str(messages)))

            data.append({
                "messages": messages,
                "type": mode,
                "size_llama": len(tokenizer.encode(str(messages)).tokens),
                "size_gpt": token_count, })
    return data


k_train_shots = load_jsonl('skills/train.jsonl')
print('First example from skills/train.jsonl')
print(k_train_shots[0])

loaded_files = {}


def nearest_neighbor_downscale(image, target_size, max_colors=16):
    # Convert the image to RGBA
    image = image.convert("RGBA")

    # If the image size is smaller than target_size, use the original size
    if image.size[0] < target_size[0] or image.size[1] < target_size[1]:
        target_size = image.size

    # Perform nearest-neighbor resizing
    image = image.resize(target_size, Image.NEAREST)

    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Reshape the array for k-means clustering
    img_reshaped = img_array.reshape((-1, 4))

    # If there are fewer than max_colors, use the original color count
    unique_colors = np.unique(img_reshaped, axis=0)
    if len(unique_colors) < max_colors:
        max_colors = len(unique_colors)

    # Perform k-means clustering to limit the number of colors
    kmeans = KMeans(n_clusters=max_colors)
    kmeans.fit(img_reshaped)

    # Replace each pixel with its nearest cluster center
    img_labels = kmeans.predict(img_reshaped)
    img_reconstructed = kmeans.cluster_centers_[img_labels].astype(np.uint8)

    # Reshape the array back to the original shape
    img_reconstructed = img_reconstructed.reshape(target_size[0], target_size[1], 4)

    # Create a PIL Image from the NumPy array
    downscaled_image = Image.fromarray(img_reconstructed)

    return downscaled_image


def load_images(files):
    global loaded_files

    result = ""
    filesnames = ""
    for idx, file in enumerate(files):
        # Open the image file
        img = Image.open(file).convert('RGB')

        # Downscale and reduce colors of the image
        img = nearest_neighbor_downscale(img, (32, 32), max_colors=16)

        img_array = np.array(img)

        # Generate the palette
        palette, _ = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0, return_inverse=True)
        palette_csv = "Key,Color\n" + "\n".join(
            f"{chr(97 + i)},#{''.join(f'{channel:02x}' for channel in color)}" for i, color in enumerate(palette))

        # Generate the image data
        image_data = ''
        for row in img_array:
            row_data = ''
            for pixel in row:
                key = chr(97 + np.where(np.all(palette == pixel, axis=1))[0][0])
                row_data += key + ','
            image_data += row_data[:-1] + '\n'  # Remove trailing comma and add newline

        # Get just the filename from the full path
        filename = os.path.basename(file)
        loaded_files[
            filename] = f'<|image|>{json.dumps({"palette": palette_csv, "image_data": image_data})}<|end_of_image|>'

        result += f"<h2>Palette of '{filename}'</h2>"
        filesnames += f"{filename},{idx}\n"

        # Yield each hex RGB color from the palette
        for color in palette:
            color_hex = "#" + "".join(f"{channel:02x}" for channel in color)
            entry = f'<span style="color: {color_hex};">██&#xfe0e;</span>'
            result += entry

            # time.sleep(0.01)
            yield result, filesnames


palette_recolor_response = gr.HTML(
    value='<span style="color: #708090;">██ The new image and palette will render here</span>')
generated_image = gr.Gallery(visible=False, elem_id="image_upload")


def handle_recolor_answer(answer, image_obj):
    # Initialize html_palette as an empty string
    html_palette = ''
    # Initialize a list to store the images
    images = []

    # Split the answer into parts by "palette.csv"
    parts = answer.split("palette.csv\n")
    for part in parts[1:]:  # Skip the first part, as it doesn't contain a palette
        # Add the "palette.csv" header back to the part
        part = "palette.csv\n" + part
        print('PARTI')
        print(part)
        # Find the start and end indices of the palette in the part
        palette_start = part.index("```csv") + len("```csv")
        palette_end = part.index("```", palette_start)
        # Extract the palette from the part
        raw_palette = part[palette_start:palette_end]

        # Split the raw palette into lines
        palette_lines = raw_palette.strip().split('\n')
        # Extract the colors from the palette
        colors = [line.split(',')[1] for line in palette_lines[1:]]

        # Generate the HTML string of colored full blocks
        html_palette = f'<h2>Generated palette</h2>' + ''.join(
            f'<span style="color: {color};">██&#xfe0e;</span>' for color in colors)

        # Create a dictionary mapping keys to colors
        palette = {key: color for key, color in (line.split(',') for line in palette_lines[1:])}
        image_data = image_obj['image_data'].split('\n')

        # Create an empty numpy array for the image
        image = np.zeros((len(image_data), len(image_data[0].split(',')), 3), dtype=np.uint8)

        # Fill in the image array with the appropriate colors
        for i, row in enumerate(image_data):
            for j, pixel in enumerate(row.strip().split(',')):
                if pixel:
                    image[i, j] = [int(palette[pixel][k:k + 2], 16) for k in (1, 3, 5)]  # Convert hex to RGB

        # Convert the numpy array to a PIL Image
        image_a = Image.fromarray(image)
        image_b = image_a.resize((512, 512), Image.NEAREST)
        # Add the image to the list of images
        images.append(image_a)
        images.append(image_b)

    return answer, html_palette, gr.update(visible=True), images, gr.update(visible=True)


def handle_creation_answer(answer, image_obj):
    # Initialize a list to store the images
    images = []

    # Split the answer into parts by "palette.csv"
    parts = answer.split("palette.csv\n")
    for part in parts[1:]:  # Skip the first part, as it doesn't contain a palette
        # Add the "palette.csv" header back to the part
        part = "palette.csv\n" + part

        # Find the start and end indices of the palette in the part
        palette_start = part.index("```csv") + len("```csv")
        palette_end = part.index("```", palette_start)
        # Extract the palette from the part
        raw_palette = part[palette_start:palette_end]

        # Split the raw palette into lines
        palette_lines = raw_palette.strip().split('\n')
        # Create a dictionary mapping keys to colors
        palette = {key: color for key, color in (line.split(',') for line in palette_lines[1:])}

        # Find the start and end indices of the image data in the part
        image_data_start = part.index("image_data.csv\n```csv\n") + len("image_data.csv\n```csv\n")
        image_data_end = part.index("```\n", image_data_start)
        # Extract the image data from the part
        raw_image_data = part[image_data_start:image_data_end]

        # Split the raw image data into lines
        image_data = raw_image_data.strip().split('\n')

        # Create an empty numpy array for the image
        image = np.zeros((len(image_data), len(image_data[0].split(',')), 3), dtype=np.uint8)

        # Fill in the image array with the appropriate colors
        for i, row in enumerate(image_data):
            for j, pixel in enumerate(row.strip().split(',')):
                if pixel:
                    image[i, j] = [int(palette[pixel][k:k + 2], 16) for k in (1, 3, 5)]  # Convert hex to RGB

        # Convert the numpy array to a PIL Image
        image_a = Image.fromarray(image)
        image_b = image_a.resize((512, 512), Image.NEAREST)
        # Add the image to the list of images
        images.append(image_a)
        images.append(image_b)

    return answer, '', None, images, gr.update(visible=True)


def handle_segmentation_answer(answer, image_obj):
    # Initialize a list to store the images
    images = []

    # Split the answer into parts by "palette.csv"
    parts = answer.split("palette.csv\n")
    for part in parts[1:]:  # Skip the first part, as it doesn't contain a palette
        # Add the "palette.csv" header back to the part
        part = "palette.csv\n" + part

        # Check if "```csv" is in the part
        if "```csv" in part:
            # Find the start and end indices of the palette in the part
            palette_start = part.index("```csv") + len("```csv")
            palette_end = part.index("```\n", palette_start)
            # Extract the palette from the part
            raw_palette = part[palette_start:palette_end]

            # Split the raw palette into lines
            palette_lines = raw_palette.strip().split('\n')
            # Create a dictionary mapping keys to colors
            palette = {key: color for key, color in (line.split(',') for line in palette_lines[1:])}

            # Find the start and end indices of the image data in the part
            image_data_start = part.index("image_data.csv\n```csv\n") + len("image_data.csv\n```csv\n")
            image_data_end = part.index("```", image_data_start)
            # Extract the image data from the part
            raw_image_data = part[image_data_start:image_data_end]

            # Split the raw image data into lines
            image_data = raw_image_data.strip().split('\n')

            # Create an empty numpy array for the image
            image = np.zeros((len(image_data), len(image_data[0].split(',')), 3), dtype=np.uint8)

            # Fill in the image array with the appropriate colors
            for i, row in enumerate(image_data):
                for j, pixel in enumerate(row.strip().split(',')):
                    if pixel:
                        image[i, j] = [int(palette[pixel][k:k + 2], 16) for k in (1, 3, 5)]  # Convert hex to RGB

            # Convert the numpy array to a PIL Image
            image_a = Image.fromarray(image)
            image_b = image_a.resize((512, 512), Image.NEAREST)
            # Add the image to the list of images
            images.append(image_a)
            images.append(image_b)
        else:
            continue  # Skip to the next part

    return answer, '', None, images, gr.update(visible=True)


def handle_inpainting_answer(answer, image_obj):
    # Initialize a list to store the images
    images = []

    # Extract the palette from the user's input
    raw_palette = image_obj['palette'].split('\n')
    palette = {key: color for key, color in (line.split(',') for line in raw_palette[1:])}  # Skip the header

    # Split the answer into parts by "image_data.csv"
    parts = answer.split("image_data.csv")
    print(parts)
    for part in parts[1:]:  # Skip the first part, as it doesn't contain image data
        # Add the "image_data.csv" header back to the part
        part = "image_data.csv" + part

        # Find the start and end indices of the image data in the part
        image_data_start = part.index("```csv") + len("```csv")
        image_data_end = part.index("```", image_data_start)
        # Extract the image data from the part
        raw_image_data = part[image_data_start:image_data_end]

        # Split the raw image data into lines
        image_data = raw_image_data.strip().split('\n')

        # Create an empty numpy array for the image
        image = np.zeros((len(image_data), len(image_data[0].split(',')), 3), dtype=np.uint8)

        # Fill in the image array with the appropriate colors
        for i, row in enumerate(image_data):
            for j, pixel in enumerate(row.strip().split(',')):
                if pixel:
                    image[i, j] = [int(palette[pixel][k:k + 2], 16) for k in (1, 3, 5)]  # Convert hex to RGB

        # Convert the numpy array to a PIL Image
        image_a = Image.fromarray(image)
        image_b = image_a.resize((512, 512), Image.NEAREST)
        # Add the image to the list of images
        images.append(image_a)
        images.append(image_b)

    return answer, '', None, images, gr.update(visible=True)


def ai_response(input_text, temperature, n_shots, n_shots_size, task_type, stream, api_input, base_url_input,
                base_model):
    global loaded_files

    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_input or openai_api_key,
        base_url=base_url_input or openai_base_url
    )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

    # Extract image tags from the input_text
    image_tags = re.findall(r'<img:(.*?)>', input_text)
    image_obj = {}

    # Replace each image tag with its corresponding image representation
    for tag in image_tags:
        # Check if the image representation exists in the loaded_files dictionary
        if tag in loaded_files:
            # Replace the image tag with the image representation in the input_text
            input_text = input_text.replace(f'<img:{tag}>', loaded_files[tag])
            region = loaded_files[tag][9:-16]
            image_obj = json.loads(region)

    p_type = task_type if task_type != 'general' else False

    # Select n_shots or fewer k_shots randomly, but only choose shots that size_gpt < 800
    scale_factor = 1
    while True:
        if p_type:
            typed_shots = [s for s in k_train_shots if
                           s['type'] == task_type and s['size_gpt'] < n_shots_size * scale_factor]
            if typed_shots or scale_factor >= 8:  # Stop increasing the scale factor after a certain point
                selected_shots = random.sample(typed_shots, min(n_shots, len(typed_shots)))
                break
        else:
            valid_shots = [s for s in k_train_shots if s['size_gpt'] < n_shots_size * scale_factor]
            if valid_shots or scale_factor >= 8:  # Stop increasing the scale factor after a certain point
                selected_shots = random.sample(valid_shots, min(n_shots, len(valid_shots)))
                break
        scale_factor *= 2  # Exponential incr.

    [(print(ss['size_gpt']), print(ss['messages'])) for ss in selected_shots]
    # Add the selected shots to the in_context_learning list
    in_context_learning = []
    for shot in selected_shots:
        in_context_learning.extend(shot['messages'])

    context = [
        {"role": "system",
         "content": 'test finished, the above messages are just examples to guide you on the following tasks. the following is totally unrelated with the above messages, it\'s a brand new interaction. don\'t mix things!'},

        {"role": "user", "name": "real user", "content": input_text}
    ]

    cot_messages = [base_system_prompt, *in_context_learning, *context]
    print(cot_messages)

    cot_completion = completion_with_backoff(
        model=base_model or openai_base_model,
        temperature=temperature,
        messages=cot_messages,
        max_tokens=5000
    )

    answer = cot_completion.choices[0].message.content
    print(answer)

    answer_parsers = {
        'recolor': handle_recolor_answer,
        'creation': handle_creation_answer,
        'segmentation': handle_segmentation_answer,
        'inpainting': handle_inpainting_answer
    }

    if task_type in answer_parsers.keys():
        recolor_messages = [
            {'role': 'system', 'content': loaded_prompts['healer/recolor_system.md']},
            {'role': 'user', 'name': 'test case', 'content': {"user_input": "[ ... ]", "buggy_answer": loaded_prompts['healer/recolor_test_input.md'], "error": "Key error: 'e'"}},
            {'role': 'assistant', 'content': loaded_prompts['healer/recolor_test_output.md']},
        ]

        try:
            _answer, _palette_html, _palette_html_v, _gallery, _gallery_v = answer_parsers[task_type](answer, image_obj)
            print("qnt, ", len(_gallery))

            # recolor task must return at least 1 palette at the first try
            if task_type == 'recolor' and len(_gallery) == 0:
                print("Recolor warning: no palette found! Trying to heal answer.")

                try:
                    recall_recolor_completion = completion_with_backoff(
                        model=base_model or openai_base_model,
                        temperature=temperature,
                        messages=[
                            *recolor_messages,
                            {'role': 'user', 'name': 'current real case',
                             'content': {"user_input": input_text, "buggy_answer": answer,
                                         "error": str('Missing palette.csv')}}
                        ],
                        max_tokens=5000
                    )

                    answer = recall_recolor_completion.choices[0].message.content
                    print(answer)

                    return answer_parsers[task_type](answer, image_obj)
                except Exception as e:
                    raise e

            return _answer, _palette_html, _palette_html_v, _gallery, _gallery_v
        except Exception as e:
            print(f"Error encountered.", repr(e))
            try:
                if task_type == 'recolor':
                    recall_recolor_completion = completion_with_backoff(
                        model=base_model or openai_base_model,
                        temperature=temperature,
                        messages=[
                            *recolor_messages,
                            {'role': 'user', 'name': 'current real case',
                             'content': {"user_input": input_text, "buggy_answer": answer.strip(), "error": repr(e)}}
                        ],
                        max_tokens=5000
                    )

                    answer = recall_recolor_completion.choices[0].message.content
                    print(answer)

                    return answer_parsers[task_type](answer, image_obj)
            except Exception as e:
                raise e
    return answer, '', None, None, None


api_input = gr.Textbox(label="Your OpenAI API key", type="password")
base_url = gr.Textbox(label="OpenAI API base URL")
base_model = gr.Textbox(label="Model")

with gr.Blocks() as demo:
    with gr.Tab("app"):
        user_input = gr.Textbox(lines=2, label="User Input (with filename autocomplete)", elem_id="prompt_input",
                                placeholder="Pick some files on the input below then write something like 'describe the following image: <img:file.ext>'")
        images_block = gr.File(label="Pixel Art Image", file_count='multiple', file_types=['.png', '.jpg', '.jpeg'])
        images_names = gr.Textbox(visible=False, elem_id="hidden_tags")

        colors = gr.HTML(label='Color palette',
                         value='<span style="color: #708090;">██ Color palettes will show here after you load an image...</span>')

        images_block.upload(fn=load_images, inputs=[images_block], outputs=[colors, images_names])


        def clear_files():
            global loaded_files
            loaded_files = {}
            return "", ""


        images_block.change(fn=clear_files, outputs=[images_names, colors])

        task_type = gr.Radio(["creation", "recolor", "segmentation", "inpainting", "caption", "general"], label="Task",
                             value="general",
                             info="Controls what k-shots will be feed to Assistant (Chat Completion). General will feed random k-shots regardless of type.")

        with gr.Row():
            temperature = gr.Slider(label="Temperature", minimum=0, maximum=2, step=0.01, value=0.01)
            n_shots = gr.Slider(label="k-shots", minimum=0, maximum=len(k_train_shots), step=1, value=1)
            n_shots_size = gr.Slider(label="k-shots max size (each, turbo tokens)", minimum=0, maximum=100000, step=1,
                                     value=800)

        completion_block = gr.Textbox(label="Assistant's response", max_lines=5, lines=5, interactive=True)

        with gr.Row():
            stream_check = gr.Checkbox(label="Stream", value=False)
            ai_btn = gr.Button("Generate AI Response")
            clear_btn = gr.Button("Clear")
            stop_btn = gr.Button("Stop")

        generation = ai_btn.click(fn=ai_response, inputs=[
            user_input,
            temperature,
            n_shots,
            n_shots_size,
            task_type,
            stream_check,
            api_input,
            base_url,
            base_model
        ], outputs=[completion_block, palette_recolor_response, palette_recolor_response, generated_image,
                    generated_image],
                                  show_progress='minimal')
        palette_recolor_response.render()
        generated_image.render()

        clear_outputs = clear_btn.click(fn=lambda: ("", "", ""), outputs=[user_input, completion_block])

        stop_btn.click(None, None, None, cancels=[generation, clear_outputs])

        gr.Markdown('# File picker examples')
        r_examples = gr.Examples(
            [[[os.path.join(os.path.dirname(__file__), "chicken.png")]],
             [[os.path.join(os.path.dirname(__file__), "skull.png")]],
             [[os.path.join(os.path.dirname(__file__), "skull.png"),
               os.path.join(os.path.dirname(__file__), "chicken.png")]]],
            [images_block],
            [colors, images_names],
            load_images,
            run_on_click=True
        )

        gr.Markdown('# User input examples + settings')
        ui_examples = gr.Examples(
            [
                [
                    'The following is an image of a [example], please change its palette to make it purple. Keep the background [example].\n\n<img:placeholder_file.ext>',
                    'recolor', 2],
                [
                    'Can you create a 12x12 image of a black cat with a white bg? Use just these two colors.',
                    'creation', 1],
                [
                    'Write a 16x16 2D \'water texture, seamless tile\' asset',
                    'creation', 2],
                [
                    'Hi there! Who are you and what can you do to me?',
                    'general', 0]
            ],
            inputs=[user_input, task_type, n_shots],
        )

    with gr.Tab("Settings"):
        api_input.render()
        base_url.render()
        base_model.render()

        gr.Examples(
            [
                ['https://openai-proxy.replicate.com/v1', 'meta/llama-2-70b-chat'],
                ['https://openrouter.ai/api/v1', 'mistralai/mixtral-8x7b-instruct'],
                ['', 'gpt-4-1106-preview'],
                ['', 'gpt-4-0613'],
                ['', 'gpt-3.5-turbo-1106']
            ],
            inputs=[base_url, base_model]
        )

if __name__ == "__main__":
    with open("script.js", "r", encoding="utf8") as jsfile:
        javascript = jsfile.read()


    def template_response(*args, **kwargs):
        res = gradio_routes_templates_response(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'<script>{javascript}</script></head>'.encode("utf8"))
        res.init_headers()
        return res


    gradio_routes_templates_response = gradio.routes.templates.TemplateResponse
    gradio.routes.templates.TemplateResponse = template_response

    demo.launch()
