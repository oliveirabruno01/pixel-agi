import random
import re
import json

dataset_out_train = 'train.jsonl'
dataset_out_validation = 'validation.jsonl'


with open('recolor.json', 'r') as f:
    recolor_refactored = json.load(f)
    print(len(recolor_refactored))

with open('captions.json', 'r') as f:
    caption_refactored = json.load(f)
    print(len(caption_refactored))

with open('creation.json', 'r') as f:
    creation_refactored = json.load(f)
    print(len(creation_refactored))

with open('segmentation.json', 'r') as f:
    segmentation_refactored = json.load(f)
    print(len(segmentation_refactored))

with open('inpainting.json', 'r') as f:
    inpainting_refactored = json.load(f)
    print(len(inpainting_refactored))


def get_wrapped_palettes(string):
    return re.findall(r'<\|palette:(\d+)\|>', string)


def get_wrapped_images(string):
    return re.findall(r'<\|image_data:(\d+)\|>', string)


def format_image_data(image_data):
    # Split the image data into lines
    lines = image_data.split('\n')
    # For each line, add commas between each character
    formatted_lines = [','.join(list(line)) for line in lines]
    # Join the lines back together
    formatted_image_data = '\n'.join(formatted_lines)
    return formatted_image_data


def mount_messages_from_dict(d: dict):
    mode = d["type"]
    messages = []

    req = d['request']
    processed_req = req

    if mode in ['recolor', 'caption', 'segmentation', 'inpainting']:
        ui_si = req.index("<|img:") + 6
        ui_ei = req.rindex("|>")
        user_image = req[ui_si:ui_ei]
        image_obj = d['images'][user_image]
        cc_obj = {"palette": image_obj['palette'], "image_data": format_image_data(image_obj['image_data'])}

        processed_req = req.replace(f'<|img:{user_image}|>', f"<|image|>{json.dumps(cc_obj)}<|end_of_image|>")

    messages.append({'role': 'user', 'content': processed_req})

    action = d['action']

    palette_matches = get_wrapped_palettes(action)
    image_matches = get_wrapped_images(action)

    for match in palette_matches:
        index = int(match)
        if 0 <= index < len(d['new_palettes']):
            action = re.sub(rf'<\|palette:{index}\|>', f'palette.csv\n```csv\n{d["new_palettes"][index]}```', action)

    for match in image_matches:
        index = int(match)
        if 0 <= index < len(d['new_images_data']):
            image = d["new_images_data"][index]
            action = re.sub(rf'<\|image_data:{index}\|>', f'image_data.csv\n```csv\n{format_image_data(image)}```', action)

    # Wrap the code block in a code block labeled as 'image_data' for inpainting examples
    if mode == 'inpainting':
        code_block_start = action.index("```") + 3
        code_block_end = action.rindex("```")
        code_block = action[code_block_start:code_block_end]
        formatted_code_block = f"image_data.csv\n```csv\n{format_image_data(code_block)}\n```"
        action = action[:code_block_start - 4] + formatted_code_block

    messages.append({'role': 'assistant', 'content': action})

    return messages, mode


if __name__ == "__main__":
    recolor_shots = [mount_messages_from_dict(mes) for mes in recolor_refactored]
    caption_shots = [mount_messages_from_dict(mes) for mes in caption_refactored]
    creation_shots = [mount_messages_from_dict(mes) for mes in creation_refactored]
    segmentation_shots = [mount_messages_from_dict(mes) for mes in segmentation_refactored]
    inpainting_shots = [mount_messages_from_dict(mes) for mes in inpainting_refactored]

    [print(c, end='\n') for c in recolor_shots]
    [print(c, end='\n') for c in caption_shots]
    [print(c, end='\n') for c in creation_shots]
    [print(c, end='\n') for c in segmentation_shots]
    [print(c, end='\n') for c in inpainting_shots]

    complete_list = recolor_shots + caption_shots + creation_shots + segmentation_shots + inpainting_shots

    random.shuffle(complete_list)
    total_examples = len(complete_list)
    validation_size = 0  # You can adjust the split ratio as needed

    with open(dataset_out_train, 'w+') as train_file, open(dataset_out_validation, 'w+') as validation_file:
        print(f"You have {total_examples} examples.")
        print(f"Train set size: {total_examples - validation_size}")
        print(f"Validation set size: {validation_size}")

        # total_list = complete_list+segmentation_shots[1:]
        # random.shuffle(total_list)

        for idx, sample in enumerate(complete_list):
            output_file = validation_file if idx < validation_size else train_file
            json.dump({"messages": sample}, output_file)
            output_file.write('\n')
