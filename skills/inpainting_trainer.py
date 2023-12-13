import tkinter as tk
from PIL import Image, ImageTk
import os
import json
import numpy as np


class Application(tk.Frame):
    def __init__(self, master=None, img_path=None, img_files=None, img_index=0):
        super().__init__(master)
        self.master = master
        self.pack()
        self.img_path = img_path
        self.img_files = img_files
        self.img_index = img_index
        self.interactions = []
        self.create_widgets()
        self.drawn = None
        self.history = []
        self.redo_list = []

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=512, height=512)
        self.load_image()
        self.canvas.pack()

        # Add a label to display the image filename
        self.filename_label = tk.Label(self, text=self.img_files[self.img_index])
        self.filename_label.pack()

        self.canvas.focus_set()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.master.bind("<Control-z>", self.undo)
        self.master.bind("<Control-y>", self.redo)
        self.description = tk.Text(self, height=2, width=30)
        self.description.pack()
        self.user_input = tk.Text(self, height=2, width=30)
        self.user_input.pack()
        self.save_button = tk.Button(self, text="SAVE", command=self.save_interaction)
        self.save_button.pack()
        self.prev_button = tk.Button(self, text="PREVIOUS", command=self.prev_image)
        self.prev_button.pack()
        self.next_button = tk.Button(self, text="NEXT", command=self.next_image)
        self.next_button.pack()

    def draw(self, event):
        if not self.drawn:
            self.drawn = self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='red')
            self.history.append(self.drawn)
        else:
            x1, y1, _, _ = self.canvas.coords(self.drawn)
            self.drawn = self.canvas.create_oval(x1, y1, event.x + 5, event.y + 5, fill='red')
            self.history.append(self.drawn)

    def reset(self, event):
        self.drawn = None

    def undo(self, event):
        if self.history:
            self.redo_list.append(self.history[-1])
            self.canvas.delete(self.history[-1])
            self.history.pop()

    def redo(self, event):
        if self.redo_list:
            self.canvas.itemconfig(self.redo_list[-1], state='normal')
            self.history.append(self.redo_list[-1])
            self.redo_list.pop()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=512, height=512)
        self.load_image()
        self.canvas.pack()
        self.canvas.focus_set()
        self.canvas.bind("<Button-1>", self.toggle_lock)
        self.canvas.bind("<Motion>", self.move_rect)
        self.canvas.bind("<MouseWheel>", self.resize_rect)
        self.description = tk.Text(self, height=2, width=30)
        self.description.pack()
        self.user_input = tk.Text(self, height=2, width=30)
        self.user_input.pack()
        self.save_button = tk.Button(self, text="SAVE", command=self.save_interaction)
        self.save_button.pack()
        self.prev_button = tk.Button(self, text="PREVIOUS", command=self.prev_image)
        self.prev_button.pack()
        self.next_button = tk.Button(self, text="NEXT", command=self.next_image)
        self.next_button.pack()
        self.locked = False

    def toggle_lock(self, event):
        self.locked = not self.locked

    def move_rect(self, event):
        if not self.locked:
            rect_coords = self.canvas.coords(self.rect)
            width = rect_coords[2] - rect_coords[0]
            height = rect_coords[3] - rect_coords[1]
            self.canvas.coords(self.rect, event.x - width / 2, event.y - height / 2, event.x + width / 2,
                               event.y + height / 2)

    def resize_rect(self, event):
        scale = 1.1 if event.delta > 0 else 0.9
        self.canvas.scale(self.rect, 0, 0, scale, scale)

    def save_interaction(self):
        rect_coords = [coord * self.original_size[0] / 512 for coord in self.canvas.coords(self.rect)]
        # Ensure the rectangle's coordinates are within the bounds of the original image
        rect_coords = [max(min(coord, self.original_size[i % 2]), 0) for i, coord in enumerate(rect_coords)]
        interaction = {
            "request": f"{self.description.get('1.0', 'end-1c')}.\n\n<|img:{self.img_files[self.img_index]}|>",
            "images": {
                self.img_files[self.img_index]: {
                    "palette": self.get_palette(),
                    "image_data": self.get_image_data_with_selected_pixels_as_white_transparent_alpha(
                        rect_coords)
                }
            },
            "action": self.user_input.get('1.0', 'end-1c') + "\n\n ```" + self.get_image_data(rect_coords) + "```",
            "type": "inpainting"
        }
        self.interactions.append(interaction)
        with open('inpainting.json', 'w') as f:
            json.dump(self.interactions, f)

    def prev_image(self):
        self.img_index = (self.img_index - 1) % len(self.img_files)
        self.canvas.delete("all")
        self.load_image()

    def next_image(self):
        self.img_index = (self.img_index + 1) % len(self.img_files)
        self.canvas.delete("all")
        self.load_image()

    def load_image(self):
        self.img = Image.open(os.path.join(self.img_path, self.img_files[self.img_index]))
        self.original_size = self.img.size
        self.img_resized = self.img.resize((512, 512), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(self.img_resized)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.rect = self.canvas.create_rectangle(50, 50, 100, 100, outline='red')

    def get_palette(self):
        img = np.array(self.img)
        palette, _ = np.unique(img.reshape(-1, img.shape[2]), axis=0, return_inverse=True)
        palette_csv = "Key,Color\n"
        for i, color in enumerate(palette):
            key = chr(97 + i)  # Generate keys as 'a', 'b', 'c', ...
            # If the color is already a transparent mask, change it to non-alpha
            if color.tolist() == [255, 255, 255, 0]:
                color = [255, 255, 255, 255]
            color_hex = '#{:02x}{:02x}{:02x}{:02x}'.format(*color)
            palette_csv += f"{key},{color_hex}\n"
        # Add a new color to the palette for the selected region
        key = chr(97 + len(palette))  # The next available key from the palette
        color_hex = '#ffffff00'  # White with transparent alpha
        palette_csv += f"{key},{color_hex}\n"
        return palette_csv.strip()

    def get_image_data(self, rect_coords):
        img = np.array(self.img)
        palette = self.get_palette().split('\n')[1:]  # Skip the header
        palette = {line.split(',')[0]: tuple(int(line.split(',')[1][i:i + 2], 16) for i in (1, 3, 5, 7)) for line in
                   palette}  # Convert hex color to RGB
        image_data = np.empty((img.shape[0], img.shape[1]), dtype='U1')
        for key, color in palette.items():
            mask = np.all(img == color, axis=2)
            image_data[mask] = key
        return '\n'.join(''.join(row) for row in image_data)

    def get_image_data_with_selected_pixels_as_white_transparent_alpha(self, rect_coords):
        img = np.array(self.img)
        # Change the #ffffff00 pixels to #ffffffff
        mask = np.all(img == [255, 255, 255, 0], axis=-1)
        img[mask] = [255, 255, 255, 255]
        palette = self.get_palette().split('\n')[1:]  # Skip the header
        palette = {line.split(',')[0]: tuple(int(line.split(',')[1][i:i + 2], 16) for i in (1, 3, 5, 7)) for line in
                   palette}  # Convert hex color to RGB
        image_data = np.empty((img.shape[0], img.shape[1]), dtype='U1')
        for key, color in palette.items():
            mask = np.all(img == color, axis=-1)
            image_data[mask] = key
        # Highlight the selected region
        x1, y1, x2, y2 = [int(coord) for coord in rect_coords]
        key = chr(97 + len(palette) - 1)  # The key for the new color in the palette
        image_data[y1:y2, x1:x2] = key
        return '\n'.join(''.join(row) for row in image_data)


def main():
    root = tk.Tk()
    img_path = 'images'  # replace with your image folder path
    img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
    app = Application(master=root, img_path=img_path, img_files=img_files)
    app.mainloop()


if __name__ == "__main__":
    main()
