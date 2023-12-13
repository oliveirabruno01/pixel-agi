import tkinter as tk
import jsonlines


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.data = []
        self.index = 0
        self.load_data()

    def create_widgets(self):
        self.user_label = tk.Label(self, text="User Content")
        self.user_label.grid(row=0, column=0)
        self.user_content = tk.Text(self, height=50, width=90)
        self.user_content.grid(row=1, column=0)

        self.assistant_label = tk.Label(self, text="Assistant Content")
        self.assistant_label.grid(row=0, column=1)
        self.assistant_content = tk.Text(self, height=50, width=90)
        self.assistant_content.grid(row=1, column=1)

        self.type_label = tk.Label(self, text="Type")
        self.type_label.grid(row=2, column=0)
        self.type_content = tk.Entry(self)
        self.type_content.grid(row=2, column=1)

        self.save_button = tk.Button(self, text="SAVE", command=self.save)
        self.save_button.grid(row=3, column=0)

        self.prev_button = tk.Button(self, text="PREVIOUS", command=self.previous)
        self.prev_button.grid(row=3, column=1)

        self.next_button = tk.Button(self, text="NEXT", command=self.next)
        self.next_button.grid(row=4, column=0)

    def load_data(self):
        with jsonlines.open('train.jsonl') as reader:
            self.data = list(reader)
        self.update_fields()

    def update_fields(self):
        self.user_content.delete("1.0", tk.END)
        self.assistant_content.delete("1.0", tk.END)
        self.type_content.delete(0, tk.END)

        self.user_content.insert(tk.END, self.data[self.index]['messages'][0][0]['content'])
        self.assistant_content.insert(tk.END, self.data[self.index]['messages'][0][1]['content'])
        self.type_content.insert(tk.END, self.data[self.index]['messages'][1])

    def save(self):
        self.data[self.index]['messages'][0][0]['content'] = self.user_content.get("1.0", tk.END).strip()
        self.data[self.index]['messages'][0][1]['content'] = self.assistant_content.get("1.0", tk.END).strip()
        self.data[self.index]['type'] = self.type_content.get()

        with jsonlines.open('train.jsonl', mode='w') as writer:
            writer.write_all(self.data)

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.update_fields()

    def next(self):
        if self.index < len(self.data) - 1:
            self.index += 1
            self.update_fields()


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
