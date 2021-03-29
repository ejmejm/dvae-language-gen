import numpy as np
import pyttsx3
from subprocess import Popen
import time
import tkinter as tk
from PIL import Image, ImageTk


class AILangGUI(tk.Tk):
    def __init__(self, explain_char_img: Image, guess_char_img: Image,
                 text_box_img: Image, arrow_img: Image, *args,
                 canvas_size=(500, 500), **kwargs):
        super().__init__(*args, **kwargs)
        self.window_margin = 10
        self.small_padding = 10
        self.med_padding = 20
        
        self.title('Super Neato Language Gen AI')
        self.canvas_width = canvas_size[0]
        self.canvas_height = canvas_size[1]
        self.char_canvas_width = self.canvas_width + self.med_padding
        self.char_canvas_height = int(1.1 * explain_char_img.size[1])
        self.text_box_x_offset = int(0.73 * explain_char_img.size[0])
        self.text_box_y_offset = self.char_canvas_height - 0.57 * explain_char_img.size[1]
        self.text_box_len = int(0.95 * text_box_img.size[0])
        self.bg_color = '#1a1a1a'
        self.button_color = '#4275c7'
        self.text_size = 18
        
        self.speech_proc = None
        
        ### Convert image formats ###

        arrow_scale_factor = (self.canvas_width * 0.4) / arrow_img.size[0]
        arrow_width = int(arrow_scale_factor * arrow_img.size[0])
        arrow_height = int(arrow_scale_factor * arrow_img.size[1])
        
        self.arrow_img = ImageTk.PhotoImage(arrow_img.resize((arrow_width, arrow_height)))
        self.explain_tk_img = ImageTk.PhotoImage(explain_char_img)
        self.guess_tk_img = ImageTk.PhotoImage(guess_char_img)
        self.text_box_img = ImageTk.PhotoImage(text_box_img)

        ### Adding image canvases ###

        # Canvas for the real img
        self.truth_canvas = tk.Canvas(
            self,
            width = self.canvas_height,
            height = self.canvas_width,
            bg = 'black',
            bd = 0,
            highlightthickness = 1)
        
        # Canvas for generated imgs
        self.gen_canvas = tk.Canvas(
            self,
            width = self.canvas_height,
            height = self.canvas_width,
            bg = 'black',
            bd = 0,
            highlightthickness = 1)
        
        # Adding arrow
        self.arrow_canvas = tk.Canvas(
            self,
            width = arrow_width,
            height = arrow_height,
            bg = self.bg_color,
            highlightthickness = 0)
        self.arrow_canvas.create_image(0, 0, anchor='nw', image=self.arrow_img)
        
        ### Adding buttons ###

        # Button to move to the next sample
        self.next_button = tk.Button(self, text='New Image',
            command=self.next_truth_img_event, bg=self.button_color,
            relief=tk.FLAT)
        
        # Button to generate a new image
        self.explain_button = tk.Button(self, text='Explain Image',
            command=self.next_gen_img_event, bg=self.button_color,
            relief=tk.FLAT)

        ### Adding character imgs ###

        # Canvas for the real img
        self.explain_char_canvas = tk.Canvas(
            self,
            width = self.char_canvas_width,
            height = self.char_canvas_height,
            bg = self.bg_color,
            highlightthickness = 0)
        
        # Canvas for generated imgs
        self.guess_char_canvas = tk.Canvas(
            self,
            width = self.char_canvas_width,
            height = self.char_canvas_height,
            bg = self.bg_color,
            highlightthickness = 0)
        
        self.text_label = None
        self.guess_label = None
        
        ### Place created items ###
        
        curr_x = self.window_margin
        curr_y = self.window_margin

        arrow_level_y = curr_y + self.canvas_height // 2 - arrow_height // 2
        button_level_y = curr_y + self.canvas_height + self.small_padding
        char_level_y = button_level_y + self.med_padding + self.small_padding
        
        self.truth_canvas.place(x=curr_x, y=curr_y)
        self.explain_char_canvas.place(x=curr_x, y=char_level_y)
        self.next_button.place(x=curr_x, y=button_level_y)
        
        curr_x += self.canvas_width + self.med_padding
        
        self.arrow_canvas.place(x=curr_x, y=arrow_level_y)
        
        curr_x += arrow_width + self.med_padding
        
        self.gen_canvas.place(x=curr_x, y=self.window_margin)
        self.guess_char_canvas.place(x=curr_x, y=char_level_y)
        self.explain_button.place(x=curr_x, y=button_level_y)
        
        curr_x += self.canvas_width + self.window_margin + 4
        curr_y = char_level_y + self.med_padding + self.char_canvas_height
        
        self.text_box_x = self.window_margin + self.text_box_x_offset + 7
        self.text_box_y = char_level_y + int(0.118 * self.char_canvas_height)
        
        self.guess_box_x = self.window_margin + self.text_box_x_offset + \
                arrow_width + self.canvas_width + 2 * self.med_padding + 150
        self.guess_box_y = char_level_y + int(0.235 * self.char_canvas_height)

        self.draw_explainer(None)
        self.draw_guesser()
        
        self.geometry('{}x{}'.format(curr_x, curr_y))
        self.configure(bg=self.bg_color)
        

    def set_img_retrieval_function(self, func):
        self.get_next_true_img = func
        
    def set_img_gen_function(self, func):
        self.get_next_gen_img = func
        
    def next_truth_img_event(self):
        img = self.get_next_true_img()
        img = img.resize((self.canvas_width, self.canvas_height))
        self.curr_real_img = ImageTk.PhotoImage(img)
        self.curr_gen_img = None
        self.truth_canvas.create_image(
            self.canvas_width // 2 + 1,
            self.canvas_height // 2 + 1,
            anchor='center',
            image=self.curr_real_img)

    def next_gen_img_event(self):
        img, text = self.get_next_gen_img()
        img = img.resize((self.canvas_width, self.canvas_height))
        self.next_gen_img = ImageTk.PhotoImage(img)
        self.animate_explainer(text)
        
    def display_next_gen_img(self):
        self.curr_gen_img = self.next_gen_img
        self.gen_canvas.create_image(
            self.canvas_width // 2 + 1,
            self.canvas_height // 2 + 1,
            anchor='center',
            image=self.curr_gen_img)
    
    def animate_explainer(self, full_text, curr_text=None, speed=20):
        delay = 1 / speed
        
        if curr_text is None:
            curr_text = ''
        
        if curr_text == '':
            self.speech_proc = Popen('python speak.py "' + full_text + '"')
        
        if curr_text == full_text:
            if self.speech_proc:
                self.speech_proc.wait()
            self.after(500, lambda: self.animate_guesser(['???', '!']))
            self.after(6150, self.draw_explainer)
        else:
            curr_text = full_text[:len(curr_text)+1]
            self.draw_explainer(curr_text)
            next_func = lambda: self.animate_explainer(full_text, curr_text, speed)
            self.after(int(1000*delay), next_func)

    def animate_guesser(self, full_text, curr_text=None, speed=1.5):
        delay = 1 / speed
        
        if isinstance(full_text, str):
            full_text = [full_text]
        if isinstance(curr_text, str):
            curr_text = [curr_text]
        if curr_text is None or len(curr_text) == 0:
            curr_text = ['']
        
        if curr_text[0] == full_text[0]:
            # All TTS finished
            if len(full_text) <= 1:
                self.display_next_gen_img()
                self.after(2000, self.draw_guesser)
            else:
                next_func = lambda: self.animate_guesser(full_text[1:], curr_text[1:], speed)
                self.after(1000, next_func)
        else:
            curr_text[0] = full_text[0][:len(curr_text[0])+1]
            self.draw_guesser(curr_text[0])
            next_func = lambda: self.animate_guesser(full_text, curr_text, speed)
            self.after(int(1000*delay), next_func)
    
    def draw_explainer(self, text=None):
        self.explain_char_canvas.delete('all')
        if self.text_label:
            self.text_label.destroy()
        self.explain_char_canvas.create_image(0, self.char_canvas_height, anchor='sw', image=self.explain_tk_img)
        
        if text is not None:
            self.explain_char_canvas.create_image(self.text_box_x_offset, self.text_box_y_offset,
                                                  anchor='sw', image=self.text_box_img)
            
            self.text_label = tk.Label(self, text=text, bg='white', font=('Helvetica', self.text_size), wraplength=self.text_box_len)
            self.text_label.place(x=self.text_box_x, y=self.text_box_y)
    
    def draw_guesser(self, text=None):
        self.guess_char_canvas.delete('all')
        if self.guess_label:
            self.guess_label.destroy()
        self.guess_char_canvas.create_image(0, self.char_canvas_height, anchor='sw', image=self.guess_tk_img)
        
        if text is not None:
            self.guess_char_canvas.create_image(self.text_box_x_offset, self.text_box_y_offset,
                                                  anchor='sw', image=self.text_box_img)
            
            self.guess_label = tk.Label(self, text=text, bg='white', font=('Helvetica', self.text_size), wraplength=self.text_box_len)
            self.guess_label.place(x=self.guess_box_x, y=self.guess_box_y, anchor='center')
        
if __name__ == '__main__':
    explain_bot_img = Image.open('assets/phil_robot.png')
    guess_bot_img = Image.open('assets/bill_robot.png')

    bot_width = int(0.4 * explain_bot_img.size[0])
    bot_height = int(0.4 * explain_bot_img.size[1])

    explain_bot_img = explain_bot_img.resize((bot_width, bot_height))
    guess_bot_img = guess_bot_img.resize((bot_width, bot_height))

    gui = AILangGUI(explain_bot_img, guess_bot_img, canvas_size=(224*2, 224*2))
    gui.set_img_retrieval_function(get_next_true_img)
    gui.mainloop()
