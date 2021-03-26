from PIL import Image

from gui import AILangGUI
from models import *

CANVAS_SIZE = (2*256, 2*256)

if __name__ == '__main__':
    explain_bot_img = Image.open('assets/phil_robot.png').convert('RGBA')
    guess_bot_img = Image.open('assets/bill_robot.png').convert('RGBA')
    arrow_img = Image.open('assets/right_arrow.png').convert('RGBA')
    text_box_img = Image.open('assets/text_box.png').convert('RGBA')

    img_scale_factor = 0.4
    
    bot_width = int(img_scale_factor * explain_bot_img.size[0])
    bot_height = int(img_scale_factor * explain_bot_img.size[1])
    text_box_width = int(img_scale_factor * text_box_img.size[0])
    text_box_height = int(img_scale_factor * text_box_img.size[1])

    explain_bot_img = explain_bot_img.resize((bot_width, bot_height))
    guess_bot_img = guess_bot_img.resize((bot_width, bot_height))
    text_box_img = text_box_img.resize((text_box_width, text_box_height))
    
    ac = AgentController()

    gui = AILangGUI(explain_bot_img, guess_bot_img, text_box_img,
                    arrow_img, canvas_size=CANVAS_SIZE)
    gui.set_img_retrieval_function(ac.get_new_img)
    gui.set_img_gen_function(ac.step_gen_img)
    gui.mainloop()