import torch
import torch.nn as nn
import yaml
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from functools import partial
import lib.Equirec2Perspec as E2P
import lib.Perspec2Equirec as P2E
import lib.multi_Perspec2Equirec as m_P2E
from model import Model, generate_basic, generate_advanced

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


if __name__=='__main__':


    example1=[
        "A room with a sofa and coffee table for relaxing.",
        "A corner sofa is surrounded by plants.",
        "A comfy sofa, bookshelf, and lamp for reading.",
        "A bright room with a sofa, TV, and games.",
        "A stylish sofa and desk setup for work.",
        "A sofa, dining table, and chairs for gatherings.",
        "A colorful sofa, art, and music fill the room.",
        "A sofa, yoga mat, and meditation corner for calm."
    ]
    example2=[
        "A room with a sofa and coffee table for relaxing, cartoon style",
        "A corner sofa is surrounded by plants, cartoon style",
        "A comfy sofa, bookshelf, and lamp for reading, cartoon style",
        "A bright room with a sofa, TV, and games, cartoon style",
        "A stylish sofa and desk setup for work, cartoon style",
        "A sofa, dining table, and chairs for gatherings, cartoon style",
        "A colorful sofa, art, and music fill the room, cartoon style",
        "A sofa, yoga mat, and meditation corner for calm, cartoon style"
    ]

    example3=[
        "A room with a sofa and coffee table for relaxing, oil painting style",
        "A corner sofa is surrounded by plants, oil painting style",
        "A comfy sofa, bookshelf, and lamp for reading, oil painting style",
        "A bright room with a sofa, TV, and games, oil painting style",
        "A stylish sofa and desk setup for work, oil painting style",
        "A sofa, dining table, and chairs for gatherings, oil painting style",
        "A colorful sofa, art, and music fill the room, oil painting style",
        "A sofa, yoga mat, and meditation corner for calm, oil painting style"
    ]

    example4=[
        "A Japanese room with muted-colored tatami mats.",
        "A Japanese room with a simple, folded futon sits to one side.",
        "A Japanese room with a low table rests in the room's center.",
        "A Japanese room with Shoji screens divide the room softly.",
        "A Japanese room with An alcove holds an elegant scroll and flowers.",
        "A Japanese room with a tea set rests on a bamboo tray.",
        "A Japanese room with a carved wooden cupboard stands against a wall.",
        "A Japanese room with a traditional lamp gently lights the room."
    ]
    example6=[
        'This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop',
        'This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop',
        'This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop',
        'To the left of the island, a stainless-steel refrigerator stands tall. ',
        'To the left of the island, a stainless-steel refrigerator stands tall. ',
        'a sink surrounded by cabinets',
        'a sink surrounded by cabinets',
        'To the right of the sink, built-in wooden cabinets painted in a muted.'
    ]

    example7= [
        "Cobblestone streets curl between old buildings.",
        "Shops and cafes display signs and emit pleasant smells.",
        "A fruit market scents the air with fresh citrus.",
        "A fountain adds calm to one side of the scene.",
        "Bicycles rest against walls and posts.",
        "Flowers in boxes color the windows.",
        "Flowers in boxes color the windows.",
        "Cobblestone streets curl between old buildings."
    ]

    example8=[
        "The patio is open and airy.",
        "A table and chairs sit in the middle.",
        "Next the table is flowers.",
        "Colorful flowers fill the planters.",
        "A grill stands ready for barbecues.",
        "A grill stands ready for barbecues.",
        "The patio overlooks a lush garden.",
        "The patio overlooks a lush garden."
    ]

    example9=[
        "A Chinese palace with roofs curve.",
        "A Chinese palace, Red and gold accents gleam in the sun.",
        "A Chinese palace with a view of mountain in the front.",
        "A view of mountain in the front.",
        "A Chinese palace with a view of mountain in the front.",
        "A Chinese palace with a tree beside.",
        "A Chinese palace with a tree beside.",
        "A Chinese palace, with a tree beside."
    ]



    example_b1="This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop, a sink surrounded by cabinets. To the left of the island, a stainless-steel refrigerator stands tall. To the right of the sink, built-in wooden cabinets painted in a muted."
    example_b2="Bursting with vibrant hues and exaggerated proportions, the cartoon-styled room sparkled with whimsy and cheer, with floating shelves crammed with oddly shaped trinkets, a comically oversized polka-dot armchair perched near a gravity-defying, tilted lamp, and the candy-striped wallpaper creating a playful backdrop to the merry chaos, exuding a sense of fun and boundless imagination."
    example_b3="Bathed in the pulsating glow of neon lights that painted stark contrasts of shadow and color, the cyberpunk room was a high-tech, low-life sanctuary, where sleek, metallic surfaces met jagged, improvised tech; a wall of glitchy monitors flickered with unending streams of data, and the buzz of electric current and the low hum of cooling fans formed a dystopian symphony, adding to the room's relentless, gritty energy."
    example_b4="Majestically rising towards the heavens, the snow-capped mountain stood, its jagged peaks cloaked in a shroud of ethereal clouds, its rugged slopes a stark contrast against the serene azure sky, and its silent grandeur exuding an air of ancient wisdom and timeless solitude, commanding awe and reverence from all who beheld it."
    example_b5='Bathed in the soft, dappled light of the setting sun, the silent street lay undisturbed, revealing the grandeur of its cobblestone texture, the rusted lampposts bearing witness to forgotten stories, and the ancient, ivy-clad houses standing stoically, their shuttered windows and weather-beaten doors speaking volumes about their passage through time.'
    example_b6='Awash with the soothing hues of an array of blossoms, the tranquil garden was a symphony of life and color, where the soft murmur of the babbling brook intertwined with the whispering willows, and the iridescent petals danced in the gentle breeze, creating an enchanting sanctuary of beauty and serenity.'
    example_b7="Canopied by a patchwork quilt of sunlight and shadows, the sprawling park was a panorama of lush green grass, meandering trails etched through vibrant wildflowers, towering oaks reaching towards the sky, and tranquil ponds mirroring the clear, blue expanse above, offering a serene retreat in the heart of nature's splendor."

    examples_basic=[example_b1, example_b2, example_b3, example_b4, example_b5, example_b6]
    examples_advanced=[example1, example2, example3, example4, example6, example7, example8, example9]

    description="The demo generates 8 perspective images, with FOV of 90 and rotation angle of 45. Please type 8 sentences corresponding to each perspective image."

    outputs=[gr.Image(shape=(484, 2048))]
    outputs.extend([gr.Image(shape=(1, 1)) for i in range(8)])

    def load_example_img(path):
        img=Image.open(path)
        img.resize((1024, 242))
        return img

    def copy(text):
        return [text]*8

    def clear():
        return None, None, None, None, None, None, None, None, None

    def load_basic(example):
        return example

    default_text='This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop, a sink surrounded by cabinets. To the left of the island, a stainless-steel refrigerator stands tall. To the right of the sink, built-in wooden cabinets painted in a muted.'
    css = """
    #warning {background-color: #000000} 
    .feedback textarea {font-size: 16px !important}
    #foo {}
    .text111 textarea {
        color: rgba(0, 0, 0, 0.5);
    }
    """

    inputs=[gr.Textbox(type="text", label='Text{}'.format(i)) for i in range(8)]

    with gr.Blocks(css=css) as demo:
        
        with gr.Row():
            gr.Markdown(
            """
            # <center>Text2Pano with MVDiffusion</center>
            """)
        with gr.Row():
            gr.Markdown(
            """
            <center>Text2Pano demonstration: Write a scene you want in Text, then click "Generate panorama". Alternatively, you can load the example text prompts below to populate text-boxes. The advanced mode allows to specify text prompts for each perspective image. It takes 3 minitues to generate one panorama.</center>
            """)
        with gr.Row():
             gr.HTML("""
                <div style='text-align: center; font-size: 25px;'>
                    <a href='https://mvdiffusion.github.io/'>Project Page</a>
                </div>
                """)
        with gr.Tab("Basic"):
            with gr.Row():
                textbox1=gr.Textbox(type="text", label='Text', value=default_text, elem_id='warning', elem_classes="feedback")

            with gr.Row():
                submit_btn = gr.Button("Generate panorama")
                clear_btn = gr.Button("Clear all texts")
                clear_btn.click(
                    clear,
                    outputs=inputs+[textbox1]
                )

            with gr.Accordion("Expand/hide examples") as acc:
                for i in range(0, len(examples_basic)):
                    with gr.Row():
                        gr.Image(load_example_img('assets/basic/img{}.png'.format(i+1)), label='example {}'.format(i+1))
                        #gr.Image('demo/assets/basic/img{}.png'.format(i+2), label='example {}'.format(i+2))
                    with gr.Row():
                        gr.Textbox(type="text", label='Example text {}'.format(i+1), value=examples_basic[i])
                        #gr.Textbox(type="text", label='Example text {}'.format(i+2), value=examples_basic[i+1])
                    # with gr.Row():
                    #     load_btn=gr.Button("Load texts to the above box")
                    #     load_btn.click(
                    #         partial(load_basic, examples_basic[i]),
                    #         outputs=[textbox1]
                    #     )
                    gr.Row()
                    gr.Row()
            
                submit_btn.click(
                    partial(generate_basic, acc),
                    inputs=textbox1,
                    outputs=[acc]+outputs
                )
                
        with gr.Tab("Advanced"):
            with gr.Row():
                for text_bar in inputs[:4]:
                    text_bar.render()
            with gr.Row():
                for text_bar in inputs[4:]:
                    text_bar.render()
            
            with gr.Row():

                submit_btn = gr.Button("Generate panorama")
                clear_btn = gr.Button("Clear all texts")
                clear_btn.click(
                    clear,
                    outputs=inputs+[textbox1],
                    queue=True,
                )
            with gr.Accordion("Expand/hide examples") as acc_advanced:
                for i, example in enumerate(examples_advanced):
                    with gr.Row():
                        gr.Image(load_example_img('assets/advanced/img{}.png'.format(i+1)), label='example {}'.format(i+1))
                    with gr.Row():
                        gr.Textbox(type="text", label='Text 1', value=example[0])
                        gr.Textbox(type="text", label='Text 2', value=example[1])
                        gr.Textbox(type="text", label='Text 3', value=example[2])
                        gr.Textbox(type="text", label='Text 4', value=example[3])
                    with gr.Row():
                        gr.Textbox(type="text", label='Text 4', value=example[4])
                        gr.Textbox(type="text", label='Text 5', value=example[5])
                        gr.Textbox(type="text", label='Text 6', value=example[6])
                        gr.Textbox(type="text", label='Text 7', value=example[7])
                    # with gr.Row():
                    #     load_btn=gr.Button("Load text to other text boxes")
                    #     load_btn.click(
                    #         partial(load_basic, example),
                    #         outputs=inputs
                    #     )
                    gr.Row()
                    gr.Row()
                submit_btn.click(
                    partial(generate_advanced, acc_advanced),
                    inputs=inputs,
                    outputs=[acc_advanced]+outputs
                )

        with gr.Row():
            outputs[0].render()
        with gr.Row():
            outputs[1].render()
            outputs[2].render()
        with gr.Row():
            outputs[3].render()
            outputs[4].render()
        with gr.Row():
            outputs[5].render()
            outputs[6].render()
        with gr.Row():
            outputs[7].render()
            outputs[8].render()
        
    demo.queue()
    demo.launch()