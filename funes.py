from interface import setup_gradio

if __name__ == "__main__":
    demo = setup_gradio()
    demo.launch(server_name="0.0.0.0")
