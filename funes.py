from interface import setup_gradio

if __name__ == "__main__":
    demo = setup_gradio()
    demo.launch(share=False,
                debug=False,
                server_name="0.0.0.0",
                server_port=7680,
                ssl_verify=False,
                ssl_certfile="/home/julio/cert.pem",
                ssl_keyfile="/home/julio/key.pem")
