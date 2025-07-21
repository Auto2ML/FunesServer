import logging
import os
from interface import setup_gradio
from config import LOGGING_CONFIG

def setup_logging():
    """
    Configure the logging system according to the settings in LOGGING_CONFIG
    """
    # Create logs directory if logging to file and directory doesn't exist
    if LOGGING_CONFIG.get('file_path'):
        log_dir = os.path.dirname(LOGGING_CONFIG['file_path'])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure the root logger
    logging_level = LOGGING_CONFIG.get('level', logging.INFO)
    
    # Disable logging if specified in config
    if not LOGGING_CONFIG.get('enable', True):
        logging.disable(logging.CRITICAL)
        return
    
    # Configure basic logging with format and level
    logging.basicConfig(
        level=logging_level,
        format=LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s'),
        datefmt=LOGGING_CONFIG.get('date_format', '%Y-%m-%d %H:%M:%S'),
        filename=LOGGING_CONFIG.get('file_path'),  # None will log to console
    )
    
    # Log startup message
    logger = logging.getLogger('Funes')
    logger.info("Funes logging system initialized")
    logger.info(f"Logging level set to: {logging.getLevelName(logging_level)}")

if __name__ == "__main__":
    # Initialize logging system
    setup_logging()
    
    # Create logger for main module
    logger = logging.getLogger('Funes')
    logger.info("Starting Funes server...")
    
    # Initialize tools and ensure embeddings are ready
    try:
        import tools
        logger.info(f"Tools initialized: {', '.join(tools.get_all_tools())}")
    except Exception as e:
        logger.error(f"Error initializing tools: {str(e)}", exc_info=True)
    
    # Initialize and launch Gradio interface
    try:
        demo = setup_gradio()
        logger.info("Launching Gradio web interface on 0.0.0.0")
        demo.launch(server_name="0.0.0.0")
    except Exception as e:
        logger.error(f"Error launching Funes: {str(e)}", exc_info=True)
        raise
