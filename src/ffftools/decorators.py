from functools import wraps
from .utils import has_valid_name, is_mandatory
import logging
import time


# Try importing colorlog, and if it fails, fall back to standard logging
try:
    import colorlog
    colorlog_available = True
except ImportError:
    colorlog_available = False

# Set up a standard log formatter
log_formatter = logging.Formatter("%(asctime)s [ %(levelname)s ] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Default to standard logging if colorlog is unavailable
if colorlog_available:
    # Set up a colored log formatter
    log_formatter = colorlog.ColoredFormatter(
        "%(asctime)s [%(log_color)s%(levelname)s%(reset)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'magenta',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red'
        }
    )

# Set up the console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Get the root logger and set the level to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

def log_and_time(func):
    """Logs and times the execution of a function."""
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        logging.info(f"Starting {func.__name__}...")
        start_time = time.time()
        result = func(df, *args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"Finished {func.__name__} after {elapsed_time:.4f} seconds.")
        return result
    return wrapper

def check_protection(*colnames):
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            protected_cols = [col for col in colnames if '!' + col in df.columns]
            
            if protected_cols:
                logging.info(f"Attempted to compute {protected_cols}, but protected custom columns were found. Skipping computation!")
                return df  # Skip computation and return the original df
            
            return func(df, *args, **kwargs)  # Call the function normally if not protected
        return wrapper
    return decorator

# Global registry to track functions that compute specific columns
COLUMN_FUNCTION_MAP = {}
def register_computation(*output_columns):
    """Registers a function that computes one or more columns passed as separate arguments."""
    def decorator(func):
        for col in output_columns:
            COLUMN_FUNCTION_MAP[col] = func  # Register each column
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            return func(df, *args, **kwargs)
        return wrapper
    return decorator

def requires(*required_columns):
    """Decorator to check and compute required columns if missing."""
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            for col in required_columns:
                if col not in df.columns:
                    logging.info(f"Column '{col}' is missing. Computing it first...")
                    if col in COLUMN_FUNCTION_MAP:
                        df = COLUMN_FUNCTION_MAP[col](df)  # Compute required column
                    else:
                        if not has_valid_name(col):
                            raise ValueError(f"Column '{col}' has not a valid fff column name!")
                        if is_mandatory(col):
                            raise ValueError(f"Column '{col}' is missing in the dataset! Mandatory columns can't be computed!")
                        raise ValueError(f"No registered function to compute '{col}'")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator


def experimental(func):
    """Decorator that logs a warning when the function is used."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.warning(f"{func.__name__} is experimental!") 
        return func(*args, **kwargs)
    return wrapper