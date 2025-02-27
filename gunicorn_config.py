# Gunicorn configuration file for memory optimization on Render

# Use a single worker for memory efficiency
workers = 1

# Use threads for concurrent requests
threads = 2

# Use the gthread worker class for better memory management
worker_class = 'gthread'

# Set temporary directory
worker_tmp_dir = '/tmp'

# Set a larger timeout for LLM processing
timeout = 120

# Recycle workers periodically to prevent memory leaks
max_requests = 50
max_requests_jitter = 10

# Log level
loglevel = 'info'

# Preload app for faster startup
preload_app = True

# Keep the server alive even without requests
keepalive = 5

# Configure access logging
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# Error log
errorlog = '-'

# Capture output
capture_output = True