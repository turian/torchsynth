"""
Default parameters
"""
DEFAULT_SAMPLE_RATE = 44100

# Modulation signals are computed at a lower sample rate
DEFAULT_CONTROL_RATE = 441

# Number of samples for fixed length synthesis.
# 4 seconds by default
DEFAULT_BUFFER_SIZE = 4 * DEFAULT_SAMPLE_RATE

# Default batch size for rendering sounds.
DEFAULT_BATCH_SIZE = 64

# Small value to avoid log errors and divide by zero
# NB I think for logs I've seen either 1e-4 or 1e-8
EPS = 1e-6

# # Equal power coefficient. 1/sqrt(2)
# EQ_POW = 0.70710678118
