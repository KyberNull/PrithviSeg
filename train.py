"""Training entrypoint wrapper."""

import signal

from config.shared import MODEL_PATH
from training.train import main
from utils import device_setup, handle_shutdown, setup_logging


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    device, pin_memory, amp_dtype = device_setup()
    main(device, MODEL_PATH, pin_memory, amp_dtype, logger=setup_logging())
