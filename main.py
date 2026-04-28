"""
ANPR (Automatic Number Plate Recognition) System
Entry point — loads config, wires up components, starts pipeline.
"""

import signal
import sys

from anpr.config import Config
from anpr.pipeline import ANPRPipeline
from anpr.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    config = Config.from_file("config.yaml")
    pipeline = ANPRPipeline(config)

    def _shutdown(signum, frame):  # noqa: ANN001
        log.info("Shutdown signal received. Stopping pipeline…")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Starting ANPR pipeline")
    pipeline.run()


if __name__ == "__main__":
    main()
