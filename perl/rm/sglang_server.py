import logging
from typing import Any, Dict, Optional

import sglang as sgl

logger = logging.getLogger("AllInOne-RM")


class SGLangManager:
    """Lifecycle manager for the SGLang offline engine."""

    def __init__(self, config):
        self.config = config
        self.engine = None

    def start(self):
        """Start the SGLang offline engine."""
        logger.info(f"🚀 Starting SGLang offline engine (Model: {self.config.model_path})...")
        self.engine = sgl.Engine(
            model_path=self.config.model_path,
            tp_size=self.config.sglang_tp_size,
            dp_size=self.config.sglang_dp_size,
            trust_remote_code=self.config.sglang_trust_remote_code,
        )

    async def wait_until_ready(self):
        """No-op for offline engine (kept for lifecycle compatibility)."""
        return

    def stop(self):
        """Shutdown the SGLang offline engine."""
        if self.engine:
            logger.info("🛑 Shutting down SGLang offline engine...")
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"Shutdown issue (may already be closed): {e}")
            logger.info("👋 SGLang offline engine stopped.")

    def generate(self, prompt: str, sampling_params: Optional[Dict[str, Any]] = None) -> str:
        """Run a single-prompt generation on the offline engine."""
        if not self.engine:
            raise RuntimeError("SGLang engine is not initialized.")
        params = sampling_params or {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 128}
        outputs = self.engine.generate([prompt], params)
        return outputs[0].get("text", "").strip()
