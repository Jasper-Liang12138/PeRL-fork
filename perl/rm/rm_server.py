import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .math_verifier import extract_boxed_answer, compute_score
from .sglang_server import SGLangManager
from .config import RMConfig

logger = logging.getLogger("AllInOne-RM")

CONFIG = RMConfig()
sglang_manager = SGLangManager(CONFIG)
OUTPUT_DIR: Path | None = None

# Concurrency limiter to prevent overwhelming SGLang.
MAX_CONCURRENT_REQUESTS = 16
sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Timeout for each extraction call (seconds).
EXTRACTION_TIMEOUT = 30


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: boot SGLang engine.
    sglang_manager.start()
    await sglang_manager.wait_until_ready()
    yield
    # Shutdown: terminate SGLang engine.
    sglang_manager.stop()


app = FastAPI(title="All-in-One RM Server", lifespan=lifespan)


class RewardRequest(BaseModel):
    prompt: str
    response: str
    label: str
    metadata: dict | None = None


async def call_qwen_extractor(text: str) -> str:
    """Call the offline engine for answer extraction."""
    extraction_prompt = (
        "You are a math answer extractor. Extract the final answer. "
        "Output ONLY the answer itself (number/expression). "
        "If possible, use \\boxed{...}. Do NOT output any explanation.\n"
        "中文提示：只输出答案本身，不要输出多余文字。\n\n"
        f"Text:\n{text}"
    )
    messages = [{"role": "user", "content": extraction_prompt}]
    return await sglang_manager.async_generate_chat(messages)


async def clean_extracted_answer(text: str) -> str:
    """Ask the engine to normalize an extracted answer to a bare value."""
    cleanup_prompt = (
        "Normalize the following to ONLY the final answer. "
        "Output just the answer (number/expression), no extra words.\n"
        "中文提示：只输出答案本身。\n\n"
        f"Text:\n{text}"
    )
    messages = [{"role": "user", "content": cleanup_prompt}]
    return await sglang_manager.async_generate_chat(messages)


async def async_save_log(
    req: RewardRequest,
    final_ans: str | None,
    score: float,
    rm_type: str,
    metadata: dict,
):
    """Persist request/response log asynchronously to avoid blocking event loop."""
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = OUTPUT_DIR / f"{timestamp}_{uuid4().hex}.json"
        payload = {
            "timestamp": timestamp,
            "rm_type": rm_type,
            "prompt": req.prompt,
            "response": req.response,
            "label": req.label,
            "metadata": metadata,
            "extracted": final_ans,
            "score": score,
        }
        content = json.dumps(payload, ensure_ascii=False)
        await anyio.to_thread.run_sync(log_path.write_text, content, "utf-8")
    except Exception as e:
        logger.error(f"Failed to save log: {e}")


@app.post("/reward")
async def calculate_reward(req: RewardRequest):
    async with sem:  # Limit concurrent requests.
        try:
            metadata = req.metadata or {}
            rm_type = metadata.get("rm_type", "math")
            final_ans = None
            score = 0.0

            if rm_type == "math":
                # Validate response format: must contain exactly one </think>.
                if not ("</think>" in req.response and req.response.count("</think>") == 1):
                    return 0.0

                response_after_think = req.response.split("</think>")[1].strip()

                # Wrap extraction calls with timeout protection.
                with anyio.fail_after(EXTRACTION_TIMEOUT):
                    qwen_res = await call_qwen_extractor(response_after_think)
                    if qwen_res:
                        if "\\boxed" in qwen_res:
                            final_ans = extract_boxed_answer(qwen_res)
                        else:
                            cleaned = await clean_extracted_answer(qwen_res)
                            final_ans = cleaned.strip() if cleaned else None

                score = compute_score(final_ans, req.label) if final_ans else 0.0

            else:
                logger.error(f"Unsupported RM type: {rm_type}")

            # Lightweight console logging.
            label_preview = req.label[:20] if req.label else ""
            logger.info(f"GT: {label_preview}... | Extracted: {final_ans} | Score: {score}")

            # Fire-and-forget async log persistence.
            if OUTPUT_DIR is not None:
                asyncio.create_task(async_save_log(req, final_ans, score, rm_type, metadata))

            return score

        except TimeoutError:
            logger.error("Reward calculation timed out!")
            return 0.0
        except Exception as e:
            logger.error(f"Reward Server Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "running"}


def run_rm_server():
    parser = argparse.ArgumentParser(description="All-in-One RM Server")
    parser.add_argument("--model-path", default=RMConfig.model_path)
    parser.add_argument("--tp-size", type=int, default=RMConfig.sglang_tp_size)
    parser.add_argument("--dp-size", type=int, default=RMConfig.sglang_dp_size)
    parser.add_argument("--rm-server-port", type=int, default=RMConfig.rm_server_port)
    parser.add_argument("--rm-host", default="0.0.0.0")
    parser.add_argument("--output-dir", default="rm_logs")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent extraction requests")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per extraction call (seconds)")
    args = parser.parse_args()

    # Initialize globals.
    global CONFIG, sglang_manager, OUTPUT_DIR, sem, MAX_CONCURRENT_REQUESTS, EXTRACTION_TIMEOUT
    CONFIG = RMConfig(
        model_path=args.model_path,
        sglang_tp_size=args.tp_size,
        sglang_dp_size=args.dp_size,
        rm_server_port=args.rm_server_port,
    )
    sglang_manager = SGLangManager(CONFIG)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    EXTRACTION_TIMEOUT = args.timeout

    print("🔥 Starting All-in-One RM service...")
    print(f"👉 HTTP port: {CONFIG.rm_server_port}")
    print(f"👉 Max concurrent: {MAX_CONCURRENT_REQUESTS}, Timeout: {EXTRACTION_TIMEOUT}s")

    uvicorn.run(app, host=args.rm_host, port=CONFIG.rm_server_port)


if __name__ == "__main__":
    run_rm_server()
