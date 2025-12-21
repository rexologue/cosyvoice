"""Minimal client for the llm_bistream backend."""

import argparse
import asyncio
import uuid

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


def build_tensor(name, arr):
    return grpcclient.InferInput(name, arr.shape, np_to_triton_dtype(arr.dtype))


async def main(args):
    client = grpcclient.InferenceServerClient(url=args.url, verbose=args.verbose)
    session_id = str(uuid.uuid4())

    async def send(text, is_start=False, is_end=False):
        inputs = []
        sid = np.array([session_id], dtype=object)
        text_bytes = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        inputs.append(build_tensor("session_id", sid))
        inputs.append(build_tensor("text", text_bytes))
        inputs.append(build_tensor("is_start", np.array([is_start], dtype=np.bool_)))
        inputs.append(build_tensor("is_end", np.array([is_end], dtype=np.bool_)))
        inputs.append(build_tensor("streaming", np.array([True], dtype=np.bool_)))
        inputs.append(build_tensor("max_tokens", np.array([args.max_tokens], dtype=np.int32)))
        inputs.append(build_tensor("temperature", np.array([1.0], dtype=np.float32)))
        inputs.append(build_tensor("top_p", np.array([0.9], dtype=np.float32)))
        inputs.append(build_tensor("top_k", np.array([50], dtype=np.int32)))

        outputs = [grpcclient.InferRequestedOutput("speech_token_ids"), grpcclient.InferRequestedOutput("is_final"), grpcclient.InferRequestedOutput("finish_reason")]
        async for response in await client.async_stream_infer(args.model, inputs, outputs=outputs, sequence_id=int(uuid.uuid4().int % 2**31)):
            speech = response.as_numpy("speech_token_ids")
            final = response.as_numpy("is_final")[0]
            print("chunk", speech.tolist(), "final", final)
            if final:
                break

    await send(args.text, is_start=True, is_end=False)
    if args.extra_text:
        await asyncio.sleep(0.1)
        await send(args.extra_text, is_start=False, is_end=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--model", default="llm_bistream")
    parser.add_argument("--text", default="hello this is cosyvoice")
    parser.add_argument("--extra_text", default=" more words arrive later")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--verbose", action="store_true")
    asyncio.run(main(parser.parse_args()))

