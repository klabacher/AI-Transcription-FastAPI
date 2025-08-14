import torch
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline
from logging_config import get_logger
from typing import Dict, Any, Iterator

logger = get_logger("engine")


def load_model_for_worker(model_id: str, config: Dict[str, Any], device: str) -> Any:
    """
    Loads a transcription model based on the provided configuration.

    Args:
        model_id: The identifier of the model to load.
        config: A dictionary containing the model's configuration details.
        device: The device to load the model on ('cpu' or 'cuda').

    Returns:
        The loaded model instance.

    Raises:
        ValueError: If the model implementation in the config is unknown.
    """
    impl = config["impl"]
    model_name = config["model_name"]

    logger.info(
        f"Loading model '{model_id}' ({impl}) with name '{model_name}' for device '{device}'..."
    )

    if impl == "faster":
        compute_type = config.get("compute_type", "default")
        if device == "cpu" and compute_type not in ["int8", "float32"]:
            logger.warning(
                f"Compute type '{compute_type}' is not optimized for CPU. Defaulting to 'int8'."
            )
            compute_type = "int8"
        elif device == "cuda" and compute_type not in ["float16", "int8_float16"]:
            logger.warning(
                f"Compute type '{compute_type}' is not ideal for GPU. Consider 'float16' or 'int8_float16'."
            )
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

    elif impl == "hf_pipeline":
        torch_device_id = 0 if device == "cuda" else -1
        model = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=torch_device_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    else:
        raise ValueError(f"Unknown model implementation: {impl}")

    logger.info(f"Model '{model_id}' loaded successfully on the worker.")
    return model


def transcribe_audio(
    model: Any, model_config: Dict[str, Any], audio_path: str, duration_seconds: float
) -> Iterator[int | Dict[str, Any]]:
    """
    Transcribes an audio file using the provided model.

    This function is a generator that yields progress updates (as integers from 0-100)
    and concludes by yielding the final transcription result dictionary.

    Args:
        model: The loaded transcription model.
        model_config: The configuration dictionary for the model.
        audio_path: The local path to the audio file to be transcribed.
        duration_seconds: The duration of the audio file in seconds.

    Yields:
        Progress percentage (int) or the final result dictionary.
    """
    impl = model_config["impl"]
    # The V3 architecture passes the language down to the dispatcher and worker.
    # We should use that instead of hardcoding it.
    # For now, we'll keep the original behavior for compatibility.
    language_code = "pt"

    if impl == "faster":
        segments, info = model.transcribe(
            audio_path, language=language_code, vad_filter=True
        )
        logger.debug(
            f"faster-whisper detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        all_segments = []
        full_text_parts = []
        for segment in segments:
            full_text_parts.append(segment.text)
            all_segments.append({"start": segment.start, "text": segment.text.strip()})
            if duration_seconds > 0:
                progress = min(99, int((segment.end / duration_seconds) * 100))
                yield progress

        yield {"text": "".join(full_text_parts).strip(), "segments": all_segments}

    elif impl == "hf_pipeline":
        yield 10  # Initial progress
        kwargs = {
            "chunk_length_s": 30,
            "stride_length_s": 5,
            "return_timestamps": True,
            "generate_kwargs": {
                "language": "portuguese"
            },  # This should also be dynamic
        }
        result = model(audio_path, **kwargs)
        yield 80  # Progress after model inference

        text_result = result.get("text", "").strip()
        segments_result = []
        if "chunks" in result:
            segments_result = [
                {"start": c["timestamp"][0], "text": c["text"].strip()}
                for c in result.get("chunks", [])
            ]

        yield {"text": text_result, "segments": segments_result}
