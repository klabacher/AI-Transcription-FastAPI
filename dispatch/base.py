from abc import ABC, abstractmethod


class AbstractJobDispatcher(ABC):
    """
    Abstract base class for a job dispatcher.

    This interface defines the contract for dispatching transcription jobs.
    Concrete implementations will handle the specific logic for different
    execution backends (e.g., local multiprocessing or a distributed queue).
    """

    @abstractmethod
    async def dispatch(
        self,
        file_content: bytes,
        internal_path: str,
        job_id: str,
        language: str,
        model_config: dict,
    ) -> None:
        """
        Dispatches a job for transcription.

        Args:
            file_content: The binary content of the audio file.
            internal_path: The original filename or internal path of the file.
            job_id: The unique identifier for the job.
            language: The language for the transcription.
            model_config: The configuration dictionary for the selected model.
        """
        pass
