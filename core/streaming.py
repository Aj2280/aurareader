import threading
import queue
from transformers import TextStreamer

# Global registry mapping threads to their respective output queues
_THREAD_QUEUES = {}
_REGISTRY_LOCK = threading.Lock()

class QueueStreamer(TextStreamer):
    """Subclass of TextStreamer that puts finalized text into a thread-safe Queue instead of printing to stdout."""
    def __init__(self, tokenizer, q, **kwargs):
        # Pass skip_prompt=True to only stream generated response tokens
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.q = q

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text:
            self.q.put(text)

def register_thread_queue(thread_obj, q):
    with _REGISTRY_LOCK:
        _THREAD_QUEUES[thread_obj] = q

def unregister_thread_queue(thread_obj):
    with _REGISTRY_LOCK:
        _THREAD_QUEUES.pop(thread_obj, None)

def get_current_thread_queue():
    with _REGISTRY_LOCK:
        return _THREAD_QUEUES.get(threading.current_thread())

def patch_model_generation(model, tokenizer):
    """
    Patch the generate method of the model instance.
    If a Queue is registered for the current calling thread, we swap out
    the streamer with a QueueStreamer pointing to that queue.
    """
    original_generate = model.generate

    def patched_generate(*args, **kwargs):
        q = get_current_thread_queue()
        if q is not None:
            # Swap with our QueueStreamer to stream directly to queue
            kwargs["streamer"] = QueueStreamer(tokenizer, q)
        return original_generate(*args, **kwargs)

    model.generate = patched_generate
