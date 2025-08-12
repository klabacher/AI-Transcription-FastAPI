import multiprocessing as mp
import logging
from logging_config import setup_worker_logging_json, get_logger
from engine import load_model, transcribe_with_faster, transcribe_with_hf_pipeline
import soundfile as sf

logger = get_logger('worker_pool')

def pool_worker_loop(task_queue, result_queue, model_id, model_cfg, device):
    setup_worker_logging_json()
    logger.info(f"Pool worker starting for model {model_id} on device {device}")
    try:
        model = load_model(model_id, model_cfg, device)
    except Exception as e:
        logger.critical(f"Failed to load model in pool worker: {e}")
        # workers that can't load should exit
        return

    while True:
        task = task_queue.get()
        if task is None:
            logger.info('Pool worker received shutdown signal')
            break
        job = task.get('job')
        audio_path = task.get('audio_path')
        job_id = job.get('id')
        try:
            duration = 0.0
            try:
                info = sf.info(audio_path)
                duration = float(info.duration or 0.0)
            except Exception:
                pass

            impl = model_cfg.get('impl')
            if impl == 'faster':
                gen = transcribe_with_faster(model, audio_path, job['config']['language'], duration)
            else:
                gen = transcribe_with_hf_pipeline(model, audio_path, job['config']['language'])

            collected_segments = []
            text_result = ''
            for item in gen:
                if isinstance(item, int):
                    # send progress update
                    result_queue.put({'type': 'PROGRESS', 'job_id': job_id, 'payload': int(item)})
                elif isinstance(item, dict) and 'segment' in item:
                    seg = item['segment']
                    collected_segments.append({'start': seg.get('start', 0), 'text': seg.get('text', '').strip()})
                elif isinstance(item, dict) and 'text' in item:
                    text_result = item.get('text', '').strip()
                    collected_segments = item.get('segments', collected_segments)

            result_queue.put({'type': 'RESULT', 'job_id': job_id, 'payload': {'text': text_result, 'segments': collected_segments, 'entities': []}})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Error processing job {job_id}: {e}\n{tb}")
            result_queue.put({'type': 'FATAL_ERROR', 'job_id': job_id, 'payload': {'error': str(e), 'traceback': tb}})

def start_pool(num_workers, model_id, model_cfg, device):
    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    for _ in range(num_workers):
        p = ctx.Process(target=pool_worker_loop, args=(task_queue, result_queue, model_id, model_cfg, device), daemon=True)
        p.start()
        workers.append(p)
    return task_queue, result_queue, workers
