import time
import edgeiq
import threading
import collections
import queue


class CircularQueue:
    """Thread-safe circular queue using a deque."""
    def __init__(self, max_size=None):
        self._queue = collections.deque(maxlen=max_size)

    def put(self, item):
        self._queue.appendleft(item)

    def get(self):
        while True:
            try:
                return self._queue.pop()
            except IndexError:
                time.sleep(0.01)

    def get_nowait(self):
        try:
            return self._queue.pop()
        except IndexError:
            raise queue.Empty


class CameraThread(threading.Thread):
    def __init__(
            self, camera_idx, engine, model_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._camera_idx = camera_idx
        self._engine = engine
        self._model_id = model_id

        self._results_q = CircularQueue(2)
        self._stop_event = threading.Event()

        self._prev_results = None

        self._fps = edgeiq.FPS()

    def stop(self):
        print("[INFO] Stopping Camera {} thread...".format(self._camera_idx))
        self._stop_event.set()

    def get_results(self, wait=True):
        if wait is True:
            return self._results_q.get()
        else:
            try:
                results = self._results_q.get_nowait()
            except queue.Empty:
                results = self._prev_results

            self._prev_results = results
            return results

    def _run_detection(self):
        obj_detect = edgeiq.ObjectDetection(self._model_id)
        obj_detect.load(engine=self._engine)

        print("Loaded model:\n{}\n".format(obj_detect.model_id))
        print("Engine: {}".format(obj_detect.engine))
        print("Accelerator: {}\n".format(obj_detect.accelerator))
        print("Labels:\n{}\n".format(obj_detect.labels))

        with edgeiq.WebcamVideoStream(cam=self._camera_idx) as video_stream:
            # Allow Webcam to warm up
            time.sleep(2.0)
            self._fps.start()

            while True:
                frame = video_stream.read()

                if self._stop_event.is_set():
                    break

                results = obj_detect.detect_objects(frame, confidence_level=.5)

                frame = edgeiq.markup_image(
                        frame, results.predictions, colors=obj_detect.colors)

                output_results = {
                        "frame": frame,
                        "results": results,
                        "model_id": obj_detect.model_id
                        }
                self._results_q.put(output_results)

                self._fps.update()

    def run(self):
        try:
            self._run_detection()
        finally:
            self._fps.stop()
            print("Camera {} Exited".format(self._camera_idx))


def main():

    cam0_thread = CameraThread(0, edgeiq.Engine.DNN, "alwaysai/mobilenet_ssd")
    cam0_thread.start()

    try:
        with edgeiq.Streamer() as streamer:
            while True:
                cam0_results = cam0_thread.get_results()

                # Generate text to display on streamer
                if cam0_results is not None:
                    text = ["Camera 0:"]
                    text.append("Model: {}".format(cam0_results["model_id"]))
                    text.append("Inference time: {:1.3f} s".format(
                        cam0_results["results"].duration))
                    text.append("Objects:")

                    for prediction in cam0_results["results"].predictions:
                        text.append("{}: {:2.2f}%".format(
                            prediction.label, prediction.confidence * 100))

                    streamer.send_data(cam0_results["frame"], text)

                if streamer.check_exit():
                    break

    finally:
        cam0_thread.stop()
        cam0_thread.join()
        print("Program Ending")


if __name__ == "__main__":
    main()
