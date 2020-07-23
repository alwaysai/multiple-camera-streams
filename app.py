import time
import edgeiq
import threading
import collections
import queue
import numpy as np


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
    """
    Performs Object Detection on a camera video stream.

    This class loads a model on a different thread, where it
    reads camera frames as they come in and performs inferencing.
    This enables the main app to get results from a camera in an
    asyncronous manner.
    """
    def __init__(
            self, camera_idx, engine, model_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.idx = camera_idx
        self._engine = engine
        self._model_id = model_id

        self._results_q = CircularQueue(2)
        self._stop_event = threading.Event()

        self._prev_results = None

        self._fps = edgeiq.FPS()

    def stop(self):
        print("Stopping Camera {} thread...".format(self.idx))
        self._stop_event.set()

    def get_results(self, wait=True):
        """
        Get the latest detection from the camera.

        When wait is set to `False` and there are no new results available
        the previous results are returned.
        """
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

        with edgeiq.WebcamVideoStream(cam=self.idx) as video_stream:
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
                        "idx": self.idx,
                        "frame": frame,
                        "results": results,
                        "model_id": obj_detect.model_id
                        }
                self._results_q.put(output_results)

                self._fps.update()

    def run(self):
        try:
            print("Camera {} starting...".format(self.idx))
            self._run_detection()
        finally:
            self._fps.stop()
            print("Camera {}: elapsed time: {:.2f}".format(
                self.idx, self._fps.get_elapsed_seconds()))
            print("Camera {}: approx. FPS: {:.2f}".format(
                self.idx, self._fps.compute_fps()))
            print("Camera {} Exited".format(self.idx))


def main():

    cameras = []
    # This is the list of camera indices to use. If you increase the length of this list,
    # you'll also need to update the `concatenate` step for displaying the frames.
    camera_idxs = [0, 1]
    for i in camera_idxs:
        cameras.append(CameraThread(i, edgeiq.Engine.DNN, "alwaysai/mobilenet_ssd"))

    for c in cameras:
        c.start()

    try:
        with edgeiq.Streamer() as streamer:
            while True:
                results = []
                for c in cameras:
                    results.append(c.get_results())

                # Generate text to display on streamer
                text = []
                for r in results:
                    if r is not None:
                        text.append("Camera {}:".format(r["idx"]))
                        text.append("Model: {}".format(r["model_id"]))
                        text.append("Inference time: {:1.3f} s".format(r["results"].duration))
                        text.append("Objects:")

                        for prediction in r["results"].predictions:
                            text.append("{}: {:2.2f}%".format(
                                prediction.label, prediction.confidence * 100))

                # Join the incoming frames vertically into a single image to be shown on
                # the Streamer
                frame = np.concatenate((results[0]["frame"], results[1]["frame"]), axis=0)

                streamer.send_data(frame, text)

                if streamer.check_exit():
                    break

    finally:
        for c in cameras:
            c.stop()
            c.join()
        print("Program Ending")


if __name__ == "__main__":
    main()
