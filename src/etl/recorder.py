import os

import numpy as np

import queue

import logging
from utils import Coordinates, ImageType
from monitor_manager import MonitorManager
from etl.model.classifier import Classifier
import time
from pathlib import Path


class EarthRecorder:
    def __init__(
        self,
        num_of_batch_to_collect: int,
        offset: int = 30,
        screenshot_width: int = 512,
        screenshot_height: int = 512,
        batch_save_size: int = 64,
        classifier: Classifier | None = None,
        delete_intermediate_saves: bool = True,
    ):
        self.num_of_batch_to_collect = num_of_batch_to_collect
        self.offset = offset
        self.screenshot_width = screenshot_width
        self.screenshot_height = screenshot_height
        self.batch_save_size = batch_save_size
        self.delete_intermediate_saves = delete_intermediate_saves
        self.classifier = classifier

        self.logger = logging.getLogger("EarthRecorder")

        self.non_visited_coords: queue.Queue[Coordinates] = queue.Queue()
        self.initial_coords: Coordinates = Coordinates(0, 0)
        self.processed: set[Coordinates] = set()
        self.processed.add(self.initial_coords)
        self.filepath_processed: list[Path] = []
        self.total_processed_sample: int = 0

        self.monitor_manager = MonitorManager()

    def _get_neighbour_coords(self, current_coord: Coordinates):
        x, y = current_coord.x, current_coord.y
        return [
            Coordinates(x, y + self.offset),
            Coordinates(x + self.offset, y),
            Coordinates(x - self.offset, y),
            Coordinates(x, y - self.offset),
        ]

    def _update_neighbour_and_processed_coords(self, current_coord: Coordinates):
        neighbour_coords = self._get_neighbour_coords(current_coord)
        for neighbour_coord in neighbour_coords:
            if neighbour_coord not in self.processed:
                self.non_visited_coords.put(neighbour_coord)
                self.processed.add(neighbour_coord)

    def _process_predictions(
        self, images: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        mask = predictions > 0.5
        return images[mask]

    def _clean_collected_data(self) -> None:
        datas = []

        for filepath in self.filepath_processed:
            try:
                data = np.load(filepath)
            except OSError:
                logging.warning(
                    "Expected npy file <%s> where no one was found", filepath
                )
                continue
            datas.append(data)
            if self.delete_intermediate_saves:
                try:
                    os.remove(filepath)
                except FileNotFoundError:
                    logging.warning(
                        "File <%s> not found for deletion, has supposed to exists",
                        filepath,
                    )

        if len(datas) == 0:
            return None

        concatenated_datas = np.concatenate(datas, axis=0)

        output_filepath = (
            Path().absolute().parent
            / "data"
            / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.npy"
        )

        np.save(output_filepath, concatenated_datas)
        self.logger.info(
            "Process finished, processed <%s> samples, data concatenated into <%s>",
            self.total_processed_sample,
            "/".join(str(output_filepath).split("/")[-5:]),
        )

    def record(self) -> None:
        current_coords = self.initial_coords
        for i in range(1, self.num_of_batch_to_collect + 1):
            self.logger.info("Batch <%s> started", i)
            images = np.empty(
                shape=(
                    self.batch_save_size,
                    self.screenshot_width,
                    self.screenshot_height,
                    3,
                ),
                dtype=np.uint8,
            )
            for img_idx in range(self.batch_save_size):
                self._update_neighbour_and_processed_coords(current_coords)
                images[
                    img_idx, :, :, :
                ] = self.monitor_manager.catpure_partial_screen_from_middle(
                    self.screenshot_width,
                    self.screenshot_height,
                    output_format=ImageType.NUMPY,
                )

                next_coords = self.non_visited_coords.get()

                self.monitor_manager.move(current_coords, next_coords)
                current_coords = next_coords
            if self.classifier is not None:
                predictions = self.classifier.predict(images)
                images = self._process_predictions(images, predictions)

            batch_save_filepath = (
                Path().absolute()
                / "temporary_data_save"
                / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_batch_{i}.npy"
            )

            if images.shape[0] != 0:
                np.save(batch_save_filepath, images)
                self.logger.info(
                    "Batch <%s> finished, keep %s/%s pictures, save it to %s",
                    i,
                    images.shape[0],
                    self.batch_save_size,
                    "/".join(str(batch_save_filepath).split("/")[-5:]),
                )
                self.filepath_processed.append(batch_save_filepath)
            else:
                self.logger.info(
                    "Batch <%s> finished, No image keep after model prediction, no saved file been created",
                    i,
                )

            self.total_processed_sample += images.shape[0]

        self._clean_collected_data()


if __name__ == "__main__":
    recorder = EarthRecorder(num_of_batch_to_collect=2, batch_save_size=5)
    time.sleep(5)
    recorder.record()
