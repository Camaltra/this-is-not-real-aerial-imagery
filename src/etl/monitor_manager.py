import numpy as np
import pyautogui

from screeninfo import get_monitors
from screeninfo.common import Monitor
import logging
from PIL import Image
from exception import UndetectedMonitor, UndetectedPrimaryMonitor
from utils import ImageType, Coordinates


logging.basicConfig(level=logging.INFO)


class MonitorManager:
    def __init__(self):
        self.logger = logging.getLogger("MonitorManager")
        monitor_width, monitor_height = self._get_monitor_size_info()
        self.logger.info(
            "Screen info: (width<%s>, height<%s>)", monitor_width, monitor_height
        )
        self.monitor_width = monitor_width
        self.monitor_height = monitor_height

    def _get_monitor_size_info(self) -> tuple[int, int]:
        """
        Get the monitor size information. It only retrive the primary screen size
        :return: (Width, Height)
        """
        monitors: list[Monitor] = get_monitors()
        if len(monitors) == 0:
            self.logger.warning("No monitor detected, abort...")
            raise UndetectedMonitor
        primary_monitor = next(
            iter([monitor for monitor in monitors if monitor.is_primary]), None
        )
        if primary_monitor is None:
            self.logger.warning("No primary monitor detected, abort...")
            raise UndetectedPrimaryMonitor

        return primary_monitor.width * 2, primary_monitor.height * 2

    def _capture_screen(self) -> Image.Image:
        self.logger.debug("Screen capture performed")
        return pyautogui.screenshot().convert("RGB")

    def catpure_partial_screen_from_middle(
        self, width: int, height: int, output_format: ImageType
    ) -> Image.Image | np.ndarray:
        left = (self.monitor_width - width) / 2
        top = (self.monitor_height - height) / 2
        right = (self.monitor_width + width) / 2
        bottom = (self.monitor_height + height) / 2

        screen_capture = self._capture_screen()
        cropped_screen_capture = screen_capture.crop((left, top, right, bottom))
        if output_format == ImageType.NUMPY:
            return np.array(cropped_screen_capture)

        return cropped_screen_capture

    def move(self, current_coords: Coordinates, next_coords: Coordinates) -> None:
        horizontal_movement, vertical_movement = current_coords - next_coords

        self.logger.debug(
            f"Moving from (%s, %s) to (%s, %s)",
            current_coords.x,
            current_coords.y,
            next_coords.x,
            next_coords.y,
        )

        if horizontal_movement > 0:
            pyautogui.press("right", presses=abs(horizontal_movement), interval=0.0)
        elif horizontal_movement < 0:
            pyautogui.press("left", presses=abs(horizontal_movement), interval=0.0)
        if vertical_movement > 0:
            pyautogui.press("up", presses=abs(vertical_movement), interval=0.0)
        elif vertical_movement < 0:
            pyautogui.press("down", presses=abs(vertical_movement), interval=0.0)
