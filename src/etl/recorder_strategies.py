from abc import ABC, abstractmethod
from dataclasses import dataclass
import queue
from enum import Enum


@dataclass(frozen=True)
class Coordinates:
    x: int
    y: int

    def __sub__(self, other: "Coordinates") -> tuple[int, int]:
        return self.x - other.x, self.y - other.y

    def __str__(self):
        print("Coordinates (x<%s>, y<%s>)", self.x, self.y)


class RecorderStrategy(ABC):
    def __init__(self) -> None:
        self.initial_coords = Coordinates(0, 0)

    @abstractmethod
    def get_next_coords(self, current_coord: Coordinates) -> Coordinates:
        ...


class DFSStrategy(RecorderStrategy):
    def __init__(self, offset: int) -> None:
        super().__init__()
        self.offset = offset
        self.non_visited_coords: queue.Queue[Coordinates] = queue.Queue()
        self.processed: set[Coordinates] = set()
        self.processed.add(self.initial_coords)

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

    def get_next_coords(self, current_coords: Coordinates) -> Coordinates:
        self._update_neighbour_and_processed_coords(current_coords)
        return self.non_visited_coords.get()


class Direction(Enum):
    RIGHT = "right"
    LEFT = "left"
    UP = "up"
    DOWN = "down"


class SnailStrategy(RecorderStrategy):
    def __init__(self, offset: int):
        super().__init__()
        self.offset = offset
        self.direction = Direction.RIGHT
        self.remaining_steps = 1
        self.steps_made = 0

    def _compute_next_coord(self, current_coord: Coordinates) -> Coordinates:
        x, y = current_coord.x, current_coord.y
        if self.direction == Direction.RIGHT:
            return Coordinates(x + self.offset, y)
        elif self.direction == Direction.UP:
            return Coordinates(x, y + self.offset)
        elif self.direction == Direction.LEFT:
            return Coordinates(x - self.offset, y)
        return Coordinates(x, y - self.offset)

    def get_next_coords(self, current_coords: Coordinates) -> Coordinates:
        if self.steps_made == self.remaining_steps:
            self.steps_made = 0
            if self.direction == Direction.RIGHT:
                self.direction = Direction.UP
            elif self.direction == Direction.UP:
                self.direction = Direction.LEFT
                self.remaining_steps += 1
            elif self.direction == Direction.LEFT:
                self.direction = Direction.DOWN
            elif self.direction == Direction.DOWN:
                self.direction = Direction.RIGHT
                self.remaining_steps += 1

        next_coord = self._compute_next_coord(current_coords)
        self.steps_made += 1

        return next_coord
