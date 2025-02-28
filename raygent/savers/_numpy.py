from typing import Any, Literal
from raygent.savers import Saver
import os

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class NumpySaver(Saver):
    """
    A saver that writes data to a .npy file.

    Example usage:
        saver = NumpySaver("my_data.npy")
    """

    def __init__(
        self,
        file_path: str,
        approach: Literal["append", "overwrite", "update"] = "append",
    ):
        """
        Args:
            file_path: The path to the .npy file.
            approach: One of `'append'`, `'overwrite'`, or `'update'`.
        """
        self.file_path = file_path
        self.approach = approach.strip().lower()

    def save(
        self, data: list[Any], indices: Any | None = None, **kwargs: dict[str, Any]
    ) -> None:
        """
        Saves or appends the data to a .npy file. In this example, we simply overwrite.
        For real use, you might want a custom append logic.

        Args:
            data: A list of results to save.
            **kwargs: Additional parameters for the saving process.
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is not installed.")

        arr = np.array(data)

        # If overwriting, simply save
        if self.approach == "overwrite":
            np.save(self.file_path, arr)

        elif self.approach == "append":
            if os.path.exists(self.file_path):
                data = np.load(self.file_path)
                combined = np.concatenate([data, arr])
                np.save(self.file_path, combined)
            else:
                np.save(self.file_path, arr)
        elif self.approach == "update":
            if os.path.exists(self.file_path):
                data = np.load(self.file_path)
                if not indices:
                    raise ValueError("`indices` cannot be None when using 'update'")
                data[indices] = arr
                np.save(self.file_path, data)
        else:
            raise ValueError(
                f"Unknown approach '{self.approach}'. Use 'append', 'update', or 'overwrite'."
            )
