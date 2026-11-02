from typing import ClassVar

from datasets import DatasetDict
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms.functional import to_tensor

from reproduction_code.data.tasks.task import Task


class FiveDatasetsTask(Task):
    IMAGE_SHAPE: ClassVar[tuple[int, int]] = (224, 224)
    RESIZE_INTERPOLATION_MODE: ClassVar[InterpolationMode] = InterpolationMode.BILINEAR

    def __init__(
        self,
        class_ids: list[int],
        data: DatasetDict,
        random_seed: int,
        label_offset: int,
    ) -> None:
        super().__init__(class_ids, data, random_seed)

        self.label_offset = label_offset

    def _preprocess_label(self, label: int) -> int:
        return label + self.label_offset

    def _preprocess_data_point(self, image: Image) -> Tensor:
        recolored_image = image.convert("RGB")
        resized_image = self._resize_image(recolored_image)
        image_tensor = self._convert_to_unit_range_tensor(resized_image)

        return image_tensor

    def _resize_image(self, image: Image) -> Image:
        return Resize(
            self.IMAGE_SHAPE,
            interpolation=self.RESIZE_INTERPOLATION_MODE,
        )(image)

    def _convert_to_unit_range_tensor(self, image: Image) -> Tensor:
        return to_tensor(image)
