from typing import ClassVar

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms.functional import to_tensor

from reproduction_code.data.tasks.task import Task


class CORe50Task(Task):
    IMAGE_SHAPE: ClassVar[tuple[int, int]] = (224, 224)
    RESIZE_INTERPOLATION_MODE: ClassVar[InterpolationMode] = InterpolationMode.BILINEAR

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
