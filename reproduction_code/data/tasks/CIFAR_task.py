from typing import ClassVar

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms.functional import to_tensor

from reproduction_code.data.tasks.task import Task


class CIFAR_task(Task):
    IMAGE_SHAPE: ClassVar[tuple[int, int]] = (224, 224)

    # NOTE: Paper did not specify which interpolation mode to use
    RESIZE_INTERPOLATION_MODE: ClassVar[InterpolationMode] = InterpolationMode.BILINEAR

    def _preprocess_data_point(self, image: Image) -> Tensor:
        resized_image = self._resize_image(image)
        image_tensor = self._convert_to_unit_range_tensor(resized_image)

        return image_tensor

    def _resize_image(self, image: Image) -> Image:
        return Resize(
            self.IMAGE_SHAPE,
            interpolation=self.RESIZE_INTERPOLATION_MODE,
        )(image)

    def _convert_to_unit_range_tensor(self, image: Image) -> Tensor:
        return to_tensor(image)
