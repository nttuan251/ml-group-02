from PIL.Image import Image


class ConvertChannels:
    """
    Utility class for converting the channels of an image to the desired mode.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(self, image: Image) -> Image:
        if image.mode != self.mode:
            return image.convert(self.mode)
        return image
