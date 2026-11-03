from PIL.Image import Image


class ConvertChannels:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(self, image: Image) -> Image:
        if image.mode != self.mode:
            return image.convert(self.mode)
        return image
