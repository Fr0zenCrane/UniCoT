import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset_MDPedit(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        data = self._init_data()
        data = self._add_text(data, row["system_prompt"], need_loss=False)
        data = self._add_text(data, row["prompt"], need_loss=False)
        data = self._add_text(data, row["think"], need_loss=False)

        data = self._add_image( # present visual state
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["source_image"]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )
        data = self._add_text(data, row["edit_instruction"], need_loss=False)
        data = self._add_image( # visual state transition
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["target_image"]))),
            need_loss=True, 
            need_vae=False, 
            need_vit=False,
        )
        return data


class UnifiedEditIterableDataset_MDPT2I(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        data = self._init_data()
        data = self._add_text(data, row["system_prompt"], need_loss=False)
        data = self._add_text(data, row["prompt"], need_loss=False)
        data = self._add_text(data, row["think"], need_loss=False)
        data = self._add_image( # visual state initialization
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image"]))),
            need_loss=True, 
            need_vae=False, 
            need_vit=False,
        )
        
        return data
    
    
class UnifiedEditIterableDataset_MDPGeo(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        data = self._init_data()
        data = self._add_text(data, row["instruction_list"][0], need_loss=False) # task prompt
        data = self._add_image( # present visual state
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][0]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True,
        )
        match row["task_type"]:
            case "step-1":
                data = data
            case "step-2":
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.open(io.BytesIO(row["image_list"][1]))),
                    need_loss=False, 
                    need_vae=True, 
                    need_vit=True,
                )
            case "step-3": # geo think text
                data = self._add_text(data, row["instruction_list"][-1], need_loss=True)
            case "step-4":
                data = self._add_image( # intermediate visual state 
                    data, 
                    pil_img2rgb(Image.open(io.BytesIO(row["image_list"][1]))),
                    need_loss=False, 
                    need_vae=True, 
                    need_vit=True,
                )
                data = self._add_text(data, row["instruction_list"][-1], need_loss=True)
            case _ :
                data = None
        
        if data is not None: # visual state transition
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][-1]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )

        return data

