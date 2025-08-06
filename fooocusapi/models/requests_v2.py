"""V2 API models"""
from typing import List
from pydantic import BaseModel, Field
from fooocusapi.models.common.requests import CommonRequest
from fooocusapi.models.common.base import (
    ControlNetType,
    EnhanceCtrlNets, OutpaintExpansion,
    ImagePrompt,
    UpscaleOrVaryMethod
)


class ImagePromptJson(BaseModel):
    """Image prompt for V2 API"""
    cn_img: str | None = Field(None, description="Input image for image prompt as base64")
    cn_stop: float | None = Field(0, ge=0, le=1, description="Stop at for image prompt, 0 for default value")
    cn_weight: float | None = Field(0, ge=0, le=2, description="Weight for image prompt, 0 for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip, description="ControlNet type for image prompt")



class ImgInpaintOrOutpaintRequestJson(CommonRequest):
    """image inpaint or outpaint request"""
    input_image: str = Field('', description="Init image for inpaint or outpaint as base64")
    input_mask: str | None = Field('', description="Inpaint or outpaint mask as base64")
    inpaint_additional_prompt: str | None = Field('', description="Describe what you want to inpaint")
    image_number: int | None = Field(1, description="Image number 1-4",ge=1, le=4)
    outpaint_selections: List[OutpaintExpansion] = []
    outpaint_distance_left: int | None = Field(-1, description="Set outpaint left distance")
    outpaint_distance_right: int | None = Field(-1, description="Set outpaint right distance")
    outpaint_distance_top: int | None = Field(-1, description="Set outpaint top distance")
    outpaint_distance_bottom: int | None = Field(-1, description="Set outpaint bottom distance")
    image_prompts: List[ImagePromptJson | ImagePrompt] = []
    



class ObjectReplaceRequestJson(BaseModel):
    image: str = Field('', description="Init image for inpaint or outpaint as base64")
    mask: str | None = Field('', description="Inpaint or outpaint mask as base64")
    prompt: str | None = Field('', description="Inpaint or outpaint mask as base64")
    guidance_scale: float = Field(default=7.0, ge=6.0, le=8.0)
    steps: str = Field(default="quality")


class BackgroundGeneration(BaseModel):
    image: str = Field('', description="Init image for inpaint or outpaint as base64")
    # mask: str | None = Field('', description="Inpaint or outpaint mask as base64")
    image_count : int = Field(default=1, description="Image Count 1-4",ge=1, le=4)
    prompt: str | None = Field(default='', description="Inpaint or outpaint mask as base64")
    # sdGuidanceScale: float = Field(default=4, ge=1.0, le=30.0)
    # sdSteps: int = Field(default=40)
    guidance_scale: float = Field(default=7.0, ge=6.0, le=8.0)
    steps: str = Field(default="quality")





class ImgPromptRequestJson(ImgInpaintOrOutpaintRequestJson):
    """img prompt request json"""
    input_image: str | None = Field(None, description="Init image for inpaint or outpaint as base64")
    image_prompts: List[ImagePromptJson | ImagePrompt]


class Text2ImgRequestWithPrompt(CommonRequest):
    """text to image request with prompt"""
    image_prompts: List[ImagePromptJson] = []


class ImgUpscaleOrVaryRequestJson(CommonRequest):
    """img upscale or vary request json"""
    uov_method: UpscaleOrVaryMethod = UpscaleOrVaryMethod.upscale_2
    upscale_value: float | None = Field(1.0, ge=1.0, le=5.0, description="Upscale custom value, 1.0 for default value")
    input_image: str = Field(description="Init image for upscale or outpaint as base64")
    image_prompts: List[ImagePromptJson | ImagePrompt] = []


class ImageEnhanceRequestJson(CommonRequest):
    """
    Image Enhance Request
    """
    enhance_input_image: str = Field(default='', description="Input image for enhance")
    enhance_checkbox: bool = Field(default=True, description="Enhance checkbox")
    enhance_uov_method: UpscaleOrVaryMethod = Field(default=UpscaleOrVaryMethod.strong_variation, description="Upscale or vary method")
    enhance_uov_processing_order: str = Field(default="Before First Enhancement", description="Upscale or vary processing order")
    enhance_uov_prompt_type: str = Field(default="Original Prompts", description="Upscale or vary prompt type")
    save_final_enhanced_image_only: bool = Field(True, description="Save Final Enhanced Image Only")
    enhance_ctrlnets: List[EnhanceCtrlNets]
