"""Generate API V2 routes

"""
import time
import base64
import shutil
from PIL import Image
import os
from io import BytesIO
import uuid
from typing import List
from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import JSONResponse
from fooocusapi.timing import server_process_time

from fooocusapi.models.common.base import EnhanceCtrlNets, GenerateMaskRequest
from fooocusapi.utils.api_utils import api_key_auth
from fooocusapi.models.requests_v1 import ImagePrompt
from fooocusapi.models.common.requests import AdvancedParams
from fooocusapi.models.requests_v2 import (
    ImageEnhanceRequestJson, ImgInpaintOrOutpaintRequestJson,
    ImgPromptRequestJson,
    Text2ImgRequestWithPrompt,
    ImgUpscaleOrVaryRequestJson, ObjectReplaceRequestJson, BackgroundGeneration
)
from fooocusapi.models.common.response import (
    AsyncJobResponse,
    GeneratedImageResult
)
from fooocusapi.utils.call_worker import (
    call_worker,
    generate_mask as gm
)
from fooocusapi.utils.img_utils import base64_to_stream
from fooocusapi.configs.default import img_generate_responses


secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)


@secure_router.post(
        path="/v2/generation/text-to-image-with-ip",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def text_to_img_with_ip(
    req: Text2ImgRequestWithPrompt,
    accept: str = Header(None),
    accept_query: str | None = Query(
        default=None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nText to image with prompt\n
    Text to image with prompt
    Arguments:
        req {Text2ImgRequestWithPrompt} -- Text to image generation request
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    default_image_prompt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_prompt)

    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
        path="/v2/generation/image-upscale-vary",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_upscale_or_vary(
    req: ImgUpscaleOrVaryRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept', description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nImage upscale or vary\n
    Image upscale or vary
    Arguments:
        req {ImgUpscaleOrVaryRequestJson} -- Image upscale or vary request
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
            Response -- img_generate_responses    
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)

    default_image_prompt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_prompt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
        path="/v2/generation/image-inpaint-outpaint",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_inpaint_or_outpaint(
    req: ImgInpaintOrOutpaintRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nInpaint or outpaint\n
    Inpaint or outpaint
    Arguments:
        req {ImgInpaintOrOutpaintRequestJson} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)
    default_image_prompt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_prompt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)



@secure_router.post(
        path="/ai/api/v1/ai_bg",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_background_change(
    req_obj: BackgroundGeneration,
    #req: ImgInpaintOrOutpaintRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nInpaint or outpaint\n
    Inpaint or outpaint
    Arguments:
        req {ImgInpaintOrOutpaintRequestJson} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query
    # GenerateMaskRequest
    start = time.time()
    generate_mask= GenerateMaskRequest(
        image= req_obj.image
    )
    masked_image=  gm(request=generate_mask)
    
    mask_time = time.time()
    # print(masked_image)
    req = ImgInpaintOrOutpaintRequestJson(
        input_image = req_obj.image,
        input_mask = masked_image,
        prompt = req_obj.prompt,
        inpaint_additional_prompt = '',
        outpaint_selections = [],
        image_number= req_obj.image_count,
        outpaint_distance_left = -1,
        outpaint_distance_right = -1,
        outpaint_distance_top = -1,
        outpaint_distance_bottom = -1,
        image_prompts = [],
        advanced_params = AdvancedParams(invert_mask_checkbox=True)

    )

    req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)
    default_image_prompt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_prompt)
    req.image_prompts = image_prompts_files

    primary_response = call_worker(req, accept)

    if primary_response:
        first_element = primary_response[0]
    else:
        first_element = None  # or handle accordingly
    output_image_url = remove_baseUrl(first_element.url)

    output_urls = []
    for image in primary_response:
        image_name = remove_baseUrl(image.url)
    
        print(len(primary_response))
        local_output_image_path = "/home/fooocus_ai_background/outputs" + remove_baseUrl(image.url)
        new_out_images_directory_name = '/ai_background/'
        new_local_out_image_directory = get_save_img_directory(new_out_images_directory_name)
        new_local_out_image_path =  new_local_out_image_directory + image_name
        move_file(local_output_image_path,new_local_out_image_directory)

        output_urls.append('/media' + new_out_images_directory_name + new_local_out_image_path.split('/')[-1])

    end = time.time()
    response_data = {
        "success": True,
        "message": "Returned output successfully",
        # "server_process_time": end-start,
        "output_image_url": output_urls,
        "server_process_time": server_process_time["preprocess_time"] + server_process_time["processing_time"],
        # "mask_generation_time": mask_time-start
    }

    return JSONResponse(content=response_data, status_code=200)



@secure_router.post(
        path="/ai/api/v1/object_replace",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_inpaint_or_outpaint(
    req_obj: ObjectReplaceRequestJson,
    #req: ImgInpaintOrOutpaintRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nInpaint or outpaint\n
    Inpaint or outpaint
    Arguments:
        req {ImgInpaintOrOutpaintRequestJson} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req = ImgInpaintOrOutpaintRequestJson(
        input_image = req_obj.image,
        input_mask = req_obj.mask,
        prompt = req_obj.prompt,
        inpaint_additional_prompt = '',
        outpaint_selections = [],
        outpaint_distance_left = -1,
        outpaint_distance_right = -1,
        outpaint_distance_top = -1,
        outpaint_distance_bottom = -1,
        image_prompts = []
    )

    req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)
    default_image_prompt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_prompt)
    req.image_prompts = image_prompts_files

    primary_response = call_worker(req, accept)

    if primary_response:
        first_element = primary_response[0]
    else:
        first_element = None  # or handle accordingly
    # output_image_url = remove_baseUrl(first_element.url)
    # local_output_image_path = "/home/Foocus_ObjectReplace/outputs" + remove_baseUrl(first_element.url)

    # new_out_images_directory_name = '/object_replace_images/'
    # new_local_out_image_directory = get_save_img_directory(new_out_images_directory_name)
    # new_local_out_image_path =  new_local_out_image_directory + output_image_url
    # move_file(local_output_image_path,new_local_out_image_directory)

    response_data = {
        "success": True,
        "message": "Returned output successfully",
        "output_image_url": '/media' + new_out_images_directory_name + new_local_out_image_path.split('/')[-1],
        "server_process_time": server_process_time["preprocess_time"] + server_process_time["processing_time"]
    }

    return JSONResponse(content=response_data, status_code=200)


def move_file(src_path: str, dest_path: str):
    try:
        if not os.path.isfile(src_path):
            return f"Error: Source file '{src_path}' does not exist."
        dest_dir = os.path.dirname(dest_path)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        new_path = shutil.move(src_path, dest_path)
        return f"File moved successfully to '{new_path}'"
    except Exception as e:
        return f"Error: {str(e)}"

def remove_baseUrl(url: str) -> str:
    return "/files" + url.split("/files", 1)[-1] if "/files" in url else url

def get_save_img_directory(directory_name):
    current_dir = '/tmp'
    img_directory = current_dir + '/.temp' + directory_name
    os.makedirs(img_directory, exist_ok=True)
    return img_directory


@secure_router.post(
        path="/v2/generation/image-inpaint-outpaint",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_Expand(
    req: ImgInpaintOrOutpaintRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nInpaint or outpaint\n
    Inpaint or outpaint
    Arguments:
        req {ImgInpaintOrOutpaintRequestJson} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)
    default_image_prompt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_prompt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)

@secure_router.post(
        path="/v2/generation/image-prompt",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_prompt(
    req: ImgPromptRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nImage prompt\n
    Image prompt generation
    Arguments:
        req {ImgPromptRequest} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if req.input_image is not None:
        req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)

    default_image_prompt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_prompt)

    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
        path="/v2/generation/image-enhance",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_enhance(
    req: ImageEnhanceRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nImage prompt\n
    Image prompt generation
    Arguments:
        req {ImageEnhanceRequestJson} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if req.enhance_input_image is not None:
        req.enhance_input_image = base64_to_stream(req.enhance_input_image)

    if len(req.enhance_ctrlnets) < 3:
        default_enhance_ctrlnet = [EnhanceCtrlNets()]
        req.enhance_ctrlnets + (default_enhance_ctrlnet * (4 - len(req.enhance_ctrlnets)))

    return call_worker(req, accept)


@secure_router.post(
    path="/v1/tools/generate_mask",
    summary="Generate mask endpoint",
    tags=["GenerateV1"])
def generate_mask(mask_options: GenerateMaskRequest) -> str:
    """
    Generate mask endpoint
    """
    return  gm(request=mask_options)
