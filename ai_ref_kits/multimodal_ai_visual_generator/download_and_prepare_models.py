import os
import shutil
import subprocess
import sys
from pathlib import Path

import openvino as ov
import torch
from huggingface_hub import hf_hub_download

model_dir = os.path.dirname(os.path.abspath(__file__)) + "/models"


def prepare_llm_model():
    if not Path(model_dir+"/llama-3-8b-instruct").exists():
        print("llama Model not downloaded.")
        cmd = "optimum-cli export openvino --model meta-llama/Meta-Llama-3-8B-Instruct --task text-generation-with-past --weight-format int4 --group-size -1 --ratio 1.0 --sym "  + model_dir + "/llama-3-8b-instruct/INT4_compressed_weights"
        print("llm download command:",cmd)
        os.system(cmd)
    else:
        print("llama Model already downloaded.")


def prepare_stable_diffusion_model():
    if not Path(model_dir+"/LCM_Dreamshaper_v7").exists():
        print("Stable Diffusion LCM Model not downloaded.")
        cmd = "optimum-cli export openvino --model SimianLuo/LCM_Dreamshaper_v7 --task stable-diffusion --weight-format fp16 " + model_dir + "/LCM_Dreamshaper_v7/FP16"
        print("llm download command:",cmd)
        os.system(cmd)
        
        def sd_force_i32_text_encoder(text_encoder_path):
            import openvino
            core = openvino.Core()
            model = core.read_model(Path(text_encoder_path + "/openvino_model.xml"))
            ppp = openvino.preprocess.PrePostProcessor(model)
            ppp.input().tensor().set_element_type(openvino.Type.i32)
            model = ppp.build()
            openvino.serialize(model, Path(text_encoder_path + "/openvino_model_i32.xml"))
        
        # We need to make sure that text encoder input tensor is of type i32.
        # So, using PPP object, set input tensor type to i32 and then replace
        # the model.
        sd_force_i32_text_encoder(f"{model_dir}/LCM_Dreamshaper_v7/FP16/text_encoder")

        # replace the original text encoder IR with the i32 tensor input version.
        shutil.move(Path(f"{model_dir}/LCM_Dreamshaper_v7/FP16/text_encoder/openvino_model_i32.xml"),
                    Path(f"{model_dir}/LCM_Dreamshaper_v7/FP16/text_encoder/openvino_model.xml"))
        shutil.move(Path(f"{model_dir}/LCM_Dreamshaper_v7/FP16/text_encoder/openvino_model_i32.bin"),
                    Path(f"{model_dir}/LCM_Dreamshaper_v7/FP16/text_encoder/openvino_model.bin"))
    else:
        print("Stable Diffusion LCM Model already downloaded.")


# Define a download file helper function
def download_file(url: str, path: Path) -> None:
    """Download file."""
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def prepare_voice_activity_detection_model():
    if not Path(model_dir+"/silero_vad.onnx").exists():
        print("Voice Activity Detection Model not downloaded.")
        download_file("https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx", Path(model_dir + "/silero_vad.onnx") )
    else:
        print("Voice Activity Detection already downloaded.")


def prepare_super_res():

    # 1032: 4x superresolution, 1033: 3x superresolution
    model_name = 'single-image-super-resolution-1033'
    base_model_dir = Path(model_dir)

    model_xml_name = f'{model_name}.xml'
    model_bin_name = f'{model_name}.bin'

    model_xml_path = base_model_dir / model_xml_name
    model_bin_path = base_model_dir / model_bin_name

    if not model_xml_path.exists():
        base_url = f'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP16/'
        model_xml_url = base_url + model_xml_name
        model_bin_url = base_url + model_bin_name

        download_file(model_xml_url, model_xml_path)
        download_file(model_bin_url, model_bin_path)
    else:
        print(f'{model_name} already downloaded to {base_model_dir}')


def prepare_whisper():
    whisper_version="whisper-base"
    if not Path(model_dir+"/"+whisper_version).exists():
       cmd = f"optimum-cli export openvino --trust-remote-code --model openai/{whisper_version} " + model_dir + "/" + whisper_version
       print("whisper download command:",cmd)
       os.system(cmd)
    else:
       print(f'{whisper_version} already downloaded.')


def clone_repo(repo_url: str, revision: str = None, add_to_sys_path: bool = True) -> Path:
    repo_path = Path(repo_url.split("/")[-1].replace(".git", ""))

    if not repo_path.exists():
        try:
            subprocess.run(["git", "clone", repo_url], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            print(f"Failed to clone the repository: {exc.stderr}")
            raise

        if revision:
            subprocess.Popen(["git", "checkout", revision], cwd=str(repo_path))
    if add_to_sys_path and str(repo_path.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_path.resolve()))

    return repo_path


def prepare_depth_anything_v2():
    if not Path(model_dir+"/depth_anything_v2_vits.xml").exists():
        repo_dir = Path(model_dir + "/Depth-Anything-V2")
        os.chdir(model_dir)

        if not repo_dir.exists():
            clone_repo("https://huggingface.co/spaces/depth-anything/Depth-Anything-V2")
        sys.path.insert(0, str(repo_dir.resolve()))
        os.chdir("..")

        print("adding this to the path: ", repo_dir)
        # sys.path.append(Path(repo_dir))
        from depth_anything_v2.dpt import DepthAnythingV2

        attention_file_path = Path(f"{model_dir}/Depth-Anything-V2/depth_anything_v2/dinov2_layers/attention.py")
        orig_attention_path = attention_file_path.parent / ("orig_" + attention_file_path.name)

        if not orig_attention_path.exists():
            attention_file_path.rename(orig_attention_path)

            with orig_attention_path.open("r") as f:
                data = f.read()
                data = data.replace("XFORMERS_AVAILABLE = True", "XFORMERS_AVAILABLE = False")
                with attention_file_path.open("w") as out_f:
                    out_f.write(data)

        encoder = "vits"
        model_type = "Small"
        model_id = f"depth_anything_v2_{encoder}"

        model_path = hf_hub_download(repo_id=f"depth-anything/Depth-Anything-V2-{model_type}",
                                     filename=f"{model_id}.pth", repo_type="model")

        model = DepthAnythingV2(encoder=encoder, features=64, out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        ov_depth_anything_path = Path(f"{model_dir}/{model_id}.xml")

        if not ov_depth_anything_path.exists():
            ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 434, 770), input=[1, 3, 434, 770])
            ov.save_model(ov_model, ov_depth_anything_path)
    else:
        print("Depth Anything V2 model already downloaded.")


if __name__ == "__main__":
    prepare_llm_model()
    prepare_stable_diffusion_model()
    prepare_voice_activity_detection_model()
    prepare_super_res()
    prepare_whisper()
    prepare_depth_anything_v2()


