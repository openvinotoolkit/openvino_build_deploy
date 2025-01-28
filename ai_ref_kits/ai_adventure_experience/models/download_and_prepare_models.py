import sys
import os
from pathlib import Path
import shutil

model_dir = os.path.dirname(os.path.abspath(__file__))

def prepare_llm_model():
    if not Path(model_dir+"/llama-3-8b-instruct").exists():
        print("llama Model not downloaded.")
        #cmd = "optimum-cli export openvino --model meta-llama/Meta-Llama-3-8B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.8 --sym " + model_dir + "/llama-3-8b-instruct/INT4_compressed_weights"
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
        
        #We need to make sure that text encoder input tensor is of type i32. 
        # So, using PPP object, set input tensor type to i32 and then replace
        # the model.
        sd_force_i32_text_encoder("LCM_Dreamshaper_v7/FP16/text_encoder")

        #replace the original text encoder IR with the i32 tensor input version.
        shutil.move(Path("LCM_Dreamshaper_v7/FP16/text_encoder/openvino_model_i32.xml"), 
                    Path("LCM_Dreamshaper_v7/FP16/text_encoder/openvino_model.xml"))
        shutil.move(Path("LCM_Dreamshaper_v7/FP16/text_encoder/openvino_model_i32.bin"), 
                    Path("LCM_Dreamshaper_v7/FP16/text_encoder/openvino_model.bin"))
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

def prepare_depth_anything_v2():
    if not Path(model_dir+"/depth_anything_v2_vits.xml").exists():
        cmd = f"python download_and_optimize_depth_anything_v2_model.py --model_dir {model_dir}"
        print("depth anything download command:",cmd)
        os.system(cmd)
    else:
        print("Depth Anything V2 model already downloaded.")

if __name__ == "__main__":
    prepare_llm_model()
    prepare_stable_diffusion_model()
    prepare_voice_activity_detection_model()
    prepare_super_res()
    prepare_whisper()
    prepare_depth_anything_v2()


