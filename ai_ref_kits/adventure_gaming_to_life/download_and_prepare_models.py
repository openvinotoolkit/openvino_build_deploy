import sys
import os
from pathlib import Path
import shutil

from optimum.intel.openvino import OVLatentConsistencyModelPipeline
from pathlib import Path
import warnings

MODEL_MAPPING = {
    "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
}
model_dir = "dnd_models"
def prepare_whisper_model() :
    if not Path(model_dir).exists():
        os.mkdir(model_dir)
    os.chdir(model_dir)
    if not Path("./whisper.cpp").exists():
      os.system("git clone https://github.com/ggerganov/whisper.cpp.git")
    os.chdir("whisper.cpp\\models")
    
    os.system("python convert-whisper-to-openvino.py --model base")  
    shutil.copy('ggml-base-encoder-openvino.bin', '../../')
    shutil.copy('ggml-base-encoder-openvino.xml', '../../')

def prepare_lcm_model() :
 warnings.filterwarnings("ignore")
 if not Path(model_dir+"/openvino_ir_lcm").exists():
    ov_pipeline = OVLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", height=512, width=512, export=True, compile=False)
    ov_pipeline.save_pretrained(model_dir+"/openvino_ir_lcm")


def prepare_llm_model():
    cmd = "optimum-cli export openvino --model meta-llama/Meta-Llama-3-8B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.8 --sym " + model_dir + "/llama-3-8b-instruct/INT4_compressed_weights"
    print(cmd)
    os.system(cmd)

# Define a download file helper function
def download_file(url: str, path: Path) -> None:
    """Download file."""
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)

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

def prepare_llava():
   cmd = "python download_and_optimize_llava_model.py --model_dir " +  model_dir
   os.system(cmd)

if __name__ == "__main__":
    prepare_super_res()
    prepare_lcm_model()
    prepare_whisper_model()   
    prepare_llm_model()
    prepare_llava()