import app
import download_and_prepare_models as models


def main():
    models.prepare_llm_model()
    models.prepare_stable_diffusion_model()
    models.prepare_voice_activity_detection_model()
    models.prepare_super_res()
    models.prepare_whisper()
    models.prepare_depth_anything_v2()

    app.main()


if __name__ == '__main__':
    main()