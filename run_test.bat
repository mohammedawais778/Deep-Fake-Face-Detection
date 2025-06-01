@echo off
cd /d "c:\Users\Admin\Desktop\deepfake_detector"
call venv_py310\Scripts\activate.bat
python verify_env.py > env_check.log 2>&1
python convert_to_torch.py --test_image "uploads\fake_183_253_4.jpg" > model_test.log 2>&1
type env_check.log
type model_test.log
