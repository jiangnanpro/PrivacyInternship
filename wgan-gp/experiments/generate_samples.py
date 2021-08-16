import os
file_path = os.path.dirname(os.path.abspath(__file__))

sample_file = os.path.join(os.path.dirname(file_path),'sample.py')

training_set_sizes = [512, 1024, 2048, 4096, 8192, 16384]
for n in training_set_sizes:
    MODEL_PATH_NIST = os.path.join(os.path.dirname(file_path),'models/wgan-gp_nist_{}'.format(n))
    MODEL_PATH_QMNIST = os.path.join(os.path.dirname(file_path),'models/wgan-gp_qmnist_{}'.format(n))
    if os.path.exists(MODEL_PATH_NIST):
        os.system('python {} --model_path {}'.format(sample_file, MODEL_PATH_NIST))
    if os.path.exists(MODEL_PATH_QMNIST):
        os.system('python {} --model_path {}'.format(sample_file, MODEL_PATH_QMNIST))