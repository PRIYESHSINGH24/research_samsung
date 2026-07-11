import torch

class Config:
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    NUM_CLASSES = 10
    IN_CHANNELS = 1
    IMAGE_SIZE  = 32

    # Paper exact settings
    TEACHER_EPOCHS = 100
    TEACHER_LR     = 0.001
    BATCH_SIZE     = 256

    # DI Generation
    NUM_DI_PER_CLASS = 2400        # MNIST: 2400, FMNIST: 4800
    DI_BATCH_SIZE    = 32
    TEMPERATURE      = 20
    DI_LR            = 0.1         # MNIST/FMNIST: 0.1, CIFAR VGG: 10.0
    DI_ITERATIONS    = 1500
    BETA_VALUES      = [1.0, 0.1]  # KD: [1.0, 0.1], DA: [0.01, 0.1]

    # KD
    STUDENT_EPOCHS = 500
    STUDENT_LR     = 0.001
    KD_TEMPERATURE = 20

    DATA_DIR       = 'data/'
    CHECKPOINT_DIR = 'checkpoints/'
    RESULTS_DIR    = 'results/'

config = Config()
print(f"Device: {config.DEVICE}")