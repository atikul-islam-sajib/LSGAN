To present the detailed documentation of your CLI script in a structured and visually appealing manner suitable for Markdown or similar documentation formats, we can organize the information into sections with tables and lists to enhance readability and accessibility. Here's how you could structure it:

## CLI for GAN Training and Testing

The script serves as a command-line interface for training and testing Generative Adversarial Networks (GANs) using PyTorch, offering customization through various command-line arguments.

### Overview

- **Command-Line Parser**: Accepts arguments for dataset management, model parameters, and training configurations.
- **Data Processing**: Utilizes `Loader` to preprocess datasets into a DataLoader.
- **Model Training**: Initializes `Trainer` with specified parameters for GAN model training.
- **Synthetic Image Generation**: Optionally generates synthetic images post-training with the `--test` flag.

### Key Arguments

| Argument         | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `--image_path`   | Path to the dataset zip file.                             |
| `--batch_size`   | Size of batches during training.                          |
| `--image_size`   | Dimensions of images for training.                        |
| `--in_channels`  | Number of input channels (e.g., 3 for RGB).               |
| `--latent_space` | Dimensionality of latent space vector.                    |
| `--num_samples`  | Number of synthetic images to generate in test phase.     |
| `--lr`           | Learning rate for the Adam optimizer.                     |
| `--epochs`       | Number of training epochs.                                |
| `--device`       | Computation device (`cuda`, `mps`, or `cpu`).             |
| `--folder`       | Flag to clean training/model directories before starting. |
| `--test`         | Flag to trigger synthetic image generation post-training. |

### Execution Flow

1. **Data Loading**: Processes and creates a DataLoader from the dataset specified by `--image_path`.
2. **Model Training**: Trains GAN models with provided parameters.
3. **Testing**: Generates synthetic images using the trained Generator model if `--test` is set.

### Example Usage

- **Training the Model**:
  ```shell
  python script.py --image_path "/path/to/dataset.zip" --batch_size 64 --image_size 64 --in_channels 3 --latent_space 100 --lr 0.0002 --epochs 10 --device cuda --folder
  ```
- **Generating Synthetic Images**:
  ```shell
  python script.py --test --latent_space 100 --num_samples 20 --device cuda
  ```

### Logging

Logs key events to `./logs/cli.log`, including data loading, training start/end, and testing phases.

### Error Handling

Checks for the presence of required arguments and logs accordingly. Improvement areas include explicit exception handling for `Loader`, `Trainer`, and `Test` classes.

This documentation format leverages tables and structured lists to clearly convey the script's capabilities, usage, and requirements, making it easier for users to understand and utilize the CLI for GAN training and testing.
