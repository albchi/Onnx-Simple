import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Neural Network Architecture
# This is a very simple model with one linear layer. It's perfect for a "Hello World"
# example where we want to learn a simple linear relationship (like y = mx + b).
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # in_features=1: The model takes one number as input (our 'x' value).
        # out_features=1: The model outputs one number (the predicted 'y' value).
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        # This defines the forward pass of the network.
        # The input 'x' is passed through the linear layer.
        return self.linear(x)

def main():
    """
    Main function to run the training and exporting process.
    """
    print("--- PyTorch GPU Training & Export Hello World ---")

    # 2. Check for NVIDIA GPU and Set the Device
    # This is the standard way to check if a CUDA-enabled GPU is available.
    # If it is, we set our device to 'cuda'; otherwise, we fall back to the 'cpu'.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU found. Using CPU for training.")

    # 3. Create a Synthetic Dataset
    # We'll create data that follows a simple line: y = 2x + 1, plus some random noise.
    # The model's job will be to learn the '2' (weight) and the '1' (bias).
    X = torch.randn(200, 1) * 10  # 200 data points, 1 feature each
    noise = torch.randn(200, 1) * 0.5 # Add some noise to make it a bit realistic
    y = 2 * X + 1 + noise

    # Move our data and labels to the selected device (GPU or CPU)
    X_train, y_train = X.to(device), y.to(device)

    # 4. Initialize the Model, Loss Function, and Optimizer
    model = SimpleNet()
    model.to(device) # <-- This moves the model's parameters to the GPU.

    criterion = nn.MSELoss() # Mean Squared Error is a good loss function for regression tasks.
    optimizer = optim.SGD(model.parameters(), lr=0.001) # Stochastic Gradient Descent

    # 5. The Training Loop
    print("\n--- Starting Training ---")
    num_epochs = 50
    for epoch in range(num_epochs):
        # Forward pass: compute predicted y by passing x to the model.
        outputs = model(X_train)

        # Compute loss
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad() # Clear the gradients from the previous step
        loss.backward()       # Compute the gradients of the loss
        optimizer.step()      # Update the model's weights

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("--- Training Finished ---\n")

    # 6. Export the Trained Model for Inference
    print("--- Exporting Model ---")
    # CRITICAL: Set the model to evaluation mode. This disables training-specific
    # layers like Dropout or BatchNorm.
    model.eval()

    # BEST PRACTICE: Move the model to the CPU before exporting. This makes the
    # exported model more portable and avoids tying it to a specific GPU device.
    model.to("cpu")

    # We need a "dummy" input tensor that has the same size and type as a typical
    # input. This is used by both TorchScript and ONNX to trace the model's architecture.
    dummy_input = torch.randn(1, 1) # (batch_size=1, num_features=1)

    # --- Export to TorchScript (.pt file) ---
    try:
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save("model.pt")
        print("✅ Model successfully exported to TorchScript: model.pt")
    except Exception as e:
        print(f"❌ Error exporting to TorchScript: {e}")

    # --- Export to ONNX (.onnx file) ---
    try:
        torch.onnx.export(
            model,                        # The model to export.
            dummy_input,                  # A dummy input for tracing.
            "model.onnx",                 # The path to save the ONNX file.
            export_params=True,           # Store the trained weights within the model file.
            opset_version=12,             # The ONNX version to use (12 is a safe default).
            do_constant_folding=True,     # Execute constant folding for optimization.
            input_names=['input'],        # The model's input names.
            output_names=['output'],      # The model's output names.
            dynamic_axes={'input' : {0 : 'batch_size'},    # Allow for variable batch sizes.
                          'output' : {0 : 'batch_size'}}
        )
        print("✅ Model successfully exported to ONNX: model.onnx")
    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")


if __name__ == '__main__':
    main()
'''

### How to Run the Code

1.  **Install Prerequisites:** Make sure you have PyTorch installed with CUDA support to use your NVIDIA GPU. You will also need ONNX.
    ```bash
    # For PyTorch, follow instructions on the official site: https://pytorch.org/get-started/locally/
    # Example for a specific CUDA version:
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install ONNX
    pip install onnx
    ```

2.  **Save the File:** Save the code above as `train_and_export.py`.

3.  **Execute from Your Terminal:**
    ```bash
    python train_and_export.py
    
'''
