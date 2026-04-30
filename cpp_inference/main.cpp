#include <torch/script.h>
#include <iostream>

int main() {
    std::cout << "Loading model..." << std::endl;

    torch::jit::script::Module model;

    try {
        model = torch::jit::load("cnn_model_torchscript.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    model.eval();

    torch::Tensor input = torch::rand({1, 3, 32, 32});

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

   // set gdb breakpoint to here
    at::Tensor output = model.forward(inputs).toTensor();

    std::cout << "Output: " << output << std::endl;

    std::cout << "Done." << std::endl;

    return 0;
}