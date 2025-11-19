#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: fanl_test.exe <model_path.pt> <image_path>\n";
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    try {
        // 1) Load TorchScript model
        auto module = torch::jit::load(model_path);
        module.eval();

        if (torch::cuda::is_available()) {
            std::cout << "✔ Using GPU (CUDA)\n";
            module.to(torch::kCUDA);
        } else {
            std::cout << "✔ Using CPU mode\n";
        }

        // 2) Read image (PathMNIST: 28x28, RGB)
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "✘ Failed to read image: " << image_path << "\n";
            return -1;
        }
        cv::resize(img, img, cv::Size(28, 28));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // 3) Convert to tensor and normalize to [-1, 1]
        torch::Tensor x = torch::from_blob(
            img.data,
            {1, 28, 28, 3},
            torch::kByte
        );

        x = x.permute({0, 3, 1, 2});          // NHWC -> NCHW
        x = x.to(torch::kFloat32).div(255.0); // [0,1]
        x = (x - 0.5f) / 0.5f;                // [-1,1]

        if (torch::cuda::is_available()) {
            x = x.to(torch::kCUDA);
        }

        // 4) Forward inference
        auto out = module.forward({x}).toTensor();

        // 5) Softmax to get probabilities
        auto probs = torch::softmax(out, 1);

        // Predict class index
        auto pred_tensor = probs.argmax(1);
        int pred = pred_tensor.item<int>();

        // Confidence score
        float conf = probs[0][pred].item<float>();

        std::vector<std::string> class_names = {
            "adipose", "background", "debris", "lymphocytes",
            "mucus", "smooth muscle", "normal colon mucosa",
            "cancer-associated stroma", "colorectal adenocarcinoma epithelium"
        };

        std::cout << "✔ Inference completed successfully!\n";
        std::cout << "Predicted class index: " << pred << "\n";
        if (pred >= 0 && pred < (int)class_names.size()) {
            std::cout << "Predicted class: " << class_names[pred] << "\n";
        }
        std::cout << "Confidence: " << conf * 100 << " %\n";

    } catch (const c10::Error& e) {
        std::cerr << "✘ Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
