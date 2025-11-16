---
title: Mnist Classifier
prev:
  text: Build from Source
  link: /install/
---

# MNIST Classifier with stochastic gradient descent

Let's begin with classifying the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset in C++ and using only CPU. In this example we use a pre-cleaned dataset in CSV. The CMake file automatically downloads the dataset from [kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). If it fails to do so, there is also the option to put the unzipped files to builds/examples/data.

We will skim through the less important parts of the code.

## Preparing the data
```cpp
#include <wolf.h>

using namespace wolf;

struct MnistSample {
    std::vector<float> x;  // 784 floats in [0,1]
    int label;             // 0..9
};

std::vector<MnistSample> load_mnist_csv(const std::string& path, int max_samples = -1) {
   // Loads CSV into a vector
}

// Convert sample to input Tensor [1 x 784]
Tensor make_input(const MnistSample& s) {
    return Tensor(s.x, 1, 784);
}

// One-hot target [1 x 10]
Tensor make_target(const MnistSample& s) {
    std::vector<float> t(10, 0.0f);
    t[s.label] = 1.0f;
    return Tensor(t, 1, 10);
}
```
wolf::Tensor is a class which wraps a flattened 1D array. Tensor(s.x, 1, 784) creates a 1x784 tensor. 1 is our batch size, while 784 is the dimensions of the array. 

Next, we load our data file with the C++17 feature std::filesystem.
```cpp

int main(int argc, char** argv) {
    std::filesystem::path exe_path = std::filesystem::canonical(argv[0]);
    std::filesystem::path exe_dir  = exe_path.parent_path();
    std::filesystem::path data_dir = exe_dir / "data";

    std::filesystem::path train_path = data_dir / "mnist_train.csv";
    std::filesystem::path test_path  = data_dir / "mnist_test.csv";

    std::println("Executable dir: {}", exe_dir.string());
    std::println("Loading MNIST train from: {}", train_path.string());
    std::println("Loading MNIST test  from: {}", test_path.string());

    auto train_data = load_mnist_csv(train_path.string());
    auto test_data  = load_mnist_csv(test_path.string());

    std::println("Loaded {} train samples, {} test samples",
                 train_data.size(), test_data.size());
```

## Building the Model
``` cpp
    // Build 2 layer model: 784 -> 128 -> ReLU -> 10
    Sequential model(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    );
    float lr     = 0.01f;
    int   epochs = 3; // Number of times the model is trained over whole train set (repeat)
```
Here we call Sequential, which gives an easy way to add layers together. Our model has one hidden ReLU layer, and takes a 784-sized input to give a 10-sized output. This 10-sized output is one-hot encoded, so the first output is confident the neural net thinks the image is a '0' and so on.  
``` cpp
    std::mt19937 gen(std::random_device{}());
    std::vector<std::size_t> indices(train_data.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::shuffle(indices.begin(), indices.end(), gen);

        float epoch_loss = 0.0f;
```
Here, we train the model on the training dataset over 3 epochs, where on each epoch the model trains over the whole dataset.
```cpp
        for (std::size_t k = 0; k < indices.size(); ++k) {
            const auto& s = train_data[indices[k]];
            Tensor x = make_input(s);
            Tensor t = make_target(s);
            Tensor y = model.pred(x);

            float loss = total_mse_loss(y, t);
            Tensor dE_dy = grad_loss(y, t);
            
            model.backward(dE_dy);
            model.step(lr);

            epoch_loss += loss;
            if ((k + 1) % 10000 == 0) {
                std::chrono::steady_clock::time_point cur_time = std::chrono::steady_clock::now();
                std::println("Epoch {} step {}/{} - running avg loss = {} t = {}",
                             epoch, k + 1, indices.size(),
                             epoch_loss / static_cast<float>(k + 1),
                             std::chrono::duration_cast<std::chrono::seconds> (cur_time - begin).count());
            }
        }

        float avg_loss = epoch_loss / static_cast<float>(indices.size());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::println("Epoch {} finished. Avg loss = {} t = {}", epoch, avg_loss, std::chrono::duration_cast<std::chrono::seconds> (end - begin).count());
    }
```
This is the core of the loop, where most of the computation time is spent. For each training time step, we make a forward prediction, calculate the gradient of the loss and feed that into the neural net.

Here we use simple mean squared error loss, which is equal to the half the sum squared difference. 

$$ 
E = \frac{1}{2} \sum (t - y)^2
$$

You might notice that there is another possible loss function for classification, the sigmoid loss function. But in this case mse loss is sufficient enough.

The neural net will do backpropogation to calculate dE/dw for each weight. Then, the step function will actually update the weights according to stochastic gradient descent:
$$ \textbf{w} \leftarrow \textbf{w} - \eta \nabla E_n(\textbf{w})  $$

This form here is one of the simplest variations of gradient descent. On every step, the weights of our neural net will move in the direction that minimizes loss the most.
## Evaluating the Data
``` cpp
    // Evaluation on test set
    int correct = 0;
    for (const auto& s : test_data) {
        Tensor x = make_input(s);
        Tensor y = model.pred(x);

        const auto& yr = y.raw();
        int pred = 0;
        for (int i = 1; i < 10; ++i) {
            if (yr[i] > yr[pred]) pred = i;
        }

        if (pred == s.label) ++correct;
    }

    float acc = 100.0f * static_cast<float>(correct) / static_cast<float>(test_data.size());
    std::println("Test accuracy: {}/{} ({:.2f}%)",
                 correct, test_data.size(), acc);

    return 0;
}

```
After the training is complete, we simply evaluate the accuracy of our neural net on the test data! 

[Link to full example file](https://github.com/warg-void/Wolf/blob/main/examples/mnistClassifier.cpp)