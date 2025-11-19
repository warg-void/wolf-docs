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

const int num_pixels = 784;
const int num_classes = 10;

std::size_t load_mnist_csv(const std::string& path, std::vector<float>& x_data, std::vector<float>& t_data)
```
MNIST dataset consists data of the form (images, label), where the images have 784 pixels and there are 10 labels. In load_mist_csv, We read the csv and flatten all the 784 pixel data across all 50000 train samples into one vector. We do the same for the labels, as well as the 10000 test data.

It is important to flatten the data as this is how GPUs internally store memory, and makes batching of data faster as well.
```cpp

int main(int argc, char** argv) {
    std::vector<float> x_data;
    std::vector<float> t_data;
    std::vector<float> x_test_data;
    std::vector<float> t_test_data;
    size_t n_train_samples = load_mnist_csv(train_path.string(), x_data, t_data);
    size_t n_test_samples = load_mnist_csv(test_path.string(), x_test_data, t_test_data);

    std::println("Loaded {} train samples, {} test samples",
                 n_train_samples, n_test_samples);
```

t stands for target, since y will stand for our model's prediction.

Before we proceed to train our neural net, let's briefly go through how we model tensors. Internally, Tensor stores a vector of data, just like x_data, along with the rows and columns. Usually, rows refer to the batch size and cols refer to the feature dimension.

Meanwhile, TensorView, is a non-owning version of Tensor, which contains the pointer to the data instead. The  TensorView API is recommended because it avoids redundant CPU - GPUs transfers. 

```cpp
class Tensor {
private:
    std::vector<float> data;
    size_t rows;
    size_t cols;
}

struct TensorView {
    float* data;
    size_t rows, cols;
}
```

## Building the Model
``` cpp
    // Build 2 layer model: 784 -> 128 -> ReLU -> 10
    Sequential model(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    );

    float lr = 0.1f;
    size_t epochs = 5;
    size_t batch_size = 5;
    OptimVariant cfg = SGD{lr};
    model.set_optimizer(cfg);

    std::mt19937 gen(std::random_device{}());
    BatchMaker batcher(n_train_samples);
```
Here we call Sequential, which gives an easy way to add layers together. Our model has one hidden ReLU layer, and takes a 784-sized input to give a 10-sized output. This 10-sized output is one-hot encoded, so the first output is confident the neural net thinks the image is a '0' and so on. 

We set the optimizer to be stochastic gradient descent. There are also other optimizers like
`Momentum{lr, momentum}` or `RMSProp{lr, alpha, eps}`

BatchMaker is a helper class to make batches efficiently.

We train the model on the training dataset over 5 epochs, and a batch size of 5. On each epoch the model trains over the whole dataset. Batch sizes greater than 1 reduce the time taken per epoch and are better able to take advantage of parallelization.
```cpp
        
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        batcher.shuffle(gen);

        for (size_t s = 0; s < n_train_samples; s += batch_size) {
            size_t current_bs = std::min(batch_size, n_train_samples - s);
            TensorView x_batch = batcher.x_batch(x_data, num_pixels, s, current_bs);
            TensorView t_batch = batcher.t_batch(t_data, num_classes, s, current_bs);
            TensorView y_batch = model.pred(x_batch);
            model.grad_loss(y_batch, t_batch);
            model.backward();
            model.step(current_bs);

            // End of core training loop
        }
    }
```
This is the core of the loop, where most of the computation time is spent. For each training time step, we make a forward prediction, calculate the gradient of the loss and feed that into the neural net.

Here, by default we use simple mean squared error loss, which is equal to the half the sum squared difference. 

$$ 
E = \frac{1}{2} \sum (t - y)^2
$$

You might notice that there is another possible loss function for classification, the cross-entropy loss function. But in this case mse loss is sufficient enough.

The neural net will do backpropogation to calculate dE/dw for each weight. Then, the step function will actually update the weights according to stochastic gradient descent:
$$ \textbf{w} \leftarrow \textbf{w} - \eta \nabla E_n(\textbf{w})  $$

This form here is one of the simplest variations of gradient descent. On every step, the weights of our neural net will move in the direction that minimizes loss the most.

We skip over the code in the example file that logs the runtime and loss over epochs and move on to testing the data. 
## Evaluating the Data
``` cpp
    // Evaluation on test set
    int correct = 0;
    for (size_t j = 0; j < n_test_samples; j++) {
        TensorView x_batch = make_batch_view(x_test_data, num_pixels, j, 1);
        TensorView y_batch = model.pred(x_batch);

        const auto& yr = y_batch.data;
        int pred = 0;
        for (int i = 1; i < num_classes; ++i) {
            if (yr[i] > yr[pred]) pred = i;
        }

        if (t_test_data[j * 10 + pred]) ++correct;
    }

    float acc = 100.0f * static_cast<float>(correct) / static_cast<float>(n_test_samples);
    std::println("Test accuracy: {}/{} ({:.2f}%)",
                 correct, n_test_samples, acc);

    return 0;
}

```
After the training is complete, we simply evaluate the accuracy of our neural net on the test data! On my testings, the example program gets an accuracy of about 96.7% accuracy after 5 seconds of training. 

[Link to full example file](https://github.com/warg-void/Wolf/blob/main/examples/mnistClassifier.cpp)