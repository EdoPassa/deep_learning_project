
import mnist_loader
import Network_1



def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network_1.Network([784, 30, 10])
    net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
    return


if __name__ == '__main__':
    main()

