from sklearn.datasets import load_digits
import matplotlib.pyplot as plot

if __name__ == "__main__":
    print("Hello World")
    digits = load_digits()
    print(digits.data.shape)
    plot.gray()
    plot.matshow(digits.images[0])
    plot.show()
