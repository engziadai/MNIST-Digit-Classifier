import torch
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms
from MNISTModel import MNISTNN

class MNISTAPP:
    def __init__(self, master):
        self.master = master
        self.model = MNISTNN()
        self.model.load_state_dict(torch.load('MNISTNN.pth'))
        self.model.eval()

        self.canvas = Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)

        self.button_frame = Frame(master)
        self.button_frame.pack()

        self.button_predict = Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.button_predict.pack(side=LEFT, padx=10)

        self.button_clear = Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(side=LEFT, padx=10)

        self.label_result = Label(master, text="Draw a digit!")
        self.label_result.pack(pady=10)

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.label_result.config(text="Draw a digit!")

    def predict_digit(self):
        image = self.image.resize((28, 28))
        image = ImageOps.invert(image)
        transform = transforms.ToTensor()
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            predicted = torch.argmax(output, 1).item()
            self.label_result.config(text=f"Predicted Digit: {predicted}")

if __name__ == "__main__":
    root = Tk()
    root.title('MNIST Digit Classifier')
    app = MNISTAPP(root)
    root.mainloop()