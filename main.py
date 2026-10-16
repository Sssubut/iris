import json
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Лабораторная работа 4: Классификация и Регрессия")
templates = Jinja2Templates(directory="templates")


# ==================== КЛАССЫ НЕЙРОННЫХ СЕТЕЙ ====================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        return sigmoid(total)


class LinearNeuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)

    def feedforward(self, inputs):
        return float(np.dot(inputs, self.weights) + self.bias)


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def feedforward(self, inputs):
        hidden_outputs = np.array([neuron.feedforward(inputs) for neuron in self.hidden_layer])
        outputs = np.array([neuron.feedforward(hidden_outputs) for neuron in self.output_layer])
        return outputs

    def predict(self, inputs):
        return int(np.argmax(self.feedforward(inputs)))


class RegressionNeuralNetwork:
    def __init__(self, input_size, hidden_size):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_neuron = LinearNeuron(hidden_size)

    def feedforward(self, inputs):
        hidden_outputs = np.array([neuron.feedforward(inputs) for neuron in self.hidden_layer])
        return self.output_neuron.feedforward(hidden_outputs)

    def predict(self, inputs):
        return self.feedforward(inputs)


# ==================== ЗАГРУЗКА ВЕСОВ ПРИ ЗАПУСКЕ ====================

# 1. Загрузка сети классификации (4 -> 8 -> 3)
with open("iris_weights.json", "r", encoding="utf-8") as f:
    cls_data = json.load(f)

cls_net = SimpleNeuralNetwork(input_size=4, hidden_size=8, output_size=3)
for neuron, params in zip(cls_net.hidden_layer, cls_data["hidden"]):
    neuron.weights = np.array(params["weights"])
    neuron.bias = params["bias"]
for neuron, params in zip(cls_net.output_layer, cls_data["output"]):
    neuron.weights = np.array(params["weights"])
    neuron.bias = params["bias"]

cls_mn = np.array(cls_data["mn"])
cls_mx = np.array(cls_data["mx"])
cls_names = cls_data["names"]

# 2. Загрузка сети регрессии (3 -> 8 -> 1)
with open("iris_regression_weights.json", "r", encoding="utf-8") as f:
    reg_data = json.load(f)

reg_net = RegressionNeuralNetwork(input_size=3, hidden_size=8)
for neuron, params in zip(reg_net.hidden_layer, reg_data["hidden"]):
    neuron.weights = np.array(params["weights"])
    neuron.bias = params["bias"]

reg_net.output_neuron.weights = np.array(reg_data["output"]["weights"])
reg_net.output_neuron.bias = reg_data["output"]["bias"]

reg_mn = np.array(reg_data["mn"])
reg_mx = np.array(reg_data["mx"])
reg_unit = reg_data.get("unit", "см")


# ==================== МАРШРУТЫ (ENDPOINTS) ====================

@app.get("/", response_class=HTMLResponse)
def get_classification(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={}
    )


@app.post("/", response_class=HTMLResponse)
def post_classification(
        request: Request,
        sepal_length: float = Form(...),
        sepal_width: float = Form(...),
        petal_length: float = Form(...),
        petal_width: float = Form(...),
):
    flower = np.array([sepal_length, sepal_width, petal_length, petal_width])
    x_norm = (flower - cls_mn) / (cls_mx - cls_mn)

    probs = cls_net.feedforward(x_norm)
    pred_idx = int(np.argmax(probs))
    predicted_class = cls_names[pred_idx]

    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={
            "prediction": predicted_class,
            "probabilities": np.round(probs, 4).tolist(),
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }
    )


@app.get("/regression", response_class=HTMLResponse)
def get_regression(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={}
    )


@app.post("/regression", response_class=HTMLResponse)
def post_regression(
        request: Request,
        sepal_length: float = Form(...),
        sepal_width: float = Form(...),
        petal_width: float = Form(...),
):
    features = np.array([sepal_length, sepal_width, petal_width])
    x_norm = (features - reg_mn) / (reg_mx - reg_mn)

    # Ответ единственного выходного нейрона (без argmax)
    pred_value = reg_net.predict(x_norm)

    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={
            "prediction": round(pred_value, 2),
            "unit": reg_unit,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_width": petal_width,
        }
    )