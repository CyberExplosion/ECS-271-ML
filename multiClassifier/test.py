# Code to test the saved model as .pt file from savedModels/ folder
# Load the first .pt model from there
from typing import List
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.utils
import torch.utils.data
from loader import CoordinateDataset
from models import MLP


class Test():
    def __init__(self, evalModel: torch.nn.Module, labeledTestData: torch.utils.data.Dataset):
        self.labeledTestData = labeledTestData
        self.evalModel = evalModel
        self.evalModel.eval()
        
    def afterTrainEval(self, batch_size: int, num_workers: int):
        evalLoader = torch.utils.data.DataLoader(
            self.labeledTestData, batch_size=batch_size, num_workers=num_workers
        )
        predList = []
        trueList = []
        with torch.no_grad():
            for coordinates, labels in iter(evalLoader):
                predictions = self.evalModel(coordinates)
                predList.extend(predictions.argmax(dim=1).tolist())
                trueList.extend(labels.tolist())
                print(f"Predictions: {predictions.argmax(dim=1)} and True: {labels}")
            acc = accuracy_score(trueList, predList)
            print(classification_report(trueList, predList))
        return acc

    def test(self, coordinates: List[torch.Tensor]) -> list:
        testResult = []
        with torch.no_grad():
            for coordinate in coordinates:
                testResult.append(self.evalModel(coordinate).argmax().item())
        return testResult


# Load the test data
if __name__ == "__main__":
    testModel = MLP()
    testModel.load_state_dict(torch.load('savedModels/earlyStop_run_1.pt'))
    labeledTestData = CoordinateDataset("data/studentsdigits-train.csv")
    test = Test(testModel, labeledTestData)
    # test.afterTrainEval(512, 8)


    testData = pd.read_csv("data/studentsdigits-test.csv")
    coordinates = [torch.tensor(row[['x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6']].values, dtype=torch.float32) for _, row in testData.iterrows()]
    output = test.test(coordinates)
    print(output)
