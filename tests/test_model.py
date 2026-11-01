import unittest

from churn_ml.data import generate_employee_data
from churn_ml.model import train_model


class ChurnModelTests(unittest.TestCase):
    def test_synthetic_data_is_reproducible(self) -> None:
        first = generate_employee_data(n_samples=200, random_state=7)
        second = generate_employee_data(n_samples=200, random_state=7)

        self.assertTrue(first.equals(second))
        self.assertEqual(set(first["churn"].unique()), {0, 1})

    def test_baseline_model_trains_and_returns_metrics(self) -> None:
        data = generate_employee_data(n_samples=500, random_state=7)
        model, metrics = train_model(data)

        self.assertEqual(set(metrics), {"accuracy", "precision", "recall", "f1", "roc_auc"})
        self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))
        self.assertEqual(len(model.predict(data.drop(columns="churn").head(3))), 3)


if __name__ == "__main__":
    unittest.main()
