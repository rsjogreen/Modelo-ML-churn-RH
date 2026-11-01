import unittest

from domain.modelo import CATEGORICAS, NUMERICAS, resolver_features


class ResolverFeaturesTests(unittest.TestCase):
    def test_all_selects_numeric_and_categorical_columns(self):
        numericas, categoricas = resolver_features("ALL")

        self.assertEqual(numericas, NUMERICAS)
        self.assertEqual(categoricas, CATEGORICAS)

    def test_numeric_selects_only_numeric_columns(self):
        numericas, categoricas = resolver_features("NUMERIC")

        self.assertEqual(numericas, NUMERICAS)
        self.assertEqual(categoricas, [])

    def test_custom_selects_only_requested_columns(self):
        numericas, categoricas = resolver_features("CUSTOM", ["Age", "Geography"])

        self.assertEqual(numericas, ["Age"])
        self.assertEqual(categoricas, ["Geography"])

    def test_custom_rejects_unknown_or_empty_columns(self):
        for custom in ([], ["Unknown"]):
            with self.subTest(custom=custom), self.assertRaises(ValueError):
                resolver_features("CUSTOM", custom)

    def test_rejects_unsupported_selection_and_duplicates(self):
        with self.assertRaises(ValueError):
            resolver_features("UNKNOWN")
        with self.assertRaises(ValueError):
            resolver_features("CUSTOM", ["Age", "Age"])


if __name__ == "__main__":
    unittest.main()
