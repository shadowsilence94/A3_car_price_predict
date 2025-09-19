import pytest
import numpy as np
from LogisticRegression import LogisticRegression

class TestLogisticRegression:
    
    def test_model_accepts_expected_input(self):
        """Test that the model takes the expected input format"""
        # Create sample data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 4, 100)
        
        # Initialize model
        model = LogisticRegression()
        
        # Test that fit accepts the input without errors
        try:
            model.fit(X, y, n_classes=4)
            assert True
        except Exception as e:
            pytest.fail(f"Model failed to accept expected input: {e}")
    
    def test_model_output_shape(self):
        """Test that the output of the model has the expected shape"""
        # Create sample data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 4, 100)
        X_test = np.random.rand(20, 5)
        
        # Initialize and train model
        model = LogisticRegression()
        model.fit(X_train, y_train, n_classes=4)
        
        # Test predict method
        predictions = model.predict(X_test)
        assert predictions.shape == (20,), f"Expected shape (20,), got {predictions.shape}"
        
        # Test that predictions are valid class labels
        assert all(pred in [0, 1, 2, 3] for pred in predictions), "Predictions contain invalid class labels"
        
        # Test accuracy method works
        accuracy = model.accuracy(y_train[:20], model.predict(X_train[:20]))
        assert 0 <= accuracy <= 1, f"Accuracy should be between 0 and 1, got {accuracy}"

if __name__ == "__main__":
    pytest.main([__file__])
