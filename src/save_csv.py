# A code snippet to help you save your results into a kaggle accepted csv
import pandas as pd

# Usage results_to_csv(clf.predict(X_test))


def results_to_csv(y_test, submit_path):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(submit_path, index_label='Id')
