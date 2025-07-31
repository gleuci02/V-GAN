import pandas as pd
import src.modules.bin.main_hics as hics
from src.modules.ss_module import HiCS, CLIQUE, ELM, CAE, PCA, UMAP, GMD
import numpy as np
from data.get_datasets import load_data
import signal
import time


def timeout_call(signum, frame):
    raise TimeoutError("Time excedeed the intended execution time")


if __name__ == "__main__":
    if __name__ == "__main__":
        X_train, X_test, y_test = load_data("20news_0")
        # signal.signal(signal.SIGALRM, handler=timeout_call)
        # signal.alarm(10)

        subspace_selection_model = UMAP(n_components=-1)
        subspace_selection_model.fit(X_train)
        df = subspace_selection_model.transform(X_train)
        print(df)
