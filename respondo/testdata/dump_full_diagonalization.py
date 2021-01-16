import adcc
import pandas as pd
from static_data import xyz
from cache import cases


def main():
    ret = {}
    for case in cases:
        n_singlets = cases[case]
        molecule, basis, method = case.split("_")
        scfres = adcc.backends.run_hf(
            "pyscf", xyz=xyz[molecule],
            basis=basis,
            # conv_tol=conv_tol,
            # multiplicity=multiplicity,
            # conv_tol_grad=conv_tol_grad,
        )
        state = adcc.run_adc(method=method, data_or_matrix=scfres, n_singlets=n_singlets)
        state.to_dataframe().to_json(f"{case}.json")
        df = pd.read_json(f"{case}.json")


if __name__ == "__main__":
    main()