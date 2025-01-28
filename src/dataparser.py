import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from numpy.distutils.fcompiler import dummy_fortran_file


class DataParser:

    def __init__(self, dir: str):
        self._read_data(dir)

    def _read_data(self, dir: str) -> None:

        paths = os.listdir(dir)

        files = [[os.path.join(dir, path, file) for file in os.listdir(os.path.join(dir, path))] for path in paths]
        files = list(itertools.chain.from_iterable(files))
        files = [file for file in files if file.split(".")[-1] in ["xlsx"]]

        dfs = []
        for file in files:

            df_tx = pd.read_excel(file, sheet_name="TX")
            df_tx["Test type"] = "TX"
            df_dss = pd.read_excel(file, sheet_name="DSS")
            df_dss["Test type"] = "DSS"
            df_file = pd.concat((df_tx, df_dss), axis=0)

            df_file = df_file.loc[df_file["X"] != "Zegveld"]
            # df_file["X"] = np.where(df_file["X"] == "Zegveld", np.nan, df_file["X"])
            # df_file["Y"] = np.where(df_file["Y"] == "Zegveld", np.nan, df_file["Y"])
            # df_file["Z [mNAP]"] = np.where(df_file["Z [mNAP]"] == "Zegveld", np.nan, df_file["Z [mNAP]"])
            # df_file["Depth [m]"] = np.where(df_file["Depth [m]"] == "Zegveld", np.nan, df_file["Depth [m]"])

            df_file.drop(columns=["Test ID"], inplace=True)
            df_file["Campaign"] = file.split(".")[0].split("\\")[-2]
            dfs.append(df_file)

        self.df = pd.concat(dfs, axis=0)

    def get_data(self) -> pd.DataFrame:
        return self.df

    def filter_df(self, soil_type: str, test_type: str):
        df = self.df.copy()
        df = df.loc[df["Test type"] == test_type]
        df = df.loc[df["Soil type"] == soil_type]
        return df

    def plot_map(self, variable: str, soil_type: str, test_type: str, dir: str) -> None:

        df = self.filter_df(soil_type, test_type)
        x = df.loc[:, "X"].values
        y = df.loc[:, "Y"].values
        data = df.loc[:, variable].values

        fig = plt.figure()
        cm = plt.cm.get_cmap('RdYlBu')
        sc = plt.scatter(x, y, c=data, cmap=cm)
        plt.xlabel("X coordinate", fontsize=12)
        plt.ylabel("Y coordinate", fontsize=12)
        cbar = plt.colorbar(sc)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(variable, rotation=270)
        fig.suptitle(f"Test type: {test_type}\nSoil type: {soil_type}")
        plt.grid()
        plt.close()
        fig.savefig(dir)

    def plot_depth(self, variable: str, soil_type: str, test_type: str, dir: str) -> None:

        df = self.filter_df(soil_type, test_type)
        depth = df.loc[:, "Depth [m]"].values
        data = df.loc[:, variable].values

        fig = plt.figure()
        sc = plt.scatter(data, depth)
        plt.xlabel(variable, fontsize=12)
        plt.ylabel("Depth [m]", fontsize=12)
        fig.suptitle(f"Test type: {test_type}\nSoil type: {soil_type}")
        plt.grid()
        plt.gca().invert_yaxis()
        plt.close()
        fig.savefig(dir)


if __name__ =="__main__":

    dir = r"c:\Users\mavritsa\OneDrive - Stichting Deltares\Documents\Projects\RESET - Heterogeneity\Data\Delft-Schiedam lab strength tests"

    parser = DataParser(dir)

    parser.plot_map("c_u [kPa]", "Clay", "TX", r"../main/figures/map_TX_clay.png")

    parser.plot_depth("c_u [kPa]", "Clay", "TX", r"../main/figures/depth_TX_clay.png")

