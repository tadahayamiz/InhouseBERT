# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

prepare dataloader

@author: tadahaya
"""
import time
import datetime
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from itertools import chain
from tqdm.auto import tqdm, trange

FIELD = [
    "Dataset",	"DiseaseType", "SpecimenID", "Enzyme", "Position", "ID",
    "Number", "X", "Y", "FITC", "mCherry", "Date"
    ]

SEP = os.sep

class Preprocess:
    """
    preprocessor, 基本的にハード
    
    ID: 2nd_H_0404_6PGDH2_1
        2ndデータセット
            1st, 2nd, astと三つある. astってなんだろうか
        健常人ラベル
            healty OR cancerで問題ない
        血液検体IDが0404
            検体IDで問題ない
        6PGDH2
            これかLDHの二つなので見ている酵素だろう
        1
            1-12までしかないので視野番号だろう
            Positionと他では表記されていた
    Number
        各視野内での輝点番号
    X, Y
        各視野内でのX, Y座標
    FITC
        プローブ蛍光由来の強度
    mCherry
        reference色素の強度
    Date
        測定日付
    
    """
    def __init__(
            self, url, fileout:str="", sep:str=",", v_name:str="FITC"
            ):
        df = pd.read_csv(url, sep=sep)
        ids = df["ID"].values.flatten().tolist()
        split = []
        n_item = 0
        for i in ids:
            tmp = i.split("_")
            if n_item==0:
                n_item = len(tmp)
            else:
                if n_item!=len(tmp):
                    print(i)
                    raise ValueError
            split.append(tmp)
        df_id = pd.DataFrame(split)
        self.id_col = ["Dataset", "DiseaseType", "SpecimenID", "Enzyme", "Position"]
        df_id.columns = self.id_col
        comb = pd.concat([df_id, df], axis=1, join="inner")
        self.data = comb
        # preprocessing
        self._fix(v_name)
        # export
        if len(fileout) == 0:
            ext = url.split(".")[-1]
            fileout = url.replace(f".{ext}", f"_modified.{ext}")
        self.data.to_csv(fileout, sep=sep)


    def _fix(self, v_name:str="FITC"):
        """ mainの値であるFITCに変なstrが混じっていたので対処 """
        before = self.data.shape
        def convert(x):
            try:
                return float(x)
            except ValueError:
                return np.nan
        self.data.loc[:, v_name] = self.data.loc[:, v_name].map(convert)
        self.data = self.data.dropna(subset=[v_name])
        # print(before, self.data.shape)


    def check_content(self):
        """ 中身を確認する """
        count = []
        for c in self.id_col:
            print("====================================", c)
            tmp = self.data[c].values.flatten().tolist()
            cntr = Counter(tmp)
            count.append(cntr)
            for k, v in cntr.items():
                print(k, v)


class Data:
    """
    data格納モジュール, 基本的にハード
        
    """
    def __init__(self, input):
        # 読み込み
        self.data = None
        if type(input) == str:
            self.data = pd.read_csv(input, index_col=0)
        elif type(input) == type(pd.DataFrame()):
            self.data = input
        else:
            raise ValueError("!! Provide url or dataframe !!")            
        # データの中身の把握
        col = list(self.data.columns)
        self.dic_components = dict()
        for c in col:
            tmp = self.data[c].values.flatten().tolist()
            self.dic_components[c] = Counter(tmp)


    def conditioned(self, condition:dict):
        """ 解析対象とするデータを条件付けする """
        for k, v in condition.items():
            try:
                self.data = self.data[self.data[k]==v]
            except KeyError:
                raise KeyError("!! Wrong key in condition: check the keys of condition !!")


    def sample(
        self, sid:int, n_sample:int=256, ratio:float=0.9,
        v_name:str="value", s_name:str="sample_id"
        ):
        """
        指定した検体からn_sampleの回の輝点のサンプリングを行う

        Parameters
        ----------
        sid: int
            Specimen ID, 検体をidentifyする

        v_name: str
            valueカラムの名称
            DPPVIの場合は素直にvalue

        """
        tmp = self.data[self.data[s_name]==sid]
        dim = int(tmp.shape[0] * ratio)
        res = np.zeros((n_sample, dim))
        for i in range(n_sample):
            res[i, :] = tmp.sample(n=dim)[v_name].values
        return res


    def prep_data(
            self, samplesize:int=10000, ratio:float=0.9,
            shuffle:bool=True, v_name:str="value", s_name:str="sample_id"
            ):
        """ 指定したsamplesizeまでサンプリングを行う """
        specimens = set(list(self.data[s_name]))
        print(f"> handling {len(specimens)} specimens")
        n_sample = samplesize // len(specimens) # n_sampleを決める
        res = []
        specimen = []
        for s in tqdm(specimens):
            tmp = self.sample(s, n_sample, ratio, v_name, s_name)
            tmp = [v[0] for v in np.split(tmp, n_sample, axis=0)]
            ## 1個中に入るため
            res.append(tmp)
            specimen.append([s] * n_sample)
        res = list(chain.from_iterable(res))
        specimen = list(chain.from_iterable(specimen))
        if shuffle:
            rng = np.random.default_rng()
            idx = list(range(len(res)))
            rng.shuffle(idx)
            res = [res[i] for i in idx]
            specimen = [specimen[i] for i in idx]
        return res, specimen
    

    def imshow(
            self, sid:int, bins:int=64, ratio:float=0.9, condition:dict=dict(),
            outdir:str="", v_name:str="value", s_name:str="sample_id",
            figsize=(), fontsize:int=16
            ):
        """ 指定したIDの画像を表示する """
        # data
        if len(condition) > 0:
            self.conditioned(condition)
        data = self.sample(sid, 2, ratio, v_name, s_name)[0]
        # show
        if len(figsize) > 0:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        plt.rcParams["font.size"] = fontsize
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        ax.hist(data, color="black", bins=bins)
        if len(outdir) > 0:
            plt.savefig(outdir + SEP + f"hist_{sid}.png")
        plt.show()


class DataMaker:
    """
    dataを読み込んでヒストグラムを表すnp配列へと変換する
    
    """
    def __init__(self, pixel:tuple=(64, 64), bins:tuple=(64, 32)):
        self.pixel = pixel
        self._dpi = 100
        self._figsize = pixel[0] / self._dpi, pixel[1] / self._dpi
        self.bins = bins
        self.data = None
        self.specimen = None
        self.limit = None


    def set_data(self, data, test_bins=None, limit:tuple=()):
        """ setter """
        self.data = data
        if len(limit) == 0:
            vmin = [np.min(v) for v in data]
            vmax = [np.max(v) for v in data]
            limit = (np.min(vmin), np.max(vmax))
        self.limit = limit
        # plot
        if test_bins is None:
            test_bins = self.bins[0]
        self._test_view(data, test_bins, limit)

    
    def set_specimen(self, specimen):
        """ specimen ID """
        self.specimen = specimen


    def main(
            self, outdir:str="", name:str="", ratio:float=0.9,
            bins:tuple=(), test_view:bool=True, limit:tuple=()
            ):
        """
        dataをまとめてhistogram arrayへと変換, npzで保存する
        input array, output arrayの順
        各arrayはsample, h, wの順
        time consuming, 1000回すのに5 min程度かかる
        
        Parameters
        ----------
        ratio: float, (0, 1)
            サンプルからデータ点を取得する割合
        
        bins: tuple
            前者がinput, 後者がoutputのヒストグラムのbinsを指定する
            AE的に組むため, 前者が大きいことが前提

        """
        assert len(outdir) > 0
        assert (ratio > 0) & (ratio < 1) 
        # dataの準備
        array0 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        array1 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        if len(bins) == 0:
            bins = self.bins
        assert bins[0] >= bins[1]
        for i, d in tqdm(enumerate(self.data)):
            # dataのhistogram化
            img0 = self.get_hist_array(d, bins=bins[0])
            img1 = self.get_hist_array(d, bins=bins[1])
            # imageのbinarize化と格納
            array0[i, :, :] = self.binarize(img0)
            array1[i, :, :] = self.binarize(img1)
        # NHWC形式になるように次元を追加
        array0 = array0[..., np.newaxis]
        array1 = array1[..., np.newaxis]
        # npzで保存
        if test_view:
            print("input example")
            self.imshow(array0[0, :, :, 0])
            print("output example")
            self.imshow(array1[0, :, :, 0])
        now = datetime.datetime.now().strftime('%Y%m%d')
        if len(name) > 0:
            fileout = outdir + SEP + f"dataset_{name}_{now}.npz"
        else:
            fileout = outdir + SEP + f"dataset_{now}.npz"
        np.savez_compressed(
            fileout, input=array0, output=array1, specimen=self.specimen
            )


    def get_hist_array(self, data, bins=30):
        """
        データをヒストグラムへと変換し, そのarrayを得る
        
        """
        # prepare histogram
        fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        # ax.hist(
        #     data, color="black", density=True, bins=bins, range=self.limit
        #     )
        # rangeを考慮しない方針
        ax.hist(data, color="black", bins=bins)
        # convert array
        fig.canvas.draw() # レンダリング
        data = fig.canvas.tostring_rgb() # rgbのstringとなっている
        w, h = fig.canvas.get_width_height()
        c = len(data) // (w * h) # channelを算出
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
        plt.close()
        return img


    # def binarize(self, data):
    #     """ 得られたarrayを二値化する """
    #     data = (data == 0).sum(axis=2) # h, w, cであり, blackなので255, 0のみとなっている
    #     data = (data > 0).astype(np.uint8)
    #     return data


    def binarize(self, data):
        """ 得られたarrayを二値化する """
        data = data.sum(axis=2) # h, w, cであり, blackなので255, 0のみとなっている
        data = np.where(data > 0, 255, 0)
        return data


    def imshow(self, data, cmap='binary_r', figsize=None):
        """ show pixelized data """
        plt.figure(figsize=figsize)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        plt.imshow(data, aspect='equal', cmap=cmap)
        plt.show()

    
    def _test_view(self, data, test_bins, limit):
        """ refer to set_data """
        num = 4
        idx = list(range(len(data)))
        np.random.shuffle(idx)
        fig, axes = plt.subplots(1, num, figsize=(2.5 * num, 2.5))
        for i in range(num):
            # axes[i].hist(
            #     data[idx[i]], color="black", density=True,
            #     bins=test_bins, range=limit
            #     )
            # rangeを考慮しない方針 230918
            axes[i].hist(
                data[idx[i]], color="black", bins=test_bins
                )

        plt.show()