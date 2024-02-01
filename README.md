# Power law spectral analysis method

### べき乗則に従うスペクトルの閾値の自動推定方法(1/n-Power Plot法、Shift-Log-Log Plot法)について

[べき乗則で解釈されるスペクトルの閾値及びべき乗数の自動解析方法](https://doi.org/10.1384/jsa.30.98)の論文内で使われたコードである。

べき乗則に従うスペクトルのべき乗数と閾値（べき乗増加の開始位置）の自動推定方法（1/n-Power Plot 法
およびShift-Log-Log Plot 法）を提案する．1/n-Power Plot 法ではべき乗数の事前情報がある場合に利用し，べ
き乗数の逆数をスペクトルに適用することで増加領域を直線化し閾値を求める．Shift-Log-Log Plot 法は事前
情報がない場合に利用し，閾値を考慮した両対数プロットにより，べき乗数を推定する．これらの方法を用
いて光電子収量スペクトルと電圧‐電流スペクトルから，閾値とべき乗数の自動推定を行った．



#### 動作環境

- Windows10Pro

- Anaconda又はMiniconda

- Python　3.10

  追加モジュール

  matplotlib、pandas、scipy、openpyxl、natsort

#### フォルダー・ファイル構成

```
|- data  							            # 論文内で利用した実験データ
|
|-pfit
|　 |-power_fit.py 					      # mainのライブラリー
|　 |-relu_abs_fit.py 				    # 1/nで直線化した後にFittingを行う関数
|　 |-sim_func.py 					      # 論文内で利用したSimulationに使った関数
|
|-reader
|　 |-datconv.py 					        # 理研計器社のACシリーズのファイルを読む関数
|　 |-SuperConlib2.py 				    # 自作の装置にて計測したデータを読む関数
|
|-inv_n_log2_template.ipynb			  # 1/n, Shift-Log-Log解析のテンプレート
|-JSA_fig_PYS_J_231012.ipynb		  # 論文内で利用した図
|-JSA_fig_simulation_231013.ipynb	# 論文内で利用した図
|-README.md	
|-requirements.txt

```
