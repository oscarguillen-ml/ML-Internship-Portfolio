\# Project 1: NASA Turbofan - ML Model Baselining \& Analysis



\*\*Objective:\*\* To investigate the viability of classical Machine Learning models (Linear, Tree, Ensemble) for predicting the Remaining Useful Life (RUL) on a complex time-series dataset.



---



\### Key Engineering Decisions



This analysis was defined by three core engineering decisions:



1\.  \*\*Business Metric:\*\* Implemented a custom asymmetric \*\*PHM Score\*\* (based on the original 2008 paper) that exponentially penalizes late (dangerous) predictions, aligning the model with real-world business risk.

2\.  \*\*Robust Validation:\*\* Used a \*\*`GroupKFold`\*\* validation strategy based on `unit\_number` (engine ID). This prevents time-series data leakage and ensures the model is validated against completely unseen engines.

3\.  \*\*Feature Engineering:\*\* Created two feature sets:

&nbsp;   \* \*\*"Myopic" Features:\*\* The raw sensor data at a single point in time.

&nbsp;   \* \*\*"Memory" Features:\*\* Used `groupby().rolling()` to generate rolling means (`\_mean`) and rolling standard deviations (`\_std`), giving the models temporal context (a PID-like approach).



---



\### Final Model Comparison



After 12 distinct experiments, the final results table clearly shows the trade-offs between model accuracy (RMSE) and business value (PHM Score).



| Model | RMSE | R² | PHM Score (Lower is Better) |

| :--- | :--- | :--- | :--- |

| \*\*Ridge (Poly2)\*\* | \*\*50.29\*\* | \*\*0.58\*\* | \*\*3.55e+08 (355 Million)\*\* |

| GBRT (Mem + Tun) | 49.63 | 0.57 | 2.06e+09 (2.06 Billion) |

| GBRT (Memory) | 49.32 | 0.57 | 3.28e+09 (3.28 Billion) |

| RandomForest (Mem + Tun)| 51.07 | 0.54 | 5.54e+09 (5.54 Billion) |

| LinearRegression | 52.26 | 0.55 | 1.21e+09 (1.21 Billion) |

| \*...and 7 others...\* | | | |



---



\### Conclusion \& Diagnosis



This analysis concluded that classical ML models are \*\*insufficient\*\* for this problem.



\* \*\*Best RMSE:\*\* `GBRT (Mem + Tun)` (\*\*49.63\*\*). This proves that memory features + ensembles are the most \*accurate on average\*.

\* \*\*Best Business Score:\*\* `Ridge (Poly2)` (\*\*355 Million\*\*). This "simpler" model won because it can \*\*extrapolate\*\* into "early" predictions (e.g., RUL = -10), which are cheap. Tree-based ensembles (RF, GBRT) cannot extrapolate beyond the training range (e.g., RUL = 0), causing them to fail dangerously "late" and resulting in catastrophic PHM scores.



\*\*Final Verdict:\*\* The high RMSE and PHM scores confirm these models are not production-ready. This problem domain is an ideal candidate for \*\*Deep Learning (LSTMs)\*\*, which are purpose-built for learning from sequences.

