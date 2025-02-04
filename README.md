Parallel Spike-Driven Transformer (PSAOT):

The ParallelSpikeDrivenTransformer class simulates multiple spike-driven neurons and applies a transformer mechanism to the heart rate signal to enhance predictions.
You apply this transformer to the BVP data (transformed_bvp), which is then used for heart rate calculation by finding the peaks in the transformed data.
Integration with Heart Rate Calculation:

The transformer-based optimization is applied to the BVP data before detecting peaks and calculating heart rates.
Metrics:

You compute metrics (e.g., MAE, RMSE, accuracy) based on the optimized heart rate values from the transformer model.
Results Storage:

All results, including metrics, are saved to an Excel file (combined_hr_results_psaot.xlsx).
By using a Parallel Spike-Driven Transformer, this approach enhances the prediction quality of heart rates from BVP data and can improve your analysis accuracy for each video segment.






Using the dataset vital videos and UBSC
