import os
import glob
import pandas as pd

collect_results = []
for checkpoint in glob.glob("checkpoints/ablations/eval_prediction/prediction_*"):
    hyp_text_path = os.path.join(checkpoint, "hyp.txt")
    try:
        hyp_text = open(hyp_text_path, "r").read()
        hyp_text = hyp_text.replace("Namespace", "")
        hyp_text = hyp_text[1:-1]
        params = hyp_text.split(", ")
        param_dict = dict()
        for one in params:
            split = one.split("=")
            param_name = split[0]
            param_value = split[1]
            param_dict[param_name] = param_value
        
        validation_result_file = os.path.join(checkpoint, "1_Prediction", "evaluation_results.csv")
        val_results = pd.read_csv(validation_result_file)
        denorm_rmse_losses = val_results["DenormRMSELoss"].values
        
        minimum_loss = min(denorm_rmse_losses)
        param_dict["min_loss"] = minimum_loss

        collect_results.append(param_dict)
    except:
        pass

df = pd.DataFrame(collect_results)
df = df.sort_values("min_loss")
print(df)
