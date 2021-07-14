import os



def main():
    count = 1
    device = "cuda:0"

    l_n_layers = [3,4,5,6,7,8]
    l_weighted_mse_min_weight=[0.2, 0.9]
    l_lr = [5e-6, 1e-5, 5e-5]
    l_weight_decay = [1e-2, 1e-5, 1e-9]
    l_grad_acc = [2, 4]

    for n_layers in l_n_layers:
        for weighted_mse_min_weight in l_weighted_mse_min_weight:
            for lr in l_lr:
                for weight_decay in l_weight_decay:
                    for gradient_accumulation in l_grad_acc:
                        os.system(f"python -m examples.ablation_rp --count={count} --device={device} --n_layers={n_layers} --weighted_mse_min_weight={weighted_mse_min_weight} --lr={lr} --weight_decay={weight_decay} --gradient_accumulation={gradient_accumulation}")
                        count += 1

main()