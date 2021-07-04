def rp_train_one_grid(n_layers, lr, weight_decay, new_checkpoint_path, one_grad_acc, use_lf):

    rp_options = {
        "model_path": "checkpoints/twostep_trf_",
        "n_layers": n_layers,
        "ratios": (0.95, 0.05, 0.0),
        "device": "cuda:0",
        "st_embed_batch_size": 32,
        "weighted_mse_min_weight": 0.5,
        "batch_size": 8,
        "epochs": 600,
        "lr": lr,
        "weight_decay": weight_decay,
        "evaluation_steps": 5000,
        "gradient_accumulation": one_grad_acc,
        "ling_subgroups": ['CKKF_', 'POSF_',  # 'PhrF_', 'TrSF_', 
              'EnDF_', 'EnGF_', 'ShaF_', 'TraF_',
              'TTRF_', 'VarF_', 'PsyF_', 'WorF_']
    }
    rp_options = edict(rp_options)

    if not use_lf:
        rp_options.ling_subgroups = ["EnGF_"]


    model = ReadabilityTransformer(
        model_path=rp_options.model_path,
        device=rp_options.device,
        new_checkpoint_path=new_checkpoint_path
    )
    
    shutil.copyfile("examples/6_rp_grid_search.py", os.path.join(new_checkpoint_path, "rp_grid_search.py"))


    commonlit_data = CommonLitDataset("train", cache=True)
    train_df, valid_df, _ = commonlit_data.split_train_valid_test(ratios=rp_options.ratios)

    # # for experiment purposes DELETE THIS AFTER
    # train_df = train_df.iloc[:50]
    # valid_df = valid_df.iloc[:50]

    ling_caches = ["train", "valid"]
    if not use_lf:
        ling_caches = ["train_no_lf", "valid_no_lf"]

    print(rp_options.ling_subgroups)
    print(ling_caches)
    train_df, valid_df = commonlit_data.apply_lingfeat_features(
        [train_df, valid_df], 
        cache_ids=ling_caches,
        cache=True,
        subgroups=rp_options.ling_subgroups,
        normalize=True
    )
    train_embed, valid_embed = commonlit_data.apply_st_embeddings(
        [train_df, valid_df],
        st_model=model.st_model,
        batch_size=rp_options.st_embed_batch_size
    )

    commonlit_data.save_parameters(rp_options.model_path) # allow for retrieval of this object later

    lingfeat_features = commonlit_data.get_lingfeat_features(train_df)
    st_embedding_size = commonlit_data.get_st_embedding_size(train_embed)

    train_reader = PredictionDataReader(train_df, train_embed, features=lingfeat_features)
    valid_reader = PredictionDataReader(valid_df, valid_embed, features=lingfeat_features)

    model.init_rp_model(
        rp_model_name="fully-connected",
        features=lingfeat_features,
        subgroups=rp_options.ling_subgroups,
        st_embedding_size=st_embedding_size,
        n_layers=rp_options.n_layers
    )

    stderr_stats = train_reader.get_standard_err_stats()
    rp_train_metric = WeightedRankingMSELoss(
        alpha=2.0, 
        min_err=stderr_stats["min"],
        max_err=stderr_stats["max"],
        min_weight=rp_options.weighted_mse_min_weight
    )

    lingfeat_maxes = commonlit_data.lingfeat_maxes
    lingfeat_mins = commonlit_data.lingfeat_mins
    target_max = lingfeat_maxes["target"]
    target_min = lingfeat_mins["target"]
    rp_evaluation_metrics = [MSELoss(reduction="mean"), DenormMSELoss(target_max, target_min)]

    model.rp_fit(
        train_reader=train_reader,
        valid_reader=valid_reader,
        train_metric=rp_train_metric,
        evaluation_metrics=rp_evaluation_metrics,
        st_embed_batch_size=rp_options.st_embed_batch_size,
        batch_size=rp_options.batch_size,
        epochs=rp_options.epochs,
        lr=rp_options.lr,
        weight_decay=rp_options.weight_decay,
        evaluation_steps=rp_options.evaluation_steps,
        save_best_model=True,
        show_progress_bar=False,
        gradient_accumulation=rp_options.gradient_accumulation
    )

if __name__ == "__main__":
    rp_train_grid_search()