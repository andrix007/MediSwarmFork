from model_lit import LitModel

model = LitModel(
    in_ch=3,
    out_ch=8,
    spatial_dims=2,
    model_type="densenet",
    lr=0.0005,
    criterion_name="BCELoss",
    num_labels=8,
    seed=3
)
print(model)