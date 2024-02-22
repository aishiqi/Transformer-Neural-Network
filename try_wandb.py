import wandb
import time

run = wandb.init(
    project="Transformer-Translation",
    notes="My Transformer Translation",
    tags=["Transformer", "Translation"]
)

# print("Upload model file")
# artifact = wandb.Artifact('model', type='model')
# artifact.add_file('checkpoints/model_0.pth')
# run.log_artifact(artifact)
#
# print("Download model file")
# time.sleep(1000)

artifact = run.use_artifact(f'aishiqi/{run.project}/model:v0', type='model')
artifact_dir = artifact.download()
print(artifact_dir)