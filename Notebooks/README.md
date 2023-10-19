Some of the experiments (as of end of AUG).

**TO DO:** Update with latest experiments.


### Multi-Rings

- `Softmax-Temperature`: Looking at some experiments across varying the architecture size and softax temperature. This notebook just has loss curves rn, the other plots get saved in the corresponding `code/figures/{config-id}` folder
- `Hungarian-multi-rings`: This notebook is ~ identical to the `Hungarian-multi-rings.ipynb`, but the models don't train long enough in the notebook to get nice figures. But... training for longer with the slurm batch system seems like some of these models are starting to converge, e.g, (although still needs more training time... which is in progress).

<img src="code/figures/2rings-sqrtD/loss-slots-iter45500-evt0.jpg"   />


### Multi-Blobs Hungarian Loss

- `Hungarian-multi-rings`: This notebook is the "bug fix" for the Hungarian loss implementation, and looks like it is working for the blobs :)
    * Not fully optimized yet... but definitely doing something
    * Also some experiments for looking at batch size and learning rate schedules

### Diagnosing a single ring example

I wanted to get a feeling where the loss was coming from

- `SA-mini`
- `Hungarian-1-ring-sanity-check`
- `Encoder-opt-pos-embed`
- `Encoder-optimization-rings`

### Lukas's Autoencoder notebook(s)

- `WTFAE.ipynb`: OG nb from Lukas (no edits)
    ^ This model was used for all of the above notebooks too (all of the pytorch models)
- `WTFAE.ipynb2`: Rerunning Lukas's nb 
- `WTFAE-Flos-data`: Rerunning Lukas's nb w/ Flo's dataset
    * Ran out of the box for blobs
    * The default hps / training times didn't work for rings
