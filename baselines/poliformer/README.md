# PoliFormer

This adapter runs PoliFormer for object navigation. Its model and dataset
dependencies are installed separately from the NavVerse simulation environment
by the setup script. Review the upstream [PoliFormer repository](https://github.com/tccoin/PoliFormer)
for model terms and hardware requirements.

```bash
bash baselines/poliformer/poliformer_setup.sh
navverse-run benchmark --preset benchmark_poliformer
```

The preset expects the checkpoint at
`baselines/submodules/poliformer/ckpt/text_nav/model.ckpt`.

Setup patches the upstream argument parser to accept NavVerse's additional
command-line options. The adapter disables gradient recording during inference
so recurrent state does not retain training graphs between observations.
