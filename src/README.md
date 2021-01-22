# Encoder requirements
1. **High speed** - encoding time should take under 0.04s
2. **High fidelity** - better performance than PCA **(MSE on pixels)**
3. **Small encoding length** - this will be used as a tie-breaker between 1. and 2.
4. **Low resource** - preferably running on RaspberryPi 

# How to quantify - our metrics
[Loss Functions for Image Restoration - Arxiv](https://arxiv.org/pdf/1511.08861.pdf) <- structural integrity metrics \
[Image Quality Assessment Techniques - paper](http://www.ijcst.com/vol23/1/sasivarnan.pdf) <- image quality metrics

# Where to start
1. Create game recordings using mario_kart_manual.py
2. Format data for encoder using *./src/preparation/make_encoder_dataset.py*

# Results
|Type           |Epoch|Train MSE|Val MSE|
|---------------|-----|---------|-------|
|PCA            |9    |0.0181   |0.0181 |
|mlp_2lyr       |99   |0.0053   |0.0047 |
|mlp_3lyr       |91   |0.0058   |0.0057 |
|mlp_2lyr + ssim|99   |0.0053   |0.0047 |

# References

## Autoencoder references
[Principal Component Analysis - DataScienceHandbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html) <- theory for how PCA works \
[Backbone Starting Point - Topbots](https://www.topbots.com/a-brief-history-of-neural-network-architectures/) <- \
[MLP Autoencoder Keras - Stackbuse](https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/) <- still PCA but done in keras and ez to modify\
[Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf) <- contrast based encoder

## Loss references
[High Fidelity Image Compression - Github](https://hific.github.io/) <- autoencoder SOTA\
[EnhanceNet Super Resolution Loss - Arxiv](https://arxiv.org/pdf/1612.07919.pdf) <- losses for contextual fidelity\
[Pytorch Image Quality - Github](https://github.com/photosynthesis-team/piq) <- losses implemented in Pytorch