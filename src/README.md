# Encoder requirements
1. **High speed** - encoding time should take under 0.04s
2. **High fidelity** - better performance than PCA **(MSE on pixels)**
3. **Small encoding length** - this will be used as a tie-breaker between 1. and 2.
4. **Low resource** - preferably running on RaspberryPi 3

# How to quantify - our metrics
## ** TBD **

# Where to start
1. Create game recordings using mario_kart_manual.py
2. Format data for encoder using *./src/preparation/make_encoder_dataset.py*

# Refferences
[Principal Component Analysis - DataScienceHandbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html) <- theory for how PCA works \
[MLP Autoencoder Keras - Stackbuse](https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/) <- still PCA but done in keras and ez to modify\
[Conv2D Autoencoder Keras - Keras.io](https://blog.keras.io/building-autoencoders-in-keras.html)\
[EnhanceNet Super Resolution Loss - Arxiv](https://arxiv.org/pdf/1612.07919.pdf) <- losses for contextual fidelity