from src.models.mnist import Encoder_conv, Encoder_mnist, Decoder_conv, Decoder_mnist
from src.models.mnist_stochman import (
    Encoder_stochman_conv,
    Encoder_stochman_mnist,
    Decoder_stochman_conv,
    Decoder_stochman_mnist,
)
from src.models.svhn_stochman import Encoder_stochman_svhn_conv, Decoder_stochman_svhn_conv
from src.models.svhn import Encoder_svhn_conv, Decoder_svhn_conv
from src.models.protein import Encoder_protein, Decoder_protein
from src.models.protein_stochman import Encoder_stochman_protein, Decoder_stochman_protein

encoders = {
    "mnist": Encoder_mnist,
    "mnist_conv": Encoder_conv,
    "cifar10": Encoder_conv,
    "svhn_conv": Encoder_svhn_conv,
    "protein": Encoder_protein,
}

stochman_encoders = {
    "mnist": Encoder_stochman_mnist,
    "mnist_conv": Encoder_stochman_conv,
    "cifar10": Encoder_stochman_conv,
    "svhn_conv": Encoder_stochman_svhn_conv,
    "protein": Encoder_stochman_protein,
}

decoders = {
    "mnist": Decoder_mnist,
    "mnist_conv": Decoder_conv,
    "cifar10": Decoder_conv,
    "svhn_conv": Decoder_svhn_conv,
    "protein": Decoder_protein,
}

stochman_decoders = {
    "mnist": Decoder_stochman_mnist,
    "mnist_conv": Decoder_stochman_conv,
    "cifar10": Decoder_stochman_conv,
    "svhn_conv": Decoder_stochman_svhn_conv,
    "protein": Decoder_stochman_protein,
}


def get_encoder(config, latent_size=2, dropout=0):

    name = config["dataset"]
    if not config["no_conv"]:
        name += "_conv"

    if "backend" in config and config["backend"] == "layer":
        encoder = stochman_encoders[name](latent_size, dropout)
    else:
        encoder = encoders[name](latent_size, dropout)

    return encoder


def get_decoder(config, latent_size=2, dropout=0):

    name = config["dataset"]
    if not config["no_conv"]:
        name += "_conv"

    if "backend" in config and config["backend"] == "layer":
        decoder = stochman_decoders[name](latent_size, dropout)
    else:
        decoder = decoders[name](latent_size, dropout)

    return decoder
