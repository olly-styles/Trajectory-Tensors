import torch.nn as nn


def conv_encoder(input_size, output_size):
    """
    The encoder from Future Person Localization in First-Person Videos - CVPR 2018. Performance
    is better without batch norm in our experiments
    """
    encoder = nn.Sequential(
        nn.Conv1d(input_size, 32, 3),
        nn.ReLU(),
        nn.Conv1d(32, 64, 3),
        nn.ReLU(),
        nn.Conv1d(64, 128, 3),
        nn.ReLU(),
        nn.Conv1d(128, output_size, 3),
        nn.ReLU(),
        Flatten(),
    )
    return encoder


def conv_decoder(input_size):
    decoder = nn.Sequential(
        nn.ConvTranspose1d(input_size, 128, (4), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(128, 64, (4), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(64, 32, (4), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(32, 32, (4), stride=2, padding=1, dilation=1),
        nn.ReLU(),
        nn.Conv1d(32, 15, (1), stride=1, padding=0, dilation=1),
    )
    return decoder


def conv_decoder_no_channel_downsample(input_size):
    decoder = nn.Sequential(
        nn.ConvTranspose1d(input_size, 128, (4), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(128, 128, (4), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(128, 128, (4), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(128, 128, (4), stride=2, padding=1, dilation=1),
        nn.ReLU(),
        nn.Conv1d(128, 128, (1), stride=1, padding=0, dilation=1),
    )
    return decoder


def cnn_3d_encoder_9(feature_size):
    encoder = nn.Sequential(
        nn.Conv3d(15, 64, (3, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((2, 2, 3)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, feature_size, (3, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((2, 1, 2)),
        nn.BatchNorm3d(feature_size),
        Flatten(),
    )
    return encoder


def cnn_3d_encoder_18(feature_size):
    encoder = nn.Sequential(
        nn.Conv3d(15, 64, (3, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, 128, (3, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((2, 2, 2)),
        nn.BatchNorm3d(128),
        nn.Conv3d(128, feature_size, (3, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 1, 4)),
        nn.BatchNorm3d(feature_size),
        Flatten(),
    )
    return encoder


def cnn_3d_encoder_27(feature_size):
    encoder = nn.Sequential(
        nn.Conv3d(15, 64, (1, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 1, 2)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, 128, (3, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2)),
        nn.BatchNorm3d(128),
        nn.Conv3d(128, 256, (3, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((2, 2, 2)),
        nn.BatchNorm3d(256),
        nn.Conv3d(256, feature_size, (3, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2)),
        nn.BatchNorm3d(feature_size),
        Flatten(),
    )
    return encoder


def cnn_2d_1d_encoder_9(feature_size):
    encoder = nn.Sequential(
        # Downsample in space
        nn.Conv3d(15, 64, (1, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, 128, (1, 2, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 5)),
        nn.BatchNorm3d(128),
        # Downsample in time
        nn.Conv3d(128, 256, (5, 1, 1), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((2, 1, 1)),
        nn.BatchNorm3d(256),
        nn.Conv3d(256, feature_size, (3, 1, 1), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(feature_size),
    )
    return encoder


def cnn_2d_1d_encoder_18(feature_size):
    encoder = nn.Sequential(
        # Downsample in space
        nn.Conv3d(15, 64, (1, 5, 5), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 3)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, 128, (1, 3, 5), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 5, 5)),
        nn.BatchNorm3d(128),
        # Downsample in time
        nn.Conv3d(128, 256, (5, 1, 1), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((2, 1, 1)),
        nn.BatchNorm3d(256),
        nn.Conv3d(256, feature_size, (3, 1, 1), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(feature_size),
    )
    return encoder


def cnn_2d_1d_encoder_27(feature_size):
    encoder = nn.Sequential(
        # Downsample in space
        nn.Conv3d(15, 32, (1, 5, 5), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2)),
        nn.BatchNorm3d(32),
        nn.Conv3d(32, 64, (1, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 3)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, 128, (1, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 3)),
        nn.BatchNorm3d(128),
        # Downsample in time
        nn.Conv3d(128, 256, (5, 1, 1), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((2, 1, 1)),
        nn.BatchNorm3d(256),
        nn.Conv3d(256, feature_size, (3, 1, 1), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(feature_size),
    )
    return encoder


def cnn_2d_encoder_9(feature_size):
    encoder = nn.Sequential(
        # Downsample in space
        nn.Conv3d(15, 64, (1, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, feature_size, (1, 2, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 5)),
        nn.BatchNorm3d(feature_size),
    )
    return encoder


def cnn_2d_encoder_18(feature_size):
    encoder = nn.Sequential(
        # Downsample in space
        nn.Conv3d(15, 64, (1, 5, 5), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 3)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, feature_size, (1, 3, 5), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 5, 5)),
        nn.BatchNorm3d(feature_size),
    )
    return encoder


def cnn_2d_encoder_27(feature_size):
    encoder = nn.Sequential(
        # Downsample in space
        nn.Conv3d(15, 32, (1, 5, 5), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2)),
        nn.BatchNorm3d(32),
        nn.Conv3d(32, 64, (1, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 3)),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, feature_size, (1, 3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 3)),
        nn.BatchNorm3d(feature_size),
    )
    return encoder


def cnn_2d_decoder(feature_size):
    decoder = nn.Sequential(
        # Spatial upsample
        nn.ConvTranspose2d(feature_size, 256, (1, 3), stride=2, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 128, (4, 3), stride=2, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 15, (3, 4), stride=2, padding=0),
    )
    return decoder


def cnn_2d_decoder_9(feature_size):
    decoder = nn.Sequential(
        # Upsample in space
        nn.UpsamplingBilinear2d(size=(2, 3)),
        nn.ConvTranspose2d(feature_size, 128, (2, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.UpsamplingBilinear2d(size=(7, 14)),
        nn.ConvTranspose2d(128, 15, (3, 3), stride=1, padding=0),
        nn.Sigmoid(),
    )
    return decoder


def cnn_2d_decoder_18(feature_size):
    decoder = nn.Sequential(
        # Upsample in space
        nn.UpsamplingBilinear2d(size=(3, 3)),
        nn.ConvTranspose2d(feature_size, 128, (3, 5), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.UpsamplingBilinear2d(size=(14, 28)),
        nn.ConvTranspose2d(128, 15, (5, 5), stride=1, padding=0),
        nn.Sigmoid(),
    )
    return decoder


def cnn_2d_decoder_27(feature_size):
    decoder = nn.Sequential(
        # Upsample in space
        nn.UpsamplingBilinear2d(size=(2, 3)),
        nn.ConvTranspose2d(feature_size, 128, (3, 3), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.UpsamplingBilinear2d(size=(9, 20)),
        nn.ConvTranspose2d(128, 64, (5, 5), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.UpsamplingBilinear2d(size=(23, 44)),
        nn.ConvTranspose2d(64, 15, (5, 5), stride=1, padding=0),
        nn.Sigmoid(),
    )
    return decoder


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def trajectory_tensor_1d_conv_decoder(input_size):
    decoder = nn.Sequential(
        nn.ConvTranspose1d(input_size, 256, (5), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(256, 128, (5), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(128, 64, (5), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.ConvTranspose1d(64, 32, (4), stride=2, padding=0, dilation=1),
        nn.ReLU(),
        nn.Conv1d(32, 15, (1), stride=1, padding=0, dilation=1),
    )
    return decoder


def conv_spatial_decoder(input_size):
    decoder = nn.Sequential(
        nn.ConvTranspose3d(input_size, 64, (1, 1, 3), stride=1, padding=0),
        nn.ReLU(),
        nn.ConvTranspose3d(64, 32, (1, 4, 3), stride=(1, 2, 2), padding=0),
        nn.ReLU(),
        nn.ConvTranspose3d(32, 15, (1, 3, 4), stride=(1, 2, 2), padding=0),
    )
    return decoder


def cnn_3d_decoder(feature_size):
    decoder = nn.Sequential(
        nn.ConvTranspose3d(feature_size, 256, (5, 1, 1), stride=2, padding=0, dilation=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(256),
        nn.ConvTranspose3d(256, 256, (5, 1, 3), stride=2, padding=0, dilation=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(256),
        nn.ConvTranspose3d(256, 128, (5, 4, 3), stride=2, padding=0, dilation=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(128),
        nn.ConvTranspose3d(128, 15, (4, 3, 4), stride=2, padding=0, dilation=1),
    )
    return decoder


def cnn_2d_1d_decoder(feature_size):
    decoder = nn.Sequential(
        # Temporal upsample
        nn.ConvTranspose3d(feature_size, 256, (7, 1, 1), stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(256),
        nn.ConvTranspose3d(256, 256, (7, 1, 1), stride=2, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(256),
        nn.ConvTranspose3d(256, 256, (6, 1, 1), stride=3, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(256),
        # Spatial upsample
        nn.ConvTranspose3d(256, 128, (1, 1, 3), stride=(1, 2, 2), padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(128),
        nn.ConvTranspose3d(128, 64, (1, 4, 3), stride=(1, 2, 2), padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm3d(64),
        nn.ConvTranspose3d(64, 15, (1, 3, 4), stride=(1, 2, 2), padding=0),
    )
    return decoder
