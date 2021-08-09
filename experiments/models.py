# Internal
from experiments.network_components import (
    conv_encoder,
    conv_decoder,
    cnn_3d_encoder_9,
    cnn_3d_encoder_18,
    cnn_3d_encoder_27,
    cnn_2d_1d_encoder_9,
    cnn_2d_1d_encoder_18,
    cnn_2d_1d_encoder_27,
    cnn_2d_encoder_9,
    cnn_2d_encoder_18,
    cnn_2d_encoder_27,
    trajectory_tensor_1d_conv_decoder,
    cnn_2d_decoder_9,
    cnn_2d_decoder_18,
    cnn_2d_decoder_27,
    conv_decoder_no_channel_downsample,
    conv_spatial_decoder,
    cnn_3d_decoder,
    cnn_2d_1d_decoder,
    cnn_2d_decoder,
)

# External
import torch
import torch.nn as nn


class FullyConnectedClassifier(nn.Module):
    def __init__(self, device, num_cameras, input_size, output_size):
        super(FullyConnectedClassifier, self).__init__()
        self.device = device
        self.classifiers = []
        self.sigmoid = nn.Sigmoid()
        for camera_num in range(num_cameras):
            self.classifiers.append(nn.Linear(input_size, output_size))
        for i, classifier in enumerate(self.classifiers):
            self.add_module(str(i), classifier)

    def forward(self, input, cam):
        outputs = []
        # Loop through batch as different input cameras are processed by differenet classifiers
        for this_input, this_camera in zip(input, cam):
            output = self.classifiers[this_camera](this_input)
            outputs += [output]
        outputs = torch.stack(outputs, 0)
        outputs = self.sigmoid(outputs)
        return outputs


class FullyConnectedTrajectoryTensorClassifier(nn.Module):
    def __init__(self, device, input_size, output_size):
        super(FullyConnectedTrajectoryTensorClassifier, self).__init__()
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.classifier(input)
        output = self.sigmoid(output)
        return output


class RecurrentEncoder(nn.Module):
    def __init__(self, device, num_cameras, input_size, num_hidden_units, recurrence_type):
        super(RecurrentEncoder, self).__init__()
        self.device = device
        self.encoders = []
        self.embeddings = []
        self.num_hidden_units = num_hidden_units
        self.recurrence_type = recurrence_type
        if recurrence_type == "gru":
            for _ in range(num_cameras):
                self.encoders.append(nn.GRUCell(num_hidden_units, num_hidden_units))
                self.embeddings.append(nn.Linear(input_size, num_hidden_units))
        elif recurrence_type == "lstm":
            for _ in range(num_cameras):
                self.encoders.append(nn.LSTMCell(num_hidden_units, num_hidden_units))
                self.embeddings.append(nn.Linear(input_size, num_hidden_units))
        else:
            print("Error: Recurrence type ", recurrence_type, " not recognized")
            exit(1)

        for i, (encoder, embedding) in enumerate(zip(self.encoders, self.embeddings)):
            self.add_module(str((i * 3)), encoder)
            self.add_module(str((i * 3) + 2), embedding)

    def forward(self, input, cam):
        outputs = []
        # Loop through batch as different input cameras are processed by differenet classifiers
        for this_input, this_camera in zip(input, cam):
            hidden_unit = torch.zeros(1, self.num_hidden_units, dtype=torch.float).to(self.device)
            # GRU Encoder
            if self.recurrence_type == "gru":
                for i in range(this_input.size()[0]):
                    inp = this_input[i, :]
                    inp = inp.unsqueeze(0)
                    inp = self.embeddings[this_camera](inp)
                    hidden_unit = self.encoders[this_camera](inp, hidden_unit)
            # LSTM encoder
            elif self.recurrence_type == "lstm":
                context_unit = torch.zeros(1, self.num_hidden_units, dtype=torch.float).to(self.device)
                for i in range(this_input.size()[0]):
                    inp = this_input[i, :]
                    inp = inp.unsqueeze(0)
                    inp = self.embeddings[this_camera](inp)
                    hidden_unit, context_unit = self.encoders[this_camera](inp, (hidden_unit, context_unit))
            else:
                print("Error: Recurrence type ", self.recurrence_type, " not recognized")
                exit(1)
            outputs += [hidden_unit]
        outputs = torch.stack(outputs, 0).squeeze()
        return outputs


class ConvolutionalEncoder(nn.Module):
    def __init__(self, device, num_cameras, input_size, output_size):
        super(ConvolutionalEncoder, self).__init__()
        self.device = device
        self.encoders = []
        for camera_num in range(num_cameras):
            self.encoders.append(conv_encoder(input_size, output_size))
        for i, classifier in enumerate(self.encoders):
            self.add_module(str(i), classifier)

    def forward(self, input, cam):
        outputs = []
        # Loop through batch as different input cameras are processed by differenet encoders
        for this_input, this_camera in zip(input, cam):
            this_input = this_input.unsqueeze(0).permute(0, 2, 1)
            output = self.encoders[this_camera](this_input)
            outputs += [output]

        outputs = torch.stack(outputs, 0)
        outputs = outputs.squeeze()
        return outputs


class CNN_3D_Encoder(nn.Module):
    def __init__(self, device, output_size, heatmap_size):
        super(CNN_3D_Encoder, self).__init__()
        self.device = device
        if heatmap_size == (9, 16):
            self.encoder = cnn_3d_encoder_9(output_size)
        if heatmap_size == (18, 32):
            self.encoder = cnn_3d_encoder_18(output_size)
        if heatmap_size == (27, 48):
            self.encoder = cnn_3d_encoder_27(output_size)

    def forward(self, input):
        outputs = self.encoder(input)
        return outputs


class CNN_3D_Decoder(nn.Module):
    def __init__(self, device, input_size):
        super(CNN_3D_Decoder, self).__init__()
        self.device = device
        self.decoder = cnn_3d_decoder(input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        outputs = self.decoder(input)
        outputs = self.sigmoid(outputs)
        return outputs


class CNN_2D_1D_Decoder(nn.Module):
    def __init__(self, device, input_size):
        super(CNN_2D_1D_Decoder, self).__init__()
        self.device = device
        self.decoder = cnn_2d_1d_decoder(input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        outputs = self.decoder(input)
        outputs = self.sigmoid(outputs)
        return outputs


class CNN_2D_1D_Encoder(nn.Module):
    def __init__(self, device, output_size, heatmap_size):
        super(CNN_2D_1D_Encoder, self).__init__()
        self.device = device
        if heatmap_size == (9, 16):
            self.encoder = cnn_2d_1d_encoder_9(output_size)
        if heatmap_size == (18, 32):
            self.encoder = cnn_2d_1d_encoder_18(output_size)
        if heatmap_size == (27, 48):
            self.encoder = cnn_2d_1d_encoder_27(output_size)

    def forward(self, input):
        outputs = self.encoder(input)
        outputs = torch.squeeze(outputs)
        return outputs


class CNN_2D_Encoder(nn.Module):
    def __init__(self, device, output_size, heatmap_size):
        super(CNN_2D_Encoder, self).__init__()
        self.device = device
        if heatmap_size == (9, 16):
            self.encoder = cnn_2d_encoder_9(output_size)
        if heatmap_size == (18, 32):
            self.encoder = cnn_2d_encoder_18(output_size)
        if heatmap_size == (27, 48):
            self.encoder = cnn_2d_encoder_27(output_size)

    def forward(self, input):
        outputs = self.encoder(input)
        outputs = torch.squeeze(outputs)
        return outputs


class Trajectory_Tensor_CNN_2D_Decoder(nn.Module):
    def __init__(self, device, feature_size):
        super(Trajectory_Tensor_CNN_2D_Decoder, self).__init__()
        self.device = device
        self.decoder = cnn_2d_decoder(feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        outputs = []

        input = input.unsqueeze(-1).unsqueeze(-1)
        for timestep in range(input.shape[2]):
            output = self.decoder(input[:, :, timestep])
            outputs += [output]

        outputs = torch.stack(outputs, 2)
        outputs = outputs.squeeze()
        outputs = self.sigmoid(outputs)
        return outputs


class TrajectoryTensorGRU(nn.Module):
    def __init__(self, device, input_size, num_hidden_units):
        super(TrajectoryTensorGRU, self).__init__()
        self.device = device
        self.gru = nn.GRUCell(input_size, num_hidden_units)
        self.num_hidden_units = num_hidden_units

    def forward(self, input):
        hidden = torch.zeros(input.size(0), self.num_hidden_units, dtype=torch.float).to(self.device)
        for i in range(input.size()[-1]):
            inp = input[:, :, i]
            hidden = self.gru(inp, hidden)

        return hidden


class ConvolutionalDecoder(nn.Module):
    def __init__(self, device, num_cameras, input_size, output_size, downsample_channels=True):
        super(ConvolutionalDecoder, self).__init__()
        self.device = device
        self.decoders = []
        self.sigmoid = nn.Sigmoid()
        for camera_num in range(num_cameras):
            if downsample_channels:
                self.decoders.append(conv_decoder(input_size))
            else:
                self.decoders.append(conv_decoder_no_channel_downsample(input_size))
        for i, classifier in enumerate(self.decoders):
            self.add_module(str(i), classifier)

    def forward(self, input, cam):
        outputs = []
        # Loop through batch as different input cameras are processed by differenet decoders
        for this_input, this_camera in zip(input, cam):
            this_input = this_input.reshape(1, 128, 2)
            output = self.decoders[this_camera](this_input)
            outputs += [output]

        outputs = torch.stack(outputs, 0)
        outputs = outputs.squeeze()
        outputs = self.sigmoid(outputs)
        return outputs


class RecurrentDecoder(nn.Module):
    def __init__(self, device, num_cameras, input_size, output_size, num_timesteps, recurrence_type):
        super(RecurrentDecoder, self).__init__()
        self.device = device
        self.decoders = []
        self.classifiers = []
        self.num_timesteps = num_timesteps
        self.recurrence_type = recurrence_type
        self.input_size = input_size
        if recurrence_type == "gru":
            for _ in range(num_cameras):
                self.decoders.append(nn.GRUCell(input_size, input_size))
                self.classifiers.append(nn.Linear(input_size, output_size))

        elif recurrence_type == "lstm":
            for _ in range(num_cameras):
                self.decoders.append(nn.LSTMCell(input_size, input_size))
                self.classifiers.append(nn.Linear(input_size, output_size))

        for i, decoder in enumerate(self.decoders):
            self.add_module(str(i), decoder)

        for i, classifier in enumerate(self.classifiers):
            self.add_module(str(i + len(self.decoders)), classifier)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, cam):
        all_outputs = []
        # Loop through batch as different input cameras are processed by differenet classifiers
        for this_input, this_camera in zip(input, cam):
            outputs = []
            this_input = this_input.unsqueeze(0)
            hidden_unit = torch.zeros(1, self.input_size, dtype=torch.float).to(self.device)
            # GRU Decoder
            if self.recurrence_type == "gru":
                for i in range(self.num_timesteps):
                    hidden_unit = self.decoders[this_camera](this_input, hidden_unit)
                    output = self.classifiers[this_camera](hidden_unit)
                    output = self.sigmoid(output)
                    outputs += [output]
            # LSTM decoder
            elif self.recurrence_type == "lstm":
                context_unit = torch.zeros(1, self.input_size, dtype=torch.float).to(self.device)
                for i in range(self.num_timesteps):
                    hidden_unit, context_unit = self.decoders[this_camera](this_input, (hidden_unit, context_unit))
                    output = self.classifiers[this_camera](hidden_unit)
                    output = self.sigmoid(output)
                    outputs += [output]
            else:
                print("Error: Recurrence type ", self.recurrence_type, " not recognized")
                exit(1)

            outputs = torch.stack(outputs, 2).squeeze()
            all_outputs += [outputs]
        all_outputs = torch.stack(all_outputs, 0).squeeze()
        return all_outputs


class RecurrentTemporalDecoder(nn.Module):
    def __init__(self, device, num_cameras, input_size, num_timesteps, recurrence_type):
        super(RecurrentTemporalDecoder, self).__init__()
        self.device = device
        self.decoders = []
        self.num_timesteps = num_timesteps
        self.recurrence_type = recurrence_type
        self.input_size = input_size
        if recurrence_type == "gru":
            for _ in range(num_cameras):
                self.decoders.append(nn.GRUCell(input_size, input_size))

        elif recurrence_type == "lstm":
            for _ in range(num_cameras):
                self.decoders.append(nn.LSTMCell(input_size, input_size))

        for i, decoder in enumerate(self.decoders):
            self.add_module(str(i), decoder)

    def forward(self, input, cam):
        all_outputs = []
        # Loop through batch as different input cameras are processed by differenet classifiers
        for this_input, this_camera in zip(input, cam):
            outputs = []
            this_input = this_input.unsqueeze(0)
            hidden_unit = torch.zeros(1, self.input_size, dtype=torch.float).to(self.device)
            # GRU Decoder
            if self.recurrence_type == "gru":
                for i in range(self.num_timesteps):
                    hidden_unit = self.decoders[this_camera](this_input, hidden_unit)
                    outputs += [hidden_unit]
            # LSTM decoder
            elif self.recurrence_type == "lstm":
                context_unit = torch.zeros(1, self.input_size, dtype=torch.float).to(self.device)
                for i in range(self.num_timesteps):
                    hidden_unit, context_unit = self.decoders[this_camera](this_input, (hidden_unit, context_unit))
                    outputs += [hidden_unit]
            else:
                print("Error: Recurrence type ", self.recurrence_type, " not recognized")
                exit(1)

            outputs = torch.stack(outputs, 2).squeeze()
            all_outputs += [outputs]
        all_outputs = torch.stack(all_outputs, 0).squeeze()
        return all_outputs


class CNN_1D_Trajectory_Tensor_Classifier(nn.Module):
    def __init__(self, device, input_size):
        super(CNN_1D_Trajectory_Tensor_Classifier, self).__init__()
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.decoder = trajectory_tensor_1d_conv_decoder(input_size)

    def forward(self, input):
        input = input.unsqueeze(2)
        outputs = self.decoder(input)
        outputs = self.sigmoid(outputs)
        return outputs


class RecurrentDecoderTrajectoryTensor(nn.Module):
    def __init__(self, device, input_size, output_size, num_timesteps, num_hidden_units):
        super(RecurrentDecoderTrajectoryTensor, self).__init__()
        self.device = device
        self.num_timesteps = num_timesteps
        self.input_size = input_size
        self.decoder = nn.GRUCell(input_size, input_size)
        self.num_hidden_units = num_hidden_units
        self.classifier = nn.Linear(num_hidden_units, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        outputs = []
        hidden_unit = torch.zeros(input.size(0), self.num_hidden_units, dtype=torch.float).to(self.device)
        # GRU Decoder
        for i in range(self.num_timesteps):
            output = self.decoder(input, hidden_unit)
            output = self.classifier(output)
            output = self.sigmoid(output)
            outputs += [output]

        outputs = torch.stack(outputs, 2).squeeze()
        return outputs


class CNN_2D_Decoder(nn.Module):
    def __init__(self, device, feature_size, num_cameras, heatmap_size=(9, 16)):
        super(CNN_2D_Decoder, self).__init__()
        self.device = device
        self.decoders = []
        self.sigmoid = nn.Sigmoid()
        for camera_num in range(num_cameras):
            self.decoders.append(conv_spatial_decoder(feature_size))
        for i, classifier in enumerate(self.decoders):
            self.add_module(str(i), classifier)

    def forward(self, input, cam):
        outputs = []
        # Loop through batch as different input cameras are processed by differenet decoders
        for this_input, this_camera in zip(input, cam):
            this_input = this_input.unsqueeze(0).unsqueeze(3).unsqueeze(4)
            output = self.decoders[this_camera](this_input)
            outputs += [output]

        outputs = torch.stack(outputs, 0)
        outputs = outputs.squeeze()
        outputs = self.sigmoid(outputs)
        return outputs
