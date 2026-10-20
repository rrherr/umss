"""The UMSS source-filter mixture autoencoder (upstream `models.py`, the one model the pretrained
checkpoints use; the U-Net baselines and helpers are left out)."""

import torch

from . import network_components as nc
from .ddsp import core, synths


class SourceFilterMixtureAutoencoder2(torch.nn.Module):

    """Autoencoder that encodes a mixture of n voices into synthesis parameters
    from which the mixture is re-synthesised. Synthesis of each voice is done with a
    source filter model """

    def __init__(self,
                 n_harmonics=101,
                 filter_order=20,
                 fft_size=512,
                 hop_size=256,
                 n_samples=64000,
                 return_sources=False,
                 harmonic_roll_off=12,
                 estimate_noise_mag=False,
                 f_ref=200,  # for harmonics roll off
                 encoder='SeparationEncoderSimple',
                 encoder_hidden_size=256,
                 embedding_size=128,
                 decoder_hidden_size=512,
                 decoder_output_size=512,
                 n_sources=2,
                 bidirectional=True,
                 voiced_unvoiced_diff=True
                 ):

        super().__init__()

        if harmonic_roll_off == -1:
            # estimate roll off
            output_splits=(('harmonic_amplitude', 1),
                           ('noise_gain', 1),
                           ('line_spectral_frequencies', filter_order + 1),
                           ('harmonic_roll_off', 1))
        else:
            output_splits=(('harmonic_amplitude', 1),
                           ('noise_gain', 1),
                           ('line_spectral_frequencies', filter_order + 1))

        # attributes
        self.return_sources = return_sources
        self.n_harmonics = n_harmonics
        self.output_splits = output_splits
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.return_synth_controls = False
        self.harmonic_roll_off = harmonic_roll_off
        self.f_ref = torch.tensor(f_ref, dtype=torch.float32)
        self.estimate_noise_mag = estimate_noise_mag
        self.return_lsf = False
        self.voiced_unvoiced_diff = voiced_unvoiced_diff

        # neural networks
        overlap = hop_size / fft_size

        if encoder == 'MixEncoderSimple':
            self.encoder = nc.MixEncoderSimple(fft_size=fft_size, overlap=overlap,
                                               hidden_size=encoder_hidden_size, embedding_size=embedding_size,
                                               n_sources=n_sources, bidirectional=bidirectional)

        self.decoder = nc.SynthParameterDecoderSimple(z_size=embedding_size,
                                                      hidden_size=decoder_hidden_size,
                                                      output_size=decoder_output_size,
                                                      bidirectional=bidirectional)
        self.dense_outs = torch.nn.ModuleList([torch.nn.Linear(decoder_output_size, v[1]) for v in output_splits])

        if self.harmonic_roll_off == -2:
            self.gru_roll_off = torch.nn.GRU(decoder_output_size, 1, batch_first=True)
        if self.estimate_noise_mag:
            self.gru_noise_mag = torch.nn.GRU(decoder_output_size, 40, batch_first=True)

        # synth
        self.source_filter_synth = synths.SourceFilterSynth2(n_samples=n_samples,
                                                             sample_rate=16000,
                                                             n_harmonics=n_harmonics,
                                                             audio_frame_size=fft_size,
                                                             hp_cutoff=500,
                                                             f_ref=f_ref,
                                                             estimate_voiced_noise_mag=estimate_noise_mag)


    @classmethod
    def from_config(cls, config: dict):
        keys = config.keys()
        filter_order = config['filter_order'] if 'filter_order' in keys else 10
        harmonic_roll_off = config['harmonic_roll_off'] if 'harmonic_roll_off' in keys else 12
        f_ref = config['f_ref_source_spec'] if 'f_ref_source_spec' in keys else 500
        encoder = config['encoder'] if 'encoder' in keys else 'SeparationEncoderSimple'
        encoder_hidden_size = config['encoder_hidden_size'] if 'encoder_hidden_size' in keys else 256
        embedding_size = config['embedding_size'] if 'embedding_size' in keys else 128
        decoder_hidden_size = config['decoder_hidden_size'] if 'decoder_hidden_size' in keys else 512
        decoder_output_size = config['decoder_output_size'] if 'decoder_output_size' in keys else 512
        n_sources = config['n_sources'] if 'n_sources' in keys else 2
        estimate_noise_mags = config['estimate_noise_mags'] if 'estimate_noise_mags' in keys else False
        bidirectional = not config['unidirectional'] if 'unidirectional' in keys else True
        voiced_unvoiced_diff = not config['voiced_unvoiced_same_noise'] if 'voiced_unvoiced_same_noise' in keys else True

        return cls(filter_order=filter_order,
                   fft_size=config['nfft'],
                   hop_size=config['nhop'],
                   harmonic_roll_off=harmonic_roll_off,
                   estimate_noise_mag=estimate_noise_mags,
                   f_ref=f_ref,
                   encoder=encoder,
                   encoder_hidden_size=encoder_hidden_size,
                   embedding_size=embedding_size,
                   decoder_hidden_size=decoder_hidden_size,
                   decoder_output_size=decoder_output_size,
                   n_sources=n_sources,
                   bidirectional=bidirectional,
                   voiced_unvoiced_diff=voiced_unvoiced_diff)


    def forward(self, audio, f0_hz):
        # audio [batch_size, n_samples]
        # f0_hz [batch_size, n_freq_frames, n_sources]

        z = self.encoder(audio, f0_hz)  # [batch_size, n_frames, n_sources, embedding_size]

        batch_size, n_frames, n_sources, embedding_size = z.shape

        f0_hz = f0_hz.transpose(1, 2)  # [batch_size, n_sources, n_freq_frames]
        f0_hz = torch.reshape(f0_hz, (batch_size*n_sources, -1))  # [batch_size * n_sources, n_freq_frames]

        f0_hz = core.resample(f0_hz, n_frames)

        if self.voiced_unvoiced_diff:
            # use different noise models for voiced and unvoiced frames (this option was not used in the experiments)
            voiced_unvoiced = torch.where(f0_hz > 0., torch.tensor(1., device=f0_hz.device),
                                                  torch.tensor(0., device=f0_hz.device))[:, :, None]
        else:
            # one noise model (this option was used in the experiments for the paper)
            voiced_unvoiced = torch.ones_like(f0_hz)[:, :, None]

        f0_hz = f0_hz[:, :, None]  # [batch_size * n_sources, n_frames, 1]

        z = z.permute(0, 2, 1, 3)
        z = z.reshape((batch_size*n_sources, n_frames, embedding_size))

        x = self.decoder(f0_hz, z)


        outputs = {}
        for layer, (key, _) in zip(self.dense_outs, self.output_splits):
            outputs[key] = layer(x)

        if self.harmonic_roll_off == -1:
            harmonic_roll_off = core.exp_sigmoid(outputs['harmonic_roll_off'], max_value=20.)
        elif self.harmonic_roll_off == -2:
            # constant value for roll off is GRU output of last frame through exponential sigmoid activation
            harmonic_roll_off = core.exp_sigmoid(4 * self.gru_roll_off(x)[0][:, -1, :], max_value=15., exponent=2.)
            harmonic_roll_off = torch.ones_like(outputs['harmonic_amplitude']) * harmonic_roll_off[:, :, None]
        else:
            harmonic_roll_off = torch.ones_like(outputs['harmonic_amplitude']) * self.harmonic_roll_off

        if self.estimate_noise_mag:
            noise_mag = self.gru_noise_mag(x)[0][:, -1, :]
            noise_mag = noise_mag[:, None, :]
        else:
            noise_mag = None

        # return synth controls for insights into how the input sources are reconstructed
        if self.return_synth_controls:
            return self.source_filter_synth.get_controls(outputs['harmonic_amplitude'],
                                                         harmonic_roll_off,
                                                         f0_hz,
                                                         outputs['noise_gain'],
                                                         voiced_unvoiced,
                                                         outputs['line_spectral_frequencies'],
                                                         noise_mag)


        # apply synthesis model
        signal = self.source_filter_synth(outputs['harmonic_amplitude'],
                                          harmonic_roll_off,
                                          f0_hz,
                                          outputs['noise_gain'],
                                          voiced_unvoiced,
                                          outputs['line_spectral_frequencies'],
                                          noise_mag)

        sources = torch.reshape(signal, (batch_size, n_sources, -1))
        mix = torch.sum(sources, dim=1)

        if self.return_sources:
            return mix, sources
        if self.return_lsf:
            lsf = core.lsf_activation(outputs['line_spectral_frequencies'])
            return mix, lsf
        return mix




# -------- U-Net Baselines -------------------------------------------------------------------------------------------
