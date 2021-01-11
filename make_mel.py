import librosa
import os
import numpy as np
import pandas as pd
import argparse
from configparser import ConfigParser
import mel_features
import resampy

def wav_to_mel(filename, parser, model):

	SAMPLE_RATE = parser.getint('mel', 'SAMPLE_RATE')
	LOG_OFFSET = parser.getfloat('mel', 'LOG_OFFSET')
	STFT_WINDOW_LENGTH_SECONDS = parser.getfloat('mel', 'STFT_WINDOW_LENGTH_SECONDS')
	STFT_HOP_LENGTH_SECONDS = parser.getfloat('mel', 'STFT_HOP_LENGTH_SECONDS')
	MEL_MIN_HZ = parser.getint('mel', 'MEL_MIN_HZ')
	MEL_MAX_HZ = parser.getint('mel', 'MEL_MAX_HZ')	

	if(model=='teacher'):
		NUM_BANDS = parser.getint('mel', 'NUM_BANDS_TEACHER')
		NUM_MEL_BINS = NUM_BANDS
	
	else:
		NUM_BANDS = parser.getint('mel', 'NUM_BANDS_STUDENT')
		NUM_MEL_BINS = NUM_BANDS

	y, sr = librosa.load(filename, mono=True, sr=None)

	if y.shape[0]<sr*1 and y.shape[0]>sr*0.0:
		y=librosa.util.fix_length(y, int(sr*1.01))

	y = y.T

	data = y
	sample_rate = sr

	if len(data.shape) > 1:
		data = np.mean(data, axis=1)	
	# Resample to the rate assumed by VGGish.
	if sample_rate != SAMPLE_RATE:
		data = resampy.resample(data, sample_rate, SAMPLE_RATE)	
	# Compute log mel spectrogram features.
	log_mel = mel_features.log_mel_spectrogram(
	        data,
	        audio_sample_rate=SAMPLE_RATE,
	        log_offset=LOG_OFFSET,
	        window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
	        hop_length_secs=STFT_HOP_LENGTH_SECONDS,
	        num_mel_bins=NUM_MEL_BINS,
	        lower_edge_hertz=MEL_MIN_HZ,
	        upper_edge_hertz=MEL_MAX_HZ)	
	
	return log_mel

def extract_mel(annotation_path, dataset_dir, output_dir, parser):
	
	print("* Loading annotations.")
	annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

	os.makedirs(output_dir+'/teacher', exist_ok=True)
	os.makedirs(output_dir+'/student', exist_ok=True)

	teacher_dir = output_dir+'/teacher'
	student_dir = output_dir+'/student'

	df = annotation_data[['split', 'audio_filename']].drop_duplicates()

	counter = 0
	for index, row in df.iterrows():
		filename = row['audio_filename']
		print('({}/{}) {}'.format(counter, len(df), filename))
		partition = row['split']
		audio_path = os.path.join(dataset_dir, partition, filename)
		mel = wav_to_mel(audio_path, parser, 'teacher')
		mel_path = os.path.join(teacher_dir, os.path.splitext(filename)[0] + '.npy')
		np.save(mel_path, mel)
		counter+=1

	counter = 0
	for index, row in df.iterrows():
		filename = row['audio_filename']
		print('({}/{}) {}'.format(counter, len(df), filename))
		partition = row['split']
		audio_path = os.path.join(dataset_dir, partition, filename)
		mel = wav_to_mel(audio_path, parser, 'student')
		mel_path = os.path.join(student_dir, os.path.splitext(filename)[0] + '.npy')
		np.save(mel_path, mel)
		counter+=1



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("annotation_path")
	parser.add_argument("dataset_dir")
	parser.add_argument("output_dir")
	args = parser.parse_args()
	config_parser = ConfigParser()
	config_parser.read('config.ini')

	extract_mel(annotation_path=args.annotation_path,
				dataset_dir=args.dataset_dir,
				output_dir=args.output_dir, parser=config_parser)



