import whisper
from whisper import _MODELS

import torch
import openvino as ov


from typing import Optional, Tuple
from functools import partial

model = whisper.load_model("base", "cpu")
model.eval()
pass

from pathlib import Path

WHISPER_ENCODER_OV = Path(f"dnd_models/ggml-base-encoder-openvino.xml")
WHISPER_DECODER_OV = Path(f"dnd_models/whisper_base_decoder.xml")

core = ov.Core()
from utils import patch_whisper_for_ov_inference, OpenVINOAudioEncoder, OpenVINOTextDecoder


patch_whisper_for_ov_inference(model)

model.encoder = OpenVINOAudioEncoder(core, WHISPER_ENCODER_OV, device="NPU")
model.decoder = OpenVINOTextDecoder(core, WHISPER_DECODER_OV, device="GPU")
print("Whisper Model loaded..")



from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import argparse
import os
import numpy as np
import signal
from threading import Thread
import speech_recognition as sr
import audioop






def record_callback(_, audio:sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)


import socket

HOST = "127.0.0.1"   # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

host_energy = "127.0.0.1"
port_energy = 65433

s_energy = socket.socket(socket.AF_INET, 
                      socket.SOCK_STREAM)
                      
s_energy.bind((host_energy, port_energy))

s_energy.listen()                      

def realtime_transcribe(phrase_time,data_queue,recorder,source,record_timeout,phrase_timeout,transcription,result_queue):

    global stop
    
    
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    c, addr = s_energy.accept() 
    c.sendall(b"speak")
    print("Listen in background",stop)
    while True:
            
        #print("Inside while loop:", stop)      
        now = datetime.utcnow()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():
            print("data queue")
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            #if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
            #    phrase_complete = True
            # This is the last time we received new audio data from the queue.
            #phrase_time = now
            
            # Combine audio data from queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()
            
            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            #energy = audioop.rms(audio_np, 4)
            c.sendall(b"continue")
            # Read the transcription.
            result = model.transcribe(audio_np, task="transcribe")
            
            #print("length of audio_np", audio_np)
            #wave = main_note(audio_data)
            #print(wave)
            text = result['text'].strip()

            # If we detected a pause between recordings, add a new item to our transcription.
            # Otherwise edit the existing one.
            #if phrase_complete:
            transcription.append(text)
            #else:
            #transcription[-1] = text
            #print(transcription)

            # Clear the console to reprint the updated transcription.
            #os.system('cls' if os.name=='nt' else 'clear')
            #for line in transcription:
            #    print(line)
            # Flush stdout.
            #print('', end='', flush=True)

            # Infinite loops are bad for processors, must sleep.
            sleep(0.1)
        if stop:
            
            # calling this function requests that the background listener stop listening
            stop_listening(wait_for_stop=False)
            c.sendall(b"stop_speak")
            print("exiting loop from transcribe and background listen is stopped",stop)
            break
    
    sentence = " ".join(transcription)
    result_queue.put(sentence)
    #print("In transcribe",result_queue.get())
    
 
 

   


with (socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s):
    s.bind((HOST, PORT))
    s.listen()
    
 
    print("Ready")
    while True:
        conn, addr = s.accept()
        print("connection accepted")
        with conn:

       
            
            data = conn.recv(1024)
        
            if data.decode() == "kill":
                os._exit(0)
            if data.decode() == "start":
                conn.sendall(data)

                 # The last time a recording was retrieved from the queue.
                phrase_time = None
                # Thread safe Queue for passing data from the threaded recording callback.
                data_queue = Queue()
                result_queue = Queue()  
                # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
                recorder = sr.Recognizer()
                recorder.energy_threshold = 300
                # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
                
                source = sr.Microphone(sample_rate=16000)

                record_timeout = 2
                phrase_timeout = 3
                transcription = ['']


                with source:
                    print("Please wait. Calibrating microphone...")  
                    recorder.adjust_for_ambient_noise(source)
                    recorder.dynamic_energy_threshold = True

                stop = False
                                          
                background_thread = Thread(target=realtime_transcribe, args=(phrase_time,data_queue,recorder,source,record_timeout,phrase_timeout,transcription,result_queue))
                print("in start")

                background_thread.start()
                                    
            if data.decode() == "stop":
                print("exiting loop")
                stop = True
                #conn.sendall(b"exiting loop")
                conn.sendall(result_queue.get().encode())
                print("FINAL")
                
                # calling this function requests that the background listener stop listening
              
    
