import numpy as np, matplotlib.pyplot as plt
from numpy import *
from scipy import signal
from scipy import interpolate
import sounddevice as sd
import queue
import threading
from time import sleep
import matplotlib.cm as cm

import bokeh.plotting as bk
from bokeh.models import GlyphRenderer
from bokeh.io import push_notebook
from IPython.display import clear_output

bk.output_notebook()

def put_data(Qout, ptrain, stop_flag):
    while( not stop_flag.is_set() ):
        if ( Qout.qsize() < 2 ):
            Qout.put( ptrain )
    
def signal_process(Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag):
    # Signal processing function for real-time sonar
    # Takes in streaming data from Qin and process them using the functions defined above
    # Uses the first 2 pulses to calculate for delay
    # Then for each Nseg segments calculate the cross correlation (uses overlap-and-add)
    # Inputs:
    #     Qin - input queue with chunks of audio data
    #     Qdata - output queue with processed data
    #     pulse_a - analytic function of pulse
    #     Nseg - length between pulses
    #     Nplot - number of samples for plotting
    #     fs - sampling frequency
    #     maxdist - maximum distance
    #     temperature - room temperature

    crossCorr = functions[2]
    findDelay = functions[3]
    dist2time = functions[4]
    
    # initialize Xrcv 
    Xrcv = zeros(2*Nseg, dtype='complex')
    cur_idx = 0 # keeps track of current index
    found_delay = False
    if ( int(dist2time(maxdist, temperature) * fs) > Nseg ):
        print("Warning: maxdist exceeds maximum distance allowed by Nseg and fs!")
        maxsamp = Nseg # replace maxsamp with Nseg
    else:
        maxsamp = int(dist2time(maxdist, temperature) * fs) # maximum samples corresponding to maximum distance
    
    while( not stop_flag.is_set() ):
        # get streaming chunk
        chunk = Qin.get()
        chunk = chunk.reshape((len(chunk),))
#         if ( sum( (chunk==chunk.max()) | (chunk==chunk.min()) ) / len(chunk) > 0.05 ):
#             print("Warning: input is clipping!")

        Xchunk = crossCorr(chunk, pulse_a) 
        
        # overlap-and-add
        Xrcv[cur_idx:(cur_idx+len(chunk)+len(pulse_a)-1)] += Xchunk
        cur_idx += len(chunk)
            
        idx = findDelay(abs(Xrcv), Nseg)

        Xrcv = roll(Xrcv, -idx)
        Xrcv[-idx:] = 0.0

        # crop a segment from Xrcv and interpolate to Nplot
        Xrcv_seg = abs(Xrcv[:maxsamp].copy()) / abs(Xrcv[0])
        interp = interpolate.interp1d(r_[:maxsamp], Xrcv_seg)
        Xrcv_seg = interp(linspace(0.0, maxsamp-1, Nplot))

        # remove segment from Xrcv
        Xrcv = roll(Xrcv, -Nseg);
        Xrcv[-Nseg:] = 0.0
        cur_idx = 0
        
        Qdata.put( Xrcv_seg )
            
def image_update(Qdata, fig, maxrep, Nplot, dBf, stop_flag):
    renderer = fig.select(dict(name="echoes", type=GlyphRenderer))
    source = renderer[0].data_source
    img = source.data["image"][0]
    
    while( not stop_flag.is_set() ):
        Xrcv_seg = Qdata.get()
        
        # convert Xcrv_seg to dBf
        Xrcv_seg_dB = 10.0*log10(Xrcv_seg)
        
        # enforce minimum dBf
        Xrcv_seg_dB[Xrcv_seg_dB < -dBf] = -dBf
        
        # convert to image intensity out of 1
        new_line = (Xrcv_seg_dB + dBf)/dBf
         
        img = np.roll(img, 1, 0)
        view = img.view(dtype=np.uint8).reshape((maxrep, Nplot, 4))
        view[0,:,:] = cm.gray(new_line) * 255
        
        source.data["image"] = [img]
        push_notebook()
        Qdata.queue.clear()

def rtsonar(f0, f1, fs, Npulse, Nseg, maxrep, Nplot, dBf, maxdist, temperature, functions):
    
    def audio_callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        Qin.put( indata )  # Global queue

        try:
            data = Qout.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)

        outdata[:] = data.reshape((len(data),1))
    
    Nrep = 1
    clear_output()
    genChirpPulse = functions[0]
    genPulseTrain = functions[1]
    
    pulse_a = genChirpPulse(Npulse, f0, f1, fs) * signal.hann(Npulse)
    pulse = pulse_a.real
    ptrain = genPulseTrain(pulse, Nrep, Nseg)
    
    # create input output FIFO queues
    Qin = queue.Queue()
    Qout = queue.Queue()
    Qdata = queue.Queue()
    
    img = zeros((maxrep,Nplot), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((maxrep, Nplot, 4))
    view[:,:,3] = 255

    # initialize plot
    fig = bk.figure(title="Sonar",  y_axis_label="Time [s]", x_axis_label="Distance [m]",
                    x_range=(0, maxdist/100), y_range=(0, maxrep*Nseg/fs), 
                    plot_height=400, plot_width=800)
    fig.image_rgba(image=[img], x=[0], y=[0], dw=[maxdist/100], dh=[maxrep*Nseg/fs], name="echoes")
    bk.show(fig, notebook_handle=True)
    
    # initialize stop_flag
    stop_flag = threading.Event()   
    
    # initialize threads
    t_put_data = threading.Thread(target=put_data, args=(Qout, ptrain, stop_flag))
    st = sd.Stream(device=(1,1), samplerate=fs, blocksize=len(ptrain), channels=1, callback=audio_callback)
    t_signal_process = threading.Thread(target=signal_process, args=(Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag))
    t_image_update = threading.Thread(target=image_update, args=(Qdata, fig, maxrep, Nplot, dBf, stop_flag))

    # start threads
    t_put_data.start()
    st.start()
    t_signal_process.start()
    t_image_update.start()
    
    return (stop_flag, st)
