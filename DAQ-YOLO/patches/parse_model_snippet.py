# Add this branch to ultralytics/nn/tasks.py::parse_model before its generic else.
# Also import: from daq_yolo.modules.d_ema import D_EMA

elif m is D_EMA:
    c1 = ch[f]
    args = [c1, *args]
    c2 = c1
