# app.py — One-click CUDA -> HIP (PoC)
import streamlit as st
from pathlib import Path
import re

# même mapping que dans axiomify.py (tu peux copier-coller le tien)
CUDA_TO_HIP = {
    r'#\s*include\s*<cuda_runtime\.h>': '#include <hip/hip_runtime.h>',
    r'\bcudaMalloc\b':'hipMalloc', r'\bcudaFree\b':'hipFree',
    r'\bcudaMemcpyAsync\b':'hipMemcpyAsync', r'\bcudaMemcpy\b':'hipMemcpy',
    r'\bcudaMemcpyHostToDevice\b':'hipMemcpyHostToDevice',
    r'\bcudaMemcpyDeviceToHost\b':'hipMemcpyDeviceToHost',
    r'\bcudaMemcpyDeviceToDevice\b':'hipMemcpyDeviceToDevice',
    r'\bcudaDeviceSynchronize\b':'hipDeviceSynchronize',
    r'\bcudaGetLastError\b':'hipGetLastError',
    r'\bcudaGetErrorString\b':'hipGetErrorString',
    r'\bcudaStream_t\b':'hipStream_t',
    r'\bcudaStreamCreate\b':'hipStreamCreate',
    r'\bcudaStreamSynchronize\b':'hipStreamSynchronize',
    r'\bcudaStreamDestroy\b':'hipStreamDestroy',
    r'\bcudaEvent_t\b':'hipEvent_t',
    r'\bcudaEventCreate\b':'hipEventCreate',
    r'\bcudaEventRecord\b':'hipEventRecord',
    r'\bcudaEventSynchronize\b':'hipEventSynchronize',
    r'\bcudaEventDestroy\b':'hipEventDestroy',
    r'\bcudaError_t\b':'hipError_t', r'\bcudaSuccess\b':'hipSuccess',
}

def translate_text(src: str) -> str:
    out = src
    for pat, repl in CUDA_TO_HIP.items():
        out = re.sub(pat, repl)
    return out

st.title("AXIOM OS — CUDA → HIP Translator (PoC)")
uploaded = st.file_uploader("Upload a .cu file", type=["cu"])

if uploaded:
    src = uploaded.read().decode("utf-8", errors="ignore")
    if st.button("Translate CUDA → AMD (HIP)"):
        dst = translate_text(src)
        st.success("Translation complete.")
        st.code(dst[:1000], language="cpp")  # aperçu
        st.download_button("Download translated file",
                           data=dst,
                           file_name=Path(uploaded.name).with_suffix(".hip.cpp").name,
                           mime="text/x-c++src")
else:
    st.info("Upload a CUDA .cu file to enable the button.")
