FROM python:2.7

# Install numpy/scipy FIRST (separate layer), so brian's setup.py can import them.
RUN pip install --no-cache-dir \
    numpy==1.16.6 \
    scipy==1.2.3 \
    matplotlib==2.2.5 \
    pandas==0.24.2 \
    cython==0.23.4 \
    numexpr==2.6.8

# Build the legacy Python-2 packages WITHOUT pip's build isolation,
# because brian's setup.py does "import numpy" at build time.
RUN pip install --no-cache-dir --no-build-isolation \
    brian==1.4.4 \
    cochlea \
    thorns

WORKDIR /app
COPY . /app
RUN pip install -e .

CMD ["bash"]
