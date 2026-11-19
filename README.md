# How to Run

First build the docker container from the project directory:

```
docker build -t mso-model .
```

Then, to run:

```
docker run -it --rm -v "$PWD":/app mso-model
python examples/calc_amp_mod.py --plot
```

# Glossary

$Delta R$ = difference in firing rate between the left and right MSO neurons (in spikes per second), defined as right - left.
sps = spikes per second
